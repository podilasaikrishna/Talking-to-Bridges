"""Tool Dispatcher for the Talking to Bridges LLM/Agent layer (Phase 2 - Week 6).

Responsibilities
----------------
1. Intent detection: map a plain-English query to a registered tool name +
   best-guess arguments using keyword heuristics (no LLM required).
2. Dataset resolution: find the active uploaded DataFrame (if any) so tools
   receive real data rather than placeholder results.
3. Tool invocation: delegate cleanly to the ToolRegistry and return a
   JSON-serializable result envelope.
4. Fallback: return None when no analytical intent is detected so the
   caller (routes.py) can fall back to pure RAG retrieval.

Architecture
------------

User query (plain text)
    |
    v
ToolDispatcher.detect_intent(query, df?)
    |
    +-- match found --> ToolDispatcher.dispatch(tool_name, args, df?)
    |                       |
    |                       v
    |                  ToolRegistry.execute(name, args, df)
    |                       |
    |                       v
    |                  handler(args, df) -> structured dict
    |                       |
    |                       v
    |                  DispatchResult(tool, status, data, llm_summary)
    |
    +-- no match --> None (caller uses pure RAG)

This module does NOT import routes.py (no circular dependency).
The routes.py file imports from here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.logging_config import logger
from analysis.tools import data_access, registry


# ---------------------------------------------------------------------------
# Result envelope
# ---------------------------------------------------------------------------

@dataclass
class DispatchResult:
    """Returned by ToolDispatcher.dispatch().

    Attributes:
        tool_name:   Name of the executed tool.
        status:      "success" or "error".
        data:        Dict from the tool handler (on success).
        error_type:  Error code string (on failure).
        message:     Human-readable description of the result/error.
        metadata:    Category and arguments used.
        llm_summary: A short prose summary injected into the LLM context block.
    """

    tool_name: str
    status: str
    data: Optional[Dict[str, Any]] = None
    error_type: Optional[str] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    llm_summary: str = ""


# ---------------------------------------------------------------------------
# Intent detection rules
# ---------------------------------------------------------------------------

# Each rule is a (tool_name, keyword_set) tuple.
# The first rule whose keywords intersect the query terms wins.
_INTENT_RULES: List[tuple] = [
    # Chart / visualization  (check BEFORE statistics to avoid 'plot average')
    # NOTE: 'show' and 'display' are intentionally omitted because they appear
    # in many non-chart queries (e.g. 'show statistics', 'show trend').
    ("chart_data", {
        "chart", "plot", "graph", "visualize", "visualise",
        "draw", "figure",
    }),
    # Anomaly
    ("anomaly_detection", {
        "anomaly", "anomalies", "outlier", "outliers",
        "alert", "spike", "spikes", "abnormal", "unusual",
    }),
    # Trend
    ("trend_analysis", {
        "trend", "trends", "increasing", "decreasing", "rising",
        "falling", "drift", "change over time", "slope",
    }),
    # Correlation
    ("correlation_analysis", {
        "correlat", "correlation", "relationship", "relation",
        "compare sensors", "compare columns",
    }),
    # Baseline / reference state
    ("baseline", {
        "baseline", "healthy", "healthy baseline", "deviation",
        "deviation ranking", "normal baseline", "reference state",
    }),
    # Prediction / model
    ("model_result", {
        "predict", "condition", "classify", "classification",
        "model", "forecast", "ml", "damage level", "damage class", "severity",
    }),
    # Summary statistics  (broadest, check LAST)
    ("summary_statistics", {
        "stat", "statistic", "statistics", "average", "mean",
        "min", "minimum", "max", "maximum", "std", "standard deviation",
        "summary", "describe", "distribution",
    }),
]

# Metric keywords matched against column names in the active dataset or known aliases
_SENSOR_ALIASES = {
    "deflection": "deflection",
    "vibration":  "vibration",
    "stress":     "stress",
    "temperature": "temperature",
    "crack":      "crack_width",
    "sensor1":    "Sensor_1",
    "sensor_1":   "Sensor_1",
    "sensor2":    "Sensor_2",
    "sensor_2":   "Sensor_2",
    "sensor3":    "Sensor_3",
    "sensor_3":   "Sensor_3",
    "sensor4":    "Sensor_4",
    "sensor_4":   "Sensor_4",
    "sensor5":    "Sensor_5",
    "sensor_5":   "Sensor_5",
}


# Comparison charts (healthy vs uploaded). A comparison word is required; the other
# words only pick which comparison chart is meant.
_COMPARE_WORDS = {"healthy", "baseline", "compare", "comparison", "compared", "versus", "vs", "against"}
_FEATURE_WORDS = {
    "feature", "features", "rms", "kurtosis", "skewness", "variance", "crest", "fft",
    "deviation", "deviations", "deviate", "range",
}
_HEATMAP_WORDS = {"heatmap", "heat"}
_SPECTRUM_WORDS = {"spectrum", "spectra", "spectral", "fft", "peaks", "peak"}
_FREQUENCY_WORDS = {"frequency", "frequencies", "modal", "natural", "f1", "f2", "f3", "mode", "modes"}


_SENSOR_PHRASE = re.compile(r"\bsensor[\s_-]*([1-9])\b")


def _named_sensor(query: str, df: Optional[pd.DataFrame]) -> Optional[str]:
    """'sensor 3' / 'Sensor-3' typed in a query -> the 'Sensor_3' column, when it exists."""
    match = _SENSOR_PHRASE.search(query.lower())
    if match and df is not None and f"Sensor_{match.group(1)}" in df.columns:
        return f"Sensor_{match.group(1)}"
    return None


def _detect_comparison(tokens: List[str]) -> Optional[str]:
    """'signals' | 'features' | 'heatmap' | 'spectrum' | 'frequencies' when the query asks to compare with the healthy state."""
    words = set(tokens)
    if not words & _COMPARE_WORDS:
        return None
    if words & _HEATMAP_WORDS:
        return "heatmap"
    if words & _SPECTRUM_WORDS:
        return "spectrum"
    if words & _FREQUENCY_WORDS:
        return "frequencies"
    if words & _FEATURE_WORDS:
        return "features"
    return "signals"


def _tokenize(text: str) -> List[str]:
    """Lower-case, alphanumeric + underscore tokenization."""
    return re.findall(r"[a-z0-9_]+", text.lower())


def _extract_metric(tokens: List[str], df: Optional[pd.DataFrame]) -> Optional[str]:
    """Return the best matching metric column from the query tokens.

    Priority:
    1. Any token that is a direct column name in df.
    2. Any token matching a known alias.
    3. First numeric column in df (fallback).
    """
    # Build alias map from the actual df columns too
    if df is not None:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        for tok in tokens:
            if tok in numeric_cols:
                return tok
            # Match via known aliases
            alias = _SENSOR_ALIASES.get(tok)
            if alias and alias in df.columns:
                return alias
        if numeric_cols:
            return numeric_cols[0]

    # Fallback without a df
    for tok in tokens:
        alias = _SENSOR_ALIASES.get(tok)
        if alias:
            return alias

    return None


def _extract_sensor_id(tokens: List[str], df: Optional[pd.DataFrame]) -> Optional[str]:
    """Extract a sensor_id from the query (e.g. 's01', 's1', 'sensor 1')."""
    # Look for patterns like s01, s1, sensor1 in tokenized query
    sensor_id_pattern = re.compile(r"^s\d+$")
    for tok in tokens:
        if sensor_id_pattern.match(tok):
            return tok.upper()
    return None


def _detect_tool(query: str) -> Optional[str]:
    """Return the matched tool name or None."""
    tokens = set(_tokenize(query))
    for tool_name, keywords in _INTENT_RULES:
        # Support multi-word keywords like "change over time"
        for kw in keywords:
            kw_tokens = set(_tokenize(kw))
            if kw_tokens.issubset(tokens) or any(tok.startswith(kw) for tok in tokens):
                # A spectrum / peaks question about the healthy state wants the chart (NEW-3).
                if tool_name == "baseline" and tokens & _SPECTRUM_WORDS:
                    return "chart_data"
                return tool_name
    return None


# ---------------------------------------------------------------------------
# NEW-4: routing for the professor's five questions
# ---------------------------------------------------------------------------
#   1 What is the current condition?          -> model_result
#   2 Is the structure damaged?               -> model_result
#   3 What parameters indicate the damage?    -> model_result (top_parameters)
#   4 What is the estimated severity?         -> model_result
#   5 How does it compare with the healthy    -> baseline + model_result + comparison chart
#     condition?  ("show me the comparison")
# Queries that clearly ask for another tool (chart words, anomalies, trend, correlation,
# statistics) keep their old routing.

_CHART_WORDS = {"chart", "plot", "graph", "visualize", "visualise", "draw", "figure"}
_OTHER_TOOL_WORDS = {
    "anomaly", "anomalies", "outlier", "outliers", "spike", "spikes",
    "trend", "trends", "slope", "drift", "increasing", "decreasing", "rising", "falling",
    "correlation", "correlate", "correlations", "relationship",
    "statistics", "statistic", "average", "variance", "std", "minimum", "maximum",
}
_ASSESS_WORDS = {
    "condition", "conditions", "damaged", "damage", "damages", "undamaged",
    "severity", "severe", "severely", "serious", "seriously",
    "crack", "cracks", "cracked", "health", "status", "state", "safe", "safety",
    "verdict", "diagnosis", "diagnose", "assessment", "assess",
    "classify", "classification", "classified", "predict", "prediction", "predicted",
    "estimate", "estimated", "indicate", "indicates", "indicating", "indicator", "indicators",
    "parameter", "parameters", "model",
}
# Questions about documents or explanations ("what does the paper say about cracks",
# "why does a frequency drop indicate damage") stay with document retrieval.
_EXPLAIN_WORDS = {
    "document", "documents", "paper", "pdf", "article", "study", "literature", "theory",
    "explain", "explains", "why", "define", "definition", "meaning", "mean", "means",
    "affect", "affects", "cause", "causes", "caused", "typically", "generally",
}
# A comparison question that names a chart type gets that chart as its primary result.
_SIGNAL_WORDS = {"signal", "signals", "waveform", "overlay"}
_COMPARE_VERBS = {"compare", "compared", "comparing", "comparison", "comparisons", "versus", "vs", "against"}
_HEALTHY_WORDS = {"healthy", "baseline", "undamaged", "reference", "normal"}
_IS_HEALTHY = re.compile(r"\b(is|are|was|were)\b[^?.!]*\bhealthy\b|\bhow healthy\b")


def _plan_tools(query: str, has_data: bool) -> Optional[List[Dict[str, Any]]]:
    """Ordered tool plan for the five assessment questions, or None for the old routing.

    Each entry: {"tool": name, "extra": {forced arguments}}.
    """
    token_list = _tokenize(query)
    tokens = set(token_list)
    if tokens & _CHART_WORDS or tokens & _OTHER_TOOL_WORDS or tokens & _EXPLAIN_WORDS:
        return None

    compare = tokens & _COMPARE_VERBS
    healthy = tokens & _HEALTHY_WORDS
    assess = tokens & _ASSESS_WORDS
    baseline_words = tokens & {"baseline", "deviation", "deviations", "ranking", "reference"}
    asks_is_healthy = bool(_IS_HEALTHY.search(query.lower())) and not baseline_words

    if compare and (healthy or assess or tokens & {"comparison", "comparisons"}):
        if not has_data:
            return [{"tool": "model_result", "extra": {}}]
        named = _detect_comparison(token_list)
        chart_named = (named not in (None, "signals")) or bool(tokens & _SIGNAL_WORDS)
        baseline_step = {"tool": "baseline", "extra": {}}
        model_step = {"tool": "model_result", "extra": {}}
        chart_step = {"tool": "chart_data", "extra": {"_comparison_default": "frequencies"}}
        if chart_named:  # e.g. "compare the spectrum with healthy": the chart leads
            return [chart_step, baseline_step, model_step]
        return [baseline_step, model_step, chart_step]
    if assess or asks_is_healthy:
        return [{"tool": "model_result", "extra": {}}]
    return None


# ---------------------------------------------------------------------------
# NEW-4: plain-language summary of the model_result tool (what the LLM sees)
# ---------------------------------------------------------------------------

_FEATURE_NAMES = {
    "f1_hz": "mode 1 frequency (f1)",
    "f2_hz": "mode 2 frequency (f2)",
    "f3_hz": "mode 3 frequency (f3)",
    "f2_over_f1": "frequency ratio f2/f1",
    "f3_over_f1": "frequency ratio f3/f1",
}


def _param_line(p: Dict[str, Any]) -> str:
    feature = p.get("feature", "")
    name = _FEATURE_NAMES.get(feature, feature)
    unit = " Hz" if feature.endswith("_hz") else ""
    parts = [name]
    if p.get("value") is not None and p.get("healthy_typical") is not None:
        parts.append(f"{p['value']:.4g}{unit} against a healthy {p['healthy_typical']:.4g}{unit}")
    if p.get("change_percent") is not None:
        change = round(p["change_percent"], 1) + 0.0  # avoids "-0.0"
        parts.append(f"{change:+.1f} percent")
    if p.get("outside") is True:
        parts.append("clearly different from healthy")
    elif p.get("outside") is False:
        parts.append("within normal healthy variation")
    return "; ".join(parts)


_ANSWER_GUIDE = (
    "How to answer: use plain language for a non-expert, 3 to 6 sentences, and only the numbers "
    "above. Say the condition and how sure the model is, name the main parameters and what "
    "changed against healthy, give the severity, and mention one limit. Do not use the words "
    "'tool' or 'JSON', and do not quote importance scores or 'spreads'. Only call a parameter "
    "an indicator of damage when it is marked 'clearly different from healthy'; say small "
    "changes marked 'within normal healthy variation' are normal. Do not claim that one "
    "frequency change proves a particular crack size.\n"
    "Report the direction of every change exactly as given above (higher or lower); in the lab "
    "data a frequency can move either way when a crack is present, so never claim that damage "
    "must lower a frequency, and do not explain the change with a physical cause unless the "
    "numbers above support it.\n"
    "Example (damaged): 'The beam looks damaged; the model is [X] percent sure. Its [feature] is "
    "[Y] percent [higher/lower] than the healthy value, and the healthy comparison also flags "
    "it as different. The estimated severity is [low/medium/high] (a [N] mm crack), with [Z] "
    "percent confidence. This only applies to this lab beam.'\n"
    "Example (healthy): 'The beam looks undamaged; the model is [X] percent sure, and the bending "
    "frequencies are inside the healthy range. There is no severity because no damage was found. "
    "Very small cracks near the free end can be hard to see, though.' Replace every [bracket] "
    "with the real numbers above; never reuse example numbers."
)


def _model_result_summary(data: Dict[str, Any]) -> str:
    """Plain-language block for the assessment questions (Interface C)."""
    lines = ["[ANALYTICAL RESULT: model_result]"]
    status = data.get("status", "ok")
    if status != "ok":
        lines.append(f"No assessment was possible: {data.get('message', '')}")
        lines.append(
            "Tell the user this plainly and, if no file is uploaded, ask them to upload an "
            "Excel sensor recording (.xlsx or .xlsm). Do not guess a condition."
        )
        return "\n".join(lines)

    baseline = data.get("baseline_comparison") or {}
    flag = baseline.get("overall_flag", "unavailable")
    n_h = baseline.get("n_healthy") or {}

    if data.get("model_available"):
        cc = data.get("condition_confidence")
        conf = f" (confidence {cc * 100:.0f} percent)" if cc is not None else ""
        lines.append(f"Condition (trained model): {data.get('condition')}{conf}")
        if data.get("condition") == "Damaged":
            sc = data.get("severity_confidence")
            sconf = f" (confidence {sc * 100:.0f} percent)" if sc is not None else ""
            if data.get("severity_label"):
                lines.append(f"Estimated severity: {data['severity_label']}{sconf}")
            else:
                lines.append("Estimated severity: the model gave no severity for this recording.")
        else:
            lines.append("Estimated severity: none, because no damage was found.")
    else:
        lines.append(
            "Condition (trained model): NOT AVAILABLE. "
            f"{data.get('model_error') or 'The model could not be used.'} "
            "Say that the damage model is not available and answer from the healthy comparison only."
        )

    params = data.get("top_parameters") or []
    if params:
        lines.append("Parameters that stand out:")
        lines.extend(f"  - {_param_line(p)}" for p in params[:6])

    if flag == "unavailable":
        lines.append(f"Healthy comparison: unavailable. {baseline.get('summary', '')}")
    else:
        lines.append(f"Healthy comparison: overall flag {flag}. {baseline.get('summary', '')}")
        if n_h:
            lines.append(
                f"The healthy baseline uses {n_h.get('recordings', 0)} healthy recording(s) "
                f"from {n_h.get('cases', 0)} case(s), so the healthy range is uncertain."
            )

    if data.get("agreement") == "disagree":
        lines.append(f"IMPORTANT, the results disagree: {data.get('agreement_note', '')} "
                     "Say this plainly and report both results.")
    elif data.get("agreement") == "agree":
        lines.append(f"Consistency: {data.get('agreement_note', '')}")

    caveats = data.get("caveats") or []
    if caveats:
        lines.append("Limits to mention (pick the most relevant one): " + " ".join(caveats))
    lines.append(_ANSWER_GUIDE)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """Routes user queries to the correct analytical tool.

    This is the bridge between the LLM/chat pipeline and the ToolRegistry.
    The LLM does not call this directly; routes.py calls detect_and_dispatch()
    before building the LLM prompt.
    """

    def __init__(self):
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_and_dispatch(
        self,
        query: str,
        df: Optional[pd.DataFrame] = None,
    ) -> Optional[DispatchResult]:
        """Detect analytical intent in query and, if found, execute the tool.

        Returns the first (primary) DispatchResult, or None when no tool matches.
        Use detect_and_dispatch_all() to get every result for the question
        (for example the baseline text AND the comparison chart).
        """
        results = self.detect_and_dispatch_all(query, df)
        return results[0] if results else None

    def detect_and_dispatch_all(
        self,
        query: str,
        df: Optional[pd.DataFrame] = None,
    ) -> List[DispatchResult]:
        """Run every tool the question needs, in order (empty list when none match).

        The five assessment questions (condition, damaged?, parameters, severity,
        comparison with healthy) use _plan_tools; everything else uses the single-tool
        keyword routing as before.
        """
        plan = _plan_tools(query, has_data=df is not None)
        if plan is None:
            tool_name = _detect_tool(query)
            if tool_name is None:
                return []
            plan = [{"tool": tool_name, "extra": {}}]

        results: List[DispatchResult] = []
        for step in plan:
            arguments = self._build_arguments(step["tool"], query, df)
            default_mode = step["extra"].get("_comparison_default")
            if step["tool"] == "chart_data" and default_mode:
                mode = arguments.get("comparison")
                explicit_signals = bool(set(_tokenize(query)) & _SIGNAL_WORDS)
                # no chart type named in the question -> the f1-f3 bars (NEW-3)
                if mode is None or (mode == "signals" and not explicit_signals):
                    arguments["comparison"] = default_mode
            results.append(self.dispatch(step["tool"], arguments, df))
        return results

    def dispatch(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
    ) -> DispatchResult:
        """Execute tool_name with arguments and wrap the result in a DispatchResult.

        Args:
            tool_name:  Registered tool name.
            arguments:  Dict of tool arguments (validated by the registry).
            df:         Optional DataFrame to pass to the tool handler.

        Returns:
            DispatchResult with status, data or error, and llm_summary.
        """
        raw = self._registry.execute(tool_name, arguments, df=df)

        if raw["status"] == "success":
            summary = self._build_llm_summary(tool_name, raw["data"], arguments)
            return DispatchResult(
                tool_name=tool_name,
                status="success",
                data=raw["data"],
                message=raw.get("message", ""),
                metadata=raw.get("metadata", {}),
                llm_summary=summary,
            )
        else:
            return DispatchResult(
                tool_name=tool_name,
                status="error",
                error_type=raw.get("error_type"),
                message=raw.get("message", "Tool execution failed."),
                metadata={},
                llm_summary=(
                    f"[Tool Error: {raw.get('error_type', 'UNKNOWN')}] "
                    f"{raw.get('message', '')}"
                ),
            )

    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Return all registered tool definitions (JSON-safe)."""
        return self._registry.list_tools()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_arguments(
        self,
        tool_name: str,
        query: str,
        df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Build the best-guess argument dict for tool_name from the query."""
        tokens = _tokenize(query)
        metric = _extract_metric(tokens, df)
        sensor_id = _extract_sensor_id(tokens, df)

        if tool_name == "summary_statistics":
            args: Dict[str, Any] = {"metric": metric or "deflection"}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "anomaly_detection":
            # Look for threshold hints in query (e.g. "threshold 3", "3 sigma")
            threshold = 5.0
            for tok in tokens:
                try:
                    v = float(tok)
                    if 0.5 <= v <= 10.0:
                        threshold = v
                        break
                except ValueError:
                    pass
            args = {"metric": metric or "vibration", "threshold": threshold}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "trend_analysis":
            args = {"metric": metric or "deflection"}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "correlation_analysis":
            # If specific columns mentioned, use them; else use all numeric cols
            args = {}
            if sensor_id:
                args["sensor_id"] = sensor_id
            return args

        if tool_name == "model_result":
            return {
                "sensor_id": sensor_id or "S01",
                "target_variable": "condition",
            }

        if tool_name == "baseline":
            args = {}
            if sensor_id:
                args["sensor_id"] = sensor_id
            if metric:
                args["metric"] = metric
            return args

        if tool_name == "chart_data":
            y_col = metric or "deflection"
            # X column: prefer Relative_Time_Sec if available in df
            x_col = "timestamp"
            if df is not None and "Relative_Time_Sec" in df.columns:
                x_col = "Relative_Time_Sec"
            y_col = _named_sensor(query, df) or y_col
            args = {"y_col": y_col, "x_col": x_col}
            comparison = _detect_comparison(tokens)
            if comparison:
                args["comparison"] = comparison
            return args

        return {}

    def _build_llm_summary(
        self,
        tool_name: str,
        data: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> str:
        """Build a concise prose summary of the tool result for injection into
        the LLM context.  The LLM uses this to compose its natural-language
        reply to the user.
        """
        metric = data.get("metric") or arguments.get("metric") or arguments.get("y_col", "")
        sensor = data.get("sensor_id") or arguments.get("sensor_id", "")
        sensor_str = f" for sensor '{sensor}'" if sensor else ""

        if tool_name == "summary_statistics":
            return (
                f"[ANALYTICAL RESULT: summary_statistics]\n"
                f"Metric: {metric}{sensor_str}\n"
                f"Count: {data.get('count')}  |  "
                f"Min: {data.get('min')}  |  Max: {data.get('max')}  |  "
                f"Mean: {data.get('mean')}  |  Median: {data.get('median')}  |  "
                f"Std: {data.get('std')}  |  RMS: {data.get('rms')}  |  "
                f"Peak-to-Peak: {data.get('p2p')}\n"
                f"Note: {data.get('_note', 'Real data result.')}"
            )

        if tool_name == "anomaly_detection":
            event_count = data.get("event_count", 0)
            max_abs_z = data.get("max_abs_z")
            events = data.get("events", [])

            lines = [
                "[ANALYTICAL RESULT: anomaly_detection]",
                f"Metric: {metric}{sensor_str}  |  Z-score threshold: {data.get('threshold_z')}",
                f"Total rows checked: {data.get('total_rows_checked')}  |  Anomalies found: {data.get('anomaly_count')} ({data.get('anomaly_fraction', 0)*100:.1f}%)",
                f"Status: {data.get('status_flag')}",
            ]
            if max_abs_z is not None:
                lines.append(f"Peak score: {max_abs_z} (max z-score)")
            if event_count > 0 or events:
                lines.append(f"Event count: {event_count} detected event(s)")
                if events:
                    ev_previews = []
                    for ev in events[:3]:
                        pz = ev.get("peak_z", ev.get("peak_score", ""))
                        pv = ev.get("peak_value", "")
                        ev_previews.append(f"Event (start: {ev.get('start_time', ev.get('start_index'))}, peak Z: {pz}, peak val: {pv})")
                    lines.append(f"Event details: {'; '.join(ev_previews)}")
            elif data.get("anomalous_timestamps"):
                lines.append(f"Anomalous timestamps (first 5): {data.get('anomalous_timestamps', [])[:5]}")
            return "\n".join(lines)

        if tool_name == "trend_analysis":
            return (
                f"[ANALYTICAL RESULT: trend_analysis]\n"
                f"Metric: {metric}{sensor_str}  |  "
                f"Time axis: {data.get('time_column')}\n"
                f"Trend direction: {data.get('trend_direction')}  |  "
                f"Slope: {data.get('slope')}  |  R-squared: {data.get('r_squared')}\n"
                f"Start value: {data.get('start_value')}  |  "
                f"End value: {data.get('end_value')}  |  "
                f"Change magnitude: {data.get('change_magnitude')}"
            )

        if tool_name == "correlation_analysis":
            return (
                f"[ANALYTICAL RESULT: correlation_analysis]\n"
                f"Columns analysed: {', '.join(data.get('columns_used', []))}\n"
                f"Strongest correlation: {data.get('strongest_pair')}\n"
                f"Weakest correlation:   {data.get('weakest_pair')}"
            )

        if tool_name == "baseline":
            devs = [d for d in data.get("deviations", []) if d.get("group") == "frequency"][:3]
            dev_str = "; ".join(
                f"{d.get('feature')}: {d.get('change_percent'):+.1f}% ({d.get('score')} spreads)" for d in devs
            )
            nh = data.get("n_healthy", {}) or {}
            return (
                f"[ANALYTICAL RESULT: baseline]\n"
                f"Status: {data.get('status')}  |  Overall flag vs healthy baseline: {data.get('overall_flag')}\n"
                f"Frequency deviations: {dev_str or 'none'}\n"
                f"Baseline built from {nh.get('recordings', 0)} healthy recordings, {nh.get('cases', 0)} case(s).\n"
                f"Summary: {data.get('summary', '')}\n"
                f"Limit to mention: {data.get('caveat', '')}"
            )

        if tool_name == "model_result":
            return _model_result_summary(data)

        if tool_name == "chart_data":
            shown = (
                "\nThe interactive chart is already displayed to the user below your reply. "
                "Describe what it shows in plain words; do not write plotting code, image "
                "links or instructions for drawing it."
                if data.get("plot_json") not in (None, "", "{}")
                else ""
            )
            return (
                f"[ANALYTICAL RESULT: chart_data]\n"
                f"Chart generated for '{data.get('y_col')}' over '{data.get('x_col')}'.\n"
                f"{data.get('explanation', '')}{shown}"
            )

        return f"[ANALYTICAL RESULT: {tool_name}]\n{data}"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

dispatcher = ToolDispatcher()