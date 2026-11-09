"""Analytical Tool Architecture for Talking to Bridges (Phase 2 - Week 6).

Provides:
- DataAccessLayer: central dataset reading and metadata inspection.
- ToolExecutionError: structured exception for tool failures.
- ToolDefinition: metadata + callable for one registered tool.
- ToolRegistry: register, lookup, validate, and execute tools.
- Six analytical tool handlers grounded in the actual professor dataset schema:
    1. summary_statistics  (Eswar's statistics interface)
    2. anomaly_detection   (Kolla's robust anomaly pipeline: analysis/anomaly_pipeline.py)
    3. trend_analysis      (linear trend over time / relative time)
    4. correlation_analysis (Pearson correlations between sensor columns)
    5. model_result        (Krishna's ML inference placeholder)
    6. chart_data          (wraps Nagarjun's visualization_service)

Professor dataset column schema (from analysis/sensor_pipeline.py):
    DateTime, Relative_Time_Sec, Sensor_1..5,
    Condition, Test_Name, Source_File,
    Damage_Level, Specimen, Test_Type, Hit_Group

Fixture / uploaded CSV schema (tests/fixtures/bridge_sensor_data.csv):
    timestamp, temperature, stress, deflection,
    vibration, crack_width, sensor_id, condition
"""

from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from analysis.anomaly_pipeline import AnomalyInputError, detect_anomalies
from app.core.config import settings

logger = logging.getLogger("ttb.analysis.tools")


# =============================================================================
# 1. DATA ACCESS LAYER
# =============================================================================

class DataAccessLayer:
    """Central data access abstraction so analytical tools do not re-implement
    file reading or path resolution.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = Path(data_dir or settings.DOCUMENTS_DIR)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_dataset(
        self,
        dataset_name_or_bytes: Union[str, bytes, Path],
        filename: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load a DataFrame from raw bytes or a disk path.

        Args:
            dataset_name_or_bytes: Raw CSV bytes, an absolute Path, or a
                filename relative to self.data_dir.
            filename: Hint for extension detection when bytes are passed.

        Returns:
            A cleaned DataFrame with stripped column names.

        Raises:
            FileNotFoundError: If a path-based lookup fails.
            ValueError: If the bytes cannot be parsed as CSV.
        """
        if isinstance(dataset_name_or_bytes, bytes):
            return self._parse_bytes(dataset_name_or_bytes, filename or "data.csv")

        path = Path(dataset_name_or_bytes)
        if not path.is_absolute():
            path = self.data_dir / path

        if not path.exists():
            alt_path = self.data_dir / path.name
            if alt_path.exists():
                path = alt_path
            else:
                raise FileNotFoundError(
                    f"Dataset file '{dataset_name_or_bytes}' not found."
                )

        return self._parse_bytes(path.read_bytes(), path.name)

    def _parse_bytes(self, content: bytes, filename: str) -> pd.DataFrame:
        """Parse raw bytes as CSV or Excel with multi-encoding fallback."""
        suffix = Path(filename).suffix.lower()
        if suffix in (".xlsx", ".xlsm"):
            try:
                df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception as exc:
                raise ValueError(f"Could not parse Excel content for '{filename}': {exc}") from exc

        for encoding in ("utf-8", "latin-1", "utf-8-sig"):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
                df.columns = [str(c).strip() for c in df.columns]
                return df
            except Exception:
                continue
        raise ValueError(f"Could not parse CSV content for '{filename}'.")

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Inspect column types, counts, and identify timestamp candidates.

        Returns:
            A dict with row_count, column_count, columns, numeric_columns,
            categorical_columns, and timestamp_column (or None).
        """
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        categorical_cols = list(df.select_dtypes(include=["object", "category"]).columns)

        timestamp_col: Optional[str] = None
        for col in df.columns:
            if "time" in col.lower() or "date" in col.lower():
                timestamp_col = col
                break

        return {
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "numeric_columns": numeric_cols,
            "categorical_columns": categorical_cols,
            "timestamp_column": timestamp_col,
        }

    def get_numeric_series(
        self,
        df: pd.DataFrame,
        column: str,
        sensor_id: Optional[str] = None,
    ) -> pd.Series:
        """Return a cleaned numeric series for a given column, optionally
        filtered by sensor_id.

        Args:
            df: Source DataFrame.
            column: Target numeric column name.
            sensor_id: If provided, only rows where df['sensor_id'] == sensor_id
                are used.  Ignored when the column is absent.

        Returns:
            pd.Series of float values with NaNs dropped.

        Raises:
            ToolExecutionError: When the column does not exist in df.
        """
        if column not in df.columns:
            raise ToolExecutionError(
                error_type="INVALID_INPUT",
                message=f"Column '{column}' not found in dataset.",
                details={"available_columns": list(df.columns)},
            )
        sub_df = df
        if sensor_id is not None and "sensor_id" in df.columns:
            sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)]
        return pd.to_numeric(sub_df[column], errors="coerce").dropna()


# Global Data Access Layer instance
data_access = DataAccessLayer()


# =============================================================================
# 2. TOOL CONTRACT SCHEMAS
# =============================================================================

class ToolExecutionError(Exception):
    """Structured exception raised during analytical tool execution."""

    def __init__(
        self,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


@dataclass
class ToolDefinition:
    """Metadata definition for a registered analytical tool."""

    name: str
    description: str
    category: str   # "statistics" | "anomaly" | "trend" | "correlation" | "prediction" | "visualization"
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    handler: Callable[[Dict[str, Any], Optional[pd.DataFrame]], Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dict (handler excluded)."""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
        }


# =============================================================================
# 3. TOOL REGISTRY
# =============================================================================

class ToolRegistry:
    """Central registry for analytical tools.

    Usage
    -----
    registry.register(ToolDefinition(...))
    registry.list_tools()          -> list of dicts (for the API)
    registry.get_tool(name)        -> ToolDefinition
    registry.execute(name, args)   -> dict with status/data/error
    registry.validate_arguments(name, args, df)  -> raises ToolExecutionError
    """

    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition) -> None:
        """Register a new analytical tool (overwrites any existing tool with the same name)."""
        self._tools[tool.name] = tool
        logger.info("Registered analytical tool: '%s' [%s]", tool.name, tool.category)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def get_tool(self, name: str) -> ToolDefinition:
        """Retrieve a tool definition by name.

        Raises:
            ToolExecutionError (UNKNOWN_TOOL) if the name is not registered.
        """
        if name not in self._tools:
            raise ToolExecutionError(
                error_type="UNKNOWN_TOOL",
                message=(
                    f"Analytical tool '{name}' is not registered. "
                    f"Available tools: {', '.join(sorted(self._tools.keys()))}"
                ),
                details={"available_tools": sorted(self._tools.keys())},
            )
        return self._tools[name]

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return metadata of all registered tools (JSON-serializable)."""
        return [tool.to_dict() for tool in self._tools.values()]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_arguments(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Validate arguments against the tool's input_schema.

        Checks:
        1. Required parameters are present and not None.
        2. If df is provided:
           - 'metric' / 'y_col' columns exist in df.
           - 'sensor_id' is a valid value in df['sensor_id'] when that column exists.
           - 'columns' list entries all exist in df.

        Raises:
            ToolExecutionError (INVALID_INPUT) on first validation failure.
        """
        tool = self.get_tool(tool_name)
        schema = tool.input_schema
        required = schema.get("required", [])

        # 1. Required parameter check
        missing = [r for r in required if r not in arguments or arguments[r] is None]
        if missing:
            raise ToolExecutionError(
                error_type="INVALID_INPUT",
                message=(
                    f"Missing required parameter(s) for '{tool_name}': "
                    + ", ".join(f"'{m}'" for m in missing)
                ),
                details={"required": required, "provided": list(arguments.keys())},
            )

        if df is None:
            return

        available_cols = list(df.columns)

        # 2a. Column existence check for 'metric' and 'y_col'
        for col_arg in ("metric", "y_col"):
            val = arguments.get(col_arg)
            if val and val not in df.columns:
                raise ToolExecutionError(
                    error_type="INVALID_INPUT",
                    message=f"Column '{val}' ('{col_arg}') not found in dataset.",
                    details={"available_columns": available_cols},
                )

        # 2b. 'columns' list check (correlation_analysis)
        cols_arg = arguments.get("columns")
        if cols_arg:
            bad = [c for c in cols_arg if c not in df.columns]
            if bad:
                raise ToolExecutionError(
                    error_type="INVALID_INPUT",
                    message=f"Columns not found in dataset: {bad}",
                    details={"available_columns": available_cols},
                )

        # 2c. sensor_id existence check
        sensor_id_arg = arguments.get("sensor_id")
        if sensor_id_arg and "sensor_id" in df.columns:
            available_ids = [str(s) for s in df["sensor_id"].dropna().unique()]
            if str(sensor_id_arg) not in available_ids and str(sensor_id_arg) not in df.columns:
                raise ToolExecutionError(
                    error_type="INVALID_INPUT",
                    message=f"sensor_id '{sensor_id_arg}' not found in dataset.",
                    details={"available_sensor_ids": available_ids},
                )

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Execute a tool with structured parameters and full error handling.

        Returns:
            A JSON-serializable dict with:
              - status: "success" | "error"
              - tool: tool name
              - data: result dict (on success)
              - error_type: error code string (on failure)
              - message: human-readable description
              - details: supplementary dict (on failure)
              - metadata: dict with category and arguments (on success)
        """
        try:
            tool = self.get_tool(tool_name)
            self.validate_arguments(tool_name, arguments, df=df)
            result_data = tool.handler(arguments, df)
            return {
                "tool": tool_name,
                "status": "success",
                "data": result_data,
                "metadata": {
                    "category": tool.category,
                    "arguments": arguments,
                },
                "message": f"Tool '{tool_name}' executed successfully.",
            }
        except ToolExecutionError as exc:
            logger.warning("Tool '%s' failed [%s]: %s", tool_name, exc.error_type, exc.message)
            return {
                "tool": tool_name,
                "status": "error",
                "error_type": exc.error_type,
                "message": exc.message,
                "details": exc.details,
            }
        except Exception as exc:
            logger.exception("Unexpected error executing tool '%s'", tool_name)
            return {
                "tool": tool_name,
                "status": "error",
                "error_type": "TOOL_EXECUTION_ERROR",
                "message": f"Unexpected error in tool '{tool_name}': {type(exc).__name__}: {exc}",
                "details": {},
            }


# Global registry singleton
registry = ToolRegistry()


# =============================================================================
# 4. ANALYTICAL TOOL HANDLERS (Week 6)
# =============================================================================

# ---------------------------------------------------------------------------
# Helper: resolve time column
# ---------------------------------------------------------------------------

def _resolve_time_column(df: pd.DataFrame, hint: Optional[str]) -> str:
    """Return the best available time column name.

    Priority:
    1. Caller's explicit hint (if it exists in df).
    2. 'Relative_Time_Sec'  (professor dataset).
    3. 'timestamp'          (fixture CSV).
    4. Falls back to the string 'index' (caller must use df.reset_index()).
    """
    if hint and hint in df.columns:
        return hint
    for candidate in ("Relative_Time_Sec", "timestamp", "DateTime"):
        if candidate in df.columns:
            return candidate
    return "index"


def _resolve_x_column(df: pd.DataFrame, hint: str) -> str:
    """Resolve the X column for chart_data.  Falls back to Relative_Time_Sec
    or timestamp if the hint is missing from the DataFrame.
    """
    if hint in df.columns:
        return hint
    for candidate in ("Relative_Time_Sec", "timestamp", "DateTime"):
        if candidate in df.columns:
            return candidate
    return hint  # return as-is; validation will catch it later


# ---------------------------------------------------------------------------
# 4.1  summary_statistics
# ---------------------------------------------------------------------------

def handle_summary_statistics(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute descriptive statistics for a sensor metric.

    Returns mean, std, min, max, median, count, and optionally rms and p2p.
    Works with both the professor's processed dataset and any uploaded CSV.
    """
    metric = arguments["metric"]
    sensor_id = arguments.get("sensor_id")

    if df is not None and metric in df.columns:
        try:
            series = data_access.get_numeric_series(df, metric, sensor_id=sensor_id)
        except ToolExecutionError:
            series = pd.Series(dtype=float)

        if not series.empty:
            arr = series.to_numpy()
            arr_clean = arr[~np.isnan(arr)]
            if len(arr_clean) == 0:
                raise ToolExecutionError(f"No valid numeric data in {metric}")
            
            rms_val = float(np.sqrt(np.mean(arr_clean ** 2)))
            p2p_val = float(arr_clean.max() - arr_clean.min())
            
            import scipy.stats as stats
            return {
                "sensor_id": sensor_id,
                "metric": metric,
                "count": int(series.count()),
                "min": round(float(arr_clean.min()), 6),
                "max": round(float(arr_clean.max()), 6),
                "mean": round(float(arr_clean.mean()), 6),
                "median": round(float(float(np.median(arr_clean))), 6),
                "std": round(float(arr_clean.std(ddof=1)) if len(arr_clean) > 1 else 0.0, 6),
                "variance": round(float(np.var(arr_clean, ddof=1)) if len(arr_clean) > 1 else 0.0, 6),
                "skewness": round(float(stats.skew(arr_clean)) if len(arr_clean) > 2 else 0.0, 6),
                "kurtosis": round(float(stats.kurtosis(arr_clean)) if len(arr_clean) > 3 else 0.0, 6),
                "q25": round(float(np.percentile(arr_clean, 25)), 6),
                "q75": round(float(np.percentile(arr_clean, 75)), 6),
                "rms": round(rms_val, 6),
                "p2p": round(p2p_val, 6),
            }

    # Deterministic placeholder when no real data is available
    return {
        "sensor_id": sensor_id,
        "metric": metric,
        "count": 30,
        "min": 1.7,
        "max": 3.8,
        "mean": 2.54,
        "median": 2.50,
        "std": 0.65,
        "variance": 0.42,
        "skewness": 0.1,
        "kurtosis": -0.2,
        "q25": 2.1,
        "q75": 3.0,
        "rms": 2.62,
        "p2p": 2.10,
        "_note": "Placeholder result: upload a dataset for real statistics.",
    }


# ---------------------------------------------------------------------------
# 4.2  anomaly_detection
# ---------------------------------------------------------------------------

def handle_anomaly_detection(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Detect anomalous readings with Kolla's robust anomaly pipeline.

    Delegates to analysis.anomaly_pipeline.detect_anomalies (robust z-score on
    median/MAD, event grouping).  The input/output schema is unchanged; a few
    extra keys (method, max_abs_z, baseline, event_count, events) are added.
    When no dataset is available a deterministic placeholder is returned.
    """
    metric = arguments["metric"]
    sensor_id = arguments.get("sensor_id")
    raw_threshold = arguments.get("threshold")

    if df is not None:
        try:
            return detect_anomalies(
                df, metric, sensor_id=sensor_id, threshold=raw_threshold
            )
        except AnomalyInputError as exc:
            raise ToolExecutionError(
                error_type=exc.error_type,
                message=exc.message,
                details=exc.details,
            )

    threshold = 2.0 if raw_threshold is None else float(raw_threshold)

    # Deterministic placeholder (no dataset uploaded)
    return {
        "sensor_id": sensor_id,
        "metric": metric,
        "threshold_z": threshold,
        "total_rows_checked": 30,
        "anomaly_count": 2,
        "anomaly_fraction": 0.067,
        "anomalous_indices": [6, 16],
        "anomalous_timestamps": ["2026-01-07", "2026-01-17"],
        "status_flag": "Alert",
        "_note": "Placeholder result: upload a dataset for real anomaly detection.",
    }


# ---------------------------------------------------------------------------
# 4.3  trend_analysis
# ---------------------------------------------------------------------------

def handle_trend_analysis(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Fit a linear trend to a sensor metric over time using numpy polyfit.

    Reports slope, intercept, R-squared, and trend direction.
    """
    metric = arguments["metric"]
    sensor_id = arguments.get("sensor_id")
    time_hint = arguments.get("time_column")

    if df is not None and metric in df.columns:
        sub_df = df
        if sensor_id and "sensor_id" in df.columns:
            sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)]

        time_col = _resolve_time_column(sub_df, time_hint)
        y = pd.to_numeric(sub_df[metric], errors="coerce")

        if time_col == "index" or time_col not in sub_df.columns:
            x = pd.Series(range(len(sub_df)), index=sub_df.index, dtype=float)
            time_col_label = "row_index"
        else:
            x = pd.to_numeric(sub_df[time_col], errors="coerce")
            time_col_label = time_col

        valid = pd.DataFrame({"x": x, "y": y}).dropna()
        n = len(valid)

        if n >= 2:
            x_arr = valid["x"].to_numpy()
            y_arr = valid["y"].to_numpy()
            slope, intercept = np.polyfit(x_arr, y_arr, 1)
            y_pred = slope * x_arr + intercept
            ss_res = float(np.sum((y_arr - y_pred) ** 2))
            ss_tot = float(np.sum((y_arr - y_arr.mean()) ** 2))
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Direction: flat if |slope| * range < 5% of mean amplitude
            x_range = float(x_arr[-1] - x_arr[0])
            amplitude_change = abs(slope * x_range)
            mean_abs = float(np.mean(np.abs(y_arr))) or 1.0
            if amplitude_change / mean_abs < 0.05:
                direction = "flat"
            elif slope > 0:
                direction = "increasing"
            else:
                direction = "decreasing"

            return {
                "metric": metric,
                "time_column": time_col_label,
                "slope": round(float(slope), 8),
                "intercept": round(float(intercept), 6),
                "r_squared": round(max(0.0, min(1.0, r_squared)), 4),
                "trend_direction": direction,
                "data_points": n,
                "start_value": round(float(y_arr[0]), 6),
                "end_value": round(float(y_arr[-1]), 6),
                "change_magnitude": round(float(y_arr[-1] - y_arr[0]), 6),
            }

    # Deterministic placeholder
    return {
        "metric": metric,
        "time_column": time_hint or "Relative_Time_Sec",
        "slope": 0.0012,
        "intercept": 2.1,
        "r_squared": 0.72,
        "trend_direction": "increasing",
        "data_points": 30,
        "start_value": 1.9,
        "end_value": 3.1,
        "change_magnitude": 1.2,
        "_note": "Placeholder result: upload a dataset for real trend analysis.",
    }


# ---------------------------------------------------------------------------
# 4.4  correlation_analysis
# ---------------------------------------------------------------------------

def handle_correlation_analysis(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compute Pearson correlation matrix between numeric sensor columns.

    Works with both professor dataset sensor columns (Sensor_1..5) and
    uploaded CSV numeric columns (deflection, vibration, stress, etc.).
    """
    sensor_id = arguments.get("sensor_id")
    requested_cols = arguments.get("columns")

    if df is not None:
        sub_df = df
        if sensor_id and "sensor_id" in df.columns:
            sub_df = df[df["sensor_id"].astype(str) == str(sensor_id)]

        # Determine which columns to correlate
        numeric_cols = list(sub_df.select_dtypes(include=["number"]).columns)
        # Exclude time/index columns
        exclude = {"Relative_Time_Sec", "DateTime", "timestamp"}
        numeric_cols = [c for c in numeric_cols if c not in exclude]

        if requested_cols:
            use_cols = [c for c in requested_cols if c in sub_df.columns]
        else:
            use_cols = numeric_cols

        if len(use_cols) >= 2:
            corr_df = sub_df[use_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
            # Build a JSON-safe nested dict
            corr_dict: Dict[str, Dict[str, float]] = {}
            for col in corr_df.columns:
                corr_dict[col] = {}
                for row in corr_df.index:
                    val = corr_df.loc[row, col]
                    corr_dict[col][row] = round(float(val), 4) if not math.isnan(val) else 0.0

            # Find strongest and weakest off-diagonal pairs
            pairs = []
            cols_list = list(corr_df.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    c1, c2 = cols_list[i], cols_list[j]
                    val = corr_df.loc[c1, c2]
                    if not math.isnan(val):
                        pairs.append((c1, c2, float(val)))

            strongest = None
            weakest = None
            if pairs:
                pairs.sort(key=lambda t: abs(t[2]), reverse=True)
                t = pairs[0]
                strongest = f"{t[0]} vs {t[1]} (r={t[2]:.4f})"
                w = pairs[-1]
                weakest = f"{w[0]} vs {w[1]} (r={w[2]:.4f})"

            return {
                "columns_used": use_cols,
                "correlation_matrix": corr_dict,
                "strongest_pair": strongest,
                "weakest_pair": weakest,
            }

    # Deterministic placeholder
    return {
        "columns_used": ["deflection", "vibration", "stress"],
        "correlation_matrix": {
            "deflection": {"deflection": 1.0, "vibration": 0.87, "stress": 0.73},
            "vibration":  {"deflection": 0.87, "vibration": 1.0,  "stress": 0.61},
            "stress":     {"deflection": 0.73, "vibration": 0.61, "stress": 1.0},
        },
        "strongest_pair": "deflection vs vibration (r=0.87)",
        "weakest_pair": "vibration vs stress (r=0.61)",
        "_note": "Placeholder result: upload a dataset for real correlation analysis.",
    }


# ---------------------------------------------------------------------------
# 4.5  model_result  (NEW-4: trained model + healthy comparison, Interface C)
# ---------------------------------------------------------------------------

def handle_model_result(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Assess the uploaded recording with the trained TRN-1 model and the healthy baseline.

    Flow: extract_features(recording) -> TRN-1 model -> compare_to_baseline (NEW-2).
    Returns the Interface C fields (condition, condition_confidence, severity_mm,
    severity_confidence, top_parameters, baseline_comparison, ...) plus the older fields
    (predicted_class, confidence, damage_class, severity, contributing_parameters,
    features_used, prediction_timestamp, model_note) so existing callers keep working.

    The label column of the uploaded data is never read.  Without an uploaded recording
    the result says so (status 'no_data'); nothing is invented.
    """
    from datetime import datetime

    from analysis.assessment import assess_recording, severity_texts

    sensor_id = str(arguments.get("sensor_id") or "recording")
    target_variable = arguments.get("target_variable", "condition")

    result = assess_recording(df)

    condition = result["condition"]
    severity_mm = result["severity_mm"]
    _, severity_word = severity_texts(severity_mm)
    if condition == "Damaged":
        damage_class = f"Damaged_{severity_mm}mm" if severity_mm else "Damaged"
        severity = severity_word
    elif condition == "Undamaged":
        damage_class, severity = "Undamaged", "None"
    else:
        damage_class, severity = None, None

    parameters = [p["feature"] for p in result["top_parameters"]]
    return {
        **result,
        "sensor_id": sensor_id,
        "target_variable": target_variable,
        "predicted_class": condition or "Unavailable",
        "confidence": result["condition_confidence"] or 0.0,
        "damage_class": damage_class,
        "severity": severity,
        "features_used": parameters,
        "contributing_parameters": parameters,
        "prediction_timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_note": result["message"] if result["message"] != "ok" else (
            "Trained model result (leave-one-case-out evaluated); see caveats."
        ),
    }


# ---------------------------------------------------------------------------
# 4.6  baseline  (NEW-2: healthy baseline + comparison, Interface B)
# ---------------------------------------------------------------------------

_BASELINE_STATUS = {"normal": "Healthy", "abnormal": "Abnormal", "unavailable": "Unavailable"}


def handle_baseline(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Compare the uploaded recording with the healthy baseline (analysis.baseline).

    Returns the Interface B fields (overall_flag, deviations, healthy_reference, n_healthy)
    plus the older fields (status, sensors_evaluated, deviation_ranking,
    healthy_baseline_summary, note) so existing callers keep working.
    The healthy spectrum is left out here (large); the charts add it themselves.
    """
    from analysis.baseline import compare_to_baseline

    if df is None:
        result = {
            "overall_flag": "unavailable",
            "deviations": [],
            "healthy_reference": {},
            "n_healthy": {"recordings": 0, "cases": 0, "case_names": []},
            "summary": "Upload a sensor recording first, then ask for the comparison with the healthy baseline.",
            "caveat": "",
        }
    else:
        result = compare_to_baseline(df, include_spectrum=False)

    sensors = [c for c in df.columns if str(c).startswith("Sensor_")] if df is not None else []
    return {
        **result,
        "status": _BASELINE_STATUS.get(result["overall_flag"], "Unavailable"),
        "sensors_evaluated": sensors,
        "deviation_ranking": result["deviations"],
        "healthy_baseline_summary": result.get("healthy_reference", {}).get("frequencies_hz", {}),
        "note": result["summary"],
    }


# ---------------------------------------------------------------------------
# 4.6  chart_data  (wraps Nagarjun's visualization_service)
# ---------------------------------------------------------------------------

CHART_ANOMALY_THRESHOLD = 5.0  # default robust z-score for chart anomaly markers


def _truthy(value: Any) -> bool:
    """Interpret True / 'true' / 'yes' / 1 as True (arguments may come from an LLM)."""
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1", "y"}
    return bool(value)


def _is_sensor_recording(df: pd.DataFrame, x_col: str, y_col: str) -> bool:
    """True for the professor's cleaned recordings (Relative_Time_Sec + Sensor_N)."""
    return x_col == "Relative_Time_Sec" and str(y_col).lower().startswith("sensor")


def _sensor_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Interactive sensor chart built with analysis.charts (Week 6).

    With show_anomalies=true, Kolla's detect_anomalies marks the flagged
    readings on the line (red x markers); otherwise it is a plain line chart.
    """
    from analysis.charts import figure_to_json, plot_anomaly_chart, plot_time_series

    if _truthy(arguments.get("show_anomalies")):
        threshold = arguments.get("threshold")
        if threshold is None:
            threshold = CHART_ANOMALY_THRESHOLD
        found = detect_anomalies(df, y_col, arguments.get("sensor_id"), threshold)
        # Mark events (start / end / peak), not the per-reading list: that list is
        # capped at 50 by the anomaly pipeline, events cover every flagged reading.
        events = found["events"]
        fig = plot_anomaly_chart(
            df,
            x_col=x_col,
            sensor_col=y_col,
            events=events,
            flagged_count=found["anomaly_count"],
        )
        shown = (
            ""
            if found["event_count"] <= len(events)
            else f" (the {len(events)} strongest of {found['event_count']} are shown)"
        )
        explanation = (
            f"{y_col} over {x_col} with anomalies marked (threshold {found['threshold_z']}). "
            f"{found['anomaly_count']} of {found['total_rows_checked']} readings flagged "
            f"in {found['event_count']} event(s); status {found['status_flag']}. "
            f"The chart marks each event's span, start/end and peak{shown}. "
            "Values are in unknown units."
        )
    else:
        fig = plot_time_series(df, x_col=x_col, y_cols=[y_col], title=f"{y_col} over time")
        explanation = (
            f"{y_col} plotted against {x_col} ({len(df)} readings, downsampled for display). "
            "Values are in unknown units."
        )
    return {
        "x_col": x_col,
        "y_col": y_col,
        "plot_json": figure_to_json(fig),
        "explanation": explanation,
    }


COMPARISON_MODES = ("signals", "features", "heatmap", "spectrum", "frequencies")
COMPARISON_TOP_N = 12


def _comparison_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    mode: str,
) -> Dict[str, Any]:
    """Healthy-vs-uploaded comparison charts (faculty requirement 5).

    signals  - healthy recording overlaid on the uploaded one, one panel per sensor
    features - feature values against the healthy range (analysis.features.rank_deviations)
    heatmap  - the same deviations as a sensor x feature grid
    spectrum    - healthy vs uploaded frequency spectrum, first three peaks marked (NEW-3)
    frequencies - f1, f2, f3 bars against the healthy values with percent change (NEW-3)

    The healthy reference is read from the reference folder (analysis.healthy_reference).
    When it is missing the result is the usual placeholder with an explanation.
    """
    from analysis.charts import (
        Z_LIMIT,
        figure_to_json,
        plot_deviation_heatmap,
        plot_feature_deviation,
        plot_recording_comparison,
        sensor_columns,
    )
    from analysis.features import build_healthy_baseline, rank_deviations
    from analysis.healthy_reference import load_healthy_recordings, reference_dir

    def _placeholder(reason: str) -> Dict[str, Any]:
        return {"x_col": x_col, "y_col": y_col, "plot_json": "{}", "explanation": reason}

    if mode in ("spectrum", "frequencies"):
        return _frequency_chart(df, x_col, y_col, mode)

    healthy = load_healthy_recordings()
    if not healthy:
        return _placeholder(
            "Cannot compare with the healthy state: no healthy reference recording is available "
            f"(expected .csv/.xlsx/.xlsm files in '{reference_dir()}')."
        )
    sensor = y_col if str(y_col).startswith("Sensor_") and y_col in df.columns else None

    if mode == "signals":
        sensors = [sensor] if sensor else None
        fig = plot_recording_comparison(healthy[0], df, sensors=sensors, x_col=x_col)
        shown = sensor or ", ".join(s for s in sensor_columns(df) if s in healthy[0].columns)
        return {
            "x_col": x_col,
            "y_col": shown,
            "comparison": mode,
            "plot_json": figure_to_json(fig),
            "explanation": (
                f"The chart is a line chart over time, one panel per sensor ({shown}): the healthy "
                "reference recording (blue) against the uploaded recording (orange), both shifted "
                "to start at t = 0. Values are in unknown units."
            ),
        }

    baseline = build_healthy_baseline(healthy)
    deviations = rank_deviations(df, baseline)
    if mode == "heatmap":
        fig = plot_deviation_heatmap(deviations)
    else:
        fig = plot_feature_deviation(deviations, top_n=COMPARISON_TOP_N, sensor=sensor)

    from analysis.charts import _prepare_deviations

    tidy = _prepare_deviations(deviations, sensor)
    outside = tidy[tidy["abs_z"] > Z_LIMIT]
    flagged = [
        f"{r.sensor} {r.feature} {r.z_score:+.1f} sigma "
        f"(uploaded {r.test_mean:.4g}, healthy {r.healthy_mean:.4g})"
        for r in outside.head(10).itertuples()
    ]
    closest = tidy[tidy["abs_z"] <= Z_LIMIT].head(3)
    near = [f"{r.sensor} {r.feature} {r.z_score:+.1f} sigma" for r in closest.itertuples()]
    if flagged:
        verdict = (
            f"Outside the healthy range ({len(outside)} of {len(tidy)} features): "
            f"{'; '.join(flagged)}."
            + (f" Every other feature is inside it (closest: {'; '.join(near)})." if near else "")
        )
    else:
        verdict = f"No feature is outside the healthy range (closest: {'; '.join(near)})."
    if mode == "heatmap":
        how = (
            "The chart is a heatmap: one row per sensor, one column per feature, each cell "
            "the deviation from the healthy mean in standard deviations (orange above, blue "
            "below, near white = close to healthy)."
        )
    else:
        how = (
            "The chart is a horizontal bar chart: one bar per feature, longest deviation on top; "
            "bar length is the deviation from the healthy mean in standard deviations (log "
            f"scale); orange bars are outside the healthy range (more than {Z_LIMIT:g} sigma), "
            "blue bars are inside it."
        )
    return {
        "x_col": x_col,
        "y_col": sensor or y_col,
        "comparison": mode,
        "plot_json": figure_to_json(fig),
        "explanation": f"{how} {verdict} Values are in unknown units.",
    }


def _frequency_chart(df: pd.DataFrame, x_col: str, y_col: str, mode: str) -> Dict[str, Any]:
    """NEW-3 charts, built from NEW-2's Interface B (analysis.baseline).

    spectrum    - healthy spectrum (blue) over the uploaded one (orange), f1-f3 marked
    frequencies - bars of f1, f2, f3 against the healthy values with percent change
    """
    from analysis.baseline import compare_to_baseline, recording_spectrum
    from analysis.charts import figure_to_json, plot_frequency_bars, plot_spectrum_overlay
    from analysis.features import extract_features
    from analysis.healthy_reference import reference_dir

    def _placeholder(reason: str) -> Dict[str, Any]:
        return {"x_col": x_col, "y_col": y_col, "comparison": mode, "plot_json": "{}", "explanation": reason}

    result = compare_to_baseline(df, include_spectrum=(mode == "spectrum"))
    if result["overall_flag"] == "unavailable":
        return _placeholder(f"Cannot compare with the healthy state: {result['summary']}")

    n = result["n_healthy"]
    caveat = result["caveat"]
    if mode == "spectrum":
        healthy_spec = result["healthy_reference"].get("spectrum")
        if not healthy_spec:
            return _placeholder(
                "Cannot draw the healthy spectrum: no healthy recording is available "
                f"(expected .csv/.xlsx/.xlsm files in '{reference_dir()}')."
            )
        row = extract_features(df)
        fig = plot_spectrum_overlay(
            healthy_spec,
            recording_spectrum(df),
            result["healthy_reference"]["frequencies_hz"],
            {k: row.get(k) for k in ("f1_hz", "f2_hz", "f3_hz")},
        )
        how = (
            "The chart is a frequency spectrum (amplitude on a log scale): the healthy "
            f"spectrum (blue, median of {healthy_spec['n_recordings']} healthy recording(s)) "
            "against the uploaded recording (orange). Diamonds mark the first three peaks "
            "(f1, f2, f3). A peak that moved sideways means a changed natural frequency."
        )
        shown = "spectrum"
    else:
        fig = plot_frequency_bars(result["deviations"])
        how = (
            "The chart shows the first three bending frequencies: healthy (blue) against "
            "uploaded (orange), one panel per mode, bars start at zero. The title of each "
            "panel is the percent change; the black whisker on the healthy bar is the healthy "
            "range (3 spreads); a red outline means the uploaded value is outside it."
        )
        shown = "f1, f2, f3"
    return {
        "x_col": x_col,
        "y_col": shown,
        "comparison": mode,
        "plot_json": figure_to_json(fig),
        "explanation": f"{how} {result['summary']} {caveat}",
    }


def handle_chart_data(
    arguments: Dict[str, Any],
    df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Generate an interactive Plotly chart payload for sensor measurements.

    Order of attempts (the output always has x_col, y_col, plot_json, explanation):
      1. Professor recordings (Relative_Time_Sec + Sensor_N): analysis.charts,
         optionally with anomalies marked (show_anomalies=true).
      2. Any other dataset: the existing visualization_service.analyze_and_plot.
      3. Placeholder ("{}") when there is no usable data or generation fails.
    """
    y_col = arguments["y_col"]
    x_col = arguments.get("x_col", "timestamp")

    if df is not None:
        resolved_x = _resolve_x_column(df, x_col)
        if resolved_x in df.columns and y_col in df.columns:
            mode = arguments.get("comparison")
            if mode in COMPARISON_MODES:
                try:
                    return _comparison_chart(df, resolved_x, y_col, mode)
                except Exception as exc:
                    logger.warning("Comparison chart (%s) failed: %s", mode, exc)
                    return {
                        "x_col": resolved_x,
                        "y_col": y_col,
                        "plot_json": "{}",
                        "explanation": f"The healthy comparison chart could not be built: {exc}",
                    }
            if y_col == resolved_x and resolved_x == "Relative_Time_Sec":
                # Query named no sensor (dispatcher picked the time column): plot all sensors.
                try:
                    from analysis.charts import figure_to_json, plot_time_series, sensor_columns

                    sensors = sensor_columns(df)
                    if sensors:
                        fig = plot_time_series(df, x_col=resolved_x, y_cols=sensors)
                        return {
                            "x_col": resolved_x,
                            "y_col": ", ".join(sensors),
                            "plot_json": figure_to_json(fig),
                            "explanation": (
                                f"All sensors ({', '.join(sensors)}) plotted against {resolved_x}. "
                                "Values are in unknown units."
                            ),
                        }
                except Exception as exc:
                    logger.warning("All-sensor chart failed: %s", exc)
            if _is_sensor_recording(df, resolved_x, y_col):
                try:
                    return _sensor_chart(df, resolved_x, y_col, arguments)
                except AnomalyInputError as exc:
                    logger.warning("Anomaly overlay failed for '%s': %s", y_col, exc.message)
                except Exception as exc:
                    logger.warning("Sensor chart failed for '%s': %s", y_col, exc)
            try:
                from app.services.visualization_service import analyze_and_plot
                fig, explanation = analyze_and_plot(df, resolved_x, y_col)
                return {
                    "x_col": resolved_x,
                    "y_col": y_col,
                    "plot_json": fig.to_json(),
                    "explanation": explanation,
                }
            except ImportError:
                logger.warning(
                    "plotly is not installed; returning chart_data placeholder for '%s'.", y_col
                )
            except Exception as exc:
                logger.warning("Chart generation failed for '%s' vs '%s': %s", y_col, resolved_x, exc)

    return {
        "x_col": x_col,
        "y_col": y_col,
        "plot_json": "{}",
        "explanation": (
            f"Chart placeholder for {y_col} over {x_col}. "
            "Upload a dataset with compatible columns to generate a real chart."
        ),
    }


# =============================================================================
# 5. REGISTER ALL TOOLS AT IMPORT TIME
# =============================================================================

registry.register(ToolDefinition(
    name="summary_statistics",
    description=(
        "Calculates descriptive statistics (count, min, max, mean, median, std, rms, p2p) "
        "for a numerical sensor measurement column. Owned by Eswar (statistics)."
    ),
    category="statistics",
    input_schema={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": (
                    "Numerical column name. Fixture columns: deflection, vibration, stress, "
                    "temperature, crack_width. Professor dataset: Sensor_1..Sensor_5."
                ),
            },
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter (e.g. 'S01').",
            },
            "start_time": {"type": "string", "description": "Optional ISO-8601 start date."},
            "end_time":   {"type": "string", "description": "Optional ISO-8601 end date."},
        },
        "required": ["metric"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sensor_id": {"type": ["string", "null"]},
            "metric":  {"type": "string"},
            "count":   {"type": "integer"},
            "min":     {"type": "number"},
            "max":     {"type": "number"},
            "mean":    {"type": "number"},
            "median":  {"type": "number"},
            "std":     {"type": "number"},
            "rms":     {"type": ["number", "null"]},
            "p2p":     {"type": ["number", "null"]},
        },
    },
    handler=handle_summary_statistics,
))

registry.register(ToolDefinition(
    name="anomaly_detection",
    description=(
        "Detects anomalous sensor readings using a robust (median/MAD) z-score "
        "against the given threshold. Returns count, fraction, timestamps and "
        "grouped events (impacts) of anomalous readings."
    ),
    category="anomaly",
    input_schema={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Numerical column to scan for anomalies.",
            },
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter.",
            },
            "threshold": {
                "type": "number",
                "description": "Z-score threshold (default 2.0).",
                "default": 2.0,
            },
        },
        "required": ["metric"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "sensor_id":          {"type": ["string", "null"]},
            "metric":             {"type": "string"},
            "threshold_z":        {"type": "number"},
            "total_rows_checked": {"type": "integer"},
            "anomaly_count":      {"type": "integer"},
            "anomaly_fraction":   {"type": "number"},
            "anomalous_indices":  {"type": "array", "items": {"type": "integer"}},
            "anomalous_timestamps": {"type": "array", "items": {"type": "string"}},
            "status_flag":        {"type": "string"},
        },
    },
    handler=handle_anomaly_detection,
))

registry.register(ToolDefinition(
    name="trend_analysis",
    description=(
        "Fits a linear trend to a sensor metric over time. "
        "Returns slope, R-squared, trend direction (increasing/decreasing/flat), "
        "and magnitude of change. Works on Relative_Time_Sec or any time column."
    ),
    category="trend",
    input_schema={
        "type": "object",
        "properties": {
            "metric": {
                "type": "string",
                "description": "Numerical column to analyse for trend.",
            },
            "time_column": {
                "type": "string",
                "description": "Optional explicit time/X-axis column name.",
            },
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter.",
            },
        },
        "required": ["metric"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "metric":           {"type": "string"},
            "time_column":      {"type": "string"},
            "slope":            {"type": "number"},
            "intercept":        {"type": "number"},
            "r_squared":        {"type": "number"},
            "trend_direction":  {"type": "string"},
            "data_points":      {"type": "integer"},
            "start_value":      {"type": "number"},
            "end_value":        {"type": "number"},
            "change_magnitude": {"type": "number"},
        },
    },
    handler=handle_trend_analysis,
))

registry.register(ToolDefinition(
    name="correlation_analysis",
    description=(
        "Computes Pearson correlation coefficients between numerical sensor columns. "
        "Reports full correlation matrix and identifies strongest/weakest pair. "
        "Works on Sensor_1..5 (professor dataset) or any numeric uploaded CSV columns."
    ),
    category="correlation",
    input_schema={
        "type": "object",
        "properties": {
            "columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Columns to correlate. Defaults to all numeric columns.",
            },
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter.",
            },
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "columns_used":       {"type": "array", "items": {"type": "string"}},
            "correlation_matrix": {"type": "object"},
            "strongest_pair":     {"type": ["string", "null"]},
            "weakest_pair":       {"type": ["string", "null"]},
        },
    },
    handler=handle_correlation_analysis,
))

registry.register(ToolDefinition(
    name="model_result",
    description=(
        "Assesses the uploaded sensor recording with the trained model and the healthy baseline: "
        "is the structure damaged or undamaged (with confidence), the estimated severity "
        "(1, 2 or 3 mm crack, with confidence), the parameters that indicate the damage, and "
        "how it compares with the healthy condition. Needs an uploaded Excel recording."
    ),
    category="prediction",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_id": {
                "type": "string",
                "description": "Sensor / specimen identifier (the whole recording is analysed).",
            },
            "target_variable": {
                "type": "string",
                "description": "Kept for compatibility (default 'condition').",
                "default": "condition",
            },
        },
        "required": ["sensor_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "model_available":         {"type": "boolean"},
            "condition":               {"type": ["string", "null"]},
            "condition_confidence":    {"type": ["number", "null"]},
            "severity_mm":             {"type": ["integer", "null"]},
            "severity_confidence":     {"type": ["number", "null"]},
            "severity_label":          {"type": ["string", "null"]},
            "top_parameters":          {"type": "array"},
            "baseline_comparison":     {"type": "object"},
            "agreement":               {"type": "string"},
            "agreement_note":          {"type": "string"},
            "status":                  {"type": "string"},
            "message":                 {"type": "string"},
            "caveats":                 {"type": "array", "items": {"type": "string"}},
            "sensor_id":               {"type": "string"},
            "target_variable":         {"type": "string"},
            "predicted_class":         {"type": "string"},
            "confidence":              {"type": "number"},
            "features_used":           {"type": "array", "items": {"type": "string"}},
            "prediction_timestamp":    {"type": "string"},
            "model_note":              {"type": "string"},
            "damage_class":            {"type": ["string", "null"]},
            "severity":                {"type": ["string", "null"]},
            "contributing_parameters": {"type": "array", "items": {"type": "string"}},
        },
    },
    handler=handle_model_result,
))

registry.register(ToolDefinition(
    name="baseline",
    description=(
        "Compares the uploaded recording with the healthy baseline (median and spread of the "
        "healthy recordings). Returns overall_flag (normal/abnormal), ranked deviations with "
        "percent change (for example 'mode 2 frequency 6 percent higher'), the healthy "
        "reference and how many healthy recordings and cases the baseline uses."
    ),
    category="statistics",
    input_schema={
        "type": "object",
        "properties": {
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter.",
            },
            "metric": {
                "type": "string",
                "description": "Optional metric or column name.",
            },
        },
        "required": [],
    },
    output_schema={
        "type": "object",
        "properties": {
            "overall_flag":             {"type": "string"},
            "deviations":               {"type": "array"},
            "healthy_reference":        {"type": "object"},
            "n_healthy":                {"type": "object"},
            "summary":                  {"type": "string"},
            "caveat":                   {"type": "string"},
            "status":                   {"type": "string"},
            "sensors_evaluated":        {"type": "array", "items": {"type": "string"}},
            "deviation_ranking":        {"type": "array"},
            "healthy_baseline_summary": {"type": "object"},
            "note":                     {"type": "string"},
        },
    },
    handler=handle_baseline,
))

registry.register(ToolDefinition(
    name="chart_data",
    description=(
        "Generates an interactive Plotly visualization for a sensor measurement over time. "
        "Sensor_N recordings use analysis.charts (optional anomaly markers); other data uses "
        "visualization_service.analyze_and_plot. Returns the Plotly figure as a JSON string and a text explanation."
    ),
    category="visualization",
    input_schema={
        "type": "object",
        "properties": {
            "y_col": {
                "type": "string",
                "description": "Numerical Y-axis column (e.g. 'deflection', 'Sensor_1').",
            },
            "x_col": {
                "type": "string",
                "description": "X-axis column (default 'timestamp' / 'Relative_Time_Sec').",
                "default": "timestamp",
            },
            "show_anomalies": {
                "type": "boolean",
                "description": "Mark anomalous readings on a Sensor_N chart (optional).",
                "default": False,
            },
            "threshold": {
                "type": "number",
                "description": "Anomaly threshold (robust z-score) used with show_anomalies.",
                "default": 5.0,
            },
            "sensor_id": {
                "type": "string",
                "description": "Optional sensor_id filter used with show_anomalies.",
            },
            "comparison": {
                "type": "string",
                "enum": ["signals", "features", "heatmap"],
                "description": (
                    "Compare the uploaded recording with the healthy reference: 'signals' "
                    "(overlay), 'features' (feature deviations vs the healthy range) or "
                    "'heatmap' (sensor x feature)."
                ),
            },
        },
        "required": ["y_col"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "x_col":       {"type": "string"},
            "y_col":       {"type": "string"},
            "plot_json":   {"type": "string"},
            "explanation": {"type": "string"},
        },
    },
    handler=handle_chart_data,
))