# Phase 2 — Week 6: Analytical Tool Interfaces

**Project:** Talking to Bridges — LLM-Based Intelligent Interface for Structural Health Monitoring  
**Author:** Pavan (Tool interface / agent layer)  
**Week:** Phase 2, Week 6  
**Status:** Implementation complete — interfaces live, tested, and integrated  

---

## 1. Overview and Scope

Week 6 implements the **tool interface layer** that allows the LLM to invoke analytical functions.

This does **not** implement the underlying statistical algorithms, ML models, or visualizations  
(those belong to Eswar, Kolla, Krishna, and Nagarjun respectively). Instead, it creates:

- Stable **input/output contracts** (schemas) that other team members plug their implementations into.
- A **ToolRegistry** for registering, discovering, and executing tools.
- A **ToolDispatcher** that routes user queries to the correct tool.
- Clean **LLM integration** so tool results are injected into the prompt context.
- Real **analytical implementations** for tools that Pavan owns (trend_analysis, correlation_analysis),  
  and **placeholder handlers** that other team members replace with their real logic.

---

## 2. Repository Architecture (Week 6 additions)

```
analysis/
├── __init__.py          # Package exports (tools, dispatcher, schemas)
├── tools.py             # ToolRegistry, DataAccessLayer, 6 tool handlers
├── dispatcher.py        # ToolDispatcher: intent detection + tool execution
├── schemas.py           # Pydantic input/output schemas for all 6 tools
├── sensor_pipeline.py   # Kolla's preprocessing pipeline (unchanged)
├── eda.py               # Krishna's EDA analysis (unchanged)
└── eda_summary.py       # EDA output (unchanged)

app/api/routes.py        # /chat updated to use ToolDispatcher
rag/prompts.py           # System prompt updated for analytical tool context
tests/
└── test_week6_tools.py  # 72 new tests for Week 6
docs/
└── phase2_week6_analytical_tools.md  # This document
```

---

## 3. Tool Interface Architecture

### 3.1 Data Flow

```
User voice / text query
    │
    ▼
frontend/voice_ui.html  (primary UI)
    │
    ▼
FastAPI /chat endpoint  (app/api/routes.py)
    │
    ├──────────────────────────────────────────┐
    │                                          │
    ▼                                          ▼
ToolDispatcher                           FAISS Retriever
(analysis/dispatcher.py)                 (RAG pipeline)
    │                                          │
    ▼                                          ▼
intent detection                         Retrieved chunks
    │
    ▼
ToolRegistry.execute(name, args, df)
(analysis/tools.py)
    │
    ├── validate_arguments()
    ├── resolve handler
    └── handler(args, df)
           │
           ▼
    DataAccessLayer
    (loads/filters DataFrame)
           │
           ▼
    Structured result dict
           │
    ┌──────┴────────────────────────────┐
    │                                   │
    ▼                                   ▼
LLM context injection              LLM context injection
(tool result block)                (RAG chunk block)
           │
           ▼
    LLM generates natural-language response
           │
           ▼
    TTS → voice_ui.html
```

### 3.2 Key Principle: RAG ≠ Analytics

| | RAG | Analytics |
|---|---|---|
| Input | Uploaded document/CSV text | Structured numerical dataset |
| Storage | FAISS vector index | Pandas DataFrame |
| Output | Retrieved text chunks | Computed statistical results |
| LLM role | Answer from text context | Interpret computed numbers |

Both can contribute to the **same** LLM call. The system prompt now explicitly instructs the LLM to use exact numbers from `[Analytical Tool: ...]` blocks and not fabricate them.

---

## 4. Tool Registry (analysis/tools.py)

### API

```python
from analysis.tools import registry

# Register a new tool (e.g. Eswar adds real stats implementation)
registry.register(ToolDefinition(
    name="summary_statistics",
    description="...",
    category="statistics",
    input_schema={...},
    output_schema={...},
    handler=my_real_handler,
))

# Look up a tool
tool = registry.get_tool("summary_statistics")

# List all available tools (JSON-serializable)
tools = registry.list_tools()

# Execute with validation and error handling
result = registry.execute("summary_statistics", {"metric": "deflection"}, df=df)
# result = {"tool": "...", "status": "success"/"error", "data": {...}, ...}
```

### Error Codes

| Code | When |
|---|---|
| `UNKNOWN_TOOL` | Tool name not registered |
| `INVALID_INPUT` | Required parameter missing, or column not in dataset |
| `TOOL_EXECUTION_ERROR` | Unexpected exception inside handler |

---

## 5. Registered Tools

### 5.1 summary_statistics

**Owner:** Eswar (statistics interface)  
**Category:** statistics

| Input | Type | Required | Description |
|---|---|---|---|
| `metric` | string | ✓ | Column name (e.g. `deflection`, `Sensor_1`) |
| `sensor_id` | string | | Filter rows by sensor ID |
| `start_time` | string | | ISO-8601 start date filter |
| `end_time` | string | | ISO-8601 end date filter |

| Output field | Type | Description |
|---|---|---|
| `count` | int | Number of valid data points |
| `min`, `max`, `mean`, `median`, `std` | float | Basic statistics |
| `rms` | float | Root-mean-square |
| `p2p` | float | Peak-to-peak range |

---

### 5.2 anomaly_detection

**Owner:** Kolla (anomaly detection interface)  
**Category:** anomaly  
**Current implementation:** Z-score thresholding placeholder  

| Input | Type | Required | Description |
|---|---|---|---|
| `metric` | string | ✓ | Column to scan |
| `sensor_id` | string | | Filter rows |
| `threshold` | float | | Z-score threshold (default 2.0, range 0.5–10.0) |

| Output field | Type | Description |
|---|---|---|
| `anomaly_count` | int | Number of anomalous readings |
| `anomaly_fraction` | float | Fraction of total rows that are anomalous |
| `anomalous_timestamps` | list[str] | Timestamps of flagged readings |
| `status_flag` | string | `"Normal"` or `"Alert"` |

**To plug in Kolla's ML detector:**
```python
from analysis.tools import registry
tool = registry.get_tool("anomaly_detection")
tool.handler = kolla_real_anomaly_handler  # same (args, df) -> dict signature
```

---

### 5.3 trend_analysis

**Owner:** Pavan (Week 6 implementation)  
**Category:** trend  

| Input | Type | Required | Description |
|---|---|---|---|
| `metric` | string | ✓ | Column to analyse |
| `time_column` | string | | Override X-axis column (defaults to `Relative_Time_Sec` or `timestamp`) |
| `sensor_id` | string | | Filter rows |

| Output field | Type | Description |
|---|---|---|
| `slope` | float | Linear regression slope |
| `r_squared` | float | Coefficient of determination |
| `trend_direction` | string | `"increasing"` / `"decreasing"` / `"flat"` |
| `change_magnitude` | float | `end_value - start_value` |

---

### 5.4 correlation_analysis

**Owner:** Pavan (Week 6 implementation)  
**Category:** correlation  

| Input | Type | Required | Description |
|---|---|---|---|
| `columns` | list[str] | | Columns to correlate (defaults to all numeric) |
| `sensor_id` | string | | Filter rows |

| Output field | Type | Description |
|---|---|---|
| `columns_used` | list[str] | Columns included in the matrix |
| `correlation_matrix` | dict[str, dict[str, float]] | Pearson correlation matrix |
| `strongest_pair` | string | Most correlated pair with r value |
| `weakest_pair` | string | Least correlated pair |

---

### 5.5 model_result

**Owner:** Krishna (ML inference interface — Week 7)  
**Category:** prediction  
**Current implementation:** Placeholder returning last known label from dataset  

| Input | Type | Required | Description |
|---|---|---|---|
| `sensor_id` | string | ✓ | Sensor/specimen identifier |
| `target_variable` | string | | Target column (default `"condition"`) |

**To plug in Krishna's ML model:**
```python
tool = registry.get_tool("model_result")
tool.handler = krishna_ml_inference_handler
```

---

### 5.6 chart_data

**Owner:** Nagarjun (visualization)  
**Category:** visualization  

| Input | Type | Required | Description |
|---|---|---|---|
| `y_col` | string | ✓ | Y-axis column |
| `x_col` | string | | X-axis column (default `"timestamp"`) |
| `show_anomalies` | boolean | | Mark anomalous readings on a `Sensor_N` chart (default `false`) |
| `threshold` | number | | Anomaly threshold used with `show_anomalies` (default `5.0`, same as the real-data default of `anomaly_detection`) |
| `sensor_id` | string | | Optional `sensor_id` filter used with `show_anomalies` |

| Output field | Type | Description |
|---|---|---|
| `x_col` | string | X column actually used |
| `y_col` | string | Y column(s) actually plotted |
| `plot_json` | string | Plotly figure serialised to JSON (`"{}"` when no chart could be built) |
| `explanation` | string | Human-readable chart summary |

**Behaviour (Week 6):**

1. Professor recordings (`Relative_Time_Sec` + `Sensor_N`): an interactive line chart built by
   `analysis/charts.py`, downsampled to about 5000 points. If the query names no sensor, all
   sensors are plotted. With `show_anomalies=true`, anomalies from `anomaly_detection`
   (Kolla's pipeline) are marked per **event**: a shaded span from each event's start to end,
   markers at its start/end and at its peak. Events are used (not the per-reading lists, which
   the pipeline caps at 50), so a recording with thousands of flagged readings is still shown
   in full; the explanation gives the true flagged count.
2. Any other dataset: the existing `app.services.visualization_service.analyze_and_plot`.
3. No usable data: the placeholder result (`plot_json = "{}"`).

The input and output contract is unchanged; the new input fields are optional.

**Chat flow:** `/chat` returns the figure as `fig` in `ChatResponse` when the dispatched tool is
`chart_data`, and the Streamlit app renders it with `st.plotly_chart`. For this the upload route
keeps the raw uploaded CSV in `DOCUMENTS_DIR` so `/chat` can load it as the active DataFrame.

**Known limitation:** the chat endpoint also sends retrieved CSV text to the LLM. For full-size
recordings (about 130 characters per row) this can exceed the Groq request size limit (HTTP 413).
Use a smaller file for demos until the context size is capped.

---

## 6. Tool Dispatcher (analysis/dispatcher.py)

The dispatcher is the bridge between the `/chat` endpoint and the tool registry.

### Intent Detection

The dispatcher uses keyword heuristics to map user queries to tools:

| Keywords (any match) | Tool |
|---|---|
| `chart`, `plot`, `graph`, `visualize`, `figure` | chart_data |
| `anomaly`, `outlier`, `alert`, `spike`, `abnormal` | anomaly_detection |
| `trend`, `trending`, `increasing`, `decreasing`, `slope` | trend_analysis |
| `correlat`, `correlation`, `relationship` | correlation_analysis |
| `predict`, `condition`, `classify`, `model`, `forecast` | model_result |
| `stat`, `average`, `mean`, `min`, `max`, `summary` | summary_statistics |

### API

```python
from analysis.dispatcher import dispatcher

# Auto-detect tool from plain text query
result = dispatcher.detect_and_dispatch("What is the trend in deflection?", df=df)
# Returns DispatchResult or None if no analytical intent found

# Manual dispatch
result = dispatcher.dispatch("trend_analysis", {"metric": "deflection"}, df=df)

# result.status       -> "success" | "error"
# result.data         -> dict from handler
# result.llm_summary  -> str injected into LLM context
# result.error_type   -> error code on failure
```

---

## 7. Pydantic Schemas (analysis/schemas.py)

All tool inputs and outputs have explicit Pydantic schemas for type safety.

```python
from analysis.schemas import SummaryStatisticsInput, AnomalyDetectionInput

# Validation before calling the dispatcher directly
inp = SummaryStatisticsInput(metric="deflection", sensor_id="S01")
# Raises pydantic.ValidationError for invalid inputs

from analysis.schemas import TOOL_INPUT_SCHEMAS
schema_class = TOOL_INPUT_SCHEMAS["trend_analysis"]
validated = schema_class(metric="Sensor_1")
```

---

## 8. LLM Integration (app/api/routes.py + rag/prompts.py)

The `/chat` endpoint now:

1. Resolves the active DataFrame from the last uploaded document.
2. Calls `dispatcher.detect_and_dispatch(query, df)`.
3. Builds two context blocks (RAG + analytical tool result).
4. Passes both to the updated system prompt.
5. The LLM receives structured numerical results directly and is instructed to use them faithfully.

The updated `SYSTEM_PROMPT` in `rag/prompts.py` explicitly tells the LLM:
> "When an [Analytical Tool] block is present, USE those exact numbers.  
> Do NOT guess, estimate, or contradict the numbers in the tool result."

---

## 9. Voice UI Integration

`frontend/voice_ui.html` remains the primary interface and is **unchanged**.

The entire Week 6 implementation is in the backend:

```
Voice query → STT → /chat → ToolDispatcher → Tool → LLM → TTS → voice_ui.html
```

The voice pipeline is fully functional with the new analytical layer because:
- The dispatcher returns `None` for non-analytical queries (RAG-only path unchanged).
- Tool errors are surfaced to the LLM as explanatory text, not exceptions.
- TTS receives the LLM's natural-language explanation of both RAG and tool results.

---

## 10. Professor Dataset Schema

The actual professor dataset (processed by Kolla's `sensor_pipeline.py`) produces:

| Column | Type | Description |
|---|---|---|
| `DateTime` | datetime | Constructed timestamp (1970-01-01 anchor) |
| `Relative_Time_Sec` | float | **Primary time axis for analysis** |
| `Sensor_1` .. `Sensor_5` | float | Sensor readings (unitless, uncalibrated) |
| `Condition` | string | `"Damaged"` / `"Undamaged"` |
| `Damage_Level` | string | `"Undamaged"`, `"1mm"`, `"2mm"`, `"3mm"` |
| `Specimen` | string | `"M1"` .. `"M7"` / `"1st"` / `"2nd"` |
| `Test_Type` | string | `"Singlehit"` / `"Multihit"` / `"Randomhit"` / `"Displacement"` |
| `Hit_Group` | string | `"2hit"` / `"3hit"` / `"4hit"` (or empty) |

**Important:** `Relative_Time_Sec` is the real time axis. `DateTime` uses a 1970-01-01 anchor date only.

---

## 11. How Team Members Plug In Their Implementations

### Replacing a handler (the only change needed)

```python
# Example: Kolla replaces the anomaly_detection placeholder
from analysis.tools import registry

def my_real_anomaly_handler(arguments, df):
    """Real ML-based anomaly detection."""
    metric = arguments["metric"]
    # ... real implementation using df ...
    return {
        "sensor_id": arguments.get("sensor_id"),
        "metric": metric,
        "threshold_z": arguments.get("threshold", 2.0),
        "total_rows_checked": len(df),
        "anomaly_count": <real_count>,
        "anomaly_fraction": <real_fraction>,
        "anomalous_indices": [...],
        "anomalous_timestamps": [...],
        "status_flag": "Alert" if <real_count> > 0 else "Normal",
    }

tool = registry.get_tool("anomaly_detection")
tool.handler = my_real_anomaly_handler
```

The schema, API endpoint, intent detection, and LLM integration all continue to work unchanged. Only the handler function changes.

---

## 12. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/tools` | List all registered tools |
| `POST` | `/api/tools/execute` | Execute a tool directly with arguments |
| `POST` | `/chat` | Main conversational endpoint (includes tool dispatch) |

### Execute a tool directly

```bash
curl -X POST http://localhost:8001/api/tools/execute \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "summary_statistics", "arguments": {"metric": "Sensor_1"}}'
```

---

## 13. Running Tests

```bash
# Week 6 tests only
python -m pytest tests/test_week6_tools.py -v

# Full analytical tool test suite
python -m pytest tests/test_week6_tools.py tests/test_analysis_tools.py -v

# All tests (excluding speech modules that require hardware)
python -m pytest tests/ --ignore=tests/test_speech.py -v
```

**Results (Week 6):** 95 tests passing, 0 failures.

---

## 14. Week 6 Definition of Done Checklist

- [x] Analytical tool interfaces exist (`analysis/tools.py`)
- [x] Tool inputs have defined Pydantic schemas (`analysis/schemas.py`)
- [x] Tool outputs have defined Pydantic schemas
- [x] Tool registry / dispatcher works (`analysis/dispatcher.py`)
- [x] LLM / chat layer can invoke the tool layer (via `/chat`)
- [x] Tool results returned to LLM as `llm_summary` context block
- [x] Tool errors handled predictably (error contracts)
- [x] Existing RAG flow still works (unchanged)
- [x] Existing CSV semantic retrieval still works (unchanged)
- [x] Existing STT/TTS flow preserved (unchanged)
- [x] `frontend/voice_ui.html` remains the primary interface (unchanged)
- [x] No Streamlit redesign introduced
- [x] Raw professor datasets remain untouched
- [x] Tests added and passing (95 pass)
- [x] Documentation updated (this file)
- [x] Architecture modular for Eswar / Kolla / Nagarjun / Krishna

---

## 15. What Week 7 Should Implement

| Who | Task |
|---|---|
| Eswar | Replace `summary_statistics` handler with real statistical implementation |
| Kolla | Replace `anomaly_detection` handler with real ML-based detector |
| Nagarjun | Replace `chart_data` handler with interactive dashboard integration |
| Krishna | Replace `model_result` handler with trained ML model inference |
| Pavan | Wire additional tools as needed; improve intent detection accuracy |
