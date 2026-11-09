"""Week 6 Test Suite - Analytical Tool Interfaces for Talking to Bridges.

Covers:
- ToolRegistry: registration, lookup, validation, execution, error handling.
- All 6 tool handlers with real fixture data and placeholder fallback.
- ToolDispatcher: intent detection, argument building, dispatch, LLM summary.
- JSON serialization of all tool results.
- Error contracts: UNKNOWN_TOOL, INVALID_INPUT, TOOL_EXECUTION_ERROR.
- RAG flow independence: tools do not break RAG retrieval path.
- Schemas: Pydantic validation with valid and invalid data.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis.tools import (
    DataAccessLayer,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    data_access,
    registry,
)
from analysis.dispatcher import (
    ToolDispatcher,
    DispatchResult,
    dispatcher,
    _detect_tool,
    _extract_metric,
    _tokenize,
)
from analysis.schemas import (
    TOOL_INPUT_SCHEMAS,
    TOOL_OUTPUT_SCHEMAS,
    SummaryStatisticsInput,
    AnomalyDetectionInput,
    TrendAnalysisInput,
    CorrelationAnalysisInput,
    ModelResultInput,
    ChartDataInput,
    ToolErrorOutput,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BRIDGE_CSV = FIXTURES_DIR / "bridge_sensor_data.csv"

# ---------------------------------------------------------------------------
# Shared fixture DataFrame (the professor-fixture CSV)
# ---------------------------------------------------------------------------

@pytest.fixture
def fixture_df():
    return data_access.load_dataset(BRIDGE_CSV)


@pytest.fixture
def sensor_df():
    """Minimal professor-style dataset with Sensor_1..5 and Relative_Time_Sec."""
    import numpy as np
    n = 50
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "DateTime": pd.date_range("1970-01-01", periods=n, freq="3ms"),
        "Relative_Time_Sec": [i * 0.003 for i in range(n)],
        "Sensor_1": rng.normal(0.5, 0.1, n),
        "Sensor_2": rng.normal(0.3, 0.05, n),
        "Sensor_3": rng.normal(0.7, 0.2, n),
        "Sensor_4": rng.normal(0.4, 0.15, n),
        "Sensor_5": rng.normal(0.6, 0.12, n),
        "Condition": ["Damaged"] * n,
        "Test_Name": ["UD-SH1"] * n,
        "Source_File": ["test/file.xlsx"] * n,
        "Damage_Level": ["3mm"] * n,
        "Specimen": ["M1"] * n,
        "Test_Type": ["Singlehit"] * n,
        "Hit_Group": [""] * n,
    })
    # Inject a clear anomaly in Sensor_1
    df.loc[25, "Sensor_1"] = 50.0
    return df


# =============================================================================
# 1. ToolRegistry
# =============================================================================

class TestToolRegistryCore:
    """Core registry operations."""

    def test_registry_contains_all_six_tools(self):
        names = {t["name"] for t in registry.list_tools()}
        assert names == {
            "summary_statistics",
            "anomaly_detection",
            "trend_analysis",
            "correlation_analysis",
            "model_result",
            "chart_data",
            "baseline",
        }

    def test_register_and_lookup_custom_tool(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="my_tool",
            description="test",
            category="test",
            input_schema={"type": "object", "required": ["x"]},
            output_schema={"type": "object"},
            handler=lambda args, df: {"x": args["x"] * 2},
        ))
        tool = reg.get_tool("my_tool")
        assert tool.name == "my_tool"
        assert tool.category == "test"

    def test_unknown_tool_returns_error_dict(self):
        res = registry.execute("no_such_tool", {})
        assert res["status"] == "error"
        assert res["error_type"] == "UNKNOWN_TOOL"
        assert "no_such_tool" in res["message"]

    def test_list_tools_is_json_serializable(self):
        tools = registry.list_tools()
        # Must not raise
        text = json.dumps(tools)
        parsed = json.loads(text)
        assert len(parsed) == 7

    def test_handler_replacement_pattern(self):
        """Simulates how Kolla/Eswar/Krishna would swap in real implementations."""
        reg = ToolRegistry()
        t = ToolDefinition(
            name="swappable",
            description="placeholder",
            category="test",
            input_schema={"type": "object", "required": ["metric"]},
            output_schema={"type": "object"},
            handler=lambda args, df: {"result": "placeholder"},
        )
        reg.register(t)
        assert reg.execute("swappable", {"metric": "x"})["data"]["result"] == "placeholder"

        # Real implementation swap
        t.handler = lambda args, df: {"result": "real_implementation"}
        assert reg.execute("swappable", {"metric": "x"})["data"]["result"] == "real_implementation"


# =============================================================================
# 2. Input Validation
# =============================================================================

class TestInputValidation:
    """Validate tool argument checking before handler is called."""

    def test_summary_statistics_missing_metric(self):
        res = registry.execute("summary_statistics", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "metric" in res["message"]

    def test_anomaly_detection_missing_metric(self):
        res = registry.execute("anomaly_detection", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"

    def test_model_result_missing_sensor_id(self):
        res = registry.execute("model_result", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"

    def test_chart_data_missing_y_col(self):
        res = registry.execute("chart_data", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"

    def test_column_not_in_df(self, fixture_df):
        res = registry.execute(
            "summary_statistics", {"metric": "nonexistent_col"}, df=fixture_df
        )
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "nonexistent_col" in res["message"]

    def test_invalid_sensor_id_in_df(self, fixture_df):
        res = registry.execute(
            "summary_statistics",
            {"metric": "deflection", "sensor_id": "S99"},
            df=fixture_df,
        )
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "S99" in res["message"]

    def test_correlation_columns_not_in_df(self, fixture_df):
        res = registry.execute(
            "correlation_analysis",
            {"columns": ["deflection", "nonexistent"]},
            df=fixture_df,
        )
        assert res["status"] == "error"
        assert "nonexistent" in res["message"]


# =============================================================================
# 3. summary_statistics handler
# =============================================================================

class TestSummaryStatistics:
    def test_real_data_fixture(self, fixture_df):
        res = registry.execute(
            "summary_statistics", {"metric": "deflection", "sensor_id": "S01"}, df=fixture_df
        )
        assert res["status"] == "success"
        d = res["data"]
        assert d["count"] > 0
        assert isinstance(d["mean"], float)
        assert isinstance(d["std"], float)
        assert d["min"] <= d["mean"] <= d["max"]
        assert "rms" in d
        assert "p2p" in d

    def test_real_data_professor_schema(self, sensor_df):
        res = registry.execute(
            "summary_statistics", {"metric": "Sensor_1"}, df=sensor_df
        )
        assert res["status"] == "success"
        d = res["data"]
        assert d["count"] == len(sensor_df)
        assert d["rms"] is not None

    def test_placeholder_fallback(self):
        res = registry.execute("summary_statistics", {"metric": "deflection"})
        assert res["status"] == "success"
        assert "_note" in res["data"]

    def test_result_json_serializable(self, fixture_df):
        res = registry.execute(
            "summary_statistics", {"metric": "deflection"}, df=fixture_df
        )
        json.dumps(res)  # Must not raise


# =============================================================================
# 4. anomaly_detection handler
# =============================================================================

class TestAnomalyDetection:
    def test_real_data_detects_injected_anomaly(self, sensor_df):
        # Sensor_1 has a 50.0 spike at row 25
        res = registry.execute(
            "anomaly_detection", {"metric": "Sensor_1", "threshold": 2.0}, df=sensor_df
        )
        assert res["status"] == "success"
        d = res["data"]
        assert d["anomaly_count"] >= 1
        assert d["status_flag"] == "Alert"
        assert isinstance(d["anomaly_fraction"], float)
        assert 0 <= d["anomaly_fraction"] <= 1.0

    def test_normal_data_returns_normal_flag(self, fixture_df):
        """S02 temperature readings are all within normal range."""
        res = registry.execute(
            "anomaly_detection",
            {"metric": "temperature", "sensor_id": "S02", "threshold": 10.0},
            df=fixture_df,
        )
        assert res["status"] == "success"
        assert res["data"]["status_flag"] == "Normal"

    def test_placeholder_fallback(self):
        res = registry.execute("anomaly_detection", {"metric": "vibration"})
        assert res["status"] == "success"
        assert res["data"]["anomaly_count"] == 2

    def test_anomalous_indices_capped_at_50(self, sensor_df):
        # Create many anomalies
        df = sensor_df.copy()
        df["Sensor_2"] = 999.0  # all rows are anomalous
        res = registry.execute(
            "anomaly_detection", {"metric": "Sensor_2", "threshold": 0.5}, df=df
        )
        assert res["status"] == "success"
        assert len(res["data"]["anomalous_indices"]) <= 50


# =============================================================================
# 5. trend_analysis handler
# =============================================================================

class TestTrendAnalysis:
    def test_increasing_trend_detected(self, fixture_df):
        res = registry.execute(
            "trend_analysis", {"metric": "deflection"}, df=fixture_df
        )
        assert res["status"] == "success"
        d = res["data"]
        assert d["data_points"] > 0
        assert isinstance(d["slope"], float)
        assert 0.0 <= d["r_squared"] <= 1.0
        assert d["trend_direction"] in ("increasing", "decreasing", "flat")

    def test_uses_relative_time_sec_for_professor_data(self, sensor_df):
        res = registry.execute(
            "trend_analysis", {"metric": "Sensor_1"}, df=sensor_df
        )
        assert res["status"] == "success"
        assert res["data"]["time_column"] == "Relative_Time_Sec"

    def test_placeholder_fallback(self):
        res = registry.execute("trend_analysis", {"metric": "deflection"})
        assert res["status"] == "success"
        assert "_note" in res["data"]

    def test_json_serializable(self, fixture_df):
        res = registry.execute("trend_analysis", {"metric": "deflection"}, df=fixture_df)
        json.dumps(res)


# =============================================================================
# 6. correlation_analysis handler
# =============================================================================

class TestCorrelationAnalysis:
    def test_all_numeric_columns_correlated(self, fixture_df):
        res = registry.execute("correlation_analysis", {}, df=fixture_df)
        assert res["status"] == "success"
        d = res["data"]
        assert len(d["columns_used"]) >= 2
        assert "correlation_matrix" in d
        # Diagonal should be 1.0
        for col in d["columns_used"]:
            assert abs(d["correlation_matrix"][col][col] - 1.0) < 1e-9

    def test_specific_columns_selection(self, fixture_df):
        res = registry.execute(
            "correlation_analysis",
            {"columns": ["deflection", "vibration"]},
            df=fixture_df,
        )
        assert res["status"] == "success"
        assert set(res["data"]["columns_used"]) == {"deflection", "vibration"}

    def test_professor_sensor_columns(self, sensor_df):
        res = registry.execute(
            "correlation_analysis",
            {"columns": ["Sensor_1", "Sensor_2", "Sensor_3"]},
            df=sensor_df,
        )
        assert res["status"] == "success"
        assert "strongest_pair" in res["data"]

    def test_no_nan_in_matrix(self, fixture_df):
        res = registry.execute("correlation_analysis", {}, df=fixture_df)
        mat = res["data"]["correlation_matrix"]
        for row in mat.values():
            for val in row.values():
                assert not math.isnan(val)

    def test_placeholder_fallback(self):
        res = registry.execute("correlation_analysis", {})
        assert res["status"] == "success"
        assert "_note" in res["data"]


# =============================================================================
# 7. model_result handler
# =============================================================================

class TestModelResult:
    def test_real_data_condition_lookup(self, fixture_df):
    # NEW-4: the label column is no longer read. This fixture is not a sensor recording.
       res = registry.execute(
        "model_result",
        {"sensor_id": "S01", "target_variable": "condition"},
        df=fixture_df,
    )
       assert res["status"] == "success"
       d = res["data"]
       assert d["status"] == "not_a_recording"
       assert d["predicted_class"] == "Unavailable"
       assert d["condition"] is None

    def test_placeholder_fallback(self):
        res = registry.execute("model_result", {"sensor_id": "S01"})
        assert res["status"] == "success"
        assert "model_note" in res["data"]

    def test_json_serializable(self):
        res = registry.execute("model_result", {"sensor_id": "S01"})
        json.dumps(res)


# =============================================================================
# 8. chart_data handler
# =============================================================================

class TestChartData:
    def test_placeholder_when_no_df(self):
        res = registry.execute("chart_data", {"y_col": "deflection"})
        assert res["status"] == "success"
        d = res["data"]
        assert "y_col" in d
        assert isinstance(d["plot_json"], str)
        assert isinstance(d["explanation"], str)

    def test_json_serializable_placeholder(self):
        res = registry.execute("chart_data", {"y_col": "deflection"})
        json.dumps(res)


# =============================================================================
# 9. ToolDispatcher intent detection
# =============================================================================

class TestIntentDetection:
    """Unit-test the intent detection logic without executing tools."""

    @pytest.mark.parametrize("query, expected_tool", [
        ("What are the statistics for deflection?", "summary_statistics"),
        ("What is the average vibration?", "summary_statistics"),
        ("Are there any anomalies in the sensor data?", "anomaly_detection"),
        ("Detect outliers in stress readings", "anomaly_detection"),
        ("Is deflection trending upward?", "trend_analysis"),
        ("What is the trend for Sensor_1?", "trend_analysis"),
        ("How correlated are the sensor readings?", "correlation_analysis"),
        ("What is the correlation between sensors?", "correlation_analysis"),
        ("What is the predicted condition?", "model_result"),
        ("Show me a chart of deflection", "chart_data"),
        ("Plot Sensor_1 over time", "chart_data"),
    ])
    def test_intent_detection(self, query, expected_tool):
        detected = _detect_tool(query)
        assert detected == expected_tool, (
            f"Query '{query}' -> got '{detected}', expected '{expected_tool}'"
        )

    def test_no_analytical_intent_returns_none(self):
        assert _detect_tool("Hello, who are you?") is None
        assert _detect_tool("What is a bridge?") is None
        assert _detect_tool("Upload a document") is None


# =============================================================================
# 10. ToolDispatcher end-to-end
# =============================================================================

class TestToolDispatcher:
    def test_dispatch_summary_returns_result(self, fixture_df):
        result = dispatcher.detect_and_dispatch(
            "What are the statistics for deflection?", df=fixture_df
        )
        assert result is not None
        assert isinstance(result, DispatchResult)
        assert result.status == "success"
        assert result.tool_name == "summary_statistics"
        assert result.llm_summary.startswith("[ANALYTICAL RESULT:")

    def test_dispatch_anomaly_returns_result(self, sensor_df):
        result = dispatcher.detect_and_dispatch(
            "Are there any anomalies in Sensor_1?", df=sensor_df
        )
        assert result is not None
        assert result.status == "success"
        assert result.tool_name == "anomaly_detection"

    def test_dispatch_trend_returns_result(self, fixture_df):
        result = dispatcher.detect_and_dispatch(
            "What is the trend in deflection?", df=fixture_df
        )
        assert result is not None
        assert result.tool_name == "trend_analysis"

    def test_dispatch_correlation_returns_result(self, fixture_df):
        result = dispatcher.detect_and_dispatch(
            "What is the correlation between sensors?", df=fixture_df
        )
        assert result is not None
        assert result.tool_name == "correlation_analysis"

    def test_no_intent_returns_none(self, fixture_df):
        result = dispatcher.detect_and_dispatch("Hello world", df=fixture_df)
        assert result is None

    def test_dispatch_without_df_uses_placeholder(self):
        result = dispatcher.detect_and_dispatch("What are the statistics for deflection?")
        assert result is not None
        assert result.status == "success"
        assert "_note" in result.data

    def test_dispatch_error_propagated_cleanly(self):
        """Requesting a metric not in df should return error DispatchResult."""
        df = pd.DataFrame({"Sensor_1": [1.0, 2.0]})
        result = dispatcher.dispatch(
            "summary_statistics", {"metric": "nonexistent"}, df=df
        )
        assert result.status == "error"
        assert result.error_type is not None

    def test_llm_summary_is_string(self, fixture_df):
        result = dispatcher.detect_and_dispatch(
            "Calculate statistics for deflection", df=fixture_df
        )
        assert isinstance(result.llm_summary, str)
        assert len(result.llm_summary) > 0

    def test_list_available_tools(self):
        tools = dispatcher.list_available_tools()
        assert len(tools) == 7
        names = {t["name"] for t in tools}
        assert "trend_analysis" in names
        assert "correlation_analysis" in names
        assert "baseline" in names


# =============================================================================
# 11. Pydantic schema validation
# =============================================================================

class TestPydanticSchemas:
    def test_summary_statistics_input_valid(self):
        inp = SummaryStatisticsInput(metric="deflection", sensor_id="S01")
        assert inp.metric == "deflection"
        assert inp.sensor_id == "S01"
        assert inp.start_time is None

    def test_summary_statistics_input_missing_metric(self):
        with pytest.raises(Exception):  # pydantic ValidationError
            SummaryStatisticsInput()

    def test_anomaly_detection_input_threshold_bounds(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            AnomalyDetectionInput(metric="vibration", threshold=0.1)  # below 0.5
        with pytest.raises(ValidationError):
            AnomalyDetectionInput(metric="vibration", threshold=11.0)  # above 10.0

    def test_trend_analysis_input_valid(self):
        inp = TrendAnalysisInput(metric="Sensor_1", time_column="Relative_Time_Sec")
        assert inp.time_column == "Relative_Time_Sec"

    def test_correlation_analysis_defaults(self):
        inp = CorrelationAnalysisInput()
        assert inp.columns is None
        assert inp.sensor_id is None

    def test_model_result_input_requires_sensor_id(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            ModelResultInput(target_variable="condition")  # sensor_id missing

    def test_chart_data_input_valid(self):
        inp = ChartDataInput(y_col="deflection", x_col="timestamp")
        assert inp.y_col == "deflection"

    def test_tool_error_output(self):
        err = ToolErrorOutput(
            tool="summary_statistics",
            error_type="INVALID_INPUT",
            message="Column 'xyz' not found.",
        )
        assert err.status == "error"
        j = json.dumps(err.model_dump())
        assert "INVALID_INPUT" in j

    def test_schema_registries_cover_all_tools(self):
        expected = {
            "summary_statistics", "anomaly_detection", "trend_analysis",
            "correlation_analysis", "model_result", "chart_data", "baseline",
        }
        assert set(TOOL_INPUT_SCHEMAS.keys()) == expected
        assert set(TOOL_OUTPUT_SCHEMAS.keys()) == expected


# =============================================================================
# 12. DataAccessLayer
# =============================================================================

class TestDataAccessLayer:
    def test_load_fixture_csv(self, fixture_df):
        assert len(fixture_df) == 30
        assert "deflection" in fixture_df.columns
        assert "sensor_id" in fixture_df.columns

    def test_inspect_dataset(self, fixture_df):
        meta = data_access.inspect_dataset(fixture_df)
        assert meta["row_count"] == 30
        assert "deflection" in meta["numeric_columns"]
        assert meta["timestamp_column"] == "timestamp"

    def test_get_numeric_series_valid(self, fixture_df):
        series = data_access.get_numeric_series(fixture_df, "deflection", sensor_id="S01")
        assert len(series) > 0
        assert series.dtype.kind == "f"

    def test_get_numeric_series_invalid_column(self, fixture_df):
        with pytest.raises(ToolExecutionError) as exc_info:
            data_access.get_numeric_series(fixture_df, "nonexistent")
        assert exc_info.value.error_type == "INVALID_INPUT"

    def test_load_from_bytes(self):
        csv_bytes = b"a,b\n1.0,2.0\n3.0,4.0\n"
        df = data_access._parse_bytes(csv_bytes, "test.csv")
        assert list(df.columns) == ["a", "b"]
        assert len(df) == 2


# =============================================================================
# 13. End-to-end: all 6 tools run successfully with fixture data
# =============================================================================

class TestAllToolsWithFixture:
    @pytest.mark.parametrize("tool_name, args", [
        ("summary_statistics", {"metric": "deflection"}),
        ("anomaly_detection",  {"metric": "vibration", "threshold": 2.0}),
        ("trend_analysis",     {"metric": "deflection"}),
        ("correlation_analysis", {}),
        ("model_result",       {"sensor_id": "S01"}),
    ])
    def test_tool_succeeds_with_fixture_df(self, fixture_df, tool_name, args):
        res = registry.execute(tool_name, args, df=fixture_df)
        assert res["status"] == "success", f"{tool_name} failed: {res.get('message')}"
        assert isinstance(res["data"], dict)

    @pytest.mark.parametrize("tool_name, args", [
        ("summary_statistics", {"metric": "deflection"}),
        ("anomaly_detection",  {"metric": "vibration"}),
        ("trend_analysis",     {"metric": "deflection"}),
        ("correlation_analysis", {}),
        ("model_result",       {"sensor_id": "S01"}),
        ("chart_data",         {"y_col": "deflection"}),
    ])
    def test_all_results_json_serializable(self, fixture_df, tool_name, args):
        df = fixture_df if tool_name != "chart_data" else None
        res = registry.execute(tool_name, args, df=df)
        # chart_data with a real df may call plotly; skip if fixture is fine without it
        text = json.dumps(res)
        assert len(text) > 0


# =============================================================================
# 14. Error contract verification
# =============================================================================

class TestErrorContracts:
    def test_unknown_tool_error_shape(self):
        res = registry.execute("ghost_tool", {})
        assert res["status"] == "error"
        assert res["error_type"] == "UNKNOWN_TOOL"
        assert isinstance(res["message"], str)
        assert isinstance(res.get("details", {}), dict)

    def test_invalid_input_error_shape(self):
        res = registry.execute("summary_statistics", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "details" in res

    def test_error_results_are_json_serializable(self):
        for err_result in [
            registry.execute("ghost_tool", {}),
            registry.execute("summary_statistics", {}),
            registry.execute("anomaly_detection", {}),
        ]:
            json.dumps(err_result)
