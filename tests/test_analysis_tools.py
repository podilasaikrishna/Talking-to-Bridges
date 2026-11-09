"""Unit tests for Week 5 Analytical Tool Architecture."""

import pytest
import pandas as pd
from pathlib import Path

from analysis.tools import (
    DataAccessLayer,
    ToolDefinition,
    ToolExecutionError,
    ToolRegistry,
    data_access,
    registry,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BRIDGE_CSV = FIXTURES_DIR / "bridge_sensor_data.csv"


class TestToolRegistry:
    """Tests for ToolRegistry registration, discovery, validation, and execution."""

    def test_tool_registration_and_listing(self):
        reg = ToolRegistry()
        def dummy_handler(args, df=None):
            return {"val": 123}

        tool_def = ToolDefinition(
            name="test_tool",
            description="Test analytical tool",
            category="test",
            input_schema={"type": "object", "required": ["param1"]},
            output_schema={"type": "object"},
            handler=dummy_handler
        )
        reg.register(tool_def)
        
        tools = reg.list_tools()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert tools[0]["category"] == "test"

    def test_valid_tool_lookup(self):
        tool = registry.get_tool("summary_statistics")
        assert tool.name == "summary_statistics"
        assert tool.category == "statistics"

    def test_unknown_tool_raises_structured_error(self):
        res = registry.execute("non_existent_tool", {})
        assert res["status"] == "error"
        assert res["error_type"] == "UNKNOWN_TOOL"
        assert "non_existent_tool" in res["message"]

    def test_input_schema_validation_missing_param(self):
        res = registry.execute("summary_statistics", {})
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "metric" in res["message"]

    def test_dataset_aware_column_validation(self):
        df = pd.DataFrame({"temperature": [31.2, 32.1], "sensor_id": ["S01", "S01"]})
        res = registry.execute("summary_statistics", {"metric": "non_existent_column"}, df=df)
        assert res["status"] == "error"
        assert res["error_type"] == "INVALID_INPUT"
        assert "non_existent_column" in res["message"]

    def test_successful_dummy_tool_execution(self):
        res = registry.execute("summary_statistics", {"metric": "deflection", "sensor_id": "S01"})
        assert res["status"] == "success"
        assert res["tool"] == "summary_statistics"
        assert res["data"]["metric"] == "deflection"
        assert "mean" in res["data"]

    def test_dummy_anomaly_detection_execution(self):
        res = registry.execute("anomaly_detection", {"metric": "vibration", "sensor_id": "S01"})
        assert res["status"] == "success"
        assert res["data"]["anomaly_count"] >= 0
        assert "status_flag" in res["data"]

    def test_dummy_model_result_execution(self):
        res = registry.execute("model_result", {"sensor_id": "S01"})
        assert res["status"] == "success"
        assert "predicted_class" in res["data"]

    def test_handler_replacement_in_registry(self):
        reg = ToolRegistry()
        
        def initial_handler(args, df=None):
            return {"result": "dummy"}
            
        def real_handler(args, df=None):
            return {"result": "real_calculation_42"}

        tool = ToolDefinition(
            name="calc",
            description="Calculation tool",
            category="math",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            handler=initial_handler
        )
        reg.register(tool)
        assert reg.execute("calc", {})["data"]["result"] == "dummy"

        # Replace handler (Simulating Week 6 replacement)
        tool.handler = real_handler
        assert reg.execute("calc", {})["data"]["result"] == "real_calculation_42"


class TestDataAccessLayer:
    """Tests for DataAccessLayer and professor dataset inspection."""

    def test_load_dataset_from_fixture(self):
        df = data_access.load_dataset(BRIDGE_CSV)
        assert len(df) == 30
        assert "deflection" in df.columns
        assert "sensor_id" in df.columns

    def test_inspect_dataset_metadata(self):
        df = data_access.load_dataset(BRIDGE_CSV)
        meta = data_access.inspect_dataset(df)
        assert meta["row_count"] == 30
        assert "deflection" in meta["numeric_columns"]
        assert meta["timestamp_column"] == "timestamp"

    def test_tool_execution_with_real_dataset(self):
        df = data_access.load_dataset(BRIDGE_CSV)
        res = registry.execute("summary_statistics", {"metric": "deflection", "sensor_id": "S01"}, df=df)
        assert res["status"] == "success"
        assert res["data"]["count"] > 0
        assert isinstance(res["data"]["mean"], float)
