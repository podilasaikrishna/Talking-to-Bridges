"""Unit and integration tests for Week 7 Pavan Tasks (Phase 2).

Tests covered:
1. model_result schema & handler extensions (damage_class, severity, confidence, contributing_parameters).
2. baseline tool registration, execution, and JSON serializability.
3. Routing of five professor questions through ToolDispatcher.
4. Inclusion of event details and peak scores in LLM summaries.
5. Verification that existing tools and RAG integration remain healthy.
"""

import json
import pytest
import pandas as pd

from analysis.tools import registry, ToolRegistry, handle_model_result, handle_baseline
from analysis.dispatcher import dispatcher, ToolDispatcher, DispatchResult
from analysis.schemas import (
    ModelResultInput, ModelResultOutput,
    BaselineInput, BaselineOutput,
    TOOL_INPUT_SCHEMAS, TOOL_OUTPUT_SCHEMAS,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame with professor-style columns and fixture columns."""
    return pd.DataFrame({
        "Relative_Time_Sec": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        "Sensor_1": [0.01, 0.02, 0.05, 1.20, 4.50, 1.10, 0.03, 0.01, 0.02, 0.01],
        "Sensor_2": [0.00, 0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.00, 0.01, 0.00],
        "Sensor_3": [0.05, 0.06, 0.05, 0.07, 0.08, 0.06, 0.05, 0.06, 0.05, 0.04],
        "condition": ["Healthy"] * 9 + ["Damaged"],
        "sensor_id": ["S01"] * 10,
        "vibration": [0.1, 0.2, 0.1, 5.2, 8.9, 4.1, 0.2, 0.1, 0.1, 0.2],
        "deflection": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
    })


# =============================================================================
# TASK 1: Model Result Extensions
# =============================================================================

class TestTask1ModelResult:
    def test_model_result_output_schema_contains_new_fields(self):
        output = ModelResultOutput(
            sensor_id="Sensor_1",
            target_variable="condition",
            predicted_class="Damaged",
            confidence=0.95,
            prediction_timestamp="2026-10-07",
            damage_class="Damaged_3mm",
            severity="High",
            contributing_parameters=["Sensor_1", "vibration"],
        )
        assert output.damage_class == "Damaged_3mm"
        assert output.severity == "High"
        assert output.confidence == 0.95
        assert "vibration" in output.contributing_parameters

    def test_handle_model_result_with_df(self, sample_df):
        res = handle_model_result({"sensor_id": "S01", "target_variable": "condition"}, df=sample_df)
        assert res["sensor_id"] == "S01"
        # NEW-4: the fixture's label column ("Damaged") is never read, and 10 rows have no
        # spectral peaks, so the real model cannot classify it.
        assert res["predicted_class"] != "Damaged" or res["model_available"] is True
        assert res["predicted_class"] in {"Damaged", "Undamaged", "Unavailable"}

    def test_handle_model_result_placeholder(self):
        res = handle_model_result({"sensor_id": "S01"})
        # NEW-4: with no uploaded recording nothing is invented.
        assert res["status"] == "no_data"
        assert res["predicted_class"] == "Unavailable"
        assert "upload" in res["message"].lower()


# =============================================================================
# TASK 2: Baseline Tool Registration & Pluggability
# =============================================================================

class TestTask2BaselineTool:
    def test_baseline_tool_is_registered(self):
        tool = registry.get_tool("baseline")
        assert tool is not None
        assert tool.name == "baseline"
        assert tool.category == "statistics"

    def test_baseline_tool_without_data_is_unavailable(self):
        res = registry.execute("baseline", {})
        assert res["status"] == "success"
        data = res["data"]
        assert data["overall_flag"] == "unavailable"
        assert data["status"] == "Unavailable"
        assert json.dumps(data) is not None

    def test_baseline_tool_with_df_has_old_and_new_fields(self, sample_df, tmp_path, monkeypatch):
        monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(tmp_path))  # no baseline here
        res = registry.execute("baseline", {"sensor_id": "Sensor_1"}, df=sample_df)
        assert res["status"] == "success"
        data = res["data"]
        for key in ("overall_flag", "deviations", "healthy_reference", "n_healthy",
                    "status", "deviation_ranking", "healthy_baseline_summary", "note"):
            assert key in data

    def test_baseline_tool_accepts_custom_swappable_handler(self):
        """Verify Eswar's real baseline function can replace handler cleanly."""
        original_handler = registry.get_tool("baseline").handler
        try:
            def eswar_baseline_fn(arguments, df=None):
                return {
                    "status": "Healthy",
                    "sensors_evaluated": ["Sensor_1", "Sensor_2"],
                    "deviation_ranking": [{"sensor": "Sensor_1", "deviation_score": 0.05, "status": "Normal"}],
                    "healthy_baseline_summary": {"Sensor_1": {"mean": 0.01}},
                    "note": "Eswar's actual healthy baseline algorithm.",
                }
            registry.get_tool("baseline").handler = eswar_baseline_fn
            res = registry.execute("baseline", {})
            assert res["data"]["note"] == "Eswar's actual healthy baseline algorithm."
        finally:
            registry.get_tool("baseline").handler = original_handler


# =============================================================================
# TASK 3: Routing the Five Professor Questions
# =============================================================================

class TestTask3ProfessorQuestionsRouting:
    def test_question_1_summary_statistics(self, sample_df):
        q = "What are the summary statistics for Sensor_1?"
        res = dispatcher.detect_and_dispatch(q, df=sample_df)
        assert res is not None
        assert res.tool_name == "summary_statistics"
        assert res.status == "success"

    def test_question_2_anomaly_detection(self, sample_df):
        q = "Are there any anomalies or spikes in Sensor_1?"
        res = dispatcher.detect_and_dispatch(q, df=sample_df)
        assert res is not None
        assert res.tool_name == "anomaly_detection"
        assert res.status == "success"

    def test_question_3_trend_analysis(self, sample_df):
        q = "What is the trend in deflection over time?"
        res = dispatcher.detect_and_dispatch(q, df=sample_df)
        assert res is not None
        assert res.tool_name == "trend_analysis"
        assert res.status == "success"

    def test_question_4_correlation_analysis(self, sample_df):
        q = "What is the correlation between sensors?"
        res = dispatcher.detect_and_dispatch(q, df=sample_df)
        assert res is not None
        assert res.tool_name == "correlation_analysis"
        assert res.status == "success"

    def test_question_5_baseline_and_model_results(self, sample_df):
        q_base = "What is the healthy baseline and deviation ranking for the sensors?"
        res_base = dispatcher.detect_and_dispatch(q_base, df=sample_df)
        assert res_base is not None
        assert res_base.tool_name == "baseline"

        q_model = "Predict the damage level and condition for Sensor_1"
        res_model = dispatcher.detect_and_dispatch(q_model, df=sample_df)
        assert res_model is not None
        assert res_model.tool_name == "model_result"


# =============================================================================
# TASK 4: Event & Peak-Score Lines in LLM Summary
# =============================================================================

class TestTask4LLMSummaryExtensions:
    def test_anomaly_summary_includes_peak_and_event_info(self, sample_df):
        res = dispatcher.detect_and_dispatch("Detect anomalies in Sensor_1", df=sample_df)
        assert res is not None
        summary = res.llm_summary
        assert "Peak score:" in summary or "max z-score" in summary
        assert "Event count:" in summary or "Event details:" in summary

    def test_model_result_summary_includes_damage_and_severity(self):
        res = dispatcher.detect_and_dispatch("What is the model prediction for S01?")
        assert res is not None
        # NEW-4: with no upload the summary asks for a file; the full condition/severity
        # summary is covered in tests/test_new4_assessment.py.
        assert "upload" in res.llm_summary.lower()

    def test_baseline_summary_includes_status_and_ranking(self):
        res = dispatcher.detect_and_dispatch("Evaluate healthy baseline")
        assert res is not None
        summary = res.llm_summary
        assert "[ANALYTICAL RESULT: baseline]" in summary
        assert "Status:" in summary
        assert "Frequency deviations:" in summary
