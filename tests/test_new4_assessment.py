"""NEW-4 tests: model + baseline assessment, routing of the five questions, summaries.

The TRN-1 model is replaced by a small fake predictor (set through the predictor cache), so
these tests run without the trained artifact.  Healthy recordings are generated into a
temporary reference folder (HEALTHY_REFERENCE_DIR), so no real data is needed either.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make "analysis" and "app" importable whether pytest or python -m pytest is used.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest

from analysis import assessment
from analysis.baseline import _profile_cache, _spectrum_cache
from analysis.dispatcher import _plan_tools, dispatcher
from analysis.schemas import ModelResultOutput
from analysis.tools import handle_model_result

assessment_freq = {"f1_hz", "f2_hz", "f3_hz", "f2_over_f1", "f3_over_f1"}  # amplitude never offered
FS = 300.0
HEALTHY_F = (7.69, 37.79, 134.8)
DAMAGED_F = (7.32, 37.79, 131.2)  # first and third frequency lowered


def make_recording(freqs, seconds=30, seed=0, label=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(0, seconds, 1.0 / FS)
    signal = sum(
        a * np.exp(-0.05 * (i + 1) * t) * np.sin(2 * np.pi * f * t)
        for i, (f, a) in enumerate(zip(freqs, (1.0, 0.6, 0.3)))
    )
    data = {"Relative_Time_Sec": t}
    for k in range(1, 6):
        data[f"Sensor_{k}"] = signal * (0.4 + 0.15 * k) + rng.normal(0, 0.01, len(t))
    df = pd.DataFrame(data)
    if label is not None:
        df["Condition"] = label
    return df


def fake_predict(row):
    """Test double for TRN-1: damaged when f1 is clearly below 7.6 Hz."""
    damaged = row["f1_hz"] < 7.55
    return {
        "condition": "Damaged" if damaged else "Undamaged",
        "condition_confidence": 0.87 if damaged else 0.91,
        "severity_mm": 2 if damaged else None,
        "severity_confidence": 0.62 if damaged else None,
        "feature_importances": {"f1_hz": 0.41, "f3_hz": 0.22, "f2_hz": 0.05},
    }


@pytest.fixture(autouse=True)
def reference_folder(tmp_path, monkeypatch):
    folder = tmp_path / "reference"
    folder.mkdir()
    make_recording((7.69, 37.79, 134.8), seed=1).to_csv(folder / "healthy_a.csv", index=False)
    make_recording((7.72, 37.82, 134.6), seed=2).to_csv(folder / "healthy_b.csv", index=False)
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(folder))
    _profile_cache.clear()
    _spectrum_cache.clear()
    assessment.reset_predictor_cache()
    assessment._predictor_cache["value"] = (fake_predict, "test:fake_predict")
    yield
    assessment.reset_predictor_cache()


# ---------------------------------------------------------------------------
# assess_recording / handle_model_result
# ---------------------------------------------------------------------------

def test_no_upload_says_so_and_invents_nothing():
    out = handle_model_result({"sensor_id": "S01"}, None)
    assert out["status"] == "no_data"
    assert out["condition"] is None and out["severity_mm"] is None
    assert out["predicted_class"] == "Unavailable"
    assert "upload" in out["message"].lower()
    ModelResultOutput(**out)


def test_non_recording_file_is_not_analysed():
    df = pd.DataFrame({"timestamp": [1, 2], "deflection": [0.1, 0.2], "condition": ["Damaged"] * 2})
    out = handle_model_result({}, df)
    assert out["status"] == "not_a_recording"
    assert out["condition"] is None  # the label column is never read


def test_damaged_recording_gives_full_interface_c():
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=3))
    assert out["status"] == "ok" and out["model_available"] is True
    assert out["condition"] == "Damaged"
    assert out["condition_confidence"] == pytest.approx(0.87)
    assert out["severity_mm"] == 2 and out["severity_confidence"] == pytest.approx(0.62)
    assert "medium" in out["severity_label"]
    assert out["damage_class"] == "Damaged_2mm" and out["severity"] == "Medium"
    assert out["baseline_comparison"]["overall_flag"] == "abnormal"
    assert out["agreement"] == "agree"
    names = [p["feature"] for p in out["top_parameters"]]
    assert names[0] == "f1_hz"                      # model's top importance first
    f1 = out["top_parameters"][0]
    assert f1["source"] == "model" and f1["change_percent"] < 0   # f1 below healthy
    baseline_params = [p for p in out["top_parameters"] if p["source"] == "baseline"]
    assert baseline_params and all(p["feature"] in assessment_freq for p in baseline_params)
    assert len(names) == len(set(names))            # no duplicates
    ModelResultOutput(**out)


def test_healthy_recording_has_no_severity():
    out = handle_model_result({}, make_recording(HEALTHY_F, seed=4))
    assert out["condition"] == "Undamaged"
    assert out["severity_mm"] is None and out["severity_confidence"] is None
    assert out["severity"] == "None" and out["damage_class"] == "Undamaged"
    assert out["agreement"] == "agree"
    assert out["baseline_comparison"]["overall_flag"] == "normal"


def test_label_column_does_not_change_the_result():
    healthy_labelled_damaged = make_recording(HEALTHY_F, seed=5, label="Damaged")
    out = handle_model_result({}, healthy_labelled_damaged)
    assert out["condition"] == "Undamaged"


def test_model_and_baseline_disagreement_is_reported():
    assessment._predictor_cache["value"] = (
        lambda row: {"condition": "Undamaged", "confidence": 0.7}, "test:always_healthy")
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=6))
    assert out["condition"] == "Undamaged"
    assert out["baseline_comparison"]["overall_flag"] == "abnormal"
    assert out["agreement"] == "disagree"
    summary = dispatcher._build_llm_summary("model_result", out, {})
    assert "disagree" in summary and "report both" in summary


def test_model_unavailable_still_returns_baseline():
    assessment._predictor_cache["value"] = (None, "no trained-model inference function was found")
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=7))
    assert out["status"] == "ok" and out["model_available"] is False
    assert out["condition"] is None and out["predicted_class"] == "Unavailable"
    assert out["baseline_comparison"]["overall_flag"] == "abnormal"
    assert "not available" in dispatcher._build_llm_summary("model_result", out, {}).lower()


def test_predictor_crash_is_contained():
    def boom(_row):
        raise RuntimeError("bad artifact")
    assessment._predictor_cache["value"] = (boom, "test:boom")
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=8))
    assert out["model_available"] is False and "bad artifact" in out["model_error"]


def test_predictor_taking_a_dataframe_is_supported():
    def frame_only(frame):
        assert isinstance(frame, pd.DataFrame)
        return {"condition": "Damaged", "confidence": 55, "severity": "3 mm"}
    assessment._predictor_cache["value"] = (frame_only, "test:frame")
    out = assessment.assess_recording(make_recording(DAMAGED_F, seed=9))
    assert out["condition"] == "Damaged"
    assert out["condition_confidence"] == pytest.approx(0.55)  # 55 percent -> 0.55
    assert out["severity_mm"] == 3


@pytest.mark.parametrize("raw,condition,mm", [
    ({"predicted_condition": "damaged", "severity_mm": "1 mm"}, "Damaged", 1),
    ({"label": "Healthy", "severity_mm": 2}, "Undamaged", None),        # severity dropped
    ({"condition": 1, "damage_level_mm": 3.0}, "Damaged", 3),
    ({"condition": "Damaged", "severity": "Damaged_2mm"}, "Damaged", 2),
    ({"condition": "Damaged", "severity": 7}, "Damaged", None),         # not 1/2/3
])
def test_normalize_prediction_variants(raw, condition, mm):
    out = assessment.normalize_prediction(raw)
    assert out["condition"] == condition and out["severity_mm"] == mm


def test_importances_accept_lists_and_pairs():
    pairs = assessment.normalize_prediction({
        "condition": "Damaged", "top_parameters": [("f3_hz", 0.2), ("f1_hz", 0.5)]})
    assert [i["feature"] for i in pairs["importances"]] == ["f1_hz", "f3_hz"]
    dicts = assessment.normalize_prediction({
        "condition": "Damaged", "feature_importances": [{"feature": "f2_hz", "importance": 0.3}]})
    assert dicts["importances"][0]["feature"] == "f2_hz"


# ---------------------------------------------------------------------------
# routing of the five questions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("question", [
    "What is the current condition?",
    "Is the structure damaged?",
    "What parameters indicate the damage?",
    "What is the estimated severity?",
    "How severe is the damage?",
    "Is the beam healthy?",
    "Is there any crack?",
])
def test_assessment_questions_use_the_model_tool(question):
    plan = _plan_tools(question, has_data=True)
    assert [step["tool"] for step in plan] == ["model_result"]


@pytest.mark.parametrize("question", [
    "How does it compare with the healthy condition?",
    "Show me the comparison",
    "compare against the baseline",
])
def test_comparison_questions_return_text_and_chart(question):
    plan = _plan_tools(question, has_data=True)
    assert [step["tool"] for step in plan] == ["baseline", "model_result", "chart_data"]


def test_comparison_without_upload_asks_for_a_file():
    plan = _plan_tools("How does it compare with the healthy condition?", has_data=False)
    assert [step["tool"] for step in plan] == ["model_result"]
    results = dispatcher.detect_and_dispatch_all("Is the structure damaged?", None)
    assert len(results) == 1 and results[0].data["status"] == "no_data"
    assert "upload" in results[0].llm_summary.lower()


@pytest.mark.parametrize("question", [
    "plot healthy vs damaged signals",       # chart words keep the NEW-3 chart path
    "show anomalies in Sensor_3",
    "what is the trend of Sensor_1",
    "correlation between sensors",
    "compare sensors",
    "give me summary statistics",
    "hello there",
    "what does the document say about cracks",          # document questions stay with retrieval
    "Why does a drop in frequency indicate damage?",
    "unusual readings",                                  # old anomaly routing kept
    "Is there any abnormal behaviour?",
])
def test_other_questions_keep_their_old_routing(question):
    assert _plan_tools(question, has_data=True) is None


def test_dispatch_all_runs_the_comparison_plan():
    df = make_recording(DAMAGED_F, seed=10)
    results = dispatcher.detect_and_dispatch_all("How does it compare with the healthy condition?", df)
    assert [r.tool_name for r in results] == ["baseline", "model_result", "chart_data"]
    assert all(r.status == "success" for r in results)
    # the old single-result API still returns the primary result
    first = dispatcher.detect_and_dispatch("How does it compare with the healthy condition?", df)
    assert first.tool_name == "baseline"
    # the comparison chart defaults to the f1-f3 frequency bars
    assert results[2].metadata["arguments"]["comparison"] == "frequencies"


def test_named_chart_type_leads_the_comparison_plan():
    df = make_recording(DAMAGED_F, seed=11)
    results = dispatcher.detect_and_dispatch_all("compare the spectrum with the healthy one", df)
    assert [r.tool_name for r in results] == ["chart_data", "baseline", "model_result"]
    assert results[0].metadata["arguments"]["comparison"] == "spectrum"
    primary = dispatcher.detect_and_dispatch("compare the spectrum with the healthy one", df)
    assert primary.tool_name == "chart_data"       # NEW-3 behaviour kept


def test_signal_overlay_can_still_be_requested_in_a_comparison():
    df = make_recording(DAMAGED_F, seed=11)
    results = dispatcher.detect_and_dispatch_all("compare the signals with the healthy one", df)
    assert results[0].tool_name == "chart_data"
    assert results[0].metadata["arguments"]["comparison"] == "signals"


# ---------------------------------------------------------------------------
# summary text the language model sees
# ---------------------------------------------------------------------------

def test_summary_is_plain_and_complete_for_a_damaged_recording():
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=12))
    text = dispatcher._build_llm_summary("model_result", out, {})
    assert "Damaged" in text and "87 percent" in text
    assert "medium" in text and "62 percent" in text
    assert "mode 1 frequency (f1)" in text and "Hz" in text
    assert "healthy" in text.lower() and "Limits to mention" in text


def test_summary_for_healthy_recording_has_no_severity_number():
    out = handle_model_result({}, make_recording(HEALTHY_F, seed=13))
    text = dispatcher._build_llm_summary("model_result", out, {})
    assert "Undamaged" in text and "severity: none" in text


def test_trn1_style_output_with_dev_features_is_mapped():
    """analysis.inference.predict returns top_parameters as dicts, model A uses dev_<feature>."""
    def trn1_like(row):
        return {
            "condition": "Damaged", "condition_confidence": 0.93,
            "severity_mm": 3, "severity_confidence": 0.7,
            "top_parameters": [
                {"feature": "dev_f1_hz", "importance": 0.5, "value": 0.05},
                {"feature": "dev_f3_hz", "importance": 0.3, "value": 0.03},
            ],
            "severity_top_parameters": None, "baseline_comparison": None,
            "model_info": {"model_a": "rf"},
        }
    assessment._predictor_cache["value"] = (trn1_like, "analysis.inference:predict")
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=14))
    first = out["top_parameters"][0]
    assert first["feature"] == "f1_hz" and first["model_feature"] == "dev_f1_hz"
    assert first["importance"] == pytest.approx(0.5) and first["change_percent"] < 0
    names = [p["feature"] for p in out["top_parameters"]]
    assert len(names) == len(set(names)) and "dev_f1_hz" not in names
    assert out["severity_mm"] == 3 and "high" in out["severity_label"]


def test_summary_separates_clear_changes_from_normal_variation():
    out = handle_model_result({}, make_recording(DAMAGED_F, seed=15))
    text = dispatcher._build_llm_summary("model_result", out, {})
    assert "clearly different from healthy" in text
    assert "spreads away" not in text and "model importance" not in text
    assert "within normal healthy variation" in text or "clearly different" in text


def test_model_result_still_requires_sensor_id():
    from analysis.tools import registry
    assert registry.execute("model_result", {})["status"] == "error"