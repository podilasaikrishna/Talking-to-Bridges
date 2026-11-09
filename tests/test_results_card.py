"""NEW-5 tests: the results card shows the assessment numbers and invents nothing."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from ui.results_card import build_card_view, parameter_name, parameter_rows

DAMAGED = {
    "model_available": True,
    "condition": "Damaged",
    "condition_confidence": 0.99,
    "severity_mm": 1,
    "severity_confidence": 0.56,
    "severity_label": "low (1 mm crack, about 20 percent of the beam depth)",
    "top_parameters": [
        {"feature": "f2_hz", "source": "model", "value": 39.6, "healthy_typical": 37.87,
         "change_percent": 4.5, "outside": True},
        {"feature": "f3_over_f1", "source": "baseline", "value": 16.6, "healthy_typical": 16.59,
         "change_percent": 0.1, "outside": False},
    ],
    "agreement": "agree",
    "agreement_note": "The model says damaged and the healthy-baseline check also finds the frequencies different.",
    "caveats": ["Only a few healthy recordings exist."],
    "baseline_flag": "abnormal",
}


def test_none_when_no_assessment():
    assert build_card_view(None) is None
    assert build_card_view({}) is None


def test_damaged_card_numbers():
    view = build_card_view(DAMAGED)
    assert view["condition"] == "Damaged"
    assert "99%" in view["condition_note"]
    assert view["severity"] == "Low (1 mm crack)"
    assert "56%" in view["severity_note"]
    assert view["baseline"] == "Abnormal"
    assert "different" in view["baseline_note"]
    assert view["is_damaged"] is True
    assert view["caveats"] == ["Only a few healthy recordings exist."]


def test_parameter_rows_keep_sign_and_status():
    rows = parameter_rows(DAMAGED["top_parameters"])
    assert rows[0]["Parameter"] == "Mode 2 frequency (Hz)"
    assert rows[0]["Change"] == "+4.5%"
    assert rows[0]["Status"] == "Clearly different from healthy"
    assert rows[0]["Found by"] == "Model"
    assert rows[1]["Status"] == "Within normal healthy variation"
    assert rows[1]["Found by"] == "Healthy comparison"


def test_negative_change_is_shown_as_measured():
    rows = parameter_rows([{"feature": "f1_hz", "value": 7.0, "healthy_typical": 7.6,
                            "change_percent": -6.1, "outside": True, "source": "model"}])
    assert rows[0]["Change"] == "-6.1%"


def test_undamaged_has_no_severity():
    view = build_card_view({**DAMAGED, "condition": "Undamaged", "severity_mm": None,
                            "severity_confidence": None, "baseline_flag": "normal"})
    assert view["severity"] == "Not applicable"
    assert view["baseline"] == "Normal"
    assert view["is_damaged"] is False


def test_model_unavailable_invents_nothing():
    view = build_card_view({"model_available": False, "condition": None, "severity_mm": None,
                            "top_parameters": [], "baseline_flag": "abnormal", "caveats": []})
    assert view["condition"] == "Not available"
    assert view["severity"] == "Not available"
    assert view["baseline"] == "Abnormal"
    assert view["parameters"] == []


def test_missing_values_show_dash_not_zero():
    rows = parameter_rows([{"feature": "f2_hz", "source": "baseline"}])
    assert rows[0]["Value"] == "-" and rows[0]["Change"] == "-" and rows[0]["Status"] == "-"


def test_unknown_feature_name_is_readable():
    assert parameter_name("rms_Sensor_3") == "rms Sensor 3"


def test_disagree_flag_passes_through():
    view = build_card_view({**DAMAGED, "agreement": "disagree", "agreement_note": "They differ."})
    assert view["agreement"] == "disagree" and view["agreement_note"] == "They differ."


def test_card_renders_in_streamlit():
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    def app():
        from ui.results_card import render_results_card
        import json
        import os

        render_results_card(json.loads(os.environ["CARD_JSON"]))

    import json, os
    os.environ["CARD_JSON"] = json.dumps(DAMAGED)
    at = AppTest.from_function(app).run(timeout=30)
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels == ["Condition", "Severity", "Healthy-baseline check"]
    assert [m.value for m in at.metric][:2] == ["Damaged", "Low (1 mm crack)"]


@pytest.mark.parametrize("page", ["ui/main.py", "frontend/app.py"])
def test_both_chat_pages_draw_the_card(page):
    """The repo has two copies of the chat page; both must show the card."""
    text = (Path(__file__).resolve().parent.parent / page).read_text(encoding="utf-8")
    assert "from ui.results_card import render_results_card" in text
    assert text.count("render_results_card(") == 2
    assert 'data.get("assessment")' in text and 'msg_data["assessment"]' in text
