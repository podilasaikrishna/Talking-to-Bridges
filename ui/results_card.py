"""Results card for the chat (NEW-5).

Shows the same numbers the assistant speaks, taken from the ``assessment`` field that
/chat returns (Interface C from analysis.assessment):

    condition + confidence, severity + confidence, healthy-baseline check,
    top parameters (value, healthy value, change), model/baseline agreement, limits.

``build_card_view`` has no Streamlit code, so it can be tested alone.
``render_results_card`` draws it. Nothing is invented: a missing value shows "not available".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

NOT_AVAILABLE = "Not available"

PARAMETER_NAMES = {
    "f1_hz": "Mode 1 frequency (Hz)",
    "f2_hz": "Mode 2 frequency (Hz)",
    "f3_hz": "Mode 3 frequency (Hz)",
    "f2_over_f1": "Frequency ratio f2/f1",
    "f3_over_f1": "Frequency ratio f3/f1",
}
SEVERITY_WORDS = {1: "Low", 2: "Medium", 3: "High"}
BASELINE_TEXT = {"abnormal": "Abnormal", "normal": "Normal"}
BASELINE_NOTE = {
    "abnormal": "Frequencies are different from the healthy beams",
    "normal": "Frequencies are inside the healthy range",
}


def _percent(value: Any) -> Optional[str]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return f"{number * 100:.0f}%"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "-"


def _change(value: Any) -> str:
    try:
        return f"{float(value):+.1f}%"
    except (TypeError, ValueError):
        return "-"


def parameter_name(feature: str) -> str:
    """Readable name for a feature (the raw name when unknown)."""
    if feature in PARAMETER_NAMES:
        return PARAMETER_NAMES[feature]
    return str(feature).replace("_", " ")


def parameter_rows(top_parameters: Optional[List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    """Table rows for the top parameters, in the order given by the assessment."""
    rows: List[Dict[str, str]] = []
    for item in top_parameters or []:
        outside = item.get("outside")
        if outside is None:
            status = "-"
        else:
            status = "Clearly different from healthy" if outside else "Within normal healthy variation"
        rows.append({
            "Parameter": parameter_name(str(item.get("feature", ""))),
            "Value": _number(item.get("value")),
            "Healthy": _number(item.get("healthy_typical")),
            "Change": _change(item.get("change_percent")),
            "Status": status,
            "Found by": "Model" if item.get("source") == "model" else "Healthy comparison",
        })
    return rows


def build_card_view(assessment: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Everything the card shows, as plain strings. None when there is nothing to show."""
    if not assessment:
        return None
    model_ok = bool(assessment.get("model_available"))
    condition = assessment.get("condition") if model_ok else None

    if condition:
        conf = _percent(assessment.get("condition_confidence"))
        condition_value = condition
        condition_note = f"Model confidence {conf}" if conf else "Model confidence not available"
    else:
        condition_value = NOT_AVAILABLE
        condition_note = "The trained model could not be used for this recording"

    severity_mm = assessment.get("severity_mm")
    if condition == "Damaged" and severity_mm in SEVERITY_WORDS:
        severity_value = f"{SEVERITY_WORDS[severity_mm]} ({severity_mm} mm crack)"
        conf = _percent(assessment.get("severity_confidence"))
        severity_note = f"Model confidence {conf}" if conf else "Model confidence not available"
    elif condition == "Undamaged":
        severity_value = "Not applicable"
        severity_note = "No damage found, so no severity"
    else:
        severity_value = NOT_AVAILABLE
        severity_note = ""

    flag = assessment.get("baseline_flag")
    return {
        "condition": condition_value,
        "condition_note": condition_note,
        "severity": severity_value,
        "severity_note": severity_note,
        "baseline": BASELINE_TEXT.get(str(flag), NOT_AVAILABLE),
        "baseline_note": BASELINE_NOTE.get(str(flag), "Healthy comparison not available"),
        "agreement": assessment.get("agreement") or "unknown",
        "agreement_note": assessment.get("agreement_note") or "",
        "parameters": parameter_rows(assessment.get("top_parameters")),
        "caveats": [str(c) for c in (assessment.get("caveats") or [])],
        "is_damaged": condition == "Damaged",
    }


def render_results_card(assessment: Optional[Dict[str, Any]], key: str = "results_card") -> bool:
    """Draw the card in Streamlit. Returns False when there was nothing to draw."""
    view = build_card_view(assessment)
    if view is None:
        return False
    import streamlit as st

    with st.container(border=True):
        st.markdown("**Assessment of the uploaded recording**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Condition", view["condition"])
        col1.caption(view["condition_note"])
        col2.metric("Severity", view["severity"])
        col2.caption(view["severity_note"])
        col3.metric("Healthy-baseline check", view["baseline"])
        col3.caption(view["baseline_note"])

        if view["agreement"] == "disagree":
            st.warning(view["agreement_note"])
        elif view["agreement_note"]:
            st.caption(view["agreement_note"])

        if view["parameters"]:
            st.markdown("**Top parameters**")
            st.dataframe(view["parameters"], hide_index=True, width="stretch")

        st.caption(
            "Confidence is the model's own score, not a measured accuracy. "
            "The healthy baseline rests on only 2 healthy beams."
        )
        if view["caveats"]:
            with st.expander("Limits of this result"):
                for text in view["caveats"]:
                    st.markdown(f"- {text}")
    return True
