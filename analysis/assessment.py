"""Structural assessment for the assistant (NEW-4): one recording -> Interface C.

Pipeline for one uploaded, cleaned recording (Sensor_1..5 + Relative_Time_Sec):

    extract_features(df)                    PRE-2   one feature row (Interface A)
        |-> TRN-1 predictor(row)            model   condition / severity + importances
        |-> compare_to_baseline(row)        NEW-2   Interface B (healthy comparison)
        v
    assess_recording(df)                    this module -> Interface C

Interface C fields returned (all JSON-safe):

    model_available, model_error
    condition, condition_confidence          "Damaged" | "Undamaged" | None
    severity_mm, severity_confidence         1, 2 or 3 when Damaged, else None
    severity_label                           plain-language severity
    top_parameters                           model importances + top baseline deviations
    baseline_comparison                      Interface B for the same recording
    agreement, agreement_note                "agree" | "disagree" | "unknown"
    caveats                                  limits the assistant must mention

The TRN-1 predictor is looked up in ONE place (PREDICTOR_CANDIDATES below). If TRN-1's
inference function lives somewhere else, either add its "module:function" to that list or
set the environment variable TTB_PREDICT_FUNC="package.module:function".

The predictor is called with ONE feature row (a dict, Interface A). It may return a dict
or an object; the field names TRN-1 uses are mapped by ``normalize_prediction``.
No prediction is ever invented: when the model cannot run, ``model_available`` is False,
``condition`` is None, and the baseline comparison is still returned.
"""
from __future__ import annotations

import importlib
import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("ttb.analysis.assessment")

# ---------------------------------------------------------------------------
# Where TRN-1's inference function lives  (EDIT HERE if yours is elsewhere)
# ---------------------------------------------------------------------------
PREDICTOR_ENV = "TTB_PREDICT_FUNC"
PREDICTOR_CANDIDATES: List[Tuple[str, str]] = [
    ("analysis.inference", "predict"),
    ("analysis.inference", "predict_recording"),
    ("analysis.inference", "predict_row"),
    ("analysis.model_inference", "predict"),
    ("analysis.predict", "predict"),
    ("analysis.train_model", "predict"),
    ("analysis.training", "predict"),
    ("analysis.trn1", "predict"),
    ("models.inference", "predict"),
    ("ml.inference", "predict"),
]

# Crack depth -> plain-language severity (Katam et al.: 1, 2, 3 mm = 20, 40, 60 % of the
# 5 mm beam depth; the paper calls them low, medium and high damage intensity).
SEVERITY_LABELS = {
    1: "low (1 mm crack, about 20 percent of the beam depth)",
    2: "medium (2 mm crack, about 40 percent of the beam depth)",
    3: "high (3 mm crack, about 60 percent of the beam depth)",
}
SEVERITY_WORDS = {1: "Low", 2: "Medium", 3: "High"}

BASE_CAVEATS = [
    "This applies to the single-crack steel cantilever beam used in the lab study only.",
    "Only a few healthy recordings exist, so the healthy range and the healthy-class accuracy are uncertain.",
    "A small crack close to the free end barely changes the first frequency and can look healthy.",
]

TOP_MODEL_PARAMS = 5
TOP_BASELINE_PARAMS = 3
TOP_AMPLITUDE_DEVIATIONS_KEPT = 5

_predictor_cache: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# predictor lookup
# ---------------------------------------------------------------------------

def _import_callable(module_name: str, func_name: str) -> Optional[Callable[..., Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception:  # noqa: BLE001 - missing module is the normal "not found" case
        return None
    func = getattr(module, func_name, None)
    return func if callable(func) else None


def load_predictor(refresh: bool = False) -> Tuple[Optional[Callable[..., Any]], str]:
    """Return (predict_function, where_it_came_from). (None, reason) when not found."""
    if not refresh and "value" in _predictor_cache:
        return _predictor_cache["value"]

    result: Tuple[Optional[Callable[..., Any]], str]
    spec = os.environ.get(PREDICTOR_ENV, "").strip()
    if spec:
        mod, _, fn = spec.partition(":")
        func = _import_callable(mod.strip(), fn.strip()) if fn else None
        result = (func, spec) if func else (None, f"{PREDICTOR_ENV}='{spec}' could not be imported")
    else:
        result = (None, "no trained-model inference function was found "
                        f"(set {PREDICTOR_ENV}=module:function or edit PREDICTOR_CANDIDATES)")
        for mod, fn in PREDICTOR_CANDIDATES:
            func = _import_callable(mod, fn)
            if func:
                result = (func, f"{mod}:{fn}")
                break
    _predictor_cache["value"] = result
    return result


def reset_predictor_cache() -> None:
    _predictor_cache.clear()


# ---------------------------------------------------------------------------
# normalising whatever TRN-1 returns
# ---------------------------------------------------------------------------

def _get(raw: Any, *names: str) -> Any:
    """First present, non-None attribute / key among names (dict, object or mapping)."""
    for name in names:
        if isinstance(raw, dict):
            if raw.get(name) is not None:
                return raw[name]
        elif getattr(raw, name, None) is not None:
            return getattr(raw, name)
    return None


def _as_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _confidence(value: Any) -> Optional[float]:
    number = _as_float(value)
    if number is None:
        return None
    if number > 1.0 and number <= 100.0:  # given as a percentage
        number /= 100.0
    return round(min(max(number, 0.0), 1.0), 4)


def normalize_condition(value: Any) -> Optional[str]:
    """'Damaged' / 'Undamaged' from the many ways a label can be written."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "Damaged" if value else "Undamaged"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "Damaged" if int(value) == 1 else "Undamaged"
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"undamaged", "healthy", "normal", "intact", "no_damage", "none", "0"}:
        return "Undamaged"
    if text.startswith("damage") or text in {"damaged", "abnormal", "cracked", "1"}:
        return "Damaged"
    return None


def normalize_severity_mm(value: Any) -> Optional[int]:
    """1, 2 or 3 from 1 / '2' / '3 mm' / 'M2' / 'Damaged_3mm' ... else None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = _as_float(value)
        return int(round(number)) if number is not None and round(number) in (1, 2, 3) else None
    text = str(value).strip().lower()
    for mm in (1, 2, 3):
        if text in {str(mm), f"{mm}mm", f"{mm} mm", f"m{mm}", f"{mm}.0"}:
            return mm
    for mm in (1, 2, 3):
        if f"{mm}mm" in text.replace(" ", "").replace("_", ""):
            return mm
    return None


def _importances(raw: Any) -> List[Dict[str, Any]]:
    """Feature importances as [{'feature', 'importance'}] sorted high to low."""
    value = _get(raw, "top_parameters", "feature_importances", "importances", "top_features",
                 "feature_importance")
    items: List[Dict[str, Any]] = []
    if isinstance(value, dict):
        pairs = list(value.items())
    elif isinstance(value, (list, tuple)):
        pairs = []
        for entry in value:
            if isinstance(entry, dict):
                name = entry.get("feature") or entry.get("name") or entry.get("parameter")
                imp = entry.get("importance", entry.get("score", entry.get("weight")))
                pairs.append((name, imp))
            elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                pairs.append((entry[0], entry[1]))
            elif isinstance(entry, str):
                pairs.append((entry, None))
    else:
        pairs = []
    for name, imp in pairs:
        if not name:
            continue
        items.append({"feature": str(name), "importance": _as_float(imp)})
    if any(i["importance"] is not None for i in items):
        items.sort(key=lambda i: -(i["importance"] if i["importance"] is not None else -1.0))
    return items


def normalize_prediction(raw: Any) -> Dict[str, Any]:
    """Map TRN-1's output to the Interface C model fields."""
    if hasattr(raw, "model_dump"):
        raw = raw.model_dump()
    elif hasattr(raw, "__dataclass_fields__"):
        raw = {k: getattr(raw, k) for k in raw.__dataclass_fields__}

    condition = normalize_condition(_get(raw, "condition", "predicted_condition", "label",
                                         "predicted_class", "prediction"))
    severity = normalize_severity_mm(_get(raw, "severity_mm", "damage_level_mm", "severity",
                                          "predicted_severity_mm"))
    if condition != "Damaged":
        severity = None  # severity only exists for damaged recordings

    return {
        "condition": condition,
        "condition_confidence": _confidence(_get(raw, "condition_confidence", "confidence",
                                                 "probability", "condition_probability")),
        "severity_mm": severity,
        "severity_confidence": (
            _confidence(_get(raw, "severity_confidence", "severity_probability"))
            if severity is not None else None
        ),
        "importances": _importances(raw),
    }


def run_predictor(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """Call TRN-1 on one feature row. Returns (normalized prediction | None, error text)."""
    func, where = load_predictor()
    if func is None:
        return None, where
    last_error = ""
    for payload in (dict(row), pd.DataFrame([row])):  # dict first, one-row frame second
        try:
            normalized = normalize_prediction(func(payload))
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if normalized["condition"] is None:
            return None, f"the model ({where}) returned no usable condition"
        return normalized, ""
    logger.warning("Model inference failed (%s): %s", where, last_error)
    return None, f"the model ({where}) failed: {last_error}"


# ---------------------------------------------------------------------------
# merging with the baseline
# ---------------------------------------------------------------------------

def _compact_baseline(result: Dict[str, Any]) -> Dict[str, Any]:
    """Interface B for the answer: every frequency deviation, the top few amplitude ones."""
    devs = result.get("deviations", []) or []
    freq = [d for d in devs if d.get("group") == "frequency"]
    amp = [d for d in devs if d.get("group") != "frequency"][:TOP_AMPLITUDE_DEVIATIONS_KEPT]
    return {
        "overall_flag": result.get("overall_flag", "unavailable"),
        "deviations": freq + amp,
        "healthy_reference": {
            k: v for k, v in (result.get("healthy_reference") or {}).items() if k != "spectrum"
        },
        "n_healthy": result.get("n_healthy", {}),
        "flag_rule": result.get("flag_rule", ""),
        "summary": result.get("summary", ""),
        "caveat": result.get("caveat", ""),
    }


def _merge_parameters(importances: List[Dict[str, Any]],
                      baseline: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Model importances first, then the largest baseline deviations (frequency first)."""
    by_feature = {d["feature"]: d for d in baseline.get("deviations", [])}
    merged: List[Dict[str, Any]] = []
    seen = set()
    for item in importances[:TOP_MODEL_PARAMS]:
        # TRN-1 model A works on "dev_<feature>" (deviation from healthy); report the
        # underlying feature so it lines up with the healthy comparison.
        name = item["feature"][4:] if item["feature"].startswith("dev_") else item["feature"]
        if name in seen:
            continue
        dev = by_feature.get(name, {})
        merged.append({
            "feature": name,
            "model_feature": item["feature"],
            "source": "model",
            "importance": item["importance"],
            "value": dev.get("value"),
            "healthy_typical": dev.get("healthy_typical"),
            "change_percent": dev.get("change_percent"),
            "score": dev.get("score"),
            "outside": dev.get("outside"),
        })
        seen.add(name)
    added = 0
    for dev in baseline.get("deviations", []):  # already sorted: frequency first, by score
        if added >= TOP_BASELINE_PARAMS:
            break
        # Amplitude features depend on sensor gain/offset: they are never offered as the
        # reason for damage unless the model itself ranked them (handled above).
        if dev["feature"] in seen or dev.get("group") != "frequency":
            continue
        merged.append({
            "feature": dev["feature"],
            "source": "baseline",
            "importance": None,
            "value": dev.get("value"),
            "healthy_typical": dev.get("healthy_typical"),
            "change_percent": dev.get("change_percent"),
            "score": dev.get("score"),
            "outside": dev.get("outside"),
        })
        seen.add(dev["feature"])
        added += 1
    return merged


def _agreement(condition: Optional[str], flag: str) -> Tuple[str, str]:
    if condition is None or flag not in {"normal", "abnormal"}:
        return "unknown", ""
    if condition == "Damaged" and flag == "abnormal":
        return "agree", "The model says damaged and the healthy-baseline check also finds the frequencies different."
    if condition == "Undamaged" and flag == "normal":
        return "agree", "The model says undamaged and the frequencies are inside the healthy range."
    if condition == "Undamaged" and flag == "abnormal":
        return "disagree", (
            "The model classes this recording as undamaged, but its frequencies are outside the "
            "healthy baseline. Both results are reported; this can also be a new healthy beam or "
            "a different test setup."
        )
    return "disagree", (
        "The model classes this recording as damaged, but its frequencies are inside the healthy "
        "baseline range. Both results are reported; small cracks (especially near the free end) "
        "change the frequencies very little."
    )


def severity_texts(severity_mm: Optional[int]) -> Tuple[Optional[str], Optional[str]]:
    """(plain label, one-word level) for 1/2/3 mm, else (None, None)."""
    if severity_mm in SEVERITY_LABELS:
        return SEVERITY_LABELS[severity_mm], SEVERITY_WORDS[severity_mm]
    return None, None


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def assess_recording(df: Optional[pd.DataFrame], folder: Any = None) -> Dict[str, Any]:
    """Interface C for one recording. Never raises: problems become fields."""
    empty: Dict[str, Any] = {
        "model_available": False,
        "model_error": "",
        "condition": None,
        "condition_confidence": None,
        "severity_mm": None,
        "severity_confidence": None,
        "severity_label": None,
        "top_parameters": [],
        "baseline_comparison": {},
        "agreement": "unknown",
        "agreement_note": "",
        "status": "no_data",
        "message": "",
        "caveats": list(BASE_CAVEATS),
    }
    if df is None:
        empty["message"] = ("No recording has been uploaded yet. Ask the user to upload an Excel "
                            "(.xlsx/.xlsm) sensor recording first.")
        return empty
    if not any(str(c).startswith("Sensor_") for c in df.columns):
        empty["status"] = "not_a_recording"
        empty["message"] = ("The active file is not a sensor recording (no Sensor_1..Sensor_5 "
                            "columns), so the damage model cannot analyse it.")
        return empty

    from analysis.baseline import compare_to_baseline
    from analysis.features import extract_features

    try:
        row = extract_features(df)
    except Exception as exc:  # noqa: BLE001
        empty["status"] = "features_failed"
        empty["message"] = f"Features could not be computed from this recording: {exc}"
        return empty

    baseline = _compact_baseline(compare_to_baseline(row, folder=folder, include_spectrum=False))
    prediction, error = run_predictor(row)

    result = dict(empty)
    result["status"] = "ok"
    result["baseline_comparison"] = baseline
    if prediction is None:
        result["model_error"] = error
        result["message"] = (f"The trained model could not be used ({error}). Only the "
                             "healthy-baseline comparison is available.")
        result["top_parameters"] = _merge_parameters([], baseline)
        return result

    label, _ = severity_texts(prediction["severity_mm"])
    agreement, note = _agreement(prediction["condition"], baseline["overall_flag"])
    result.update({
        "model_available": True,
        "condition": prediction["condition"],
        "condition_confidence": prediction["condition_confidence"],
        "severity_mm": prediction["severity_mm"],
        "severity_confidence": prediction["severity_confidence"],
        "severity_label": label,
        "top_parameters": _merge_parameters(prediction["importances"], baseline),
        "agreement": agreement,
        "agreement_note": note,
        "message": "ok",
    })
    return result