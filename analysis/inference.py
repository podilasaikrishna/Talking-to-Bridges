"""
TRN-1 inference function (Interface C).

    from analysis.inference import predict
    result = predict(feature_row)          # dict, pandas Series, or one-row DataFrame

Returns a dict with:
    condition                "Damaged" | "Undamaged"
    condition_confidence     0..1   (probability of the predicted condition)
    severity_mm              1 | 2 | 3 when Damaged, else None
    severity_confidence      0..1 when Damaged, else None
    top_parameters           features that most influenced the condition decision, with the
                             model's importance and this recording's value
    severity_top_parameters  same, for the severity model (None when Undamaged)
    baseline_comparison      None here; NEW-4 fills it from the NEW-2 baseline tool
    model_info               which models and feature lists were used

The feature row must come from the same feature code used for training
(extract_features). Label, file-name, case and sampling-rate columns are ignored.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

DEFAULT_BUNDLE = Path(__file__).resolve().parents[1] / "models" / "damage_model_bundle.joblib"


@lru_cache(maxsize=2)
def _load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Model bundle not found: {p}. Run `python -m analysis.train_models` first.")
    return joblib.load(p)


def load_bundle(path: str | Path | None = None) -> dict:
    return _load(str(path or DEFAULT_BUNDLE))


def _to_row(features: Mapping[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    if isinstance(features, pd.DataFrame):
        if len(features) != 1:
            raise ValueError(f"predict() takes exactly one feature row, got {len(features)}.")
        return features.reset_index(drop=True)
    if isinstance(features, pd.Series):
        return features.to_frame().T.reset_index(drop=True)
    if isinstance(features, Mapping):
        return pd.DataFrame([dict(features)])
    raise TypeError("features must be a dict, pandas Series or one-row DataFrame.")


def _prepare(row: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    missing = [c for c in cols if c not in row.columns]
    if missing:
        raise KeyError(f"Feature row is missing required features: {missing}")
    X = row[cols].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    # The feature extractor writes 0.0 when no spectral peak is found. A 0.0, negative or NaN
    # frequency must never reach the model as if it were a real measurement.
    for c in cols:
        if re.match(r"^f[1-9]_hz$", c, re.IGNORECASE) or "_over_" in c:
            X.loc[X[c] <= 0, c] = np.nan
    # f1-f3 carry almost all of the decision, so a median fill would invent the answer.
    absent = [c for c in cols if re.match(r"^f[1-9]_hz$", c, re.IGNORECASE) and X[c].isna().any()]
    if absent:
        raise ValueError(
            f"No valid spectral peak found for {absent}; cannot classify this recording. "
            "Check the recording (length, noise, hit) or the modal search bands."
        )
    return X   # any other NaN (for example a ratio) is filled with the training median by the saved pipeline


def _top(importances: Mapping[str, float], row: pd.DataFrame, k: int) -> list[dict]:
    out = []
    for name, imp in list(importances.items())[:k]:
        v = row[name].iloc[0] if name in row.columns else np.nan
        v = None if pd.isna(v) else float(v)
        out.append({"feature": name, "importance": float(imp), "value": v})
    return out


def predict(features, bundle: dict | None = None, top_k: int = 5) -> dict:
    b = bundle or load_bundle()
    row = _to_row(features)

    XA = _prepare(row, b["features_a"])
    tr = b.get("a_transform")
    if tr:   # Model A works on two-sided deviation from the healthy reference
        ref = tr["reference"]
        XA = pd.DataFrame({f"dev_{c}": (XA[c] - ref[c]).abs() / ref[c] for c in tr["columns"]}, index=XA.index)
        row = row.copy()
        for c in XA.columns:
            row[c] = XA[c].values
    pA = b["model_a"].predict_proba(XA)[0]
    classes_a = list(b["model_a"].named_steps["clf"].classes_)
    p_damaged = float(pA[classes_a.index(1)])
    thr = float(b.get("threshold_a", 0.5))
    damaged = p_damaged >= thr
    # rescale so the chosen threshold maps to 0.5: the confidence of the predicted class is then always >= 0.5
    p_adj = 0.5 * p_damaged / thr if p_damaged < thr else 0.5 + 0.5 * (p_damaged - thr) / (1.0 - thr)
    result = {
        "condition": "Damaged" if damaged else "Undamaged",
        "condition_confidence": p_adj if damaged else 1.0 - p_adj,
        "severity_mm": None,
        "severity_confidence": None,
        "top_parameters": _top(b["importances_a"], row, top_k),
        "severity_top_parameters": None,
        "baseline_comparison": None,
        "model_info": {
            "model_a": b["model_a_name"], "model_b": b["model_b_name"],
            "features_a": b["features_a"], "features_b": b["features_b"],
            "p_damaged": p_damaged, "threshold": thr,
        },
    }
    if damaged:
        XB = _prepare(row, b["features_b"])
        pB = b["model_b"].predict_proba(XB)[0]
        classes_b = list(b["model_b"].named_steps["clf"].classes_)
        i = int(np.argmax(pB))
        result["severity_mm"] = int(classes_b[i])
        result["severity_confidence"] = float(pB[i])
        result["severity_top_parameters"] = _top(b["importances_b"], row, top_k)
        result["model_info"]["severity_probabilities"] = {int(c): float(p) for c, p in zip(classes_b, pB)}
    return result