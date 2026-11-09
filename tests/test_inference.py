"""Tests for analysis/inference.py (TRN-1). Run: python -m pytest tests/test_inference.py -q

They build a small bundle in memory (same structure train_models.py saves), so they are fast and
do not need the real data or the real model. One extra test checks the real saved bundle if it exists.
"""
import copy
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.inference import DEFAULT_BUNDLE, load_bundle, predict
from analysis.train_models import FREQ3, dev_features, fit_model

HEALTHY_REF = {"f1_hz": 7.625, "f2_hz": 37.867, "f3_hz": 126.5}


def _make_df(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for sev, f2, n in [(0, 37.9, 40), (1, 39.7, 40), (2, 40.0, 40), (3, 39.0, 40)]:
        for _ in range(n):
            rows.append(dict(f1_hz=7.6 - 0.08 * sev + rng.normal(0, .05), f2_hz=f2 + rng.normal(0, .4),
                             f3_hz=126.5 - 1.5 * sev + rng.normal(0, 1.5), sev=sev))
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def bundle():
    df = _make_df()
    y = (df["sev"] > 0).astype(int).values
    model_a = fit_model("logreg", dev_features(df, HEALTHY_REF), y)
    dmg = df[df["sev"] > 0].reset_index(drop=True)
    model_b = fit_model("logreg", dmg[FREQ3], dmg["sev"].values)
    return {
        "model_a": model_a, "model_a_name": "logreg", "features_a": FREQ3,
        "importances_a": {"dev_f2_hz": 0.6, "dev_f1_hz": 0.3, "dev_f3_hz": 0.1},
        "a_transform": {"type": "abs_rel_dev", "reference": HEALTHY_REF, "columns": FREQ3},
        "model_b": model_b, "model_b_name": "logreg", "features_b": FREQ3,
        "importances_b": {"f1_hz": 0.5, "f2_hz": 0.3, "f3_hz": 0.2},
        "threshold_a": 0.7,
    }


HEALTHY = {"f1_hz": 7.62, "f2_hz": 37.9, "f3_hz": 126.5}
DAMAGED = {"f1_hz": 7.45, "f2_hz": 40.0, "f3_hz": 124.0}


def test_healthy_and_damaged(bundle):
    h, d = predict(HEALTHY, bundle=bundle), predict(DAMAGED, bundle=bundle)
    assert h["condition"] == "Undamaged" and h["severity_mm"] is None and h["severity_confidence"] is None
    assert h["severity_top_parameters"] is None
    assert d["condition"] == "Damaged" and d["severity_mm"] in (1, 2, 3)
    assert 0.0 <= d["severity_confidence"] <= 1.0 and d["severity_top_parameters"]


def test_output_fields_and_confidence_range(bundle):
    for row in (HEALTHY, DAMAGED, {"f1_hz": 7.6, "f2_hz": 38.8, "f3_hz": 126.0}):
        r = predict(row, bundle=bundle)
        for k in ("condition", "condition_confidence", "severity_mm", "severity_confidence",
                  "top_parameters", "severity_top_parameters", "baseline_comparison", "model_info"):
            assert k in r
        assert 0.5 <= r["condition_confidence"] <= 1.0      # confidence of the predicted class is never below 0.5
        assert r["baseline_comparison"] is None
        assert len(r["top_parameters"]) <= 5 and all(set(t) == {"feature", "importance", "value"} for t in r["top_parameters"])


@pytest.mark.parametrize("feature", FREQ3)
@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan"), float("inf"), "abc", None])
def test_failed_peak_is_rejected_not_guessed(bundle, feature, bad):
    row = dict(HEALTHY)
    row[feature] = bad
    with pytest.raises(ValueError, match="No valid spectral peak"):
        predict(row, bundle=bundle)


def test_missing_feature_column(bundle):
    with pytest.raises(KeyError):
        predict({"f1_hz": 7.6}, bundle=bundle)


def test_wrong_input_types(bundle):
    with pytest.raises(ValueError):
        predict(pd.DataFrame([HEALTHY, DAMAGED]), bundle=bundle)      # two rows
    with pytest.raises(ValueError):
        predict(pd.DataFrame(columns=FREQ3), bundle=bundle)           # zero rows
    with pytest.raises(TypeError):
        predict([7.6, 38.0, 126.0], bundle=bundle)


def test_input_forms_agree(bundle):
    a = predict(DAMAGED, bundle=bundle)
    b = predict(pd.Series(DAMAGED), bundle=bundle)
    c = predict(pd.DataFrame([DAMAGED]), bundle=bundle)
    for r in (b, c):
        assert r["condition"] == a["condition"] and r["severity_mm"] == a["severity_mm"]
        assert math.isclose(r["condition_confidence"], a["condition_confidence"], rel_tol=1e-12)


def test_numeric_strings_and_extra_columns_are_fine(bundle):
    row = {"f1_hz": "7.45", "f2_hz": "40.0", "f3_hz": "124.0", "case_id": "x", "condition": "Damaged", "sampling_rate_hz": 303.0}
    assert predict(row, bundle=bundle)["condition"] == "Damaged"


def test_extra_columns_do_not_change_the_answer(bundle):
    plain = predict(DAMAGED, bundle=bundle)
    noisy = predict({**DAMAGED, "case_id": "healthy", "condition": "Undamaged", "sampling_rate_hz": 333.3, "damage_level_mm": 0}, bundle=bundle)
    assert plain["condition"] == noisy["condition"] and plain["condition_confidence"] == noisy["condition_confidence"]


def test_deterministic(bundle):
    assert predict(DAMAGED, bundle=bundle) == predict(DAMAGED, bundle=bundle)


def test_threshold_is_respected_and_confidence_stays_valid(bundle):
    probe = {"f1_hz": 7.6, "f2_hz": 38.9, "f3_hz": 126.0}
    p = predict(probe, bundle=bundle)["model_info"]["p_damaged"]
    for thr in (0.2, 0.5, 0.7, 0.95):
        b = copy.copy(bundle)
        b["threshold_a"] = thr
        r = predict(probe, bundle=b)
        assert (r["condition"] == "Damaged") == (p >= thr)
        assert 0.5 <= r["condition_confidence"] <= 1.0


def test_ratio_columns_not_used_by_model_do_not_matter(bundle):
    row = {**HEALTHY, "f3_over_f1": 0.0, "f2_over_f1": float("nan")}
    assert predict(row, bundle=bundle)["condition"] == "Undamaged"


@pytest.mark.skipif(not Path(DEFAULT_BUNDLE).exists(), reason="no saved model bundle")
def test_real_saved_bundle_structure():
    b = load_bundle()
    for k in ("model_a", "model_b", "features_a", "features_b", "importances_a", "importances_b", "threshold_a", "a_transform"):
        assert k in b
    assert 0.0 < b["threshold_a"] < 1.0
    assert set(b["features_a"]) >= set(FREQ3)
    r = predict({"f1_hz": 7.62, "f2_hz": 37.9, "f3_hz": 126.5})
    assert r["condition"] in ("Damaged", "Undamaged")