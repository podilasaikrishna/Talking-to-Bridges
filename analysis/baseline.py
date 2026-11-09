"""Healthy baseline and comparison tool (NEW-2).

Compares one recording with the healthy state and returns "Interface B":

    overall_flag       "normal" | "abnormal" | "unavailable"
    deviations         ranked list; each item has feature, value, healthy_typical,
                       healthy_spread, change_percent, score (+ direction, outside, group)
    healthy_reference  healthy f1-f3 and (optional) a healthy reference spectrum
    n_healthy          how many healthy recordings and cases the baseline is built from

The healthy profile is the median and the robust spread (1.4826 * MAD) of every feature
over the healthy recordings. It is read from data/reference/healthy_baseline.json
(written by analysis.build_features_dataset). If that file is missing, it is built on the
fly from the healthy recordings in the reference folder.

Flag rule (default): "abnormal" when any frequency feature (f1_hz, f2_hz, f3_hz,
f2_over_f1, f3_over_f1) is more than FLAG_SPREADS spreads away from the healthy median.
Amplitude features (rms, variance, ...) are ranked after the frequency features and are
shown for information only - they do not change the flag.

Limits that must be shown with the result: the baseline comes from only two healthy
beams, and a new healthy beam can also be flagged.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from analysis.features import compute_spectrum, extract_features
from analysis.healthy_reference import load_healthy_recordings, reference_dir

logger = logging.getLogger("ttb.analysis.baseline")

FREQUENCY_FEATURES = ["f1_hz", "f2_hz", "f3_hz", "f2_over_f1", "f3_over_f1"]
FLAG_SPREADS = 3.0  # more than this many spreads from the healthy median = outside
BASELINE_FILE = "healthy_baseline.json"
SPECTRUM_MAX_HZ = 150.0
SPECTRUM_STEP_HZ = 0.1
SPECTRUM_SENSOR = "combined"  # same channel that extract_features uses for f1-f3

CAVEAT = (
    "The healthy baseline comes from only a few healthy recordings of two beams, so a "
    "new healthy beam can also be flagged. A flag means 'different from this baseline', "
    "not proof of damage."
)

_NICE = {
    "f1_hz": "mode 1 frequency (f1)",
    "f2_hz": "mode 2 frequency (f2)",
    "f3_hz": "mode 3 frequency (f3)",
    "f2_over_f1": "frequency ratio f2/f1",
    "f3_over_f1": "frequency ratio f3/f1",
}

_lock = threading.Lock()
_profile_cache: Dict[tuple, Optional[Dict[str, Any]]] = {}
_spectrum_cache: Dict[tuple, Optional[Dict[str, Any]]] = {}


# ---------------------------------------------------------------------------
# healthy profile
# ---------------------------------------------------------------------------

def _folder_key(folder: Path) -> tuple:
    """Changes when the reference folder content changes (so caches refresh)."""
    items = []
    if folder.is_dir():
        for p in sorted(folder.iterdir()):
            if p.is_file():
                st = p.stat()
                items.append((p.name, st.st_mtime_ns, st.st_size))
    return (str(folder), tuple(items))


def _robust_spread(values: np.ndarray) -> float:
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med))) * 1.4826
    if mad > 0:
        return mad
    std = float(np.std(values))
    return std if std > 0 else 0.01


def build_profile_from_recordings(recordings: List[pd.DataFrame]) -> Optional[Dict[str, Any]]:
    """Healthy profile (same layout as healthy_baseline.json) from cleaned recordings."""
    rows = []
    for rec in recordings:
        try:
            rows.append(extract_features(rec))
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the rest
            logger.warning("Healthy recording skipped (features failed): %s", exc)
    if not rows:
        return None
    table = pd.DataFrame(rows)
    features: Dict[str, Any] = {}
    for col in table.columns:
        if col == "sampling_rate_hz" or not pd.api.types.is_numeric_dtype(table[col]):
            continue
        vals = table[col].dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        features[col] = {
            "typical_median": float(np.median(vals)),
            "spread_mad": _robust_spread(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return {
        "n_healthy_recordings": len(rows),
        "healthy_cases": ["reference recordings (case names not recorded)"],
        "features": features,
        "source": "built from the healthy recordings in the reference folder",
    }


def load_profile(folder: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Healthy profile: healthy_baseline.json if present, else built from the recordings.

    Returns None when neither exists.
    """
    folder = Path(folder) if folder else reference_dir()
    key = _folder_key(folder)
    with _lock:
        if key in _profile_cache:
            return _profile_cache[key]
    profile: Optional[Dict[str, Any]] = None
    path = folder / BASELINE_FILE
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data.get("features"), dict) and data["features"]:
                profile = dict(data)
                profile["source"] = f"{BASELINE_FILE}"
        except Exception as exc:  # noqa: BLE001
            logger.warning("%s could not be read: %s", path, exc)
    if profile is None:
        recs = load_healthy_recordings(folder)
        if recs:
            profile = build_profile_from_recordings(recs)
    with _lock:
        _profile_cache[key] = profile
    return profile


def _spread_of(entry: Dict[str, Any]) -> float:
    for k in ("spread_mad", "std"):
        v = entry.get(k)
        if v is not None and float(v) > 0:
            return float(v)
    return 0.01


# ---------------------------------------------------------------------------
# healthy reference spectrum
# ---------------------------------------------------------------------------

def healthy_spectrum(folder: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Median healthy spectrum on a fixed 0-150 Hz grid (None without healthy recordings)."""
    folder = Path(folder) if folder else reference_dir()
    key = _folder_key(folder)
    with _lock:
        if key in _spectrum_cache:
            return _spectrum_cache[key]
    recs = load_healthy_recordings(folder)
    grid = np.arange(0.0, SPECTRUM_MAX_HZ + SPECTRUM_STEP_HZ / 2, SPECTRUM_STEP_HZ)
    curves = []
    for rec in recs:
        try:
            freqs, amps = compute_spectrum(rec, sensor=SPECTRUM_SENSOR)
            curves.append(np.interp(grid, freqs, amps))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Healthy spectrum skipped for one recording: %s", exc)
    result = None
    if curves:
        result = {
            "frequencies_hz": [round(float(x), 3) for x in grid],
            "amplitude": [float(x) for x in np.median(np.vstack(curves), axis=0)],
            "n_recordings": len(curves),
            "sensor": SPECTRUM_SENSOR,
        }
    with _lock:
        _spectrum_cache[key] = result
    return result


def recording_spectrum(df: pd.DataFrame) -> Dict[str, Any]:
    """Spectrum of one recording on the same grid as the healthy one."""
    grid = np.arange(0.0, SPECTRUM_MAX_HZ + SPECTRUM_STEP_HZ / 2, SPECTRUM_STEP_HZ)
    freqs, amps = compute_spectrum(df, sensor=SPECTRUM_SENSOR)
    return {
        "frequencies_hz": [round(float(x), 3) for x in grid],
        "amplitude": [float(x) for x in np.interp(grid, freqs, amps)],
        "sensor": SPECTRUM_SENSOR,
    }


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------

def _group(feature: str) -> str:
    return "frequency" if feature in FREQUENCY_FEATURES else "amplitude"


def _describe(d: Dict[str, Any]) -> str:
    name = _NICE.get(d["feature"], d["feature"])
    unit = " Hz" if d["feature"].endswith("_hz") else ""
    word = "higher" if d["direction"] == "higher" else "lower"
    return (
        f"{name} is {abs(d['change_percent']):.1f} percent {word} than healthy "
        f"({d['value']:.4g}{unit} vs {d['healthy_typical']:.4g}{unit}, {d['score']:.1f} spreads)"
    )


def _unavailable(reason: str, profile: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return {
        "overall_flag": "unavailable",
        "deviations": [],
        "healthy_reference": {},
        "n_healthy": _n_healthy(profile),
        "summary": reason,
        "caveat": CAVEAT,
    }


def _n_healthy(profile: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not profile:
        return {"recordings": 0, "cases": 0, "case_names": []}
    cases = list(profile.get("healthy_cases") or [])
    return {
        "recordings": int(profile.get("n_healthy_recordings") or 0),
        "cases": len(cases),
        "case_names": cases,
    }


def compare_to_baseline(
    data: Union[pd.DataFrame, Dict[str, Any], pd.Series],
    folder: Optional[Path] = None,
    include_spectrum: bool = True,
    flag_spreads: float = FLAG_SPREADS,
) -> Dict[str, Any]:
    """Compare one recording (cleaned DataFrame) or one feature row with the healthy state.

    Returns Interface B (see module docstring) plus a plain-language ``summary``,
    ``flag_rule`` and ``caveat``. Never raises for missing data: returns
    overall_flag="unavailable" with the reason in ``summary``.
    """
    profile = load_profile(folder)
    if not profile:
        return _unavailable(
            "No healthy baseline is available: put healthy_baseline.json or healthy "
            f"recordings in '{Path(folder) if folder else reference_dir()}'."
        )

    row: Dict[str, Any]
    df: Optional[pd.DataFrame] = None
    if isinstance(data, pd.DataFrame):
        df = data
        if not any(str(c).startswith("Sensor_") for c in df.columns):
            return _unavailable("The data has no Sensor_1..Sensor_5 columns, so it cannot be compared.", profile)
        try:
            row = extract_features(df)
        except Exception as exc:  # noqa: BLE001
            return _unavailable(f"Features could not be computed from this recording: {exc}", profile)
    else:
        row = dict(data)

    feats = profile["features"]
    deviations: List[Dict[str, Any]] = []
    for name, entry in feats.items():
        if name not in row or row[name] is None:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(value):
            continue
        if name in FREQUENCY_FEATURES and value <= 0:
            continue  # no peak found in the band: not a measurement
        typical = float(entry["typical_median"])
        spread = _spread_of(entry)
        signed = (value - typical) / spread
        change = (value - typical) / abs(typical) * 100.0 if abs(typical) > 1e-12 else 0.0
        deviations.append({
            "feature": name,
            "group": _group(name),
            "value": round(value, 6),
            "healthy_typical": round(typical, 6),
            "healthy_spread": round(spread, 6),
            "change_percent": round(change, 2),
            "score": round(abs(signed), 2),
            "direction": "higher" if signed >= 0 else "lower",
            "outside": bool(abs(signed) > flag_spreads),
        })

    freq_devs = [d for d in deviations if d["group"] == "frequency"]
    if not freq_devs:
        return _unavailable("No usable frequency features were found in the data.", profile)

    # Frequency features first (they decide the flag), amplitude features after.
    deviations.sort(key=lambda d: (d["group"] != "frequency", -d["score"]))
    outside = [d for d in freq_devs if d["outside"]]
    flag = "abnormal" if outside else "normal"
    amp_outside = sum(1 for d in deviations if d["group"] == "amplitude" and d["outside"])

    healthy_ref: Dict[str, Any] = {
        "frequencies_hz": {
            k: round(float(feats[k]["typical_median"]), 4) for k in FREQUENCY_FEATURES if k in feats
        },
        "profile_source": profile.get("source", BASELINE_FILE),
    }
    if include_spectrum:
        healthy_ref["spectrum"] = healthy_spectrum(folder)

    n_h = _n_healthy(profile)
    if outside:
        worst = sorted(outside, key=lambda d: -d["score"])[:3]
        summary = "Different from the healthy baseline: " + "; ".join(_describe(d) for d in worst) + "."
    else:
        closest = max(freq_devs, key=lambda d: d["score"])
        summary = (
            "Inside the healthy range for all frequency features "
            f"(largest difference: {_describe(closest)})."
        )
    summary += (
        f" Baseline: {n_h['recordings']} healthy recordings from {n_h['cases']} healthy case(s)."
    )

    return {
        "overall_flag": flag,
        "deviations": deviations,
        "healthy_reference": healthy_ref,
        "n_healthy": n_h,
        "flag_rule": (
            f"abnormal if any of {', '.join(FREQUENCY_FEATURES)} is more than "
            f"{flag_spreads:g} spreads from the healthy median"
        ),
        "amplitude_features_outside": amp_outside,
        "summary": summary,
        "caveat": CAVEAT,
    }
