"""Anomaly-detection pipeline (Phase 2 - Week 6, Kolla).

Replaces the plain z-score placeholder behind the ``anomaly_detection`` tool.
The tool's input and output contract (analysis/schemas.py) is NOT changed:
every key of ``AnomalyDetectionOutput`` is still returned with the same type.
A few extra keys are added (method, max_abs_z, baseline, event_count, events);
they are additive, so existing code that ignores them keeps working.

What the pipeline does for one metric (e.g. ``Sensor_1`` or ``vibration``):
    1. Filters rows by ``sensor_id`` when that column exists.
    2. Converts the metric to numbers and drops missing / non-finite values.
       Nothing is dropped silently: ``total_rows_checked`` is the number of
       valid readings that were actually scanned.
    3. Scores every reading with a ROBUST z-score (median and MAD instead of
       mean and standard deviation), so a few large spikes cannot inflate the
       spread and hide themselves. When the MAD is zero (most readings are
       identical) the mean absolute deviation is used instead; when the signal
       is perfectly constant nothing is flagged.
    4. Flags readings whose |score| is above ``threshold``.
    5. Groups flagged readings into events (one impact = one event) and
       reports where each event starts, ends and peaks. Flagged readings that
       are separated by at most ``merge_gap`` unflagged readings (default 25)
       are one event, because a vibrating structure swings through zero and
       would otherwise split a single hit into many fragments.

Row numbers (``anomalous_indices``) are POSITIONS in the DataFrame that was
passed in (0 = first row), so they stay correct after a sensor_id filter.
``anomalous_timestamps`` is read from the same rows, using
Relative_Time_Sec, then timestamp, then DateTime; the row number when none
exists.

This module does not import analysis.tools, so there is no circular import.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("ttb.analysis.anomaly")

METHOD_NAME = "robust_zscore"
DEFAULT_THRESHOLD = 2.0
MIN_POINTS = 5               # fewer valid readings than this: no reliable decision
MAX_REPORTED_ANOMALIES = 50  # cap on anomalous_indices / anomalous_timestamps
MAX_REPORTED_EVENTS = 20     # cap on the events list
MERGE_GAP = 25               # flagged readings separated by <= this many unflagged
                             # readings belong to the same event (a hit makes the
                             # signal swing through zero, which splits one impact
                             # into many short runs)
_TIME_COLUMNS = ("Relative_Time_Sec", "timestamp", "DateTime")
_MAD_TO_SIGMA = 0.6745       # makes the MAD score comparable to a normal z-score
_MEANAD_TO_SIGMA = 1.253314  # same, when the MAD is zero (Iglewicz & Hoaglin)
_REL_EPS = 1e-12             # spread below this (relative) counts as "constant"


class AnomalyInputError(Exception):
    """Raised for bad input. tools.py converts it into a ToolExecutionError."""

    def __init__(self, error_type: str, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_column(df: pd.DataFrame, name: str) -> pd.Series:
    """Return df[name] as a Series even if the column name is duplicated."""
    col = df[name]
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    return col


def _resolve_threshold(value: Any) -> float:
    if value is None:
        return DEFAULT_THRESHOLD
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise AnomalyInputError(
            "INVALID_INPUT", f"threshold must be a number, got {value!r}."
        )
    if not np.isfinite(threshold) or threshold <= 0:
        raise AnomalyInputError(
            "INVALID_INPUT", f"threshold must be a positive finite number, got {value!r}."
        )
    return threshold


def _select_valid_readings(
    df: pd.DataFrame, metric: str, sensor_id: Optional[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (positions, values) of the valid readings for the metric.

    positions are row numbers in the original df (0-based).
    """
    if metric not in df.columns:
        raise AnomalyInputError(
            "INVALID_INPUT",
            f"Column '{metric}' not found in dataset.",
            {"available_columns": [str(c) for c in df.columns]},
        )

    positions = np.arange(len(df))
    if sensor_id is not None and "sensor_id" in df.columns:
        keep = (_first_column(df, "sensor_id").astype(str) == str(sensor_id)).to_numpy()
        positions = positions[keep]

    try:
        numeric = pd.to_numeric(_first_column(df, metric), errors="coerce")
        values = numeric.to_numpy(dtype=float)[positions]
    except (TypeError, ValueError) as exc:
        raise AnomalyInputError(
            "INVALID_INPUT", f"Column '{metric}' is not numeric: {exc}"
        )

    finite = np.isfinite(values)
    return positions[finite], values[finite]


def robust_scores(values: np.ndarray) -> Tuple[np.ndarray, float, float, str]:
    """Return (scores, median, scale, scale_type) for a 1-D float array.

    scores are signed robust z-scores. A constant signal gives all zeros.
    """
    median = float(np.median(values))
    dev = values - median
    tol = _REL_EPS * max(1.0, abs(median))

    mad = float(np.median(np.abs(dev)))
    if mad > tol:
        return _MAD_TO_SIGMA * dev / mad, median, mad, "MAD"

    mean_ad = float(np.mean(np.abs(dev)))
    if mean_ad > tol:
        return dev / (_MEANAD_TO_SIGMA * mean_ad), median, mean_ad, "mean absolute deviation"

    return np.zeros_like(values, dtype=float), median, 0.0, "constant signal"


def _resolve_time_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in _TIME_COLUMNS:
        if candidate in df.columns:
            return candidate
    return None


def _format_time(value: Any, fallback: int) -> str:
    try:
        if pd.isna(value):
            return str(fallback)
    except (TypeError, ValueError):
        pass
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _group_events(flagged: np.ndarray, merge_gap: int) -> List[np.ndarray]:
    """Group flagged clean-array indices into events.

    Two flagged readings are in the same event when at most ``merge_gap``
    unflagged readings lie between them (merge_gap=0: strict neighbours only).
    """
    if flagged.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(flagged) - 1 > merge_gap) + 1
    return np.split(flagged, breaks)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_anomalies(
    df: pd.DataFrame,
    metric: str,
    sensor_id: Optional[str] = None,
    threshold: Any = DEFAULT_THRESHOLD,
    merge_gap: int = MERGE_GAP,
) -> Dict[str, Any]:
    """Scan one metric of df for anomalous readings.

    Returns a JSON-serialisable dict (see module docstring for the keys).

    Raises:
        AnomalyInputError: INVALID_INPUT (bad column / threshold) or
            DATA_UNAVAILABLE (no valid readings after filtering).
    """
    thr = _resolve_threshold(threshold)
    try:
        gap = max(0, int(merge_gap))
    except (TypeError, ValueError):
        raise AnomalyInputError(
            "INVALID_INPUT", f"merge_gap must be a whole number, got {merge_gap!r}."
        )
    positions, values = _select_valid_readings(df, metric, sensor_id)
    total = int(values.size)

    result: Dict[str, Any] = {
        "sensor_id": sensor_id,
        "metric": metric,
        "threshold_z": thr,
        "total_rows_checked": total,
        "anomaly_count": 0,
        "anomaly_fraction": 0.0,
        "anomalous_indices": [],
        "anomalous_timestamps": [],
        "status_flag": "Normal",
        "method": METHOD_NAME,
        "max_abs_z": 0.0,
        "baseline": {"median": None, "scale": None, "scale_type": None},
        "event_count": 0,
        "events": [],
    }

    if total == 0:
        raise AnomalyInputError(
            "DATA_UNAVAILABLE",
            f"No valid numeric readings for '{metric}'"
            + (f" and sensor_id '{sensor_id}'." if sensor_id is not None else "."),
        )
    if total < MIN_POINTS:
        result["_note"] = (
            f"Only {total} valid readings; at least {MIN_POINTS} are needed for a "
            "reliable anomaly decision, so none were flagged."
        )
        return result

    scores, median, scale, scale_type = robust_scores(values)
    abs_scores = np.abs(scores)
    flagged = np.flatnonzero(abs_scores > thr)  # indices into the clean arrays

    time_col = _resolve_time_column(df)
    time_series = _first_column(df, time_col) if time_col else None

    def time_at(clean_idx: int) -> str:
        pos = int(positions[clean_idx])
        if time_series is None:
            return str(pos)
        return _format_time(time_series.iloc[pos], pos)

    # Reported list: if there are more than the cap, keep the most severe ones,
    # but always present them in row order.
    reported = flagged
    if flagged.size > MAX_REPORTED_ANOMALIES:
        top = np.argsort(-abs_scores[flagged], kind="stable")[:MAX_REPORTED_ANOMALIES]
        reported = np.sort(flagged[top])

    # Events (one impact = one event)
    events: List[Dict[str, Any]] = []
    for group in _group_events(flagged, gap):
        peak = int(group[np.argmax(abs_scores[group])])
        events.append({
            "start_index": int(positions[group[0]]),
            "end_index": int(positions[group[-1]]),
            "points": int(group.size),
            "start_time": time_at(int(group[0])),
            "end_time": time_at(int(group[-1])),
            "peak_index": int(positions[peak]),
            "peak_value": round(float(values[peak]), 6),
            "peak_z": round(float(scores[peak]), 3),
            "direction": "high" if scores[peak] > 0 else "low",
        })
    event_count = len(events)
    if event_count > MAX_REPORTED_EVENTS:
        events.sort(key=lambda e: abs(e["peak_z"]), reverse=True)
        events = sorted(events[:MAX_REPORTED_EVENTS], key=lambda e: e["start_index"])

    count = int(flagged.size)
    result.update({
        "anomaly_count": count,
        "anomaly_fraction": round(count / total, 4),
        "anomalous_indices": [int(positions[i]) for i in reported],
        "anomalous_timestamps": [time_at(int(i)) for i in reported],
        "status_flag": "Alert" if count > 0 else "Normal",
        "max_abs_z": round(float(abs_scores.max()), 3),
        "baseline": {
            "median": round(median, 6),
            "scale": round(scale, 6),
            "scale_type": scale_type,
        },
        "event_count": event_count,
        "events": events,
    })
    return result