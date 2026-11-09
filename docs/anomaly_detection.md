# Week 6 - Anomaly detection pipeline

Replaces the z-score placeholder behind the `anomaly_detection` tool.
Code: `analysis/anomaly_pipeline.py`; handler wiring: `handle_anomaly_detection` in `analysis/tools.py`.
Tests: `tests/test_anomaly_pipeline.py`.

## Contract (unchanged)
Input: `metric` (required), `sensor_id`, `threshold` (default 2.0).
Output: every key of `AnomalyDetectionOutput` with the same types.
Added keys (extra, safe to ignore): `method`, `max_abs_z`, `baseline`, `event_count`, `events`.

## Method
1. Filter by `sensor_id` (when that column exists), convert to numbers, drop NaN/inf.
2. Robust z-score: `0.6745 * (x - median) / MAD`. If MAD is 0, mean absolute deviation is used; a constant signal flags nothing.
3. Flag `|score| > threshold`. Fewer than 5 valid readings: nothing flagged, `_note` explains why.
4. Neighbouring flagged readings are grouped into events (start, end, peak value, peak score, direction).

## Behaviour notes
- `anomalous_indices` are row positions (0-based) in the DataFrame passed in, so they stay correct after a sensor filter.
- `anomalous_timestamps` come from Relative_Time_Sec, then timestamp, then DateTime, else the row number.
- Lists are capped at 50; when capped, the most severe readings are kept, in row order.
- No dataset: the old deterministic placeholder is returned (with `_note`).
- Errors: unknown column or bad threshold -> `INVALID_INPUT`; no valid readings -> `DATA_UNAVAILABLE`.