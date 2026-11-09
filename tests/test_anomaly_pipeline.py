"""Week 6 tests - Kolla's anomaly-detection pipeline (analysis/anomaly_pipeline.py)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.anomaly_pipeline import (
    MAX_REPORTED_ANOMALIES,
    MIN_POINTS,
    AnomalyInputError,
    detect_anomalies,
    robust_scores,
)
from analysis.dispatcher import dispatcher
from analysis.schemas import AnomalyDetectionOutput
from analysis.tools import registry

FIXTURE = Path(__file__).parent / "fixtures" / "bridge_sensor_data.csv"


@pytest.fixture
def sensor_df():
    """Professor-style table with one clear spike at row 25 and a 3-row impact at 40-42."""
    n = 60
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "Relative_Time_Sec": [i * 0.003 for i in range(n)],
        "Sensor_1": rng.normal(0.5, 0.01, n),
        "Sensor_2": rng.normal(0.3, 0.01, n),
    })
    df.loc[25, "Sensor_1"] = 50.0
    df.loc[40:42, "Sensor_1"] = [30.0, 35.0, 31.0]
    return df


@pytest.fixture
def fixture_df():
    return pd.read_csv(FIXTURE)


class TestRobustScores:
    def test_constant_signal_scores_zero(self):
        scores, median, scale, kind = robust_scores(np.full(10, 3.0))
        assert np.all(scores == 0)
        assert kind == "constant signal"

    def test_mad_zero_falls_back_to_mean_absolute_deviation(self):
        values = np.array([1.0] * 9 + [10.0])
        scores, _, scale, kind = robust_scores(values)
        assert kind == "mean absolute deviation"
        assert abs(scores[-1]) > 2.0
        assert np.all(np.isfinite(scores))

    def test_spike_does_not_hide_itself(self):
        values = np.array([1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 100.0, 90.0, 95.0])
        scores, *_ = robust_scores(values)
        assert np.all(np.abs(scores[-3:]) > 3.0)


class TestDetectAnomalies:
    def test_finds_spike_and_flags_alert(self, sensor_df):
        d = detect_anomalies(sensor_df, "Sensor_1", threshold=3.0)
        assert d["status_flag"] == "Alert"
        assert 25 in d["anomalous_indices"]
        assert {40, 41, 42}.issubset(d["anomalous_indices"])
        assert d["total_rows_checked"] == 60
        assert d["anomaly_count"] == len(d["anomalous_indices"])
        assert d["anomaly_fraction"] == round(d["anomaly_count"] / 60, 4)

    def test_events_group_neighbouring_points(self, sensor_df):
        d = detect_anomalies(sensor_df, "Sensor_1", threshold=5.0, merge_gap=0)
        starts = [e["start_index"] for e in d["events"]]
        assert starts == [25, 40]
        impact = d["events"][1]
        assert (impact["end_index"], impact["points"], impact["peak_index"]) == (42, 3, 41)
        assert impact["peak_value"] == 35.0
        assert impact["direction"] == "high"
        assert d["event_count"] == 2

    def test_close_events_are_merged_by_default(self, sensor_df):
        # spike at row 25 and impact at 40-42 are 14 readings apart -> one event
        d = detect_anomalies(sensor_df, "Sensor_1", threshold=5.0)
        assert d["event_count"] == 1
        ev = d["events"][0]
        assert (ev["start_index"], ev["end_index"], ev["points"]) == (25, 42, 4)
        assert ev["peak_index"] == 25 and ev["peak_value"] == 50.0

    def test_far_apart_events_stay_separate(self):
        x = np.zeros(200)
        x[10] = 50.0
        x[150] = -50.0
        x += np.random.default_rng(0).normal(0, 0.01, 200)
        d = detect_anomalies(pd.DataFrame({"x": x}), "x", threshold=5.0)
        assert d["event_count"] == 2
        assert [e["direction"] for e in d["events"]] == ["high", "low"]

    def test_ring_down_after_one_hit_is_one_event(self):
        rng = np.random.default_rng(3)
        n, dt = 5719, 0.003
        t = np.arange(n) * dt
        sig = rng.normal(0, 0.01, n)
        m = t >= 2.42
        sig[m] += -3.0 * np.exp(-(t[m] - 2.42) / 0.6) * np.cos(2 * np.pi * 14 * (t[m] - 2.42) + 0.3)
        df = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": sig})
        strict = detect_anomalies(df, "Sensor_1", threshold=4.0, merge_gap=0)
        merged = detect_anomalies(df, "Sensor_1", threshold=4.0)
        assert strict["event_count"] > 20      # fragments
        assert merged["event_count"] <= 3      # one hit
        assert merged["anomaly_count"] == strict["anomaly_count"]

    def test_bad_merge_gap(self, sensor_df):
        with pytest.raises(AnomalyInputError):
            detect_anomalies(sensor_df, "Sensor_1", merge_gap="x")

    def test_timestamps_use_relative_time(self, sensor_df):
        d = detect_anomalies(sensor_df, "Sensor_1", threshold=5.0)
        assert d["anomalous_timestamps"][0] == str(sensor_df["Relative_Time_Sec"].iloc[25])

    def test_clean_signal_is_normal(self, sensor_df):
        d = detect_anomalies(sensor_df, "Sensor_2", threshold=10.0)
        assert d["status_flag"] == "Normal"
        assert d["anomaly_count"] == 0
        assert d["anomalous_indices"] == [] and d["events"] == []

    def test_constant_signal_is_normal(self):
        df = pd.DataFrame({"Sensor_1": [999.0] * 20})
        d = detect_anomalies(df, "Sensor_1", threshold=0.5)
        assert d["status_flag"] == "Normal" and d["anomaly_count"] == 0
        assert d["baseline"]["scale_type"] == "constant signal"

    def test_nan_and_inf_are_dropped_not_crashing(self):
        vals = [1.0, 1.1, 0.9, np.nan, 1.0, np.inf, -np.inf, 1.05, 0.95, 40.0, 1.0]
        d = detect_anomalies(pd.DataFrame({"x": vals}), "x", threshold=3.0)
        assert d["total_rows_checked"] == 8
        assert d["anomalous_indices"] == [9]  # position in the ORIGINAL frame

    def test_text_values_are_coerced_or_dropped(self):
        df = pd.DataFrame({"x": ["1", "1.1", "oops", "0.9", "1.0", "1.05", "60"]})
        d = detect_anomalies(df, "x", threshold=3.0)
        assert d["total_rows_checked"] == 6
        assert d["anomalous_indices"] == [6]

    def test_sensor_filter_keeps_original_row_positions(self, fixture_df):
        df = fixture_df.copy()
        df.loc[df.index[df["sensor_id"] == "S02"][5], "vibration"] = 9.9
        pos = int(df.index[df["sensor_id"] == "S02"][5])
        d = detect_anomalies(df, "vibration", sensor_id="S02", threshold=3.0)
        assert d["anomalous_indices"] == [pos]
        assert d["anomalous_timestamps"] == [str(df["timestamp"].iloc[pos])]
        assert d["total_rows_checked"] == int((df["sensor_id"] == "S02").sum())

    def test_non_default_index_is_handled(self, sensor_df):
        df = sensor_df.copy()
        df.index = np.arange(1000, 1000 + len(df))
        d = detect_anomalies(df, "Sensor_1", threshold=5.0)
        assert 25 in d["anomalous_indices"]

    def test_cap_keeps_most_severe_in_row_order(self):
        rng = np.random.default_rng(1)
        vals = rng.normal(0, 1, 400)
        vals[300] = 500.0
        d = detect_anomalies(pd.DataFrame({"x": vals}), "x", threshold=0.5)
        assert d["anomaly_count"] > MAX_REPORTED_ANOMALIES
        assert len(d["anomalous_indices"]) == MAX_REPORTED_ANOMALIES
        assert len(d["anomalous_timestamps"]) == MAX_REPORTED_ANOMALIES
        assert 300 in d["anomalous_indices"]
        assert d["anomalous_indices"] == sorted(d["anomalous_indices"])

    def test_too_few_points_returns_note_not_error(self):
        d = detect_anomalies(pd.DataFrame({"x": [1.0, 2.0, 100.0]}), "x")
        assert d["status_flag"] == "Normal"
        assert str(MIN_POINTS) in d["_note"]

    def test_duplicate_column_names_do_not_crash(self):
        df = pd.DataFrame([[1.0, 1.0]] * 9 + [[80.0, 1.0]], columns=["x", "x"])
        d = detect_anomalies(df, "x", threshold=3.0)
        assert d["anomalous_indices"] == [9]

    def test_datetime_time_column(self):
        df = pd.DataFrame({
            "DateTime": pd.date_range("2026-01-01", periods=12, freq="D"),
            "x": [1.0] * 11 + [50.0],
        })
        d = detect_anomalies(df, "x", threshold=3.0)
        assert d["anomalous_timestamps"] == [pd.Timestamp("2026-01-12").isoformat()]

    def test_no_time_column_uses_row_number(self):
        d = detect_anomalies(pd.DataFrame({"x": [1.0] * 9 + [50.0]}), "x", threshold=3.0)
        assert d["anomalous_timestamps"] == ["9"]

    def test_result_is_json_serialisable_and_matches_schema(self, sensor_df):
        d = detect_anomalies(sensor_df, "Sensor_1", threshold=3.0)
        json.dumps(d)
        AnomalyDetectionOutput(**{k: v for k, v in d.items() if k in AnomalyDetectionOutput.model_fields})

    def test_input_df_is_not_modified(self, sensor_df):
        before = sensor_df.copy()
        detect_anomalies(sensor_df, "Sensor_1")
        pd.testing.assert_frame_equal(sensor_df, before)


class TestErrors:
    def test_unknown_column(self, sensor_df):
        with pytest.raises(AnomalyInputError) as e:
            detect_anomalies(sensor_df, "nope")
        assert e.value.error_type == "INVALID_INPUT"

    @pytest.mark.parametrize("bad", ["abc", -1, 0, float("nan"), float("inf")])
    def test_bad_threshold(self, sensor_df, bad):
        with pytest.raises(AnomalyInputError) as e:
            detect_anomalies(sensor_df, "Sensor_1", threshold=bad)
        assert e.value.error_type == "INVALID_INPUT"

    def test_all_values_missing(self):
        with pytest.raises(AnomalyInputError) as e:
            detect_anomalies(pd.DataFrame({"x": [np.nan, np.nan]}), "x")
        assert e.value.error_type == "DATA_UNAVAILABLE"

    def test_sensor_filter_matches_no_rows(self, fixture_df):
        with pytest.raises(AnomalyInputError) as e:
            detect_anomalies(fixture_df, "vibration", sensor_id="S99")
        assert e.value.error_type == "DATA_UNAVAILABLE"


class TestRegistryIntegration:
    def test_tool_returns_real_result(self, sensor_df):
        res = registry.execute("anomaly_detection", {"metric": "Sensor_1", "threshold": 3.0}, df=sensor_df)
        assert res["status"] == "success"
        assert res["data"]["status_flag"] == "Alert"
        assert "_note" not in res["data"]
        json.dumps(res)

    def test_threshold_is_optional(self, sensor_df):
        res = registry.execute("anomaly_detection", {"metric": "Sensor_1"}, df=sensor_df)
        assert res["status"] == "success"
        assert res["data"]["threshold_z"] == 2.0

    def test_no_dataset_gives_placeholder(self):
        res = registry.execute("anomaly_detection", {"metric": "vibration"})
        assert res["status"] == "success"
        assert res["data"]["anomaly_count"] == 2 and "_note" in res["data"]

    def test_empty_metric_surfaces_data_unavailable(self):
        df = pd.DataFrame({"x": [np.nan] * 6})
        res = registry.execute("anomaly_detection", {"metric": "x"}, df=df)
        assert res["status"] == "error" and res["error_type"] == "DATA_UNAVAILABLE"

    def test_missing_metric_is_invalid_input(self, sensor_df):
        res = registry.execute("anomaly_detection", {"metric": "zzz"}, df=sensor_df)
        assert res["status"] == "error" and res["error_type"] == "INVALID_INPUT"

    def test_dispatcher_end_to_end(self, sensor_df):
        out = dispatcher.detect_and_dispatch("Are there any anomalies in Sensor_1?", df=sensor_df)
        assert out is not None and out.status == "success"
        assert out.tool_name == "anomaly_detection"
        assert out.data["anomaly_count"] >= 1
        assert "anomaly_detection" in out.llm_summary