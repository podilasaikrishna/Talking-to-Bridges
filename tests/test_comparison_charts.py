"""Healthy-vs-uploaded comparison charts (faculty requirement 5)."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis.charts import (
    plot_deviation_heatmap,
    plot_feature_deviation,
    plot_recording_comparison,
)
from analysis.dispatcher import _detect_comparison, dispatcher
from analysis.features import build_healthy_baseline, rank_deviations
from analysis.healthy_reference import load_healthy_recordings
from analysis.schemas import ChartDataInput
from analysis.tools import registry

SENSORS = [f"Sensor_{i}" for i in range(1, 6)]


def _recording(n=3000, scale=1.0, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"Relative_Time_Sec": 10.0 + np.arange(n) / 303.0})
    for s in SENSORS:
        df[s] = rng.normal(1.5 + shift, 0.05 * scale, n)
    return df


@pytest.fixture
def healthy():
    return _recording(seed=1)


@pytest.fixture
def damaged():
    return _recording(scale=6.0, seed=2, shift=0.2)


@pytest.fixture
def ref_dir(tmp_path, healthy, monkeypatch):
    folder = tmp_path / "reference"
    folder.mkdir()
    healthy.to_csv(folder / "healthy.csv", index=False)
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(folder))
    return folder


@pytest.fixture
def deviations(healthy, damaged):
    return rank_deviations(damaged, build_healthy_baseline([healthy]))


def _fig(result):
    assert result["data"]["plot_json"] != "{}", result["data"]["explanation"]
    return json.loads(result["data"]["plot_json"])


# ---- chart functions -------------------------------------------------------

def test_recording_comparison_has_two_lines_per_sensor(healthy, damaged):
    fig = plot_recording_comparison(healthy, damaged)
    assert len(fig.data) == 2 * len(SENSORS)
    assert {t.name for t in fig.data} == {"Healthy", "Uploaded"}
    assert sum(1 for t in fig.data if t.showlegend) == 2  # one legend entry each


def test_recording_comparison_starts_both_at_zero(healthy, damaged):
    fig = plot_recording_comparison(healthy, damaged, sensors=["Sensor_1"])
    assert fig.data[0].x[0] == 0 and fig.data[1].x[0] == 0


def test_recording_comparison_single_sensor(healthy, damaged):
    assert len(plot_recording_comparison(healthy, damaged, sensors=["Sensor_3"]).data) == 2


def test_recording_comparison_rejects_bad_input(healthy):
    with pytest.raises(ValueError):
        plot_recording_comparison(healthy, pd.DataFrame())
    with pytest.raises(ValueError):
        plot_recording_comparison(healthy, healthy[["Relative_Time_Sec"]])


def test_feature_deviation_marks_outside_range(deviations):
    fig = plot_feature_deviation(deviations, top_n=5)
    assert len(fig.data[0].x) == 5
    assert "outside the range" in fig.layout.title.text
    assert all(c == "#eb6834" for c in fig.data[0].marker.color[-1:])  # biggest is outside


def test_feature_deviation_sensor_filter_and_unknown_sensor(deviations):
    fig = plot_feature_deviation(deviations, sensor="Sensor_2")
    assert all(label.startswith("Sensor_2") for label in fig.data[0].y)
    with pytest.raises(ValueError):
        plot_feature_deviation(deviations, sensor="Sensor_9")


def test_fft_peak_freq_rows_are_split_correctly(deviations):
    """rank_deviations leaves 'Sensor_1_fft_peak_freq' unsplit; the chart must still label it."""
    labels = list(plot_feature_deviation(deviations, top_n=40).data[0].y)
    assert "Sensor_1 fft_peak_freq" in labels


def test_heatmap_is_sensor_by_feature(deviations):
    fig = plot_deviation_heatmap(deviations)
    assert len(fig.data[0].y) == 5 and len(fig.data[0].x) == 8


def test_empty_deviations_raise():
    with pytest.raises(ValueError):
        plot_feature_deviation(pd.DataFrame())


# ---- healthy reference loader ----------------------------------------------

def test_loader_reads_csv_and_caches(ref_dir):
    first = load_healthy_recordings()
    assert len(first) == 1 and "Sensor_1" in first[0].columns
    assert load_healthy_recordings()[0] is first[0]


def test_loader_missing_folder_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(tmp_path / "nope"))
    assert load_healthy_recordings() == []


def test_loader_skips_unreadable_files(ref_dir):
    (ref_dir / "broken.xlsx").write_bytes(b"not an excel file")
    assert len(load_healthy_recordings()) == 1


# ---- chart_data tool --------------------------------------------------------

@pytest.mark.parametrize("mode,kind", [("signals", "scattergl"), ("features", "bar"), ("heatmap", "heatmap")])
def test_tool_comparison_modes(ref_dir, damaged, mode, kind):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "comparison": mode}, damaged)
    assert res["status"] == "success"
    assert _fig(res)["data"][0]["type"] == kind
    assert res["data"]["comparison"] == mode


def test_tool_features_explanation_names_top_deviation(ref_dir, damaged):
    data = registry.execute("chart_data", {"y_col": "Relative_Time_Sec", "x_col": "Relative_Time_Sec",
                                           "comparison": "features"}, damaged)["data"]
    assert "Outside the healthy range" in data["explanation"] and "sigma" in data["explanation"]


def test_tool_without_reference_returns_placeholder_with_reason(tmp_path, monkeypatch, damaged):
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(tmp_path / "empty"))
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "comparison": "signals"}, damaged)
    assert res["status"] == "success" and res["data"]["plot_json"] == "{}"
    assert "no healthy reference" in res["data"]["explanation"]


def test_tool_plain_chart_unchanged_without_comparison(ref_dir, damaged):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "x_col": "Relative_Time_Sec"}, damaged)
    assert [t["name"] for t in _fig(res)["data"]] == ["Sensor_1"]


def test_schema_comparison_field():
    assert ChartDataInput(y_col="Sensor_1").comparison is None
    assert ChartDataInput(y_col="Sensor_1", comparison="features").comparison == "features"
    with pytest.raises(ValueError):
        ChartDataInput(y_col="Sensor_1", comparison="nonsense")
    props = registry.get_tool("chart_data").input_schema["properties"]
    assert props["comparison"]["enum"] == ["signals", "features", "heatmap"]


# ---- dispatcher -------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("plot sensor_3 healthy vs uploaded", "signals"),
    ("compare healthy and damaged sensor_2 chart", "signals"),
    ("plot features against the healthy range", "features"),
    ("graph the kurtosis deviation compared to baseline", "features"),
    ("chart a heatmap of deviations versus healthy", "heatmap"),
    ("plot sensor_1", None),
    ("plot the variance of sensor_1", None),
])
def test_detect_comparison(query, expected):
    from analysis.dispatcher import _tokenize

    assert _detect_comparison(_tokenize(query)) == expected


def test_dispatcher_returns_comparison_chart(ref_dir, damaged):
    result = dispatcher.detect_and_dispatch("plot sensor_3 healthy vs uploaded", damaged)
    assert result.tool_name == "chart_data" and result.status == "success"
    assert result.data["comparison"] == "signals" and result.data["plot_json"] != "{}"


def test_dispatcher_plain_chart_has_no_comparison(damaged):
    result = dispatcher.detect_and_dispatch("plot sensor_3", damaged)
    assert "comparison" not in result.metadata["arguments"]


@pytest.mark.parametrize("query", ["plot sensor 3", "plot Sensor-3 healthy vs uploaded", "chart sensor_3"])
def test_sensor_named_with_space_is_recognised(ref_dir, damaged, query):
    result = dispatcher.detect_and_dispatch(query, damaged)
    assert result.metadata["arguments"]["y_col"] == "Sensor_3"


def test_llm_summary_says_chart_is_already_shown(ref_dir, damaged):
    shown = dispatcher.detect_and_dispatch("plot features against healthy range", damaged)
    assert "already displayed" in shown.llm_summary
    placeholder = dispatcher.detect_and_dispatch("plot sensor_1", None)
    assert "already displayed" not in placeholder.llm_summary
