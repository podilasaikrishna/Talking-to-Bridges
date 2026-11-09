import numpy as np
import pandas as pd
import plotly.io as pio
import pytest

from analysis.charts import (
    figure_to_json,
    plot_anomaly_chart,
    plot_time_series,
    sensor_columns,
)


@pytest.fixture
def df():
    n = 20000
    return pd.DataFrame({"Relative_Time_Sec": np.arange(n) / 300.0,
                         **{f"Sensor_{i}": np.random.randn(n) for i in range(1, 4)}})


def test_sensor_columns(df):
    assert sensor_columns(df) == ["Sensor_1", "Sensor_2", "Sensor_3"]


def test_time_series_auto_detects_and_downsamples(df):
    fig = plot_time_series(df)
    assert len(fig.data) == 3
    assert len(fig.data[0].x) <= 5000


def test_time_series_selected_sensor(df):
    assert len(plot_time_series(df, y_cols=["Sensor_2"]).data) == 1


@pytest.mark.parametrize("bad", [pd.DataFrame(), None])
def test_empty_data_raises(bad):
    with pytest.raises(ValueError):
        plot_time_series(bad)


def test_missing_column_raises(df):
    with pytest.raises(ValueError, match="deflection"):
        plot_time_series(df, y_cols=["deflection"])


def test_anomaly_from_column_survives_downsampling(df):
    df["is_anomaly"] = False
    df.loc[[7, 12345], "is_anomaly"] = True
    fig = plot_anomaly_chart(df)
    assert len(fig.data) == 2 and len(fig.data[1].x) == 2


def test_anomaly_from_x_values(df):
    fig = plot_anomaly_chart(df, anomaly_x=[df["Relative_Time_Sec"][10]])
    assert len(fig.data[1].x) == 1


def test_no_anomalies_gives_line_only(df):
    assert len(plot_anomaly_chart(df).data) == 1


def test_json_round_trip(df):
    fig = pio.from_json(figure_to_json(plot_time_series(df)))
    assert len(fig.data) == 3


def test_events_mark_spans_and_peaks(df):
    events = [
        {"start_index": 100, "end_index": 400, "peak_index": 250},
        {"start_index": 5000, "end_index": 5100, "peak_index": 5050},
    ]
    fig = plot_anomaly_chart(df, events=events, flagged_count=2420)
    names = [t.name for t in fig.data]
    assert "Event peak" in names and "Event start/end" in names
    assert len(fig.layout.shapes) == 2
    assert "2420 flagged in 2 event(s)" in fig.layout.title.text


def test_empty_events_gives_line_only(df):
    fig = plot_anomaly_chart(df, events=[])
    assert len(fig.data) == 1 and len(fig.layout.shapes) == 0
