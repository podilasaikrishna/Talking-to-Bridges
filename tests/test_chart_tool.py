"""Week 6 - chart_data tool: interactive charts from the tool result (Nagarjun)."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis.schemas import ChartDataInput
from analysis.tools import registry


@pytest.fixture
def recording():
    """Professor-style cleaned recording with one impact on Sensor_1."""
    n = 6000
    rng = np.random.default_rng(0)
    s1 = rng.normal(0, 0.01, n)
    s1[3000:3010] += 5.0
    return pd.DataFrame({
        "Relative_Time_Sec": np.arange(n) / 100.0,
        "Sensor_1": s1,
        "Sensor_2": rng.normal(0, 0.01, n),
    })


def _fig(result):
    data = result["data"]
    assert data["plot_json"] != "{}"
    return json.loads(data["plot_json"])


def test_sensor_chart_is_real_plotly_json(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "x_col": "Relative_Time_Sec"}, recording)
    assert res["status"] == "success"
    fig = _fig(res)
    assert len(fig["data"]) == 1
    assert fig["data"][0]["name"] == "Sensor_1"
    assert len(fig["data"][0]["y"]) <= 5001  # downsampled


def test_default_x_falls_back_to_relative_time(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_2"}, recording)
    assert res["data"]["x_col"] == "Relative_Time_Sec"
    assert _fig(res)["data"][0]["name"] == "Sensor_2"


def test_anomaly_overlay_marks_the_impact(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "show_anomalies": True}, recording)
    assert res["status"] == "success"
    fig = _fig(res)
    names = [t["name"] for t in fig["data"]]
    assert "Event peak" in names
    assert len(fig["layout"]["shapes"]) >= 1  # one shaded span per event
    assert "flagged" in res["data"]["explanation"]


def test_anomaly_overlay_accepts_string_flag(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "show_anomalies": "true"}, recording)
    assert "Event peak" in [t["name"] for t in _fig(res)["data"]]


def test_output_contract_keys(recording):
    data = registry.execute("chart_data", {"y_col": "Sensor_1"}, recording)["data"]
    assert set(data) >= {"x_col", "y_col", "plot_json", "explanation"}
    assert isinstance(data["plot_json"], str) and isinstance(data["explanation"], str)


def test_no_data_still_returns_placeholder():
    res = registry.execute("chart_data", {"y_col": "Sensor_1"})
    assert res["status"] == "success"
    assert res["data"]["plot_json"] == "{}"


def test_unknown_column_is_invalid_input(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_99"}, recording)
    assert res["status"] == "error"
    assert res["error_type"] == "INVALID_INPUT"


def test_schema_accepts_new_optional_fields():
    m = ChartDataInput(y_col="Sensor_1", show_anomalies=True, threshold=3.0)
    assert m.show_anomalies is True and m.threshold == 3.0
    assert ChartDataInput(y_col="Sensor_1").show_anomalies is False


def test_default_threshold_is_5():
    assert ChartDataInput(y_col="Sensor_1").threshold == 5.0
    schema = registry.get_tool("chart_data").input_schema
    assert schema["properties"]["threshold"]["default"] == 5.0


def test_threshold_omitted_uses_5(recording):
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "show_anomalies": True}, recording)
    assert "threshold 5.0" in res["data"]["explanation"]


def test_many_flagged_readings_still_marked_as_events():
    """The anomaly tool caps its index list at 50; the chart must not."""
    import numpy as np
    import pandas as pd

    n = 6000
    rng = np.random.default_rng(1)
    s1 = rng.normal(0, 1, n)
    s1[1000:1300] += 40  # 300 flagged readings in one event
    s1[4000:4200] -= 40  # 200 more in a second event
    df = pd.DataFrame({"Relative_Time_Sec": np.arange(n) / 100.0, "Sensor_1": s1})
    res = registry.execute("chart_data", {"y_col": "Sensor_1", "show_anomalies": True}, df)
    fig = _fig(res)
    assert len(fig["layout"]["shapes"]) == 2
    peaks = [t for t in fig["data"] if t["name"] == "Event peak"]
    assert len(peaks) == 1 and len(peaks[0]["x"]) == 2
    assert "events" in res["data"]["explanation"] or "event(s)" in res["data"]["explanation"]
    assert "500 of" in res["data"]["explanation"]


def test_no_sensor_named_plots_all_sensors(recording):
    """Dispatcher falls back to the first numeric column (the time column)."""
    res = registry.execute(
        "chart_data", {"y_col": "Relative_Time_Sec", "x_col": "Relative_Time_Sec"}, recording
    )
    assert res["status"] == "success"
    assert [t["name"] for t in _fig(res)["data"]] == ["Sensor_1", "Sensor_2"]
