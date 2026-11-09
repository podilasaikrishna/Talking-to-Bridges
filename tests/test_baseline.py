"""NEW-2 / NEW-3 tests: healthy baseline comparison (Interface B) and the new charts."""
import json

import numpy as np
import pandas as pd
import pytest

from analysis import baseline as bl
from analysis.charts import plot_frequency_bars, plot_spectrum_overlay
from analysis.tools import registry

FS = 300.0


def make_recording(f1=7.6, f2=37.9, f3=126.5, seed=0, n=9000):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / FS
    data = {"Relative_Time_Sec": t}
    for i in range(1, 6):
        sig = (
            3.0 * np.sin(2 * np.pi * f1 * t)
            + 1.5 * np.sin(2 * np.pi * f2 * t + i)
            + 0.8 * np.sin(2 * np.pi * f3 * t)
            + rng.normal(0, 0.05, n)
        )
        data[f"Sensor_{i}"] = 1.5 + 0.01 * sig
    return pd.DataFrame(data)


@pytest.fixture()
def ref_dir(tmp_path):
    """Reference folder with four slightly different healthy recordings (CSV only)."""
    for k in range(4):
        make_recording(f1=7.6 + 0.03 * k, f2=37.9 + 0.1 * k, f3=126.5 + 0.3 * k, seed=k).to_csv(
            tmp_path / f"healthy_{k}.csv", index=False
        )
    return tmp_path


def test_profile_built_from_recordings(ref_dir):
    profile = bl.load_profile(ref_dir)
    assert profile["n_healthy_recordings"] == 4
    assert {"f1_hz", "f2_hz", "f3_hz", "f2_over_f1", "f3_over_f1"} <= set(profile["features"])
    assert profile["features"]["f2_hz"]["spread_mad"] > 0


def test_profile_prefers_json_file(ref_dir):
    (ref_dir / "healthy_baseline.json").write_text(json.dumps({
        "n_healthy_recordings": 40,
        "healthy_cases": ["A", "B"],
        "features": {"f1_hz": {"typical_median": 7.6, "spread_mad": 0.05}},
    }))
    profile = bl.load_profile(ref_dir)
    assert profile["n_healthy_recordings"] == 40
    assert bl._n_healthy(profile) == {"recordings": 40, "cases": 2, "case_names": ["A", "B"]}


def test_healthy_like_recording_is_normal(ref_dir):
    res = bl.compare_to_baseline(make_recording(f1=7.64, f2=37.95, f3=126.7, seed=9), ref_dir)
    assert res["overall_flag"] == "normal"
    assert res["n_healthy"]["recordings"] == 4


def test_shifted_mode_2_is_abnormal_with_percent(ref_dir):
    res = bl.compare_to_baseline(make_recording(f2=40.4, seed=11), ref_dir)
    assert res["overall_flag"] == "abnormal"
    top = res["deviations"][0]
    assert top["feature"] in ("f2_hz", "f2_over_f1")
    f2 = next(d for d in res["deviations"] if d["feature"] == "f2_hz")
    assert f2["direction"] == "higher" and f2["outside"]
    assert 4.0 < f2["change_percent"] < 9.0
    assert "mode 2 frequency" in res["summary"]
    assert res["caveat"]


def test_interface_b_fields_and_json(ref_dir):
    res = bl.compare_to_baseline(make_recording(seed=3), ref_dir)
    for key in ("overall_flag", "deviations", "healthy_reference", "n_healthy"):
        assert key in res
    for d in res["deviations"]:
        for k in ("feature", "value", "healthy_typical", "healthy_spread", "change_percent", "score"):
            assert k in d
    assert set(res["healthy_reference"]["frequencies_hz"]) >= {"f1_hz", "f2_hz", "f3_hz"}
    assert res["healthy_reference"]["spectrum"]["n_recordings"] == 4
    json.dumps(res)


def test_frequency_features_ranked_before_amplitude(ref_dir):
    res = bl.compare_to_baseline(make_recording(seed=4), ref_dir)
    groups = [d["group"] for d in res["deviations"]]
    assert groups == sorted(groups, key=lambda g: g != "frequency")


def test_feature_row_input(ref_dir):
    row = bl.extract_features(make_recording(f2=40.4, seed=5))
    assert bl.compare_to_baseline(row, ref_dir)["overall_flag"] == "abnormal"


def test_unavailable_without_baseline(tmp_path):
    res = bl.compare_to_baseline(make_recording(), tmp_path)
    assert res["overall_flag"] == "unavailable" and res["deviations"] == []


def test_unavailable_without_sensor_columns(ref_dir):
    res = bl.compare_to_baseline(pd.DataFrame({"a": [1, 2, 3]}), ref_dir)
    assert res["overall_flag"] == "unavailable"


# ---- tool + charts ---------------------------------------------------------

def test_baseline_tool_returns_interface_b_and_old_fields(ref_dir, monkeypatch):
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(ref_dir))
    res = registry.execute("baseline", {}, df=make_recording(f2=40.4, seed=6))
    assert res["status"] == "success"
    data = res["data"]
    assert data["overall_flag"] == "abnormal" and data["status"] == "Abnormal"
    assert data["deviation_ranking"] == data["deviations"]
    assert "spectrum" not in data["healthy_reference"]
    json.dumps(data)


def test_baseline_tool_without_data():
    data = registry.execute("baseline", {})["data"]
    assert data["overall_flag"] == "unavailable"


@pytest.mark.parametrize("mode", ["spectrum", "frequencies"])
def test_chart_modes(ref_dir, monkeypatch, mode):
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(ref_dir))
    df = make_recording(f2=40.4, seed=7)
    res = registry.execute(
        "chart_data", {"y_col": "Sensor_5", "x_col": "Relative_Time_Sec", "comparison": mode}, df=df
    )
    assert res["status"] == "success", res
    data = res["data"]
    assert data["plot_json"] not in ("", "{}")
    assert "percent" in data["explanation"]
    assert json.loads(data["plot_json"])["data"]


def test_chart_modes_without_reference(tmp_path, monkeypatch):
    monkeypatch.setenv("HEALTHY_REFERENCE_DIR", str(tmp_path))
    res = registry.execute(
        "chart_data",
        {"y_col": "Sensor_5", "x_col": "Relative_Time_Sec", "comparison": "spectrum"},
        df=make_recording(),
    )
    assert res["data"]["plot_json"] == "{}"
    assert "healthy" in res["data"]["explanation"].lower()


def test_chart_builders_validate_input():
    with pytest.raises(ValueError):
        plot_spectrum_overlay({}, {}, {}, {})
    with pytest.raises(ValueError):
        plot_frequency_bars([])


@pytest.mark.parametrize(
    "query,expected",
    [
        ("plot the frequency spectrum compared to healthy", "spectrum"),
        ("chart f1 f2 f3 against healthy", "frequencies"),
        ("plot the first three frequencies versus the healthy beam", "frequencies"),
        ("graph the kurtosis deviation compared to baseline", "features"),
    ],
)
def test_dispatcher_picks_comparison_mode(query, expected):
    from analysis.dispatcher import _detect_comparison, _detect_tool, _tokenize

    assert _detect_tool(query) == "chart_data"
    assert _detect_comparison(_tokenize(query)) == expected


def test_spectrum_question_goes_to_chart_not_baseline_text():
    from analysis.dispatcher import _detect_tool

    assert _detect_tool("show spectrum vs healthy baseline") == "chart_data"
    assert _detect_tool("is this beam different from the healthy baseline?") == "baseline"
