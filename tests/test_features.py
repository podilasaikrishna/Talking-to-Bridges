import numpy as np
import pandas as pd
import pytest

from analysis.features import (
    calculate_sampling_rate,
    compute_spectrum,
    extract_modal_frequencies,
    extract_features,
    extract_window_features,
    build_healthy_baseline,
    rank_deviations,
    NOMINAL_FS_HZ,
    NON_FEATURE_COLS,
    SENSOR_COLS,
)


def test_calculate_sampling_rate():
    # 1. 303 Hz case (dt = 0.0033)
    t_303 = np.arange(0, 100) * 0.0033
    df_303 = pd.DataFrame({"Relative_Time_Sec": t_303})
    fs_303 = calculate_sampling_rate(df_303)
    assert pytest.approx(fs_303, abs=0.5) == 303.03

    # 2. 333 Hz case (dt = 0.0030)
    t_333 = np.arange(0, 100) * 0.0030
    df_333 = pd.DataFrame({"Relative_Time_Sec": t_333})
    fs_333 = calculate_sampling_rate(df_333)
    assert pytest.approx(fs_333, abs=0.5) == 333.33

    # 3. Fallback when time column is missing
    df_empty = pd.DataFrame({"Sensor_1": [1.0, 2.0, 3.0]})
    assert calculate_sampling_rate(df_empty, default_fs=250.0) == 250.0


def test_compute_spectrum():
    duration = 5.0  # 5 seconds at nominal 300 Hz
    n = int(duration * NOMINAL_FS_HZ)
    t = np.arange(0, n) / NOMINAL_FS_HZ

    # Pure sine wave at 8.0 Hz on Sensor_5
    sig5 = 5.0 * np.sin(2 * np.pi * 8.0 * t)
    df = pd.DataFrame({
        "Relative_Time_Sec": t,
        "Sensor_5": sig5,
    })

    freqs, amps = compute_spectrum(df, sensor="Sensor_5")
    assert len(freqs) == len(amps)
    assert len(freqs) > 100

    # Max peak should be exactly at 8.0 Hz
    peak_freq = freqs[np.argmax(amps)]
    assert pytest.approx(peak_freq, abs=0.25) == 8.0


def test_extract_modal_frequencies_synthetic():
    duration = 10.0
    n = int(duration * NOMINAL_FS_HZ)
    t = np.arange(0, n) / NOMINAL_FS_HZ

    # Signal with known 3 bending frequencies: 7.6 Hz, 38.0 Hz, 135.0 Hz
    sig = (
        3.0 * np.sin(2 * np.pi * 7.6 * t)
        + 1.5 * np.sin(2 * np.pi * 38.0 * t)
        + 0.8 * np.sin(2 * np.pi * 135.0 * t)
    )

    df = pd.DataFrame({
        "Relative_Time_Sec": t,
        "Sensor_1": sig,
        "Sensor_5": sig,
    })

    modals = extract_modal_frequencies(df)
    assert "f1_hz" in modals
    assert "f2_hz" in modals
    assert "f3_hz" in modals

    assert pytest.approx(modals["f1_hz"], abs=0.15) == 7.6
    assert pytest.approx(modals["f2_hz"], abs=0.2) == 38.0
    assert pytest.approx(modals["f3_hz"], abs=0.2) == 135.0


def test_modal_bands_reject_nyquist_and_fixture_noise():
    """Verify that refined modal search bands reject Nyquist edge spikes (149.5 Hz)
    and fixture impact bounce spikes (45.9 Hz) even when spurious noise has high amplitude.
    """
    duration = 10.0
    n = int(duration * NOMINAL_FS_HZ)
    t = np.arange(0, n) / NOMINAL_FS_HZ

    # True bending modes + spurious fixture / Nyquist artifacts
    sig = (
        2.0 * np.sin(2 * np.pi * 7.60 * t)
        + 1.2 * np.sin(2 * np.pi * 38.50 * t)
        + 0.8 * np.sin(2 * np.pi * 126.50 * t)
        + 2.5 * np.sin(2 * np.pi * 45.90 * t)   # Fixture impact noise at ~46 Hz
        + 3.0 * np.sin(2 * np.pi * 149.50 * t)  # Nyquist edge spike at ~150 Hz
    )

    df = pd.DataFrame({
        "Relative_Time_Sec": t,
        "Sensor_1": sig,
        "Sensor_5": sig,
    })

    modals = extract_modal_frequencies(df)
    # The picker must pick the true bending modes, rejecting 45.9 Hz and 149.5 Hz
    assert pytest.approx(modals["f1_hz"], abs=0.15) == 7.60
    assert pytest.approx(modals["f2_hz"], abs=0.20) == 38.50
    assert pytest.approx(modals["f3_hz"], abs=0.30) == 126.50



def test_extract_features_interface_a():
    n = 1500
    t = np.arange(n) * 0.0033  # simulates damaged file time axis
    sig = np.sin(2 * np.pi * 7.6 * (np.arange(n) / NOMINAL_FS_HZ))
    data = {"Relative_Time_Sec": t}
    for s in SENSOR_COLS:
        data[s] = sig + np.random.normal(0, 0.02, n)

    df = pd.DataFrame(data)
    features = extract_features(df)

    # Core modal frequencies & invariant ratios present
    assert "f1_hz" in features
    assert "f2_hz" in features
    assert "f3_hz" in features
    assert "f2_over_f1" in features
    assert "f3_over_f1" in features
    assert "sampling_rate_hz" in features

    # Modal peak near 7.6 Hz
    assert pytest.approx(features["f1_hz"], abs=0.2) == 7.6

    # All sensors have statistical features
    for s in SENSOR_COLS:
        assert f"{s}_rms" in features
        assert f"{s}_kurtosis" in features
        assert f"{s}_skewness" in features
        assert f"{s}_variance" in features
        assert f"{s}_peak" in features
        assert f"{s}_crest_factor" in features
        assert f"{s}_p2p" in features


def test_sampling_rate_rounding_invariance():
    """NEW: Verify that the same true physical signal sampled with dt=0.003 (undamaged export)
    and dt=0.0033 (damaged export) yields identical modal frequencies (within 0.05 Hz)
    and identical modal ratios, eliminating the 10% export rounding bias.
    """
    n_samples = 3000
    # Physical signal sampled at true 300 Hz
    t_true = np.arange(n_samples) / NOMINAL_FS_HZ
    sig = (
        2.5 * np.sin(2 * np.pi * 7.61 * t_true)
        + 1.2 * np.sin(2 * np.pi * 37.94 * t_true)
        + 0.6 * np.sin(2 * np.pi * 125.1 * t_true)
    )

    # File A: written with dt=0.0030 s (333.33 Hz nominal export)
    df_333 = pd.DataFrame({
        "Relative_Time_Sec": np.arange(n_samples) * 0.0030,
        "Sensor_5": sig,
    })

    # File B: written with dt=0.0033 s (303.03 Hz nominal export)
    df_303 = pd.DataFrame({
        "Relative_Time_Sec": np.arange(n_samples) * 0.0033,
        "Sensor_5": sig,
    })

    feats_333 = extract_features(df_333)
    feats_303 = extract_features(df_303)

    # Both must identify the exact same physical frequencies within 0.05 Hz
    assert abs(feats_333["f1_hz"] - feats_303["f1_hz"]) <= 0.05
    assert abs(feats_333["f2_hz"] - feats_303["f2_hz"]) <= 0.05
    assert abs(feats_333["f3_hz"] - feats_303["f3_hz"]) <= 0.05

    # Ratios f2/f1 and f3/f1 must be identical
    assert pytest.approx(feats_333["f2_over_f1"], abs=0.02) == feats_303["f2_over_f1"]
    assert pytest.approx(feats_333["f3_over_f1"], abs=0.02) == feats_303["f3_over_f1"]


def test_ratio_features_and_no_metadata_in_features():
    """NEW: Verify that ratio columns exist and no metadata or label column leaks into features."""
    df = pd.DataFrame({
        "Relative_Time_Sec": np.arange(600) * 0.0033,
        "Sensor_1": np.sin(np.linspace(0, 20, 600)),
        "Sensor_5": np.cos(np.linspace(0, 20, 600)),
    })

    feats = extract_features(df)
    assert "f2_over_f1" in feats
    assert "f3_over_f1" in feats

    # Disallowed metadata/label columns must not be in model features
    baseline = build_healthy_baseline([df], use_recording_features=True)
    for col in NON_FEATURE_COLS:
        assert col not in baseline, f"Metadata/label '{col}' leaked into healthy baseline features!"


def test_backward_compatibility_extract_window_features():
    df = pd.DataFrame({
        "Relative_Time_Sec": np.linspace(0, 1, 300),
        "Sensor_1": np.sin(np.linspace(0, 10, 300)),
        "Sensor_2": np.cos(np.linspace(0, 10, 300)),
    })

    feats = extract_window_features(df, window_size=150)
    assert len(feats) == 2

    for sensor in ["Sensor_1", "Sensor_2"]:
        for metric in ["rms", "variance", "skewness", "kurtosis", "peak", "crest_factor", "fft_peak_mag", "fft_peak_freq"]:
            assert f"{sensor}_{metric}" in feats.columns


def test_build_healthy_baseline_and_rank_deviations():
    t = np.linspace(0, 2, 600)
    df1 = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600)})
    df2 = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600)})

    baseline = build_healthy_baseline([df1, df2], use_recording_features=True)
    assert "Sensor_1_rms" in baseline
    assert "mean" in baseline["Sensor_1_rms"]
    assert "std" in baseline["Sensor_1_rms"]

    test_df = pd.DataFrame({"Relative_Time_Sec": t, "Sensor_1": np.random.randn(600) * 10})
    rankings = rank_deviations(test_df, baseline, use_recording_features=True)

    assert not rankings.empty
    assert "sensor" in rankings.columns
    assert "feature" in rankings.columns
    assert "z_score" in rankings.columns
    assert "abs_deviation" in rankings.columns
