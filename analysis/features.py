"""Feature extraction module for Structural Health Monitoring (Phase 2 / Week 7 - PRE-2).

Author: Eswar
Task: PRE-2 (Frequency-analysis preparation)
Authoritative Reference: Talking to Bridges Revised Plan v2 (November 2026)

Key Responsibilities:
1. Nominal Sampling Rate (NOMINAL_FS_HZ = 300.0): In accordance with Katam et al.,
   Structures 71 (2025), experimental data acquisition was conducted at 300 Hz.
   Observed deltas (0.0033 s vs 0.0030 s) are software export rounding artifacts.
   Spectral frequency calculations (compute_spectrum, extract_modal_frequencies,
   extract_features) uniformly evaluate at NOMINAL_FS_HZ = 300.0 Hz by default.
2. Dynamic Sampling Rate for Reporting: calculate_sampling_rate() measures the nominal/
   empirical interval (303.03 Hz vs 333.33 Hz) and records it as metadata (sampling_rate_hz).
   It is strictly excluded from model feature sets.
3. Whole-Recording FFT: High spectral resolution (delta_f <= 0.05 Hz) over full signal length.
4. Natural Bending Frequencies & Invariant Ratios:
   - f1_hz, f2_hz, f3_hz
   - f2_over_f1 (f2_hz / f1_hz) and f3_over_f1 (f3_hz / f1_hz)
5. Time-Domain Amplitude & Energy Metrics: Per-sensor RMS, variance, skewness, kurtosis,
   peak, crest factor, and peak-to-peak.
6. Backward Compatibility: Preserves extract_window_features, build_healthy_baseline,
   and rank_deviations.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.fft import rfft, rfftfreq

logger = logging.getLogger("ttb.analysis.features")

# Constant nominal sampling rate cited in Katam et al., Structures 71 (2025).
# Can be overridden via function parameter if required by faculty confirmation.
NOMINAL_FS_HZ: float = 300.0

# Standard sensor column names from clamp (Sensor_1) to tip (Sensor_5)
SENSOR_COLS: List[str] = ["Sensor_1", "Sensor_2", "Sensor_3", "Sensor_4", "Sensor_5"]

# Metadata and label columns that must NEVER be used as model input features
NON_FEATURE_COLS: set[str] = {
    "recording_id",
    "case_id",
    "source_file",
    "specimen",
    "test_type",
    "sampling_rate_hz",
    "condition",
    "damage_level_mm",
    "duration_sec",
    "sample_count",
}

# Search bands for the first three transverse bending natural frequencies (Hz).
# Based on Euler-Bernoulli/Timoshenko cantilever dynamics and Katam et al., Structures 71 (2025):
# Mode 1 ~ 7.0 - 7.7 Hz; Mode 2 ~ 36.8 - 41.5 Hz; Mode 3 ~ 124.0 - 138.0 Hz.
# Bounds are set using physical signal criteria to avoid Nyquist skirts (near 150 Hz)
# and fixture/transient impact spikes (near 46 Hz):
DEFAULT_MODAL_BANDS: Dict[str, Tuple[float, float]] = {
    "f1_hz": (6.5, 8.8),
    "f2_hz": (36.0, 43.5),
    "f3_hz": (120.0, 142.0),
}


# ---------------------------------------------------------------------------
# 1. Measured Sampling Rate (Metadata / Reporting Only)
# ---------------------------------------------------------------------------

def calculate_sampling_rate(
    df: pd.DataFrame,
    default_fs: float = NOMINAL_FS_HZ,
    time_col: Optional[str] = None,
) -> float:
    """Calculate empirical sampling rate fs = 1 / median(dt) from the recording's time column.

    Used for metadata auditing and reporting. Model frequency calculations default
    to NOMINAL_FS_HZ (300.0 Hz) to eliminate export-rounding bias.
    """
    candidate_cols = [time_col] if time_col else ["Relative_Time_Sec", "relative_time", "timestamp", "Time"]
    target_col = next((c for c in candidate_cols if c and c in df.columns), None)

    if target_col is not None:
        try:
            numeric_times = pd.to_numeric(df[target_col], errors="coerce").dropna()
            if len(numeric_times) > 1:
                diffs = numeric_times.diff().dropna()
                valid_diffs = diffs[diffs > 0]
                if not valid_diffs.empty:
                    median_dt = float(np.median(valid_diffs))
                    if median_dt > 0:
                        return float(round(1.0 / median_dt, 4))
        except Exception as exc:
            logger.warning("Error estimating sampling rate from column '%s': %s", target_col, exc)

    return float(default_fs)


# ---------------------------------------------------------------------------
# 2. Whole-Recording Spectral Analysis
# ---------------------------------------------------------------------------

def compute_spectrum(
    df: pd.DataFrame,
    sensor: Optional[str] = "Sensor_5",
    fs: float = NOMINAL_FS_HZ,
    detrend: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute high-resolution FFT amplitude spectrum over the whole recording.

    Args:
        df: Cleaned recording DataFrame containing sensor columns.
        sensor: Sensor column name ('Sensor_1'..'Sensor_5'), or 'combined'
                for sum of spectra across all available sensors.
        fs: Sampling rate in Hz (defaults to NOMINAL_FS_HZ = 300.0).
        detrend: Whether to subtract the DC mean component before FFT.

    Returns:
        (frequencies, amplitudes): Tuple of 1D numpy arrays.
    """
    available_sensors = [c for c in SENSOR_COLS if c in df.columns]
    if not available_sensors:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        available_sensors = [c for c in numeric_cols if c not in {"Relative_Time_Sec", "timestamp", "DateTime"}]

    if not available_sensors:
        return np.array([0.0]), np.array([0.0])

    if sensor == "combined" or sensor is None:
        target_sensors = available_sensors
    elif sensor in available_sensors:
        target_sensors = [sensor]
    else:
        target_sensors = [available_sensors[-1]]

    combined_amp: Optional[np.ndarray] = None
    freqs: Optional[np.ndarray] = None

    for col in target_sensors:
        arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(arr) < 2:
            continue
        if detrend:
            arr = arr - np.mean(arr)

        yf = np.abs(rfft(arr))
        xf = rfftfreq(len(arr), 1.0 / fs)

        if combined_amp is None:
            combined_amp = yf
            freqs = xf
        else:
            if len(yf) == len(combined_amp):
                combined_amp += yf

    if combined_amp is None or freqs is None:
        return np.array([0.0]), np.array([0.0])

    return freqs, combined_amp


# ---------------------------------------------------------------------------
# 3. Peak-Picking for Bending Natural Frequencies (f1, f2, f3)
# ---------------------------------------------------------------------------

def extract_modal_frequencies(
    df: pd.DataFrame,
    sensor: Optional[str] = "combined",
    fs: float = NOMINAL_FS_HZ,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, float]:
    """Identify the first three natural bending frequencies from FFT peaks.

    Args:
        df: Recording DataFrame.
        sensor: Sensor column or 'combined' (default 'combined').
        fs: Sampling rate (Hz), defaults to NOMINAL_FS_HZ = 300.0.
        bands: Dictionary of search bands {mode_name: (min_hz, max_hz)}.

    Returns:
        Dict with keys 'f1_hz', 'f2_hz', 'f3_hz'.
    """
    search_bands = bands or DEFAULT_MODAL_BANDS
    freqs, amps = compute_spectrum(df, sensor=sensor, fs=fs, detrend=True)

    modal_peaks: Dict[str, float] = {k: 0.0 for k in search_bands}

    if len(freqs) < 2 or len(amps) < 2:
        return modal_peaks

    for mode_name, (f_min, f_max) in search_bands.items():
        band_mask = (freqs >= f_min) & (freqs <= f_max)
        if np.any(band_mask):
            sub_freqs = freqs[band_mask]
            sub_amps = amps[band_mask]
            best_idx = int(np.argmax(sub_amps))
            modal_peaks[mode_name] = float(round(sub_freqs[best_idx], 4))
        else:
            modal_peaks[mode_name] = 0.0

    return modal_peaks


# ---------------------------------------------------------------------------
# 4. Interface A: Unified Feature Row Extraction
# ---------------------------------------------------------------------------

def extract_features(
    df: pd.DataFrame,
    sensor_for_spectrum: str = "combined",
    fs: float = NOMINAL_FS_HZ,
) -> Dict[str, Any]:
    """Extract complete Interface A feature dictionary for a single recording.

    Excludes labels and metadata from model features while including:
    - First three bending frequencies f1_hz, f2_hz, f3_hz evaluated at nominal fs
    - Rate-independent modal ratios: f2_over_f1 (f2/f1) and f3_over_f1 (f3/f1)
    - Dynamic sampling_rate_hz (metadata only, strictly NOT a model feature)
    - Per-sensor statistical features: RMS, variance, skewness, kurtosis, peak, crest_factor, p2p

    Args:
        df: Cleaned recording DataFrame.
        sensor_for_spectrum: Channel used for modal identification (default 'combined').
        fs: FFT sampling rate (default NOMINAL_FS_HZ = 300.0).

    Returns:
        Dictionary containing all features and metadata for Interface A.
    """
    measured_fs = calculate_sampling_rate(df, default_fs=fs)
    modal_freqs = extract_modal_frequencies(df, sensor=sensor_for_spectrum, fs=fs)

    f1 = float(modal_freqs.get("f1_hz", 0.0))
    f2 = float(modal_freqs.get("f2_hz", 0.0))
    f3 = float(modal_freqs.get("f3_hz", 0.0))

    # Rate-independent modal ratios with division-by-zero protection
    f2_over_f1 = round(f2 / f1, 6) if f1 > 1e-6 else 0.0
    f3_over_f1 = round(f3 / f1, 6) if f1 > 1e-6 else 0.0

    feature_dict: Dict[str, Any] = {
        # Metadata only (NEVER used as model input)
        "sampling_rate_hz": measured_fs,

        # Modal features
        "f1_hz": f1,
        "f2_hz": f2,
        "f3_hz": f3,
        "f2_over_f1": f2_over_f1,
        "f3_over_f1": f3_over_f1,
    }

    # Extract time-domain statistics for all available physical sensors
    for col in SENSOR_COLS:
        if col in df.columns:
            arr = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(arr) > 0:
                abs_arr = np.abs(arr)
                peak_val = float(np.max(abs_arr))
                rms_val = float(np.sqrt(np.mean(arr**2)))
                var_val = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
                skew_val = float(stats.skew(arr)) if len(arr) > 2 else 0.0
                kurt_val = float(stats.kurtosis(arr)) if len(arr) > 2 else 0.0
                crest_val = float(peak_val / rms_val) if rms_val > 1e-9 else 0.0
                p2p_val = float(np.ptp(arr))

                feature_dict[f"{col}_rms"] = round(rms_val, 6)
                feature_dict[f"{col}_variance"] = round(var_val, 6)
                feature_dict[f"{col}_skewness"] = round(skew_val, 6)
                feature_dict[f"{col}_kurtosis"] = round(kurt_val, 6)
                feature_dict[f"{col}_peak"] = round(peak_val, 6)
                feature_dict[f"{col}_crest_factor"] = round(crest_val, 6)
                feature_dict[f"{col}_p2p"] = round(p2p_val, 6)
            else:
                feature_dict[f"{col}_rms"] = 0.0
                feature_dict[f"{col}_variance"] = 0.0
                feature_dict[f"{col}_skewness"] = 0.0
                feature_dict[f"{col}_kurtosis"] = 0.0
                feature_dict[f"{col}_peak"] = 0.0
                feature_dict[f"{col}_crest_factor"] = 0.0
                feature_dict[f"{col}_p2p"] = 0.0

    return feature_dict


# ---------------------------------------------------------------------------
# 5. Sliding-Window Extraction (Preserved for Backward Compatibility)
# ---------------------------------------------------------------------------

def extract_window_features(
    df: pd.DataFrame,
    window_size: int = 300,
    fs: float = NOMINAL_FS_HZ,
) -> pd.DataFrame:
    """Extract time and frequency domain features from sliding windows of sensor data.

    Preserved for backward compatibility with frontend showcase and existing tests.
    Defaults to NOMINAL_FS_HZ = 300.0.
    """
    cols = [c for c in SENSOR_COLS if c in df.columns]
    if not cols:
        numeric_cols = list(df.select_dtypes(include=["number"]).columns)
        cols = [c for c in numeric_cols if c not in {"Relative_Time_Sec", "timestamp", "DateTime"}]

    if not cols:
        return pd.DataFrame()

    df_clean = df[cols].dropna().reset_index(drop=True)
    n_samples = len(df_clean)
    if n_samples < window_size:
        return pd.DataFrame()

    features_list: List[Dict[str, float]] = []

    for start in range(0, n_samples - window_size + 1, window_size):
        window = df_clean.iloc[start : start + window_size]
        window_feats: Dict[str, float] = {}

        for col in cols:
            arr = window[col].to_numpy(dtype=float)
            peak = float(np.max(np.abs(arr)))
            rms = float(np.sqrt(np.mean(arr**2)))
            variance = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
            skewness = float(stats.skew(arr)) if len(arr) > 2 else 0.0
            kurt = float(stats.kurtosis(arr)) if len(arr) > 2 else 0.0
            crest_factor = peak / rms if rms > 1e-9 else 0.0

            yf = np.abs(rfft(arr))
            xf = rfftfreq(len(arr), 1.0 / fs)

            if len(yf) > 1:
                peak_idx = int(np.argmax(yf[1:])) + 1
                fft_peak_mag = float(yf[peak_idx])
                fft_peak_freq = float(xf[peak_idx])
            else:
                fft_peak_mag = 0.0
                fft_peak_freq = 0.0

            window_feats[f"{col}_rms"] = rms
            window_feats[f"{col}_variance"] = variance
            window_feats[f"{col}_skewness"] = skewness
            window_feats[f"{col}_kurtosis"] = kurt
            window_feats[f"{col}_peak"] = peak
            window_feats[f"{col}_crest_factor"] = crest_factor
            window_feats[f"{col}_fft_peak_mag"] = fft_peak_mag
            window_feats[f"{col}_fft_peak_freq"] = fft_peak_freq

        features_list.append(window_feats)

    return pd.DataFrame(features_list)


# ---------------------------------------------------------------------------
# 6. Baseline Profile & Deviation Ranking (Interface B Foundation)
# ---------------------------------------------------------------------------

def build_healthy_baseline(
    undamaged_dfs: Sequence[pd.DataFrame],
    window_size: int = 300,
    use_recording_features: bool = False,
    fs: float = NOMINAL_FS_HZ,
) -> Dict[str, Dict[str, float]]:
    """Build healthy baseline profile (mean and std) from undamaged recordings.

    Args:
        undamaged_dfs: Collection of DataFrames from undamaged tests.
        window_size: Used if use_recording_features is False.
        use_recording_features: If True, extracts one whole-recording Interface A
                                feature row per DataFrame.
        fs: Nominal sampling frequency for FFT (default NOMINAL_FS_HZ = 300.0).

    Returns:
        Dict mapping feature name to {'mean': float, 'std': float}.
    """
    all_features: List[pd.DataFrame] = []

    for df in undamaged_dfs:
        if use_recording_features:
            feat_row = extract_features(df, fs=fs)
            # Filter out non-numeric and disallowed metadata/label keys
            clean_feats = {
                k: v for k, v in feat_row.items()
                if isinstance(v, (int, float)) and k not in NON_FEATURE_COLS
            }
            all_features.append(pd.DataFrame([clean_feats]))
        else:
            feats = extract_window_features(df, window_size=window_size, fs=fs)
            if not feats.empty:
                all_features.append(feats)

    if not all_features:
        raise ValueError("No valid undamaged data provided to build baseline.")

    combined = pd.concat(all_features, ignore_index=True)

    baseline: Dict[str, Dict[str, float]] = {}
    for col in combined.columns:
        series = pd.to_numeric(combined[col], errors="coerce").dropna()
        if not series.empty:
            mean_v = float(series.mean())
            std_v = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            baseline[col] = {
                "mean": round(mean_v, 6),
                "std": round(std_v, 6),
            }

    return baseline


def rank_deviations(
    test_df: pd.DataFrame,
    baseline: Dict[str, Dict[str, float]],
    window_size: int = 300,
    use_recording_features: bool = False,
    fs: float = NOMINAL_FS_HZ,
) -> pd.DataFrame:
    """Compare a test recording against the healthy baseline using z-scores.

    Args:
        test_df: The recording to evaluate.
        baseline: Precomputed healthy baseline dictionary.
        window_size: Sliding window size if use_recording_features is False.
        use_recording_features: If True, compares whole-recording feature row.
        fs: Nominal sampling rate for FFT (default NOMINAL_FS_HZ = 300.0).

    Returns:
        DataFrame sorted by absolute deviation (highest z-score first).
    """
    if use_recording_features:
        raw_feats = extract_features(test_df, fs=fs)
        test_mean_feats = {
            k: v for k, v in raw_feats.items()
            if isinstance(v, (int, float)) and k not in NON_FEATURE_COLS
        }
    else:
        test_feats = extract_window_features(test_df, window_size=window_size, fs=fs)
        if test_feats.empty:
            raw_feats = extract_features(test_df, fs=fs)
            test_mean_feats = {
                k: v for k, v in raw_feats.items()
                if isinstance(v, (int, float)) and k not in NON_FEATURE_COLS
            }
        else:
            test_mean_feats = test_feats.mean().to_dict()

    deviations: List[Dict[str, Any]] = []
    for feat_name, base_stats in baseline.items():
        if feat_name in NON_FEATURE_COLS:
            continue
        if feat_name in test_mean_feats:
            test_val = float(test_mean_feats[feat_name])
            base_mean = float(base_stats["mean"])
            base_std = float(base_stats["std"])

            z_score = (test_val - base_mean) / base_std if base_std > 1e-9 else 0.0

            sensor_id = "Global"
            for s in SENSOR_COLS:
                if feat_name.startswith(f"{s}_"):
                    sensor_id = s
                    break

            metric_name = feat_name.replace(f"{sensor_id}_", "")
            change_pct = ((test_val - base_mean) / base_mean * 100.0) if abs(base_mean) > 1e-9 else 0.0

            deviations.append({
                "sensor": sensor_id,
                "feature": metric_name if sensor_id != "Global" else feat_name,
                "healthy_mean": round(base_mean, 4),
                "test_mean": round(test_val, 4),
                "change_percent": round(change_pct, 2),
                "z_score": round(z_score, 4),
                "abs_deviation": round(abs(z_score), 4),
            })

    dev_df = pd.DataFrame(deviations)
    if not dev_df.empty:
        dev_df = dev_df.sort_values(by="abs_deviation", ascending=False).reset_index(drop=True)

    return dev_df
