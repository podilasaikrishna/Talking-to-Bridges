"""Unit tests for analysis/time_audit.py (Task PRE-1).

Validates:
- Uniform time intervals
- Damaged and undamaged sampling-rate calculations (~303 Hz vs ~333 Hz)
- Duplicate timestamps (zero dt)
- Out-of-order timestamps (negative dt)
- Missing / invalid time values
- Irregular time intervals (> 5% deviation)
- Large recording gaps (> 1.5 * median dt)
- Missing, invalid, or flat sensor columns
- Wide summary tables detection and skipping
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis.time_audit import (
    audit_recording_file,
    audit_relative_time,
    run_dataset_time_audit,
)


def test_uniform_intervals_damaged_example():
    """Verify uniform interval detection and ~303.03 Hz rate calculation."""
    n = 100
    times = np.round(np.arange(n) * 0.0033, 4)
    audit = audit_relative_time(times)

    assert audit.sample_count == n
    assert audit.is_strictly_increasing is True
    assert audit.is_uniformly_sampled is True
    assert audit.zero_dt_count == 0
    assert audit.negative_dt_count == 0
    assert audit.large_gap_count == 0
    assert audit.irregular_interval_count == 0
    assert audit.median_dt == pytest.approx(0.0033, abs=1e-6)
    assert audit.mean_dt == pytest.approx(0.0033, abs=1e-6)
    assert audit.sampling_rate_hz == pytest.approx(303.0303, abs=0.01)
    assert audit.cv_percent == pytest.approx(0.0, abs=1e-3)


def test_uniform_intervals_undamaged_example():
    """Verify uniform interval detection and ~333.33 Hz rate calculation."""
    n = 100
    times = np.round(np.arange(n) * 0.003, 4)
    audit = audit_relative_time(times)

    assert audit.sample_count == n
    assert audit.is_strictly_increasing is True
    assert audit.is_uniformly_sampled is True
    assert audit.zero_dt_count == 0
    assert audit.negative_dt_count == 0
    assert audit.large_gap_count == 0
    assert audit.irregular_interval_count == 0
    assert audit.median_dt == pytest.approx(0.003, abs=1e-6)
    assert audit.mean_dt == pytest.approx(0.003, abs=1e-6)
    assert audit.sampling_rate_hz == pytest.approx(333.3333, abs=0.01)


def test_theoretical_300hz_intervals():
    """Verify 300 Hz exact intervals (dt = 1/300)."""
    n = 300
    times = np.arange(n) / 300.0
    audit = audit_relative_time(times)

    assert audit.sample_count == n
    assert audit.is_strictly_increasing is True
    assert audit.is_uniformly_sampled is True
    assert audit.median_dt == pytest.approx(1.0 / 300.0, rel=1e-6)
    assert audit.sampling_rate_hz == pytest.approx(300.0, abs=0.01)


def test_duplicate_timestamps_flagged():
    """Verify duplicate timestamps (dt == 0) are identified."""
    times = [0.0, 0.0033, 0.0033, 0.0066, 0.0099]
    audit = audit_relative_time(times)

    assert audit.zero_dt_count == 1
    assert audit.is_strictly_increasing is False
    assert audit.is_uniformly_sampled is False


def test_out_of_order_timestamps_flagged():
    """Verify out-of-order timestamps (dt < 0) are identified."""
    times = [0.0, 0.0066, 0.0033, 0.0099, 0.0132]
    audit = audit_relative_time(times)

    assert audit.negative_dt_count == 1
    assert audit.is_strictly_increasing is False
    assert audit.is_uniformly_sampled is False
    assert audit.min_dt < 0


def test_irregular_time_intervals_flagged():
    """Verify intervals deviating by > 5% from median are flagged."""
    # 0.0033, then jump of 0.0040 (deviation > 20%), then 0.0033
    times = [0.0, 0.0033, 0.0073, 0.0106, 0.0139]
    audit = audit_relative_time(times, tolerance_pct=0.05)

    assert audit.irregular_interval_count >= 1
    assert audit.is_uniformly_sampled is False
    assert audit.is_strictly_increasing is True


def test_large_recording_gaps_flagged():
    """Verify gaps exceeding 1.5 * median dt are detected."""
    # Median is 0.0033, gap of 0.0100 is ~3x median
    times = [0.0, 0.0033, 0.0066, 0.0166, 0.0199]
    audit = audit_relative_time(times, gap_factor=1.5)

    assert audit.large_gap_count == 1
    assert audit.is_uniformly_sampled is False
    assert audit.max_dt == pytest.approx(0.0100, abs=1e-5)


def test_missing_time_values_handled():
    """Verify NaN values in time array are handled gracefully."""
    times = np.array([0.0, 0.0033, np.nan, 0.0066, 0.0099])
    audit = audit_relative_time(times)

    # NaNs filtered out from dt diff calculation
    assert audit.sample_count == 4
    assert audit.median_dt == pytest.approx(0.0033, abs=1e-5)


def test_insufficient_samples():
    """Verify arrays with < 2 samples return safe null metrics."""
    audit_empty = audit_relative_time([])
    assert audit_empty.sample_count == 0
    assert np.isnan(audit_empty.median_dt)

    audit_single = audit_relative_time([0.0])
    assert audit_single.sample_count == 1
    assert np.isnan(audit_single.median_dt)


def test_audit_recording_file_damaged_layout(tmp_path: Path):
    """Test full workbook audit on simulated 7-column damaged layout."""
    n = 20
    rel = np.round(np.arange(n) * 0.0033, 4)
    data = {
        0: [f"12:00:00.{i:03d}000" for i in range(n)],
        1: rel,
        2: np.random.randn(n),
        3: np.random.randn(n),
        4: np.random.randn(n),
        5: np.random.randn(n),
        6: np.random.randn(n),
    }
    df = pd.DataFrame(data)
    test_file = tmp_path / "Damaged_Beams" / "1mm" / "M1" / "Singlehit" / "D-SH 1-2.xlsx"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(test_file, header=False, index=False)

    rec = audit_recording_file(test_file)
    assert rec.condition == "Damaged"
    assert rec.layout == "Damaged"
    assert rec.damage_level == "1mm"
    assert rec.specimen == "M1"
    assert rec.test_type == "Singlehit"
    assert rec.validation_status == "valid"
    assert rec.sampling_rate_hz == pytest.approx(303.0303, abs=0.01)
    assert rec.sample_count == n


def test_audit_recording_file_undamaged_layout(tmp_path: Path):
    """Test full workbook audit on simulated 9-column undamaged layout."""
    n = 20
    rel = np.round(np.arange(n) * 0.003, 4)
    data = {
        0: [10] * n,
        1: [30] * n,
        2: [15.0 + r for r in rel],
        3: rel,
        4: np.random.randn(n),
        5: np.random.randn(n),
        6: np.random.randn(n),
        7: np.random.randn(n),
        8: np.random.randn(n),
    }
    df = pd.DataFrame(data)
    test_file = tmp_path / "Undamaged_Data_of_2_cantilever" / "1st" / "Singlehit" / "UD-SH 1-2.xlsx"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(test_file, header=False, index=False)

    rec = audit_recording_file(test_file)
    assert rec.condition == "Undamaged"
    assert rec.layout == "Undamaged"
    assert rec.specimen == "1st"
    assert rec.test_type == "Singlehit"
    assert rec.validation_status == "valid"
    assert rec.sampling_rate_hz == pytest.approx(333.3333, abs=0.01)
    assert rec.sample_count == n


def test_audit_recording_file_wide_summary_skipped(tmp_path: Path):
    """Test wide summary tables (> 20 columns) are marked as skipped."""
    df_wide = pd.DataFrame(np.zeros((5, 3001)))
    test_file = tmp_path / "Damaged_Beams" / "1mm.xlsx"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_excel(test_file, header=False, index=False)

    rec = audit_recording_file(test_file)
    assert rec.layout == "Wide_Summary"
    assert rec.validation_status == "skipped"
    assert "Wide summary table" in rec.notes


def test_audit_recording_file_with_non_numeric_header(tmp_path: Path):
    """Test file with non-numeric header rows (e.g. '->' artifact)."""
    n = 20
    # First 3 rows non-numeric in time column
    times = ["->", "->", "->"] + [f"{0.0033 * i:.4f}" for i in range(n - 3)]
    data = {
        0: [f"12:00:00.{i:03d}000" for i in range(n)],
        1: times,
        2: [1.5] * n,
        3: [2.0 + i for i in range(n)],
        4: [3.0] * n,
        5: [4.0] * n,
        6: [5.0] * n,
    }
    df = pd.DataFrame(data)
    test_file = tmp_path / "Damaged_Beams" / "1mm" / "M2" / "Displacement" / "3cm.xlsx"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(test_file, header=False, index=False)

    rec = audit_recording_file(test_file)
    assert rec.invalid_missing_count == 3
    assert rec.validation_status == "irregular"
    assert "3 non-numeric/missing time cells" in rec.notes
    assert rec.sample_count == n - 3
    assert rec.sampling_rate_hz == pytest.approx(303.0303, abs=0.01)

