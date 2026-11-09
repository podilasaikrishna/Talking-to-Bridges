"""Dataset inspection and time validation audit module (Task PRE-1).

Performs rigorous, independent time-quality validation and sampling rate analysis
on raw sensor recordings across the entire Talking to Bridges dataset.

Validates:
- Monotonicity, strict ordering, and out-of-order steps (dt < 0)
- Uniformity and sampling interval variations (median, mean, std, min, max, CV%)
- Duplicate timestamps (dt == 0)
- Discontinuities and large time gaps (dt > 1.5 * median_dt)
- Sampling rate estimation (1 / median_valid_dt)
- Non-numeric / invalid / missing values
- Discrepancy between Damaged (~303 Hz) and Undamaged (~333 Hz) vs paper (300 Hz)
"""

from __future__ import annotations

import argparse
import io
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import openpyxl
import pandas as pd

from analysis.sensor_pipeline import (
    MAX_RAW_COLUMNS,
    SENSOR_COLUMNS,
    _looks_like_hour_minute,
    _read_raw,
    parse_path_metadata,
    resolve_condition,
)

logger = logging.getLogger("ttb.time_audit")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesAudit:
    """Detailed time-quality audit results for a relative time array."""

    sample_count: int
    duration_sec: float
    median_dt: float
    mean_dt: float
    std_dt: float
    min_dt: float
    max_dt: float
    cv_percent: float
    sampling_rate_hz: float
    zero_dt_count: int
    negative_dt_count: int
    large_gap_count: int
    irregular_interval_count: int
    is_strictly_increasing: bool
    is_uniformly_sampled: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecordingAudit:
    """Audit record for a single recording or summary file."""

    source_file: str
    condition: str
    damage_level: str
    specimen: str
    test_type: str
    hit_group: str
    layout: str
    sample_count: int
    duration_sec: Optional[float]
    median_time_interval_sec: Optional[float]
    mean_time_interval_sec: Optional[float]
    std_time_interval_sec: Optional[float]
    sampling_rate_hz: Optional[float]
    min_time_interval_sec: Optional[float]
    max_time_interval_sec: Optional[float]
    cv_time_interval_pct: Optional[float]
    duplicate_count: int
    invalid_missing_count: int
    out_of_order_count: int
    large_gap_count: int
    irregular_interval_count: int
    sensor_invalid_cells: int
    flat_sensors: str
    validation_status: str  # "valid", "irregular", "skipped", "failed"
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Core Time Validation Logic
# ---------------------------------------------------------------------------

def audit_relative_time(
    rel_time_values: Sequence[float] | np.ndarray | pd.Series,
    gap_factor: float = 1.5,
    tolerance_pct: float = 0.05,
) -> TimeSeriesAudit:
    """Perform detailed time-quality calculations on a series of relative time values.

    Parameters:
        rel_time_values: Ordered or raw numeric time values in seconds.
        gap_factor: Multiplier over median dt defining an unusually large gap.
                    Default is 1.5 (indicating a missed sample period).
        tolerance_pct: Maximum relative deviation from median dt for an interval
                       to be considered uniform. Default is 0.05 (5%).

    Returns:
        TimeSeriesAudit with interval metrics, gap counts, and flags.
    """
    arr = np.asarray(rel_time_values, dtype=np.float64)
    # Filter out NaNs if any present in array
    valid_arr = arr[~np.isnan(arr)]

    n_samples = int(len(valid_arr))
    if n_samples < 2:
        return TimeSeriesAudit(
            sample_count=n_samples,
            duration_sec=0.0,
            median_dt=np.nan,
            mean_dt=np.nan,
            std_dt=np.nan,
            min_dt=np.nan,
            max_dt=np.nan,
            cv_percent=np.nan,
            sampling_rate_hz=np.nan,
            zero_dt_count=0,
            negative_dt_count=0,
            large_gap_count=0,
            irregular_interval_count=0,
            is_strictly_increasing=False,
            is_uniformly_sampled=False,
        )

    dt_all = np.diff(valid_arr)
    zero_dt = int((dt_all == 0).sum())
    negative_dt = int((dt_all < 0).sum())
    valid_dt = dt_all[dt_all > 0]

    min_dt = float(np.min(dt_all))
    max_dt = float(np.max(dt_all))
    duration = float(valid_arr[-1] - valid_arr[0])

    if len(valid_dt) == 0:
        return TimeSeriesAudit(
            sample_count=n_samples,
            duration_sec=duration,
            median_dt=np.nan,
            mean_dt=np.nan,
            std_dt=np.nan,
            min_dt=min_dt,
            max_dt=max_dt,
            cv_percent=np.nan,
            sampling_rate_hz=np.nan,
            zero_dt_count=zero_dt,
            negative_dt_count=negative_dt,
            large_gap_count=0,
            irregular_interval_count=len(dt_all),
            is_strictly_increasing=False,
            is_uniformly_sampled=False,
        )

    median_dt = float(np.median(valid_dt))
    mean_dt = float(np.mean(valid_dt))
    std_dt = float(np.std(valid_dt, ddof=1)) if len(valid_dt) > 1 else 0.0
    cv_percent = float((std_dt / mean_dt * 100.0)) if mean_dt > 0 else 0.0
    sampling_rate_hz = float(round(1.0 / median_dt, 4)) if median_dt > 0 else np.nan

    # Gaps and irregularities
    large_gaps = int((dt_all > gap_factor * median_dt).sum())
    irregular_intervals = int((np.abs(dt_all - median_dt) > tolerance_pct * median_dt).sum())

    is_strictly_increasing = bool(zero_dt == 0 and negative_dt == 0)
    is_uniformly_sampled = bool(irregular_intervals == 0 and large_gaps == 0 and is_strictly_increasing)

    return TimeSeriesAudit(
        sample_count=n_samples,
        duration_sec=duration,
        median_dt=median_dt,
        mean_dt=mean_dt,
        std_dt=std_dt,
        min_dt=min_dt,
        max_dt=max_dt,
        cv_percent=cv_percent,
        sampling_rate_hz=sampling_rate_hz,
        zero_dt_count=zero_dt,
        negative_dt_count=negative_dt,
        large_gap_count=large_gaps,
        irregular_interval_count=irregular_intervals,
        is_strictly_increasing=is_strictly_increasing,
        is_uniformly_sampled=is_uniformly_sampled,
    )


# ---------------------------------------------------------------------------
# File & Workbook Audit
# ---------------------------------------------------------------------------

def read_raw_fast(file_path: Union[str, Path]) -> pd.DataFrame:
    """Read first sheet of workbook using openpyxl read_only (fast) with fallback."""
    p = Path(file_path)
    ext = p.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(p, header=None)

    if ext in (".xlsx", ".xlsm"):
        try:
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            sheet = wb.worksheets[0]
            rows = list(sheet.iter_rows(values_only=True))
            wb.close()
            return pd.DataFrame(rows)
        except Exception:
            # Fallback to standard read_raw if read_only fails
            with open(p, "rb") as f:
                return _read_raw(f.read(), str(p))

    raise ValueError(f"Unsupported format: {ext}")


def audit_recording_file(file_path: Union[str, Path]) -> RecordingAudit:
    """Audit one recording file (raw Excel workbook or CSV)."""
    p = Path(file_path)
    source_file = str(p).replace("\\", "/")
    meta = parse_path_metadata(source_file)
    cond = resolve_condition(None, source_file, p.name) or "Unknown"

    damage_level = meta.get("Damage_Level") or ("None" if cond == "Undamaged" else "Unknown")
    specimen = meta.get("Specimen") or "Unknown"
    test_type = meta.get("Test_Type") or "Unknown"
    hit_group = meta.get("Hit_Group") or "None"

    try:
        raw = read_raw_fast(p)
    except Exception as exc:
        return RecordingAudit(
            source_file=source_file,
            condition=cond,
            damage_level=damage_level,
            specimen=specimen,
            test_type=test_type,
            hit_group=hit_group,
            layout="Unknown",
            sample_count=0,
            duration_sec=None,
            median_time_interval_sec=None,
            mean_time_interval_sec=None,
            std_time_interval_sec=None,
            sampling_rate_hz=None,
            min_time_interval_sec=None,
            max_time_interval_sec=None,
            cv_time_interval_pct=None,
            duplicate_count=0,
            invalid_missing_count=0,
            out_of_order_count=0,
            large_gap_count=0,
            irregular_interval_count=0,
            sensor_invalid_cells=0,
            flat_sensors="",
            validation_status="failed",
            notes=f"Read error: {exc}",
        )

    # Check empty rows
    non_empty_rows = raw.dropna(how="all")
    if non_empty_rows.empty:
        return RecordingAudit(
            source_file=source_file,
            condition=cond,
            damage_level=damage_level,
            specimen=specimen,
            test_type=test_type,
            hit_group=hit_group,
            layout="Unknown",
            sample_count=0,
            duration_sec=None,
            median_time_interval_sec=None,
            mean_time_interval_sec=None,
            std_time_interval_sec=None,
            sampling_rate_hz=None,
            min_time_interval_sec=None,
            max_time_interval_sec=None,
            cv_time_interval_pct=None,
            duplicate_count=0,
            invalid_missing_count=0,
            out_of_order_count=0,
            large_gap_count=0,
            irregular_interval_count=0,
            sensor_invalid_cells=0,
            flat_sensors="",
            validation_status="failed",
            notes="Empty file: no data rows.",
        )

    # Check for wide summary table
    non_empty_cols = int(raw.notna().any().sum())
    if non_empty_cols > MAX_RAW_COLUMNS:
        return RecordingAudit(
            source_file=source_file,
            condition=cond,
            damage_level=damage_level,
            specimen=specimen,
            test_type=test_type,
            hit_group=hit_group,
            layout="Wide_Summary",
            sample_count=int(len(non_empty_rows)),
            duration_sec=None,
            median_time_interval_sec=None,
            mean_time_interval_sec=None,
            std_time_interval_sec=None,
            sampling_rate_hz=None,
            min_time_interval_sec=None,
            max_time_interval_sec=None,
            cv_time_interval_pct=None,
            duplicate_count=0,
            invalid_missing_count=0,
            out_of_order_count=0,
            large_gap_count=0,
            irregular_interval_count=0,
            sensor_invalid_cells=0,
            flat_sensors="",
            validation_status="skipped",
            notes=f"Wide summary table ({non_empty_cols} cols). Deliberately skipped as non-raw recording.",
        )

    # Detect layout
    layout = "Undamaged" if _looks_like_hour_minute(raw) else "Damaged"
    time_col_idx = 3 if layout == "Undamaged" else 1
    sensor_indices = list(range(4, 9)) if layout == "Undamaged" else list(range(2, 7))

    # Drop completely empty rows and reset index
    raw = non_empty_rows.reset_index(drop=True)

    rel_raw = raw.iloc[:, time_col_idx]
    rel_num = pd.to_numeric(rel_raw, errors="coerce")
    invalid_time_count = int(rel_num.isna().sum())

    # Check sensor columns
    sensor_invalid = 0
    flat_sensors_list = []
    for idx, s_col_idx in enumerate(sensor_indices, start=1):
        if s_col_idx < raw.shape[1]:
            s_series = pd.to_numeric(raw.iloc[:, s_col_idx], errors="coerce")
            sensor_invalid += int(s_series.isna().sum())
            valid_s = s_series.dropna()
            if len(valid_s) > 1 and valid_s.nunique() <= 1:
                flat_sensors_list.append(f"Sensor_{idx}")
        else:
            sensor_invalid += len(raw)
            flat_sensors_list.append(f"Sensor_{idx}_missing")

    # Audit the valid time values
    valid_time = rel_num.dropna().to_numpy(dtype=np.float64)
    audit = audit_relative_time(valid_time)

    # Determine validation status
    notes_parts = []
    if invalid_time_count > 0:
        notes_parts.append(f"{invalid_time_count} non-numeric/missing time cells")
    if audit.zero_dt_count > 0:
        notes_parts.append(f"{audit.zero_dt_count} duplicate time steps")
    if audit.negative_dt_count > 0:
        notes_parts.append(f"{audit.negative_dt_count} out-of-order time steps")
    if audit.large_gap_count > 0:
        notes_parts.append(f"{audit.large_gap_count} gaps > 1.5*median")
    if audit.irregular_interval_count > 0:
        notes_parts.append(f"{audit.irregular_interval_count} intervals deviate > 5%")
    if flat_sensors_list:
        notes_parts.append(f"Flat sensors: {','.join(flat_sensors_list)}")

    if not audit.is_strictly_increasing or not audit.is_uniformly_sampled or invalid_time_count > 0:
        validation_status = "irregular"
    else:
        validation_status = "valid"

    notes = "; ".join(notes_parts) if notes_parts else "Uniform and strictly increasing."

    return RecordingAudit(
        source_file=source_file,
        condition=cond,
        damage_level=damage_level,
        specimen=specimen,
        test_type=test_type,
        hit_group=hit_group,
        layout=layout,
        sample_count=audit.sample_count,
        duration_sec=audit.duration_sec,
        median_time_interval_sec=audit.median_dt,
        mean_time_interval_sec=audit.mean_dt,
        std_time_interval_sec=audit.std_dt,
        sampling_rate_hz=audit.sampling_rate_hz,
        min_time_interval_sec=audit.min_dt,
        max_time_interval_sec=audit.max_dt,
        cv_time_interval_pct=audit.cv_percent,
        duplicate_count=audit.zero_dt_count,
        invalid_missing_count=invalid_time_count,
        out_of_order_count=audit.negative_dt_count,
        large_gap_count=audit.large_gap_count,
        irregular_interval_count=audit.irregular_interval_count,
        sensor_invalid_cells=sensor_invalid,
        flat_sensors=",".join(flat_sensors_list),
        validation_status=validation_status,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Dataset-Wide Audit & Summary Generation
# ---------------------------------------------------------------------------

def run_dataset_time_audit(
    data_dir: Union[str, Path] = "data",
    max_files: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Scan all raw recording files under data_dir and generate recording-level
    and summary audit tables.

    Returns:
        (recording_audit_df, dataset_summary_df)
    """
    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    all_files = sorted(
        [
            p
            for p in root.rglob("*")
            if p.is_file()
            and not p.name.startswith(("~$", "."))
            and p.suffix.lower() in (".xlsx", ".xlsm", ".csv")
        ]
    )

    if max_files is not None:
        all_files = all_files[:max(0, max_files)]

    logger.info("Found %d file(s) to audit under %s", len(all_files), root)

    records: List[RecordingAudit] = []
    for f in all_files:
        rec = audit_recording_file(f)
        records.append(rec)

    audit_df = pd.DataFrame([r.to_dict() for r in records])

    # Build grouped summary
    summary_rows: List[Dict[str, Any]] = []

    # 1. Condition counts & sampling rate stats
    cond_groups = audit_df[audit_df["validation_status"].isin(["valid", "irregular"])].groupby("condition")
    for cond_name, grp in cond_groups:
        summary_rows.append({
            "Category": "Condition",
            "Group": cond_name,
            "Recording_Count": len(grp),
            "Total_Samples": grp["sample_count"].sum(),
            "Mean_Duration_Sec": round(float(grp["duration_sec"].mean()), 2),
            "Sampling_Rate_Min_Hz": grp["sampling_rate_hz"].min(),
            "Sampling_Rate_Median_Hz": grp["sampling_rate_hz"].median(),
            "Sampling_Rate_Mean_Hz": round(float(grp["sampling_rate_hz"].mean()), 4),
            "Sampling_Rate_Max_Hz": grp["sampling_rate_hz"].max(),
            "Sampling_Rate_Std_Hz": round(float(grp["sampling_rate_hz"].std()), 6) if len(grp) > 1 else 0.0,
        })

    # 2. Damage Level counts
    dmg_counts = audit_df[audit_df["validation_status"].isin(["valid", "irregular"])].groupby("damage_level")
    for dmg_lvl, grp in dmg_counts:
        summary_rows.append({
            "Category": "Damage_Level",
            "Group": dmg_lvl,
            "Recording_Count": len(grp),
            "Total_Samples": grp["sample_count"].sum(),
            "Mean_Duration_Sec": round(float(grp["duration_sec"].mean()), 2),
            "Sampling_Rate_Min_Hz": grp["sampling_rate_hz"].min(),
            "Sampling_Rate_Median_Hz": grp["sampling_rate_hz"].median(),
            "Sampling_Rate_Mean_Hz": round(float(grp["sampling_rate_hz"].mean()), 4),
            "Sampling_Rate_Max_Hz": grp["sampling_rate_hz"].max(),
            "Sampling_Rate_Std_Hz": round(float(grp["sampling_rate_hz"].std()), 6) if len(grp) > 1 else 0.0,
        })

    # 3. Specimen counts
    spec_counts = audit_df[audit_df["validation_status"].isin(["valid", "irregular"])].groupby("specimen")
    for spec, grp in spec_counts:
        summary_rows.append({
            "Category": "Specimen",
            "Group": spec,
            "Recording_Count": len(grp),
            "Total_Samples": grp["sample_count"].sum(),
            "Mean_Duration_Sec": round(float(grp["duration_sec"].mean()), 2),
            "Sampling_Rate_Min_Hz": grp["sampling_rate_hz"].min(),
            "Sampling_Rate_Median_Hz": grp["sampling_rate_hz"].median(),
            "Sampling_Rate_Mean_Hz": round(float(grp["sampling_rate_hz"].mean()), 4),
            "Sampling_Rate_Max_Hz": grp["sampling_rate_hz"].max(),
            "Sampling_Rate_Std_Hz": round(float(grp["sampling_rate_hz"].std()), 6) if len(grp) > 1 else 0.0,
        })

    # 4. Test Type counts
    ttype_counts = audit_df[audit_df["validation_status"].isin(["valid", "irregular"])].groupby("test_type")
    for ttype, grp in ttype_counts:
        summary_rows.append({
            "Category": "Test_Type",
            "Group": ttype,
            "Recording_Count": len(grp),
            "Total_Samples": grp["sample_count"].sum(),
            "Mean_Duration_Sec": round(float(grp["duration_sec"].mean()), 2),
            "Sampling_Rate_Min_Hz": grp["sampling_rate_hz"].min(),
            "Sampling_Rate_Median_Hz": grp["sampling_rate_hz"].median(),
            "Sampling_Rate_Mean_Hz": round(float(grp["sampling_rate_hz"].mean()), 4),
            "Sampling_Rate_Max_Hz": grp["sampling_rate_hz"].max(),
            "Sampling_Rate_Std_Hz": round(float(grp["sampling_rate_hz"].std()), 6) if len(grp) > 1 else 0.0,
        })

    # 5. Validation Status counts
    status_counts = audit_df.groupby("validation_status")
    for st, grp in status_counts:
        summary_rows.append({
            "Category": "Validation_Status",
            "Group": st,
            "Recording_Count": len(grp),
            "Total_Samples": grp["sample_count"].sum(),
            "Mean_Duration_Sec": round(float(grp["duration_sec"].mean()), 2) if grp["duration_sec"].notna().any() else 0.0,
            "Sampling_Rate_Min_Hz": grp["sampling_rate_hz"].min(),
            "Sampling_Rate_Median_Hz": grp["sampling_rate_hz"].median(),
            "Sampling_Rate_Mean_Hz": round(float(grp["sampling_rate_hz"].mean()), 4) if grp["sampling_rate_hz"].notna().any() else np.nan,
            "Sampling_Rate_Max_Hz": grp["sampling_rate_hz"].max(),
            "Sampling_Rate_Std_Hz": round(float(grp["sampling_rate_hz"].std()), 6) if grp["sampling_rate_hz"].notna().any() else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    return audit_df, summary_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit dataset timestamps and sampling intervals.")
    parser.add_argument("--data-dir", default="data", help="Directory containing raw data files.")
    parser.add_argument("--audit-out", default="data/audit/pre1_recording_time_audit.csv", help="Output CSV for recording audit.")
    parser.add_argument("--summary-out", default="data/audit/pre1_dataset_summary.csv", help="Output CSV for dataset summary.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional limit for quick testing.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    Path(args.audit_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)

    audit_df, summary_df = run_dataset_time_audit(args.data_dir, max_files=args.max_files)
    audit_df.to_csv(args.audit_out, index=False)
    summary_df.to_csv(args.summary_out, index=False)

    print("\n=== TIME AUDIT COMPLETE ===")
    print(f"Total files audited: {len(audit_df)}")
    print(f"Audit output: {args.audit_out}")
    print(f"Summary output: {args.summary_out}")
    print("\nValidation Status counts:")
    print(audit_df["validation_status"].value_counts().to_string())
    print("\nCondition breakdown:")
    print(audit_df[audit_df["validation_status"].isin(["valid", "irregular"])]["condition"].value_counts().to_string())

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

