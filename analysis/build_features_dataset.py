"""Batch Feature Extraction & Baseline Generator (Task PRE-3 & PRE-4).

Author: Eswar
Task: PRE-3 (Dataset and feature integration) & PRE-4 (Frequency validation)
Reference: Talking to Bridges Revised Plan v2 (Interface A & Interface B)

This script:
1. Scans all cleaned recordings in data/processed/sensors/ (both undamaged and damaged).
2. Extracts high-resolution modal frequencies (f1, f2, f3) and time-domain features.
3. Assembles the Interface A feature table with standardized metadata and labels.
4. Generates data/processed/features_dataset.csv.
5. Computes and saves the healthy baseline profile (data/reference/healthy_baseline.json).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.features import extract_features, SENSOR_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ttb.build_features")


def process_recording_file(file_path: Path) -> Dict[str, Any]:
    """Process a single cleaned recording CSV and return its Interface A row."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"File {file_path.name} is empty.")

    # Extract physics and statistical features using PRE-2 module
    feats = extract_features(df)

    first_row = df.iloc[0]
    condition = str(first_row.get("Condition", "Undamaged")).strip()

    raw_level = str(first_row.get("Damage_Level", "")).strip()
    damage_level_mm: Optional[float] = None
    if condition == "Damaged":
        if "1mm" in raw_level.lower():
            damage_level_mm = 1.0
        elif "2mm" in raw_level.lower():
            damage_level_mm = 2.0
        elif "3mm" in raw_level.lower():
            damage_level_mm = 3.0
        else:
            try:
                damage_level_mm = float(raw_level.replace("mm", "").strip())
            except ValueError:
                damage_level_mm = 1.0

    specimen = str(first_row.get("Specimen", "")).strip()
    if not specimen or specimen.lower() == "nan":
        specimen = "unknown"

    test_type = str(first_row.get("Test_Type", "")).strip()
    if not test_type or test_type.lower() == "nan":
        test_type = "dynamic"

    source_file = str(first_row.get("Source_File", file_path.name)).strip()

    # Define grouped case_id for leave-one-case-out cross-validation
    if condition == "Damaged":
        case_id = f"{raw_level}_{specimen}" if raw_level else f"Damaged_{specimen}"
    else:
        case_id = f"Undamaged_{specimen}"

    row = {
        # Metadata (Not used as model input)
        "recording_id": file_path.stem,
        "case_id": case_id,
        "source_file": source_file,
        "specimen": specimen,
        "test_type": test_type,
        "sampling_rate_hz": feats.get("sampling_rate_hz", 300.0),

        # Labels (Training targets only)
        "condition": condition,
        "damage_level_mm": damage_level_mm,

        # Physics & Statistical Features (Model Inputs)
        "f1_hz": feats.get("f1_hz", 0.0),
        "f2_hz": feats.get("f2_hz", 0.0),
        "f3_hz": feats.get("f3_hz", 0.0),
        "f2_over_f1": feats.get("f2_over_f1", 0.0),
        "f3_over_f1": feats.get("f3_over_f1", 0.0),
    }

    # Add all per-sensor statistical features
    for col in SENSOR_COLS:
        for metric in ["rms", "variance", "skewness", "kurtosis", "peak", "crest_factor", "p2p"]:
            k = f"{col}_{metric}"
            row[k] = feats.get(k, 0.0)

    return row


def build_healthy_baseline_profile(
    df_features: pd.DataFrame,
    output_path: Path,
) -> Dict[str, Any]:
    """Compute robust healthy baseline profile (median and MAD) from undamaged rows."""
    undamaged_df = df_features[df_features["condition"] == "Undamaged"]
    if undamaged_df.empty:
        logger.warning("No undamaged recordings found to build baseline.")
        return {}

    disallowed = {
        "recording_id", "case_id", "source_file", "specimen",
        "test_type", "sampling_rate_hz", "condition", "damage_level_mm",
        "duration_sec", "sample_count",
    }
    feature_cols = [c for c in undamaged_df.columns if c not in disallowed]
    assert not any(c in disallowed for c in feature_cols), "Metadata or label column found in feature list!"

    profile: Dict[str, Any] = {
        "n_healthy_recordings": int(len(undamaged_df)),
        "healthy_cases": sorted(list(undamaged_df["case_id"].unique())),
        "features": {},
    }

    for col in feature_cols:
        series = pd.to_numeric(undamaged_df[col], errors="coerce").dropna()
        if not series.empty:
            med_val = float(series.median())
            # Median Absolute Deviation (MAD) scaled for normal consistency (1.4826 * MAD)
            mad_val = float(np.median(np.abs(series - med_val))) * 1.4826
            std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0
            mean_val = float(series.mean())

            profile["features"][col] = {
                "typical_median": round(med_val, 4),
                "spread_mad": round(mad_val if mad_val > 1e-6 else (std_val if std_val > 1e-6 else 0.01), 4),
                "mean": round(mean_val, 4),
                "std": round(std_val, 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    logger.info("Saved healthy baseline profile to %s", output_path)
    return profile


REPO_ROOT = Path(__file__).resolve().parent.parent

def run_batch_extraction(
    processed_dir: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    baseline_json: Optional[Path] = None,
) -> pd.DataFrame:
    """Run full batch feature extraction across all processed sensor files."""
    if processed_dir is None:
        processed_dir = REPO_ROOT / "data" / "processed" / "sensors"
    if output_csv is None:
        output_csv = REPO_ROOT / "data" / "processed" / "features_dataset.csv"
    if baseline_json is None:
        baseline_json = REPO_ROOT / "data" / "reference" / "healthy_baseline.json"
    undamaged_dir = processed_dir / "undamaged"
    damaged_dir = processed_dir / "damaged"

    undamaged_files = sorted(list(undamaged_dir.glob("*.csv"))) if undamaged_dir.exists() else []
    damaged_files = sorted(list(damaged_dir.glob("*.csv"))) if damaged_dir.exists() else []

    all_files = undamaged_files + damaged_files
    logger.info("Found %d undamaged and %d damaged files (total: %d)", len(undamaged_files), len(damaged_files), len(all_files))

    rows: List[Dict[str, Any]] = []
    for idx, f in enumerate(all_files, start=1):
        try:
            row = process_recording_file(f)
            rows.append(row)
            if idx % 50 == 0 or idx == len(all_files):
                logger.info("Processed %d / %d files...", idx, len(all_files))
        except Exception as exc:
            logger.error("Failed processing %s: %s", f.name, exc)

    df_out = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    logger.info("Saved %d feature rows to %s", len(df_out), output_csv)

    # Generate the healthy baseline profile
    build_healthy_baseline_profile(df_out, baseline_json)

    return df_out


if __name__ == "__main__":
    run_batch_extraction()
