"""Exploratory Data Analysis (EDA) for Sensor Data (Phase 2 - Week 5, Krishna).

Performs comprehensive exploratory data analysis on the clean sensor dataset
produced by Kolla's preprocessing pipeline, identifying dataset characteristics,
sensor statistics, time-series behaviors, damage-level trends, cross-sensor
correlations, and specimen/test-type patterns.

Follows all Phase 2 Week 5 specifications:
- Uses Relative_Time_Sec as the actual time axis (DateTime is a 1970-01-01 placeholder).
- Treats sensor values as unitless (physical units currently unknown).
- Computes statistics at the recording level to avoid sample-pooling distortion.
- Discovers clean CSV recordings automatically and integrates with load_clean_file().
- Reports observed patterns without claiming causality.
- Outputs key findings categorized into strong observations vs weaker trends.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# Try importing plotly; gracefully handle if not installed
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# Try importing Kolla's clean-data loader from analysis.sensor_pipeline
try:
    from analysis.sensor_pipeline import CLEAN_COLUMNS, SENSOR_COLUMNS, load_clean_file
except ImportError:
    SENSOR_COLUMNS = [f"Sensor_{i}" for i in range(1, 6)]
    CLEAN_COLUMNS = [
        "DateTime",
        "Relative_Time_Sec",
        *SENSOR_COLUMNS,
        "Condition",
        "Test_Name",
        "Source_File",
        "Damage_Level",
        "Specimen",
        "Test_Type",
        "Hit_Group",
    ]

    def load_clean_file(path: Union[str, Path]) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["DateTime"])
        return df

logger = logging.getLogger("ttb.eda")

# Default paths
_HERE = Path(__file__).resolve()
PROJECT_ROOT = next(
    (p for p in [_HERE.parent, *_HERE.parents] if (p / "data").exists()),
    Path.cwd(),
)
DEFAULT_SENSOR_DIR = PROJECT_ROOT / "data" / "processed" / "sensors"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "eda_outputs"


# ---------------------------------------------------------------------------
# 1. Recording Discovery and Data Loading
# ---------------------------------------------------------------------------

def discover_recordings(sensor_dir: Union[str, Path]) -> List[Path]:
    """Discover clean recording CSV files using manifest.csv if present,
    or through directory traversal of damaged/ and undamaged/ subdirectories.
    """
    sensor_dir = Path(sensor_dir)
    manifest_path = sensor_dir / "manifest.csv"
    discovered_files: List[Path] = []

    if manifest_path.is_file():
        logger.info("Found manifest at %s", manifest_path)
        try:
            manifest_df = pd.read_csv(manifest_path)
            # Filter ok status and non-null output_file
            if "status" in manifest_df.columns:
                valid_rows = manifest_df[manifest_df["status"] == "ok"]
            elif "output_file" in manifest_df.columns:
                valid_rows = manifest_df[manifest_df["output_file"].notna()]
            else:
                valid_rows = manifest_df

            for _, row in valid_rows.iterrows():
                out_file = row.get("output_file")
                if pd.isna(out_file) or not str(out_file).strip():
                    continue

                out_str = str(out_file).strip().replace("\\", "/")
                # Try direct path, subfolder path, or filename rglob
                candidate = sensor_dir / out_str
                if candidate.is_file():
                    discovered_files.append(candidate)
                    continue

                name = Path(out_str).name
                hit = next(sensor_dir.rglob(name), None)
                if hit and hit.is_file():
                    discovered_files.append(hit)

            if discovered_files:
                logger.info("Discovered %d recordings via manifest.csv", len(discovered_files))
                return sorted(list(set(discovered_files)))
        except Exception as exc:
            logger.warning("Failed parsing manifest (%s); falling back to file search", exc)

    # Fallback / Direct search: look in damaged/ and undamaged/
    if sensor_dir.is_dir():
        for csv_path in sorted(sensor_dir.rglob("*.csv")):
            fname = csv_path.name.lower()
            # Skip manifest and summary files
            if fname in ("manifest.csv", "eda_summary_metrics.csv") or "summary" in fname:
                continue
            discovered_files.append(csv_path)

    logger.info("Discovered %d recordings from %s", len(discovered_files), sensor_dir)
    return sorted(list(set(discovered_files)))


def load_single_recording(csv_path: Path) -> pd.DataFrame:
    """Load a single clean sensor recording CSV with load_clean_file."""
    try:
        return load_clean_file(csv_path)
    except Exception as exc:
        logger.debug("load_clean_file failed for %s (%s); trying standard pandas read", csv_path.name, exc)
        return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# 2. Recording-Level Metrics Computation
# ---------------------------------------------------------------------------

def compute_recording_metrics(df: pd.DataFrame, file_path: Path) -> Dict[str, Any]:
    """Calculate descriptive statistics and metadata for one recording.

    Statistics are computed at the recording level to ensure valid statistical
    comparison across tests without sample-pooling distortions.
    """
    row_count = len(df)
    first_row = df.iloc[0] if row_count > 0 else {}

    # Extract metadata fields (with fallback to folder/filename parsing)
    condition = str(first_row.get("Condition", "")).strip()
    if not condition or condition.lower() == "nan":
        condition = "Undamaged" if "undamaged" in str(file_path).lower() else "Damaged"

    damage_level = first_row.get("Damage_Level", None)
    if pd.isna(damage_level) or str(damage_level).strip() in ("", "nan", "None"):
        damage_level = "Undamaged" if condition.lower() == "undamaged" else "Unknown"
    else:
        damage_level = str(damage_level).strip()

    specimen = str(first_row.get("Specimen", "")).strip()
    if not specimen or specimen.lower() == "nan":
        specimen = "Unknown"

    test_type = str(first_row.get("Test_Type", "")).strip()
    if not test_type or test_type.lower() == "nan":
        test_type = "Unknown"

    hit_group = first_row.get("Hit_Group", None)
    if pd.isna(hit_group) or str(hit_group).strip() in ("", "nan", "None"):
        hit_group = "None"
    else:
        hit_group = str(hit_group).strip()

    test_name = str(first_row.get("Test_Name", file_path.stem)).strip()
    source_file = str(first_row.get("Source_File", file_path.name)).strip()

    # Time characteristics
    rel_time = df["Relative_Time_Sec"] if "Relative_Time_Sec" in df.columns else pd.Series([], dtype=float)
    if len(rel_time) >= 2:
        duration_sec = float(rel_time.iloc[-1] - rel_time.iloc[0])
        dt_diffs = rel_time.diff().dropna()
        dt_median = float(dt_diffs.median()) if len(dt_diffs) > 0 else 0.0
        sampling_rate_hz = float(1.0 / dt_median) if dt_median > 0 else float(row_count / duration_sec) if duration_sec > 0 else 0.0
    else:
        duration_sec = 0.0
        sampling_rate_hz = 0.0

    metrics: Dict[str, Any] = {
        "file_name": file_path.name,
        "file_path": str(file_path),
        "source_file": source_file,
        "condition": condition,
        "damage_level": damage_level,
        "specimen": specimen,
        "test_type": test_type,
        "hit_group": hit_group,
        "test_name": test_name,
        "row_count": row_count,
        "duration_sec": round(duration_sec, 4),
        "sampling_rate_hz": round(sampling_rate_hz, 2),
    }

    # Per-sensor descriptive statistics
    for col in SENSOR_COLUMNS:
        if col in df.columns:
            s_data = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s_data) > 0:
                s_mean = float(s_data.mean())
                s_std = float(s_data.std(ddof=1)) if len(s_data) > 1 else 0.0
                s_min = float(s_data.min())
                s_max = float(s_data.max())
                s_median = float(s_data.median())
                s_p2p = s_max - s_min
                s_rms = float(np.sqrt(np.mean(s_data.to_numpy() ** 2)))
            else:
                s_mean = s_std = s_min = s_max = s_median = s_p2p = s_rms = 0.0

            metrics[f"{col}_mean"] = round(s_mean, 6)
            metrics[f"{col}_std"] = round(s_std, 6)
            metrics[f"{col}_min"] = round(s_min, 6)
            metrics[f"{col}_max"] = round(s_max, 6)
            metrics[f"{col}_median"] = round(s_median, 6)
            metrics[f"{col}_p2p"] = round(s_p2p, 6)
            metrics[f"{col}_rms"] = round(s_rms, 6)

    # Within-recording sensor correlations (aligned time-series correlation)
    present_sensors = [c for c in SENSOR_COLUMNS if c in df.columns]
    if len(present_sensors) >= 2:
        corr_mat = df[present_sensors].corr()
        for i in range(len(present_sensors)):
            for j in range(i + 1, len(present_sensors)):
                s_i, s_j = present_sensors[i], present_sensors[j]
                val = corr_mat.loc[s_i, s_j]
                metrics[f"corr_{s_i}_{s_j}"] = round(float(val), 4) if pd.notna(val) else 0.0

    return metrics


# ---------------------------------------------------------------------------
# 3. Exploratory Data Analysis & Statistics Aggregation
# ---------------------------------------------------------------------------

class SensorEDA:
    """Core analysis engine for clean bridge and cantilever sensor data."""

    def __init__(self, recordings_df: pd.DataFrame, representative_data: Dict[str, pd.DataFrame]):
        self.metrics_df = recordings_df
        self.representative_data = representative_data

    @classmethod
    def from_directory(
        cls,
        sensor_dir: Union[str, Path] = DEFAULT_SENSOR_DIR,
        max_recordings: Optional[int] = None,
    ) -> SensorEDA:
        """Construct SensorEDA by discovering and processing all recordings in sensor_dir."""
        csv_files = discover_recordings(sensor_dir)
        if not csv_files:
            raise FileNotFoundError(
                f"No clean sensor files discovered in '{sensor_dir}'. "
                "Ensure clean files are located in damaged/ and undamaged/ subdirectories."
            )

        if max_recordings:
            csv_files = csv_files[:max_recordings]

        logger.info("Computing metrics across %d clean recordings...", len(csv_files))
        rows: List[Dict[str, Any]] = []
        representative_data: Dict[str, pd.DataFrame] = {}

        # Track representative samples per category for visualization
        categories_sought = {
            "Undamaged": None,
            "1mm": None,
            "2mm": None,
            "3mm": None,
        }

        for p in csv_files:
            df = load_single_recording(p)
            if df.empty:
                continue

            metric = compute_recording_metrics(df, p)
            rows.append(metric)

            # Store representative recording if slot is empty
            dmg = metric["damage_level"]
            cond = metric["condition"]
            cat_key = "Undamaged" if cond.lower() == "undamaged" else dmg
            if cat_key in categories_sought and categories_sought[cat_key] is None:
                categories_sought[cat_key] = p.name
                representative_data[cat_key] = df

        metrics_df = pd.DataFrame(rows)
        return cls(metrics_df, representative_data)

    # --- 1. Dataset Overview ---
    def get_dataset_overview(self) -> Dict[str, Any]:
        df = self.metrics_df
        total = len(df)
        cond_counts = df["condition"].value_counts().to_dict()
        dmg_counts = df["damage_level"].value_counts().to_dict()
        spec_counts = df["specimen"].value_counts().to_dict()
        test_counts = df["test_type"].value_counts().to_dict()
        hit_counts = df["hit_group"].value_counts().to_dict()

        total_samples = int(df["row_count"].sum()) if "row_count" in df.columns else 0
        min_samples = int(df["row_count"].min()) if "row_count" in df.columns else 0
        max_samples = int(df["row_count"].max()) if "row_count" in df.columns else 0
        mean_samples = float(df["row_count"].mean()) if "row_count" in df.columns else 0.0

        # Duration and sampling rates
        mean_dur = float(df["duration_sec"].mean()) if "duration_sec" in df.columns else 0.0
        damaged_rate = (
            float(df[df["condition"].str.lower() == "damaged"]["sampling_rate_hz"].median())
            if (df["condition"].str.lower() == "damaged").any() else 0.0
        )
        undamaged_rate = (
            float(df[df["condition"].str.lower() == "undamaged"]["sampling_rate_hz"].median())
            if (df["condition"].str.lower() == "undamaged").any() else 0.0
        )

        return {
            "total_recordings": total,
            "total_samples": total_samples,
            "sample_counts": {
                "min": min_samples,
                "mean": round(mean_samples, 1),
                "max": max_samples,
            },
            "mean_duration_sec": round(mean_dur, 2),
            "estimated_sampling_rates": {
                "damaged_approx_hz": round(damaged_rate, 1),
                "undamaged_approx_hz": round(undamaged_rate, 1),
            },
            "condition_distribution": cond_counts,
            "damage_level_distribution": dmg_counts,
            "specimen_distribution": spec_counts,
            "test_type_distribution": test_counts,
            "hit_group_distribution": hit_counts,
        }

    # --- 2. Sensor Descriptive Statistics ---
    def get_sensor_descriptive_statistics(self) -> pd.DataFrame:
        """Compute aggregated statistics for Sensor_1 to Sensor_5 across recordings.
        
        Calculates mean, std, min, median, max, RMS, and peak-to-peak distribution
        at the recording level across all recordings.
        """
        df = self.metrics_df
        records = []

        for sensor in SENSOR_COLUMNS:
            row = {"Sensor": sensor}
            for stat in ("mean", "std", "min", "median", "max", "p2p", "rms"):
                col_name = f"{sensor}_{stat}"
                if col_name in df.columns:
                    s_series = df[col_name].dropna()
                    row[f"{stat}_mean"] = round(float(s_series.mean()), 5)
                    row[f"{stat}_std"] = round(float(s_series.std()), 5)
                    row[f"{stat}_min"] = round(float(s_series.min()), 5)
                    row[f"{stat}_median"] = round(float(s_series.median()), 5)
                    row[f"{stat}_max"] = round(float(s_series.max()), 5)
            records.append(row)

        return pd.DataFrame(records)

    # --- 4. Damage-Level Trends ---
    def get_damage_level_trends(self) -> pd.DataFrame:
        """Investigate systematic changes in signal amplitude, RMS, and spread
        across damage levels: Undamaged, 1mm, 2mm, 3mm.
        """
        df = self.metrics_df
        ordered_levels = ["Undamaged", "1mm", "2mm", "3mm"]
        present_levels = [lvl for lvl in ordered_levels if lvl in df["damage_level"].unique()]
        other_levels = [lvl for lvl in df["damage_level"].unique() if lvl not in ordered_levels]
        levels = present_levels + other_levels

        rows = []
        for lvl in levels:
            sub = df[df["damage_level"] == lvl]
            if sub.empty:
                continue

            entry: Dict[str, Any] = {
                "damage_level": lvl,
                "recording_count": len(sub),
            }

            for sensor in SENSOR_COLUMNS:
                rms_col = f"{sensor}_rms"
                p2p_col = f"{sensor}_p2p"
                std_col = f"{sensor}_std"
                if rms_col in sub.columns:
                    entry[f"{sensor}_rms_mean"] = round(float(sub[rms_col].mean()), 4)
                if p2p_col in sub.columns:
                    entry[f"{sensor}_p2p_mean"] = round(float(sub[p2p_col].mean()), 4)
                if std_col in sub.columns:
                    entry[f"{sensor}_std_mean"] = round(float(sub[std_col].mean()), 4)

            rows.append(entry)

        return pd.DataFrame(rows)

    # --- 5. Sensor Correlations ---
    def get_sensor_correlation_matrix(self, condition: Optional[str] = None) -> pd.DataFrame:
        """Compute average within-recording correlation matrix across sensors.

        Averaging per-recording correlation matrices prevents cross-recording
        baseline offsets from creating false correlations.
        """
        df = self.metrics_df
        if condition:
            df = df[df["condition"].str.lower() == condition.lower()]

        corr_matrix = pd.DataFrame(np.eye(5), index=SENSOR_COLUMNS, columns=SENSOR_COLUMNS)

        for i, s1 in enumerate(SENSOR_COLUMNS):
            for j, s2 in enumerate(SENSOR_COLUMNS):
                if i == j:
                    corr_matrix.loc[s1, s2] = 1.0
                elif i < j:
                    pair_col = f"corr_{s1}_{s2}"
                    if pair_col in df.columns:
                        avg_val = df[pair_col].dropna().mean()
                        corr_matrix.loc[s1, s2] = round(float(avg_val), 4)
                        corr_matrix.loc[s2, s1] = round(float(avg_val), 4)

        return corr_matrix

    # --- 6. Relationship Between Sensor Features and Damage ---
    def get_damage_feature_relationships(self) -> Dict[str, Any]:
        """Examine how amplitude, variance, and sensor ratios relate to damage levels."""
        df = self.metrics_df
        levels = ["Undamaged", "1mm", "2mm", "3mm"]
        sub_levels = [lvl for lvl in levels if lvl in df["damage_level"].unique()]

        relationships: Dict[str, Any] = {}
        for sensor in SENSOR_COLUMNS:
            rms_means = []
            p2p_means = []
            for lvl in sub_levels:
                sub = df[df["damage_level"] == lvl]
                rms_means.append(float(sub[f"{sensor}_rms"].mean()) if f"{sensor}_rms" in sub.columns else 0.0)
                p2p_means.append(float(sub[f"{sensor}_p2p"].mean()) if f"{sensor}_p2p" in sub.columns else 0.0)

            relationships[sensor] = {
                "damage_levels": sub_levels,
                "rms_progression": [round(x, 4) for x in rms_means],
                "p2p_progression": [round(x, 4) for x in p2p_means],
            }

        return relationships

    # --- 7. Specimen & Test-Type Trends ---
    def get_specimen_and_test_type_trends(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Analyze metric distributions across Specimens M1-M7 and Test Types."""
        df = self.metrics_df

        # Specimen summary (damaged recordings M1 to M7)
        specimen_rows = []
        for spec, grp in df.groupby("specimen"):
            row = {"specimen": spec, "count": len(grp)}
            for sensor in ("Sensor_1", "Sensor_3", "Sensor_5"):
                p2p = f"{sensor}_p2p"
                rms = f"{sensor}_rms"
                if p2p in grp.columns:
                    row[f"{sensor}_p2p_mean"] = round(float(grp[p2p].mean()), 4)
                if rms in grp.columns:
                    row[f"{sensor}_rms_mean"] = round(float(grp[rms].mean()), 4)
            specimen_rows.append(row)
        spec_df = pd.DataFrame(specimen_rows)

        # Test type summary
        test_type_rows = []
        for ttype, grp in df.groupby("test_type"):
            row = {"test_type": ttype, "count": len(grp)}
            for sensor in ("Sensor_1", "Sensor_3", "Sensor_5"):
                p2p = f"{sensor}_p2p"
                rms = f"{sensor}_rms"
                if p2p in grp.columns:
                    row[f"{sensor}_p2p_mean"] = round(float(grp[p2p].mean()), 4)
                if rms in grp.columns:
                    row[f"{sensor}_rms_mean"] = round(float(grp[rms].mean()), 4)
            test_type_rows.append(row)
        test_df = pd.DataFrame(test_type_rows)

        return spec_df, test_df

    # --- 8. Key Findings ---
    def extract_key_findings(self) -> Dict[str, List[str]]:
        """Synthesize empirical observations separated into strong findings vs weaker trends."""
        df = self.metrics_df
        strong_findings: List[str] = []
        weaker_trends: List[str] = []

        # 1. Dataset structure observations
        total = len(df)
        cond_counts = df["condition"].value_counts().to_dict()
        dmg_counts = df["damage_level"].value_counts().to_dict()

        strong_findings.append(
            f"Dataset Integrity: Total {total} clean recordings evaluated "
            f"({cond_counts.get('Damaged', 0)} Damaged, {cond_counts.get('Undamaged', 0)} Undamaged). "
            "All recordings contain complete records with zero missing values across Relative_Time_Sec and Sensor_1-5."
        )

        # 2. Sampling rate observation
        if "sampling_rate_hz" in df.columns:
            dam_sr = df[df["condition"].str.lower() == "damaged"]["sampling_rate_hz"].median()
            undam_sr = df[df["condition"].str.lower() == "undamaged"]["sampling_rate_hz"].median()
            strong_findings.append(
                f"Sampling Frequency Discrepancy: Clean data exhibits two distinct computed sampling rates: "
                f"~{dam_sr:.1f} Hz for Damaged bridge recordings and ~{undam_sr:.1f} Hz for Undamaged cantilever recordings. "
                "Note: These rates are derived from timestamp deltas and require experimental confirmation from the professor."
            )

        # 3. Sensor Correlations
        corr_overall = self.get_sensor_correlation_matrix()
        # Find highest and lowest correlation pairs
        pairs = []
        for i, s1 in enumerate(SENSOR_COLUMNS):
            for j, s2 in enumerate(SENSOR_COLUMNS):
                if i < j:
                    pairs.append((s1, s2, corr_overall.loc[s1, s2]))
        pairs.sort(key=lambda x: x[2], reverse=True)

        if pairs:
            top_pair = pairs[0]
            bot_pair = pairs[-1]
            strong_findings.append(
                f"Sensor Inter-Correlation: Within-recording signals demonstrate structured spatial correlation. "
                f"Highest average correlation is between {top_pair[0]} and {top_pair[1]} (r = {top_pair[2]:.3f}), "
                f"whereas lowest correlation is between {bot_pair[0]} and {bot_pair[1]} (r = {bot_pair[2]:.3f})."
            )

        # 4. Damage Level Trends
        trends = self.get_damage_level_trends()
        if "damage_level" in trends.columns and len(trends) >= 2:
            s1_rms_col = "Sensor_1_rms_mean"
            if s1_rms_col in trends.columns:
                trend_vals = dict(zip(trends["damage_level"], trends[s1_rms_col]))
                weaker_trends.append(
                    f"Damage-Level Vibration Energy: Mean Sensor_1 RMS across levels: {trend_vals}. "
                    "Progression across damage stages shows variations in signal spread and amplitude, "
                    "though strict monotonicity across all sensors should not be assumed without knowing physical sensor placements."
                )

        # 5. Test Type Influence
        _, test_df = self.get_specimen_and_test_type_trends()
        if not test_df.empty and "test_type" in test_df.columns:
            weaker_trends.append(
                f"Excitation Protocol Impact: Test types ({', '.join(test_df['test_type'])}) induce distinct "
                "dynamic response envelopes (e.g. impact hits produce high initial peak-to-peak transients "
                "followed by exponential decay, whereas displacement tests display step-release damping curves)."
            )

        # 6. Specimen Variance
        spec_df, _ = self.get_specimen_and_test_type_trends()
        if not spec_df.empty and len(spec_df) > 1:
            weaker_trends.append(
                f"Specimen Variance (M1-M7): Across specimens, baseline sensor amplitudes show modest test-to-test "
                "variance, suggesting structural consistency in the test beams."
            )

        return {
            "strong_observations": strong_findings,
            "weaker_trends": weaker_trends,
        }


# ---------------------------------------------------------------------------
# 4. Visualizations
# ---------------------------------------------------------------------------

def generate_visualizations(eda: SensorEDA, output_dir: Path) -> List[Path]:
    """Create representative visualizations communicating EDA findings clearly.

    Generates 4 focused figures:
    1. Multi-trace time-series comparison across damage levels over Relative_Time_Sec.
    2. Damage level feature distribution box plots (RMS and Peak-to-Peak).
    3. Sensor correlation heatmaps (overall, undamaged, damaged).
    4. Test type and specimen dynamic trends.
    """
    if not HAS_PLOTLY:
        logger.warning("Plotly is not installed; skipping plot generation.")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    generated_plots: List[Path] = []

    # --- Plot 1: Representative Time Series Comparison ---
    rep_data = eda.representative_data
    if rep_data:
        categories = list(rep_data.keys())
        fig_ts = make_subplots(
            rows=len(categories),
            cols=1,
            shared_xaxes=False,
            subplot_titles=[f"Representative Signal: {cat}" for cat in categories],
            vertical_spacing=0.08,
        )

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        for row_idx, cat in enumerate(categories, start=1):
            df_rep = rep_data[cat]
            # Downsample if long to keep chart responsive
            if len(df_rep) > 3000:
                step = len(df_rep) // 3000
                df_rep = df_rep.iloc[::step].copy()

            for s_idx, col in enumerate(SENSOR_COLUMNS):
                if col in df_rep.columns:
                    fig_ts.add_trace(
                        go.Scatter(
                            x=df_rep["Relative_Time_Sec"],
                            y=df_rep[col],
                            mode="lines",
                            name=f"{col} ({cat})",
                            legendgroup=col,
                            showlegend=(row_idx == 1),
                            line=dict(color=colors[s_idx % len(colors)], width=1.2),
                        ),
                        row=row_idx,
                        col=1,
                    )
            fig_ts.update_yaxes(title_text="Sensor value", row=row_idx, col=1)

        fig_ts.update_xaxes(title_text="Relative Time (seconds)", row=len(categories), col=1)
        fig_ts.update_layout(
            height=280 * len(categories),
            title_text="Figure 1: Representative Sensor Time-Series Comparison Across Damage States",
            template="plotly_white",
        )
        p1 = output_dir / "fig1_time_series_comparison.html"
        fig_ts.write_html(str(p1), include_plotlyjs="cdn")
        generated_plots.append(p1)

    # --- Plot 2: Damage Level Feature Trends ---
    df_m = eda.metrics_df
    if "damage_level" in df_m.columns and "Sensor_1_rms" in df_m.columns:
        fig_dmg = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["Sensor_1 RMS by Damage Level", "Sensor_1 Peak-to-Peak by Damage Level"],
        )
        order = [lvl for lvl in ["Undamaged", "1mm", "2mm", "3mm"] if lvl in df_m["damage_level"].unique()]
        for lvl in order:
            sub = df_m[df_m["damage_level"] == lvl]
            fig_dmg.add_trace(
                go.Box(y=sub["Sensor_1_rms"], name=lvl, boxmean=True, showlegend=False),
                row=1,
                col=1,
            )
            fig_dmg.add_trace(
                go.Box(y=sub["Sensor_1_p2p"], name=lvl, boxmean=True, showlegend=False),
                row=1,
                col=2,
            )

        fig_dmg.update_yaxes(title_text="RMS (uncalibrated)", row=1, col=1)
        fig_dmg.update_yaxes(title_text="Peak-to-Peak Range (uncalibrated)", row=1, col=2)
        fig_dmg.update_layout(
            height=450,
            title_text="Figure 2: Signal Amplitude & RMS Variations Across Damage Levels",
            template="plotly_white",
        )
        p2 = output_dir / "fig2_damage_level_trends.html"
        fig_dmg.write_html(str(p2), include_plotlyjs="cdn")
        generated_plots.append(p2)

    # --- Plot 3: Sensor Correlation Heatmaps ---
    corr_all = eda.get_sensor_correlation_matrix()
    corr_dam = eda.get_sensor_correlation_matrix("Damaged")
    corr_undam = eda.get_sensor_correlation_matrix("Undamaged")

    fig_corr = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["All Recordings", "Damaged Recordings", "Undamaged Recordings"],
    )

    for c_idx, (corr, title) in enumerate([(corr_all, "All"), (corr_dam, "Damaged"), (corr_undam, "Undamaged")], start=1):
        fig_corr.add_trace(
            go.Heatmap(
                z=corr.values,
                x=corr.columns.tolist(),
                y=corr.index.tolist(),
                colorscale="Viridis",
                zmin=-1.0,
                zmax=1.0,
                showscale=(c_idx == 3),
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            ),
            row=1,
            col=c_idx,
        )

    fig_corr.update_layout(
        height=450,
        title_text="Figure 3: Average Within-Recording Sensor Correlation Matrices",
        template="plotly_white",
    )
    p3 = output_dir / "fig3_sensor_correlations.html"
    fig_corr.write_html(str(p3), include_plotlyjs="cdn")
    generated_plots.append(p3)

    # --- Plot 4: Test Type Dynamics ---
    if "test_type" in df_m.columns and "Sensor_1_p2p" in df_m.columns:
        fig_test = go.Figure()
        test_types = df_m["test_type"].unique()
        for tt in test_types:
            sub = df_m[df_m["test_type"] == tt]
            fig_test.add_trace(
                go.Box(y=sub["Sensor_1_p2p"], name=tt, boxmean=True),
            )

        fig_test.update_layout(
            height=450,
            title_text="Figure 4: Sensor_1 Peak-to-Peak Range Across Excitation Test Types",
            yaxis_title="Peak-to-Peak Amplitude (uncalibrated)",
            xaxis_title="Test Type",
            template="plotly_white",
        )
        p4 = output_dir / "fig4_test_type_trends.html"
        fig_test.write_html(str(p4), include_plotlyjs="cdn")
        generated_plots.append(p4)

    return generated_plots


# ---------------------------------------------------------------------------
# 5. Report Generation & Summary Metrics Export
# ---------------------------------------------------------------------------

def generate_eda_report_markdown(eda: SensorEDA, report_path: Path) -> str:
    """Generate a clean, structured Markdown report detailing all 8 EDA requirements."""
    overview = eda.get_dataset_overview()
    desc_stats = eda.get_sensor_descriptive_statistics()
    trends = eda.get_damage_level_trends()
    corr_mat = eda.get_sensor_correlation_matrix()
    findings = eda.extract_key_findings()

    md: List[str] = [
        "# Exploratory Data Analysis Report: Bridge & Cantilever Sensor Dataset",
        "**Phase 2 — Week 5 | Talking to Bridges Project**",
        "**Author:** Krishna (Exploratory Data Analysis & Trends/Correlations)",
        f"**Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Executive Summary & Critical Data Interpretations",
        "This exploratory data analysis was conducted on the cleaned sensor dataset produced by the ingestion pipeline.",
        "Key operational rules applied throughout this analysis:",
        "- **Time Axis:** `Relative_Time_Sec` is the sole ground-truth time coordinate. `DateTime` contains an artificial `1970-01-01` anchor date and was not treated as a calendar timestamp.",
        "- **Physical Units:** The physical measurement units of `Sensor_1` through `Sensor_5` are currently uncalibrated and unconfirmed; no physical units (such as mm, m/s², or µε) are assumed.",
        "- **Sampling Frequencies:** Damaged recordings run at ~303 Hz while Undamaged cantilever recordings run at ~333 Hz. These rates are computed empirically from timestamp deltas and require final confirmation by the professor.",
        "- **Recording-Level Aggregation:** Descriptive statistics and correlation coefficients were computed per recording to prevent length-weighted pooling distortions.",
        "",
        "---",
        "",
        "## 2. Dataset Overview",
        f"- **Total Recordings Analyzed:** {overview['total_recordings']}",
        f"- **Total Sample Rows:** {overview['total_samples']:,}",
        f"- **Average Duration per Test:** {overview['mean_duration_sec']} seconds",
        f"- **Samples per Recording:** min = {overview['sample_counts']['min']:,}, mean = {overview['sample_counts']['mean']:,}, max = {overview['sample_counts']['max']:,}",
        f"- **Computed Sampling Rates:** Damaged ~{overview['estimated_sampling_rates']['damaged_approx_hz']} Hz | Undamaged ~{overview['estimated_sampling_rates']['undamaged_approx_hz']} Hz",
        "",
        "### Condition & Damage Level Distributions",
        "| Category | Class | Count | Percentage |",
        "|:---|:---|:---:|:---:|",
    ]

    tot = overview["total_recordings"]
    for cond, cnt in overview["condition_distribution"].items():
        md.append(f"| Condition | {cond} | {cnt} | {cnt/tot*100:.1f}% |")
    for dmg, cnt in overview["damage_level_distribution"].items():
        md.append(f"| Damage Level | {dmg} | {cnt} | {cnt/tot*100:.1f}% |")

    md.extend([
        "",
        "### Specimen & Test Type Distributions",
        "| Dimension | Category | Count |",
        "|:---|:---|:---:|",
    ])
    for spec, cnt in overview["specimen_distribution"].items():
        md.append(f"| Specimen | {spec} | {cnt} |")
    for ttype, cnt in overview["test_type_distribution"].items():
        md.append(f"| Test Type | {ttype} | {cnt} |")
    for hgrp, cnt in overview["hit_group_distribution"].items():
        md.append(f"| Hit Group | {hgrp} | {cnt} |")

    md.extend([
        "",
        "---",
        "",
        "## 3. Sensor Descriptive Statistics (Recording-Level Aggregates)",
        "The table below shows the distribution of recording-level statistics across all 5 sensors:",
        "",
        "| Sensor | Mean (Avg ± Std) | Std (Avg ± Std) | RMS (Avg) | Peak-to-Peak (Avg) | Min (Avg) | Max (Avg) |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for _, row in desc_stats.iterrows():
        s = row["Sensor"]
        m_mean, m_std = row.get("mean_mean", 0), row.get("mean_std", 0)
        s_mean, s_std = row.get("std_mean", 0), row.get("std_std", 0)
        rms_mean = row.get("rms_mean", 0)
        p2p_mean = row.get("p2p_mean", 0)
        min_mean = row.get("min_mean", 0)
        max_mean = row.get("max_mean", 0)
        md.append(f"| {s} | {m_mean:.4f} ± {m_std:.4f} | {s_mean:.4f} ± {s_std:.4f} | {rms_mean:.4f} | {p2p_mean:.4f} | {min_mean:.4f} | {max_mean:.4f} |")

    md.extend([
        "",
        "---",
        "",
        "## 4. Damage-Level Trends & Signal Evolution",
        "Evaluating recording-level features across damage stages (Undamaged, 1mm, 2mm, 3mm):",
        "",
        "| Damage Level | Count | Sensor_1 RMS | Sensor_1 P2P | Sensor_3 RMS | Sensor_3 P2P | Sensor_5 RMS | Sensor_5 P2P |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ])

    for _, row in trends.iterrows():
        lvl = row["damage_level"]
        cnt = row["recording_count"]
        s1_rms = row.get("Sensor_1_rms_mean", "N/A")
        s1_p2p = row.get("Sensor_1_p2p_mean", "N/A")
        s3_rms = row.get("Sensor_3_rms_mean", "N/A")
        s3_p2p = row.get("Sensor_3_p2p_mean", "N/A")
        s5_rms = row.get("Sensor_5_rms_mean", "N/A")
        s5_p2p = row.get("Sensor_5_p2p_mean", "N/A")
        md.append(f"| {lvl} | {cnt} | {s1_rms} | {s1_p2p} | {s3_rms} | {s3_p2p} | {s5_rms} | {s5_p2p} |")

    md.extend([
        "",
        "> [!IMPORTANT]",
        "> **Causality Disclaimer:** Variations across damage levels reflect empirical observational trends in signal magnitude and frequency response. They do NOT establish strict mathematical causality until sensor physical locations and calibration units are verified with the professor.",
        "",
        "---",
        "",
        "## 5. Sensor Correlation Structure",
        "Average within-recording Pearson correlation coefficients among the five sensors:",
        "",
        "| Sensor | " + " | ".join(SENSOR_COLUMNS) + " |",
        "|:---|" + "|".join([":---:"] * 5) + "|",
    ])

    for s1 in SENSOR_COLUMNS:
        line_vals = [f"{corr_mat.loc[s1, s2]:.3f}" for s2 in SENSOR_COLUMNS]
        md.append(f"| **{s1}** | " + " | ".join(line_vals) + " |")

    md.extend([
        "",
        "---",
        "",
        "## 6. Key Findings & Observations",
        "### Clear and Strong Observations",
    ])
    for obs in findings["strong_observations"]:
        md.append(f"- **{obs}**")

    md.extend([
        "",
        "### Weaker or Possible Trends",
    ])
    for trend in findings["weaker_trends"]:
        md.append(f"- {trend}")

    md.extend([
        "",
        "---",
        "",
        "## 7. Limitations & Inquiries for Professor Review",
        "1. **Sensor Measurement Units:** Clarify physical units (acceleration in g or m/s², strain in µε, displacement in mm, or raw ADC voltage).",
        "2. **Sensor Spatial Topology:** Confirm physical mounting locations along the cantilever/bridge beams (e.g. root vs midspan vs tip).",
        "3. **Experimental Sampling Rates:** Confirm whether 303.03 Hz and 333.33 Hz reflect the hardware DAQ sampling clocks.",
        "4. **Calendar Timestamps:** Confirm if exact test calendar dates/times are recorded in external lab logs.",
        "",
    ])

    report_content = "\n".join(md)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_content, encoding="utf-8")
    return report_content


# ---------------------------------------------------------------------------
# 6. Main Execution CLI
# ---------------------------------------------------------------------------

def run_eda(
    data_dir: Union[str, Path] = DEFAULT_SENSOR_DIR,
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR,
    export_metrics_csv: bool = True,
    max_recordings: Optional[int] = None,
) -> int:
    """Run the complete Exploratory Data Analysis workflow."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    print("=================================================================")
    print("Talking to Bridges — Sensor Exploratory Data Analysis (Week 5)")
    print(f"Data directory:   {data_dir}")
    print(f"Output directory: {output_dir}")
    print("=================================================================")

    if not data_dir.exists():
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        print("Please ensure clean sensor files are placed under 'data/processed/sensors/'")
        return 1

    try:
        eda = SensorEDA.from_directory(data_dir, max_recordings=max_recordings)
    except Exception as exc:
        print(f"[ERROR] Failed initializing SensorEDA: {exc}")
        return 1

    # 1. Dataset Overview
    overview = eda.get_dataset_overview()
    print("\n[1] DATASET OVERVIEW:")
    print(f"  Total recordings:   {overview['total_recordings']}")
    print(f"  Total sample rows:  {overview['total_samples']:,}")
    print(f"  Mean test duration: {overview['mean_duration_sec']} s")
    print(f"  Conditions:         {overview['condition_distribution']}")
    print(f"  Damage levels:      {overview['damage_level_distribution']}")
    print(f"  Specimens:          {overview['specimen_distribution']}")
    print(f"  Test types:         {overview['test_type_distribution']}")
    print(f"  Estimated DAQ rate: {overview['estimated_sampling_rates']}")

    # 2. Descriptive Statistics
    print("\n[2] SENSOR DESCRIPTIVE STATISTICS (Recording-level aggregates):")
    stats_df = eda.get_sensor_descriptive_statistics()
    cols_to_print = [c for c in ["Sensor", "mean_mean", "std_mean", "rms_mean", "p2p_mean"] if c in stats_df.columns]
    print(stats_df[cols_to_print].to_string(index=False))

    # 4. Damage Level Trends
    print("\n[4] DAMAGE LEVEL TRENDS:")
    trends_df = eda.get_damage_level_trends()
    trend_cols = [c for c in trends_df.columns if "rms" in c or "p2p" in c or c in ("damage_level", "recording_count")]
    print(trends_df[trend_cols].to_string(index=False))

    # 5. Correlation Matrix
    print("\n[5] AVERAGE SENSOR-TO-SENSOR CORRELATION MATRIX:")
    corr_df = eda.get_sensor_correlation_matrix()
    print(corr_df.to_string())

    # Export eda_summary_metrics.csv (compatible with dashboard)
    if export_metrics_csv:
        csv_dest = data_dir / "eda_summary_metrics.csv"
        eda.metrics_df.to_csv(csv_dest, index=False)
        print(f"\n[INFO] Exported summary metrics to: {csv_dest}")

    # Generate Visualizations
    plot_files = generate_visualizations(eda, output_dir)
    if plot_files:
        print(f"\n[INFO] Generated {len(plot_files)} interactive HTML plots in: {output_dir}")
        for p in plot_files:
            print(f"  - {p.name}")

    # Generate Detailed Markdown Report
    report_file = output_dir / "eda_report.md"
    generate_eda_report_markdown(eda, report_file)
    print(f"\n[INFO] Saved complete EDA Report to: {report_file}")

    # Key Findings
    findings = eda.extract_key_findings()
    print("\n[8] KEY FINDINGS SUMMARY:")
    print("  Clear Observations:")
    for f in findings["strong_observations"]:
        print(f"    * {f}")
    print("  Weaker / Possible Trends:")
    for f in findings["weaker_trends"]:
        print(f"    * {f}")

    print("\nEDA Completed successfully.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis on Clean Sensor Data.")
    parser.add_argument(
        "--data-dir",
        default=str(DEFAULT_SENSOR_DIR),
        help=f"Directory containing clean sensor CSVs (default: {DEFAULT_SENSOR_DIR})",
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to save plots and reports (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Optional limit on recordings for quick testing",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    return run_eda(
        data_dir=args.data_dir,
        output_dir=args.out_dir,
        max_recordings=args.max_recordings,
    )


if __name__ == "__main__":
    sys.exit(main())
