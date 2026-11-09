"""Excel upload + Data Audit UI for Talking-to-Bridges (Kolla – pre-training prep).

Accepts .xlsx and .xlsm files, cleans them with the Week 5 pipeline
(analysis/sensor_pipeline.py) and renders the five audit tables required
before training:
  1. Files per condition (Damaged / Undamaged)
  2. Specimen & Damage Level cross-table
  3. Sampling-rate consistency check (must be the same 5 sensors @ same rate)
  4. Skipped / failed file log
  5. Train / test split recommendation by specimen
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import streamlit as st

# Import pipeline symbols – handled gracefully so the rest of the app still
# loads even if the analysis package is not on sys.path.
try:
    from analysis.sensor_pipeline import (
        FileReport,
        SkippedFileError,
        SensorDataError,
        clean_sensor_bytes,
        SENSOR_COLUMNS,
    )
    _PIPELINE_OK = True
except ImportError:
    _PIPELINE_OK = False
    FileReport = None
    SkippedFileError = Exception
    SensorDataError = Exception
    clean_sensor_bytes = None
    SENSOR_COLUMNS = [f"Sensor_{i}" for i in range(1, 6)]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry-point called from frontend/app.py
# ---------------------------------------------------------------------------

def render_data_audit() -> None:
    """Render the Excel-upload + data-audit section inside the Streamlit app."""

    st.markdown("## 📂 Data Upload & Training Prep Audit")
    st.caption("Upload the raw professor Excel files (.xlsx / .xlsm). "
               "The Week 5 pipeline cleans them and generates the five audit "
               "tables needed before training.")

    if not _PIPELINE_OK:
        st.error(
            "⚠️ `analysis.sensor_pipeline` could not be imported. "
            "Make sure the repository root is on `sys.path` and all "
            "dependencies are installed (`pip install -r requirements.txt`)."
        )
        return

    # --- File uploader --------------------------------------------------
    uploaded_files = st.file_uploader(
        "Upload raw sensor recordings",
        type=["xlsx", "xlsm"],
        accept_multiple_files=True,
        help="You may select multiple files at once. "
             "Files that are wide summary tables (not raw recordings) are "
             "automatically detected and reported as skipped.",
    )

    if not uploaded_files:
        st.info("👆 Upload one or more Excel files to begin the audit.")
        return

    if not st.button("▶ Run Pipeline & Generate Audit", type="primary"):
        return

    # --- Run pipeline ---------------------------------------------------
    reports: list[FileReport] = []
    clean_dfs: list[pd.DataFrame | None] = []

    progress = st.progress(0, text="Starting pipeline…")
    total = len(uploaded_files)

    for idx, f in enumerate(uploaded_files, start=1):
        progress.progress(idx / total, text=f"Processing {f.name} ({idx}/{total})…")
        file_bytes = f.read()

        try:
            # condition=None → pipeline auto-detects from file content (layout).
            # This is correct: a bare .xlsx filename does NOT reliably indicate
            # the condition; the column structure does.
            clean_df, report = clean_sensor_bytes(
                file_bytes, f.name, condition=None
            )
            reports.append(report)
            clean_dfs.append(clean_df)

        except SkippedFileError as exc:
            # Wide summary tables are deliberately skipped, not errors.
            log.warning("Skipped %s: %s", f.name, exc)
            r = FileReport(
                source_file=f.name,
                status="skipped",
                message=str(exc),
            )
            reports.append(r)
            clean_dfs.append(None)

        except SensorDataError as exc:
            log.error("Pipeline error for %s: %s", f.name, exc)
            r = FileReport(
                source_file=f.name,
                status="error",
                message=str(exc),
            )
            reports.append(r)
            clean_dfs.append(None)

        except Exception as exc:  # unexpected – still report, don't crash
            log.exception("Unexpected error for %s", f.name)
            r = FileReport(
                source_file=f.name,
                status="error",
                message=f"Unexpected: {exc}",
            )
            reports.append(r)
            clean_dfs.append(None)

    progress.empty()

    if not reports:
        st.warning("No files were processed.")
        return

    st.success(f"Pipeline finished — {len(reports)} file(s) processed.")

    # Build the master audit DataFrame
    audit_df = pd.DataFrame([r.to_dict() for r in reports])

    # Collect per-file specimen metadata (only from successful runs)
    specimen_rows: list[dict] = []
    for df in clean_dfs:
        if df is not None and not df.empty:
            row: dict = {
                "Specimen": df["Specimen"].iloc[0] if "Specimen" in df.columns else "",
                "Damage_Level": df["Damage_Level"].iloc[0] if "Damage_Level" in df.columns else "",
                "Condition": df["Condition"].iloc[0] if "Condition" in df.columns else "",
                "Test_Type": df["Test_Type"].iloc[0] if "Test_Type" in df.columns else "",
            }
            specimen_rows.append(row)

    # ===================================================================
    # AUDIT SECTION 1 – Files per condition
    # ===================================================================
    st.markdown("---")
    st.markdown("### 1️⃣ Files per Condition")
    ok_df = audit_df[audit_df["status"] == "ok"]
    if ok_df.empty:
        st.warning("No files were successfully cleaned.")
    else:
        cond_counts = ok_df["condition"].value_counts(dropna=False).reset_index()
        cond_counts.columns = ["Condition", "File Count"]
        st.dataframe(cond_counts, width='content')

    # ===================================================================
    # AUDIT SECTION 2 – Damage level & specimen cross-table
    # ===================================================================
    st.markdown("### 2️⃣ Specimen × Damage Level")
    if specimen_rows:
        spec_df = pd.DataFrame(specimen_rows)
        cross = (
            spec_df.groupby(["Specimen", "Damage_Level", "Condition"])
            .size()
            .reset_index(name="File Count")
        )
        st.dataframe(cross, width='stretch')
    else:
        st.info("No specimen metadata found. Make sure the file paths contain "
                "folder names like '1mm', 'M1', 'M2', etc.")

    # ===================================================================
    # AUDIT SECTION 3 – Sampling rate & sensor consistency
    # ===================================================================
    st.markdown("### 3️⃣ Sensor & Sampling-Rate Consistency")
    if "sampling_rate_hz" in audit_df.columns:
        rate_counts = (
            ok_df["sampling_rate_hz"]
            .value_counts(dropna=False)
            .reset_index()
        )
        rate_counts.columns = ["Sampling Rate (Hz)", "File Count"]
        st.dataframe(rate_counts, width='content')

        n_rates = ok_df["sampling_rate_hz"].nunique(dropna=True)
        if n_rates == 1:
            st.success("✅ All files share the same sampling rate — consistent.")
        elif n_rates > 1:
            st.warning(
                f"⚠️ {n_rates} different sampling rates detected. "
                "Ensure you re-sample to a common rate before training."
            )

    # Each clean file always has exactly SENSOR_COLUMNS (guaranteed by pipeline)
    st.caption(f"All clean files contain the standard 5 sensor columns: "
               f"{', '.join(SENSOR_COLUMNS)}")

    # ===================================================================
    # AUDIT SECTION 4 – Skipped / failed files
    # ===================================================================
    st.markdown("### 4️⃣ Skipped / Failed Files")
    bad_df = audit_df[audit_df["status"] != "ok"]
    if bad_df.empty:
        st.success("✅ All files were processed successfully — no skips or errors.")
    else:
        st.warning(f"{len(bad_df)} file(s) were not fully processed:")
        st.dataframe(
            bad_df[["source_file", "status", "message"]].reset_index(drop=True),
            width='stretch',
        )

    # ===================================================================
    # AUDIT SECTION 5 – Train / test split recommendation by specimen
    # ===================================================================
    st.markdown("### 5️⃣ Train / Test Split Recommendation (by Specimen)")
    if specimen_rows:
        unique_specimens = spec_df["Specimen"].dropna().unique().tolist()
        unique_specimens = [s for s in unique_specimens if s]  # drop blanks

        if len(unique_specimens) >= 2:
            test_specimen = unique_specimens[-1]
            train_specimens = unique_specimens[:-1]
            st.markdown(
                f"- **Train specimens:** `{', '.join(train_specimens)}`\n"
                f"- **Test specimen:** `{test_specimen}`"
            )
            st.info(
                "🔒 The test specimen is held-out entirely. "
                "The model will never see its data during training, "
                "preventing data leakage and giving a fair generalisation estimate."
            )
        elif len(unique_specimens) == 1:
            st.warning(
                f"Only one specimen found (`{unique_specimens[0]}`). "
                "Upload files from at least two different specimens to get "
                "a proper specimen-based train/test split."
            )
        else:
            st.info(
                "Specimen names could not be parsed from the file paths. "
                "Ensure your files are inside folders named M1, M2, 1st, 2nd, etc."
            )
    else:
        st.info("No successful files — split recommendation unavailable.")
