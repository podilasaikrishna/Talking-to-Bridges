# PRE-1: Dataset Inspection and Time Validation Report

**Project:** Talking to Bridges  
**Author:** Pavan  
**Task:** PRE-1 (Dataset Inspection and Time Validation)  
**Authoritative Plan Reference:** `Talking_to_Bridges_Revised_Plan_v2.pdf`  
**Date:** November 2026  
**Status:** Complete  

---

## Executive Summary

As part of Task PRE-1, an exhaustive, independent dataset inspection and time-quality audit was performed across all **464** files in the repository's `data/` directory.

Key empirical findings:
1. **Total Files:** Exactly 464 non-temporary Excel (`.xlsx`) files exist in the dataset:
   - **460 raw dynamic recordings** (420 Damaged, 40 Undamaged).
   - **4 wide summary tables** (3,001–3,002 columns: `1mm.xlsx`, `2mm.xlsx`, `3mm.xlsx`, and `Processed Data.xlsx`), which are deliberately skipped by the raw ingestion pipeline.
   - **0 failed or unreadable files**.
2. **Sampling Frequency Discrepancy Verified:**
   - **All 420 Damaged recordings** exhibit an empirical relative time step of exactly $\Delta t = 0.0033\text{ s}$ ($\text{std} \approx 0$), resulting in an estimated sampling rate of **$303.0303\text{ Hz}$** ($1 / 0.0033$).
   - **All 40 Undamaged recordings** exhibit an empirical relative time step of exactly $\Delta t = 0.0030\text{ s}$ ($\text{std} \approx 0$), resulting in an estimated sampling rate of **$333.3333\text{ Hz}$** ($1 / 0.0030$).
   - The research paper cites a nominal experimental rate of **$300\text{ Hz}$** ($\Delta t = 1/300 \approx 0.003333...\text{ s}$).
   - **Root-Cause Hypothesis:** The discrepancy between 303.03 Hz and 333.33 Hz is consistent with software-level decimal rounding/formatting of a nominal 300 Hz sampling period (e.g., `round(1/300, 4) = 0.0033` vs. `round(1/300, 3) = 0.003`). Because the hardware clock configuration is not in the repository, this remains an open question requiring professor confirmation.
   - **Data Leakage Prohibition:** Because sampling rate cleanly and perfectly separates Damaged from Undamaged recordings, `sampling_rate_hz` and `dt` are strictly excluded from machine learning feature vectors.
3. **Data Quality & Problematic Files:**
   - **454 recordings** are completely strictly increasing and uniform with 0 missing, 0 duplicate, and 0 irregular intervals.
   - **6 recordings** are flagged as `irregular` due to non-numeric header cells (`'->'`) in the relative time column for the first 14–17 rows. Once those non-numeric header rows are dropped, the subsequent relative times start cleanly at $0.0000\text{ s}$ with $\Delta t = 0.0033\text{ s}$.
   - **0 recordings** contain out-of-order timestamps ($\Delta t < 0$), duplicate timestamps ($\Delta t = 0$), or large discontinuities/gaps ($\Delta t > 1.5 \times \text{median}$).
4. **Handoff for Eswar (PRE-2):**
   - In `analysis/features.py`, FFT frequencies currently assume a hardcoded `1 / 303.0` sampling interval (`rfftfreq(len(arr), 1 / 303.0)`). For PRE-2, Eswar must make frequency extraction dynamic and decouple FFT modal extraction from fixed sampling assumptions.

---

## 1. Dataset Structure and Raw File Formats

The raw dataset files are organized under two root directories:
- `data/Damaged_Beams/` (423 files: 420 recordings + 3 summary tables)
- `data/Undamaged_Data_of_2_cantilever/` (41 files: 40 recordings + 1 summary table)

### Raw Layouts

The raw recordings follow two distinct layouts, both headerless Excel files:

| Layout Name | Raw Column Count | Column Structure | Primary Occurrence |
| :--- | :---: | :--- | :--- |
| **9-Column Layout** | 9 (or 10–12 with notes) | Col 0: Hour<br>Col 1: Minute<br>Col 2: Second<br>Col 3: `Relative_Time_Sec`<br>Cols 4–8: Sensors 1–5 | `Undamaged_Data_of_2_cantilever/1st` (20 files) |
| **7-Column Layout** | 7 | Col 0: Absolute Time String (`HH:MM:SS.ffffff`)<br>Col 1: `Relative_Time_Sec`<br>Cols 2–6: Sensors 1–5 | `Damaged_Beams/` (all 420 files)<br>`Undamaged_Data_of_2_cantilever/2nd` (all 20 files) |
| **Wide Summary Tables** | > 3,000 | 1 row per sensor, 1 column per time sample | `1mm.xlsx`, `2mm.xlsx`, `3mm.xlsx`, `Processed Data.xlsx` |

> [!IMPORTANT]
> **Critical Layout Finding:** Raw layout does **NOT** determine damage condition.
> - While all 420 Damaged recordings use the 7-column layout, the Undamaged 2nd cantilever recordings (20 files) **also use the 7-column layout**.
> - The condition must always be inferred from verified folder/path metadata (`Damaged_Beams` vs. `Undamaged_Data_of_2_cantilever`), never solely from the column count.

---

## 2. Sensor-Column Mapping

Both layouts provide 5 synchronized dynamic sensor channels representing physical sensors along the beam:

| Raw 7-Column Layout | Raw 9-Column Layout | Standardized Column Name | Description |
| :---: | :---: | :---: | :--- |
| Column 2 (Index 2) | Column 4 (Index 4) | `Sensor_1` | Dynamic sensor channel 1 (uncalibrated units) |
| Column 3 (Index 3) | Column 5 (Index 5) | `Sensor_2` | Dynamic sensor channel 2 (uncalibrated units) |
| Column 4 (Index 4) | Column 6 (Index 6) | `Sensor_3` | Dynamic sensor channel 3 (uncalibrated units) |
| Column 5 (Index 5) | Column 7 (Index 7) | `Sensor_4` | Dynamic sensor channel 4 (uncalibrated units) |
| Column 6 (Index 6) | Column 8 (Index 8) | `Sensor_5` | Dynamic sensor channel 5 (uncalibrated units) |

- **Extra Columns:** Several undamaged files contain stray columns after column 8 (e.g., Col 11 containing notes such as `"Single hit between S1-S2"`). These are safely ignored by the pipeline and logged in the audit report.
- **Physical Calibration:** Physical engineering units (e.g., $g$, $\text{m/s}^2$, $\mu\varepsilon$, or $\text{mm}$) remain unconfirmed by the professor; no arbitrary unit assumptions are made.

---

## 3. Dataset Counts by Condition, Damage Level, Specimen, and Test Type

All counts represent **recordings**, not individual sensor time steps.

### A. Condition Distribution

| Condition | Recordings | Total Samples | Mean Duration (s) | Sampling Rate (Hz) |
| :--- | :---: | :---: | :---: | :---: |
| **Damaged** | 420 | 5,653,583 | 44.42 | 303.0303 |
| **Undamaged** | 40 | 334,133 | 25.06 | 333.3333 |
| **Summary Tables (Skipped)** | 4 | 764 | N/A | N/A |
| **Total** | **464** | **5,988,480** | **42.73** | — |

### B. Damage Level Distribution

| Damage Level | Condition | Recordings | Percentage of Dataset |
| :--- | :--- | :---: | :---: |
| **None** | Undamaged | 40 | 8.7% |
| **1mm** | Damaged | 140 | 30.4% |
| **2mm** | Damaged | 140 | 30.4% |
| **3mm** | Damaged | 140 | 30.4% |
| **Total Valid/Irregular** | — | **460** | **100.0%** |

### C. Specimen Distribution

| Specimen Identifier | Condition | Damage Levels | Recordings | Test Types per Specimen |
| :--- | :--- | :--- | :---: | :--- |
| **1st** | Undamaged | None | 20 | Singlehit (4), Displacement (5), Randomhit (4), Multihit (7) |
| **2nd** | Undamaged | None | 20 | Singlehit (4), Displacement (5), Randomhit (4), Multihit (7) |
| **M1** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M2** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M3** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M4** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M5** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M6** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **M7** | Damaged | 1mm, 2mm, 3mm | 60 | 20 tests × 3 damage levels |
| **Unknown (Summary Tables)** | N/A | None | 4 | Wide summary workbooks |

> [!NOTE]
> **Specimen Naming Ambiguity:** Undamaged specimens are labelled `"1st"` and `"2nd"` (representing two physical cantilever test specimens), whereas damaged specimens are labelled `"M1"` through `"M7"`. Whether `M1`–`M7` correspond to distinct physical beams or sensor/cut configurations must be clarified with the professor before finalizing `case_id` mappings in PRE-3.

### D. Test Type Distribution

Each of the 23 experimental configurations (2 Undamaged + 21 Damaged [7 specimens × 3 damage levels]) contains exactly 20 test recordings:

| Test Type | Hit Group / Subtype | Damaged Count | Undamaged Count | Total Recordings |
| :--- | :--- | :---: | :---: | :---: |
| **Displacement** | 1cm, 2cm, 3cm, 4cm, 5cm | 105 (21 × 5) | 10 (2 × 5) | 115 |
| **Singlehit** | S1-S2, S2-S3, S3-S4, S4-S5 | 84 (21 × 4) | 8 (2 × 4) | 92 |
| **Randomhit** | RH 1-3-4, 2-4, 3-2-4, 4-1-2 | 84 (21 × 4) | 8 (2 × 4) | 92 |
| **Multihit** | 2-Hit (4), 3-Hit (2), 4-Hit (1) | 147 (21 × 7) | 14 (2 × 7) | 161 |
| **Total** | — | **420** | **40** | **460** |

---

## 4. Time-Quality Validation Methodology

### Ground-Truth Time Axis
- `Relative_Time_Sec` is established as the **sole ground-truth time coordinate**.
- The synthesized `DateTime` column is **not** used for time differences or sampling rates because it relies on an artificial date anchor (`1970-01-01`).

### Mathematical Audit Rules
For every raw recording, the raw relative time column $t = [t_0, t_1, \dots, t_{N-1}]$ is inspected:

1. **Consecutive Time Difference:**
   $$\Delta t_i = t_i - t_{i-1}, \quad \text{for } i = 1, \dots, N-1$$
2. **Problematic Interval Tracking:**
   - Zero intervals (duplicate timestamps): $\Delta t_i = 0$
   - Negative intervals (out-of-order timestamps): $\Delta t_i < 0$
   - Valid positive intervals: $\Delta t_{\text{valid}} = \{ \Delta t_i \mid \Delta t_i > 0 \}$
3. **Sampling Interval Statistics:**
   - Median interval: $\widetilde{\Delta t} = \text{median}(\Delta t_{\text{valid}})$
   - Mean interval: $\mu_{\Delta t} = \frac{1}{M}\sum \Delta t_{\text{valid}}$
   - Standard deviation: $\sigma_{\Delta t} = \sqrt{\frac{1}{M-1}\sum (\Delta t_{\text{valid}} - \mu_{\Delta t})^2}$
   - Coefficient of Variation (CV%): $\text{CV} = \frac{\sigma_{\Delta t}}{\mu_{\Delta t}} \times 100\%$
   - Estimated Sampling Frequency:
     $$f_s = \frac{1}{\widetilde{\Delta t}}$$
4. **Irregularity & Gap Definitions:**
   - **Unusually Large Gap:** Defined as any interval $\Delta t_i > 1.5 \times \widetilde{\Delta t}$ (representing a missed sample period $\ge 50\%$).
   - **Irregular Interval:** Defined as any interval where $|\Delta t_i - \widetilde{\Delta t}| > 0.05 \times \widetilde{\Delta t}$ (deviation $> 5\%$).
   - **Strict Monotonicity:** Strictly increasing if and only if zero count $= 0$ and negative count $= 0$.
   - **Uniform Sampling:** Uniform if and only if strictly increasing, large gap count $= 0$, and irregular interval count $= 0$.

---

## 5. Per-Recording Sampling Interval and Frequency Findings

Across all 460 raw dynamic recordings:

| Metric | Damaged Recordings ($N=420$) | Undamaged Recordings ($N=40$) |
| :--- | :---: | :---: |
| **Nominal Interval ($\Delta t$)** | **$0.0033\text{ s}$** | **$0.0030\text{ s}$** |
| **Minimum Interval** | $0.003299999999995862\text{ s}$ | $0.002999999999996561\text{ s}$ |
| **Maximum Interval** | $0.003300000000010073\text{ s}$ | $0.003000000000000114\text{ s}$ |
| **Standard Deviation ($\sigma_{\Delta t}$)** | $< 1.0 \times 10^{-15}\text{ s}$ (machine precision) | $< 1.0 \times 10^{-15}\text{ s}$ (machine precision) |
| **Coefficient of Variation (CV%)** | $0.0000\%$ | $0.0000\%$ |
| **Estimated Sampling Rate ($f_s$)** | **$303.0303\text{ Hz}$** | **$333.3333\text{ Hz}$** |
| **Sampling Rate Std ($\sigma_{f_s}$)** | **$0.000000\text{ Hz}$** | **$0.000000\text{ Hz}$** |

Every single recording within the Damaged group is uniformly spaced at 303.0303 Hz. Every single recording within the Undamaged group is uniformly spaced at 333.3333 Hz. There is zero within-file jitter or rate drift.

---

## 6. Investigation of the 300 Hz vs 303 Hz vs 333 Hz Discrepancy

### Evidence Analysis

1. **Experimental Paper Statement:**
   The research paper states that experimental acceleration/displacement data was collected at **300 Hz**.
2. **Observed In-Data Values:**
   - Undamaged files: $\Delta t = 0.003\text{ s} \implies f_s = \frac{1}{0.003} = 333.3333\text{ Hz}$.
   - Damaged files: $\Delta t = 0.0033\text{ s} \implies f_s = \frac{1}{0.0033} = 303.0303\text{ Hz}$.
3. **Decimal Truncation / Rounding Analysis:**
   If the physical DAQ hardware was configured to sample at $300\text{ Hz}$, the theoretical time increment is:
   $$\Delta t_{\text{true}} = \frac{1}{300} = 0.003333333333333...\text{ s}$$
   - When rounded/truncated to **3 decimal places**:
     $$\text{round}\left(\frac{1}{300}, 3\right) = 0.003\text{ s} \implies \frac{1}{0.003} = 333.3333\text{ Hz}$$
   - When rounded/truncated to **4 decimal places**:
     $$\text{round}\left(\frac{1}{300}, 4\right) = 0.0033\text{ s} \implies \frac{1}{0.0033} = 303.0303\text{ Hz}$$
4. **Alternative Plausible Explanations:**
   - **Clock Configuration:** The DAQ hardware was set to an integer millisecond period (e.g., $3.0\text{ ms} = 333.33\text{ Hz}$ for undamaged cantilever tests, and $3.3\text{ ms} = 303.03\text{ Hz}$ for damaged beam tests).
   - **Export Driver Software:** Different versions or export presets of the DAQ software (or Excel exporter) were used for the two test batches.
5. **Operational Rule for ML & Feature Engineering:**
   - **Strict Data Leakage Constraint:** Because $f_s$ provides 100% discriminatory power between Damaged and Undamaged classes due to experimental/export artifacts, **neither $f_s$ nor $\Delta t$ may be used as a feature in any model**.
   - Raw recordings must **not** be artificially resampled or forced to 300 Hz in this phase.

---

## 7. Missing, Duplicate, Invalid, and Irregular Data Findings

Across the 460 recordings:
- **Duplicate Timestamps ($\Delta t = 0$):** 0 across all files.
- **Out-of-Order Rows ($\Delta t < 0$):** 0 across all files.
- **Discontinuities / Large Gaps ($\Delta t > 1.5 \times \widetilde{\Delta t}$):** 0 across all files.
- **Invalid / Missing Sensor Readings:** 0 within valid data regions.
- **Flat Sensors (dead channels):** 0 flat sensor channels found across the entire dataset.

---

## 8. List of Problematic Recordings

Six recordings were identified with non-numeric header cells in the Relative Time column:

| # | Source File | Condition | Damage Level | Specimen | Test Type | Invalid Time Rows | Nature of Defect |
| :-: | :--- | :--- | :---: | :---: | :--- | :---: | :--- |
| 1 | `data/Damaged_Beams/1mm/M2/Displacement/3cm.xlsx` | Damaged | 1mm | M2 | Displacement | 14 | Initial 14 rows contain string `'->'` in Col 1 |
| 2 | `data/Damaged_Beams/1mm/M7/Randomhit/D-RH 3-2-4.xlsx` | Damaged | 1mm | M7 | Randomhit | 17 | Initial 17 rows contain string `'->'` in Col 1 |
| 3 | `data/Damaged_Beams/2mm/M4/Multihit/3hit/D-MH 2-3-4-5.xlsx` | Damaged | 2mm | M4 | Multihit | 14 | Initial 14 rows contain string `'->'` in Col 1 |
| 4 | `data/Damaged_Beams/2mm/M6/Displacement/1cm.xlsx` | Damaged | 2mm | M6 | Displacement | 16 | Initial 16 rows contain string `'->'` in Col 1 |
| 5 | `data/Damaged_Beams/2mm/M6/Multihit/3hit/D-MH 1-2-3-4.xlsx` | Damaged | 2mm | M6 | Multihit | 15 | Initial 15 rows contain string `'->'` in Col 1 |
| 6 | `data/Damaged_Beams/2mm/M6/Singlehit/D-SH 2-3.xlsx` | Damaged | 2mm | M6 | Singlehit | 15 | Initial 15 rows contain string `'->'` in Col 1 |

### Handling & Pipeline Resilience
- In all 6 files, the `'->'` characters precede the start of the experimental recording.
- Once coerced to NaN and dropped by `analysis/sensor_pipeline.py`, the remaining rows start at relative time $0.0000\text{ s}$ and proceed uniformly at $303.0303\text{ Hz}$ without data loss.
- These 6 files are tracked in `data/audit/pre1_recording_time_audit.csv` with status `irregular` and explicit notes.

### Deliberately Skipped Files (Wide Summary Tables)
The following 4 workbooks are summary matrices (one row per sensor, ~3,000 columns per recording), not raw continuous recordings:
1. `data/Damaged_Beams/1mm.xlsx` (3,002 columns)
2. `data/Damaged_Beams/2mm.xlsx` (3,001 columns)
3. `data/Damaged_Beams/3mm.xlsx` (3,001 columns)
4. `data/Undamaged_Data_of_2_cantilever/Processed Data.xlsx` (3,001 columns)

These are categorized as `skipped` in accordance with repository design.

---

## 9. Comparison Against the Existing Pipeline & EDA Report

| Evaluation Dimension | Existing `sensor_pipeline.py` | Independent Time Audit (`analysis/time_audit.py`) | Status / Assessment |
| :--- | :--- | :--- | :--- |
| **Sampling Rate** | Computes $1 / \text{median}(\text{steps})$ rounded to 3 decimals | Computes $1 / \text{median}(\Delta t_{\text{valid}})$ to 4 decimals directly on raw deltas | Consistent ($303.03\text{ Hz}$ vs $333.33\text{ Hz}$) |
| **Sampling Variation** | Not calculated | Mean, std, min, max, CV% calculated | **Enhanced** in audit module |
| **Gaps & Discontinuities** | `max_gap_sec` only | `max_gap_sec` + count of gaps $> 1.5 \times \text{median}$ | **Enhanced** in audit module |
| **Out-of-Order** | Boolean `was_out_of_order` | Exact negative step count ($\Delta t < 0$) | **Enhanced** in audit module |
| **Duplicate Timestamps** | Rows dropped as duplicate | Exact duplicate step count ($\Delta t = 0$) | Consistent |
| **Invalid Time Cells** | Aggregated with invalid sensor cells | Separated into invalid time vs invalid sensor cells | **Enhanced** in audit module |
| **EDA Report Alignment** | Cites ~303 Hz (damaged) and ~333 Hz (undamaged) | Proves 100% of files have 0 std dev and exact rates | Confirmed & validated |

---

## 10. Unresolved Questions for the Professor

1. **Hardware DAQ Sampling Clock:**  
   What was the actual clock frequency configured on the data acquisition system?
   - Was the hardware DAQ set to $300\text{ Hz}$, with Excel/software export rounding intervals to 3 decimals ($0.003\text{ s}$) for undamaged tests and 4 decimals ($0.0033\text{ s}$) for damaged tests?
   - Or were the undamaged cantilever tests physically recorded at $333.33\text{ Hz}$ ($3\text{ ms}$) and the damaged beam tests at $303.03\text{ Hz}$ ($3.3\text{ ms}$)?
2. **Physical Sensor Units:**  
   What are the engineering physical units for `Sensor_1` through `Sensor_5` (e.g., acceleration in $g$ or $\text{m/s}^2$, strain in $\mu\varepsilon$, displacement in $\text{mm}$, or raw ADC voltage)?
3. **Sensor Layout Topology:**  
   What are the physical mounting positions of sensors 1 to 5 along the beam (e.g., sensor 1 near clamp/fixed end, sensor 5 at cantilever tip)?
4. **Specimen M1–M7 Identity:**  
   Are `M1` through `M7` 7 distinct physical aluminum/steel beams, or a single specimen tested under 7 different boundary/mass/cut conditions?

---

## 11. Handoff Notes for Eswar's PRE-2 Work

> [!WARNING]
> **PRE-2 Handoff Issue — Hardcoded 303 Hz Assumption in Feature Extraction:**
> In `analysis/features.py`, line 47 currently contains:
> ```python
> xf = rfftfreq(len(arr), 1 / 303.0)  # Assuming 303Hz sample rate from previous EDA
> ```
> And line 15 assumes window size 300 corresponds to ~1 second at 303 Hz.
>
> **Action Required for Eswar (PRE-2):**
> 1. Undamaged recordings run at **333.33 Hz**, not 303 Hz. Applying `1 / 303.0` to undamaged signals induces a frequency scaling error of approximately $+10\%$:
>    $$f_{\text{calc}} = f_{\text{true}} \times \frac{303.03}{333.33} \approx 0.909 \times f_{\text{true}}$$
> 2. PRE-2 must accept the sampling interval dynamically (or retrieve it from the file audit metadata / `Relative_Time_Sec`), rather than hardcoding 303 Hz.
> 3. Modal frequency extraction ($f_1, f_2, f_3$) must be evaluated with window lengths appropriate to the actual sampling frequency of each recording.

