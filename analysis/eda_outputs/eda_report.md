# Exploratory Data Analysis Report: Bridge & Cantilever Sensor Dataset
**Phase 2 — Week 5 | Talking to Bridges Project**
**Author:** Krishna (Exploratory Data Analysis & Trends/Correlations)
**Date Generated:** 2026-10-09 13:36:02

---

## 1. Executive Summary & Critical Data Interpretations
This exploratory data analysis was conducted on the cleaned sensor dataset produced by the ingestion pipeline.
Key operational rules applied throughout this analysis:
- **Time Axis:** `Relative_Time_Sec` is the sole ground-truth time coordinate. `DateTime` contains an artificial `1970-01-01` anchor date and was not treated as a calendar timestamp.
- **Physical Units:** The physical measurement units of `Sensor_1` through `Sensor_5` are currently uncalibrated and unconfirmed; no physical units (such as mm, m/s², or µε) are assumed.
- **Sampling Frequencies:** Damaged recordings run at ~303 Hz while Undamaged cantilever recordings run at ~333 Hz. These rates are computed empirically from timestamp deltas and require final confirmation by the professor.
- **Recording-Level Aggregation:** Descriptive statistics and correlation coefficients were computed per recording to prevent length-weighted pooling distortions.

---

## 2. Dataset Overview
- **Total Recordings Analyzed:** 257
- **Total Sample Rows:** 3,074,613
- **Average Duration per Test:** 39.09 seconds
- **Samples per Recording:** min = 3,946, mean = 11,963.5, max = 29,324
- **Computed Sampling Rates:** Damaged ~303.0 Hz | Undamaged ~333.3 Hz

### Condition & Damage Level Distributions
| Category | Class | Count | Percentage |
|:---|:---|:---:|:---:|
| Condition | Damaged | 217 | 84.4% |
| Condition | Undamaged | 40 | 15.6% |
| Damage Level | 1mm | 140 | 54.5% |
| Damage Level | 2mm | 77 | 30.0% |
| Damage Level | Undamaged | 40 | 15.6% |

### Specimen & Test Type Distributions
| Dimension | Category | Count |
|:---|:---|:---:|
| Specimen | M1 | 40 |
| Specimen | M2 | 40 |
| Specimen | M3 | 40 |
| Specimen | M4 | 37 |
| Specimen | M5 | 20 |
| Specimen | M6 | 20 |
| Specimen | M7 | 20 |
| Specimen | 1st | 20 |
| Specimen | 2nd | 20 |
| Test Type | Multihit | 91 |
| Test Type | Displacement | 65 |
| Test Type | Randomhit | 52 |
| Test Type | Singlehit | 49 |
| Hit Group | None | 173 |
| Hit Group | 2hit | 48 |
| Hit Group | 3hit | 24 |
| Hit Group | 4hit | 12 |

---

## 3. Sensor Descriptive Statistics (Recording-Level Aggregates)
The table below shows the distribution of recording-level statistics across all 5 sensors:

| Sensor | Mean (Avg ± Std) | Std (Avg ± Std) | RMS (Avg) | Peak-to-Peak (Avg) | Min (Avg) | Max (Avg) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Sensor_1 | 1.5665 ± 0.0909 | 0.4965 ± 0.1810 | 1.6524 | 9.2497 | -3.7764 | 5.4733 |
| Sensor_2 | 1.5307 ± 0.0924 | 0.3133 ± 0.1528 | 1.5697 | 5.9428 | -1.6781 | 4.2647 |
| Sensor_3 | 1.6200 ± 0.1372 | 0.2632 ± 0.0709 | 1.6433 | 7.0823 | -2.2718 | 4.8105 |
| Sensor_4 | 1.4987 ± 0.0650 | 0.1548 ± 0.0416 | 1.5073 | 5.4955 | -1.4971 | 3.9984 |
| Sensor_5 | 1.6041 ± 0.0999 | 0.0498 ± 0.0219 | 1.6051 | 2.7057 | 0.1225 | 2.8282 |

---

## 4. Damage-Level Trends & Signal Evolution
Evaluating recording-level features across damage stages (Undamaged, 1mm, 2mm, 3mm):

| Damage Level | Count | Sensor_1 RMS | Sensor_1 P2P | Sensor_3 RMS | Sensor_3 P2P | Sensor_5 RMS | Sensor_5 P2P |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Undamaged | 40 | 1.5243 | 9.077 | 1.3709 | 5.6048 | 1.4968 | 2.9278 |
| 1mm | 140 | 1.7081 | 9.2784 | 1.7179 | 7.511 | 1.6761 | 2.7679 |
| 2mm | 77 | 1.6179 | 9.2871 | 1.6491 | 7.0705 | 1.5323 | 2.4773 |

> [!IMPORTANT]
> **Causality Disclaimer:** Variations across damage levels reflect empirical observational trends in signal magnitude and frequency response. They do NOT establish strict mathematical causality until sensor physical locations and calibration units are verified with the professor.

---

## 5. Sensor Correlation Structure
Average within-recording Pearson correlation coefficients among the five sensors:

| Sensor | Sensor_1 | Sensor_2 | Sensor_3 | Sensor_4 | Sensor_5 |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Sensor_1** | 1.000 | 0.682 | 0.260 | 0.089 | 0.112 |
| **Sensor_2** | 0.682 | 1.000 | 0.726 | 0.474 | 0.075 |
| **Sensor_3** | 0.260 | 0.726 | 1.000 | 0.795 | 0.104 |
| **Sensor_4** | 0.089 | 0.474 | 0.795 | 1.000 | 0.339 |
| **Sensor_5** | 0.112 | 0.075 | 0.104 | 0.339 | 1.000 |

---

## 6. Key Findings & Observations
### Clear and Strong Observations
- **Dataset Integrity: Total 257 clean recordings evaluated (217 Damaged, 40 Undamaged). All recordings contain complete records with zero missing values across Relative_Time_Sec and Sensor_1-5.**
- **Sampling Frequency Discrepancy: Clean data exhibits two distinct computed sampling rates: ~303.0 Hz for Damaged bridge recordings and ~333.3 Hz for Undamaged cantilever recordings. Note: These rates are derived from timestamp deltas and require experimental confirmation from the professor.**
- **Sensor Inter-Correlation: Within-recording signals demonstrate structured spatial correlation. Highest average correlation is between Sensor_3 and Sensor_4 (r = 0.795), whereas lowest correlation is between Sensor_2 and Sensor_5 (r = 0.075).**

### Weaker or Possible Trends
- Damage-Level Vibration Energy: Mean Sensor_1 RMS across levels: {'Undamaged': 1.5243, '1mm': 1.7081, '2mm': 1.6179}. Progression across damage stages shows variations in signal spread and amplitude, though strict monotonicity across all sensors should not be assumed without knowing physical sensor placements.
- Excitation Protocol Impact: Test types (Displacement, Multihit, Randomhit, Singlehit) induce distinct dynamic response envelopes (e.g. impact hits produce high initial peak-to-peak transients followed by exponential decay, whereas displacement tests display step-release damping curves).
- Specimen Variance (M1-M7): Across specimens, baseline sensor amplitudes show modest test-to-test variance, suggesting structural consistency in the test beams.

---

## 7. Limitations & Inquiries for Professor Review
1. **Sensor Measurement Units:** Clarify physical units (acceleration in g or m/s², strain in µε, displacement in mm, or raw ADC voltage).
2. **Sensor Spatial Topology:** Confirm physical mounting locations along the cantilever/bridge beams (e.g. root vs midspan vs tip).
3. **Experimental Sampling Rates:** Confirm whether 303.03 Hz and 333.33 Hz reflect the hardware DAQ sampling clocks.
4. **Calendar Timestamps:** Confirm if exact test calendar dates/times are recorded in external lab logs.
