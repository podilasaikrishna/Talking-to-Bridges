# TRN-1 training report

Generated 2026-11-06 09:17:08 | scikit-learn 1.9.1 | seed 42

## Data
- 460 recordings, 23 cases: 2 healthy, 21 damaged.
- Every evaluation below is leave-one-case-out: all recordings of a case are held out together.
- Metrics are computed once on the pooled out-of-fold predictions; per-case results follow.
- Inputs never include label, file-name, case or sampling-rate columns.

## Model A: Undamaged vs Damaged
Selected (frequency-based sets only unless --allow-amplitude): **logreg** on feature set **freq_dev** (3 features).
`freq_dev` = two-sided deviation of f1-f3 from the healthy reference (median of the healthy training recordings), so a frequency drop and a rise both count as abnormal.

| model | feature_set | n_features | balanced_accuracy | f1_macro | roc_auc |
|---|---|---|---|---|---|
| logreg | freq_amp | 40 | 0.982 | 0.961 | 1.000 |
| gradient_boosting | freq_amp | 40 | 0.975 | 0.986 | 0.988 |
| logreg | freq_dev | 3 | 0.906 | 0.865 | 0.979 |
| random_forest | freq_amp | 40 | 0.875 | 0.923 | 1.000 |
| logreg | freq_ratio | 5 | 0.854 | 0.824 | 0.958 |
| logreg | freq | 3 | 0.849 | 0.807 | 0.965 |
| gradient_boosting | freq | 3 | 0.808 | 0.851 | 0.848 |
| gradient_boosting | freq_ratio | 5 | 0.787 | 0.804 | 0.643 |
| random_forest | freq | 3 | 0.771 | 0.826 | 0.935 |
| random_forest | freq_ratio | 5 | 0.755 | 0.758 | 0.962 |
| gradient_boosting | freq_dev | 3 | 0.598 | 0.640 | 0.650 |
| random_forest | freq_dev | 3 | 0.561 | 0.588 | 0.839 |

Pooled result for the selected model: balanced accuracy 0.949, macro F1 0.865, ROC-AUC 0.979, recall on undamaged 0.950, recall on damaged 0.948.
Always answering "Damaged" would score 0.913 plain accuracy but 0.500 balanced accuracy.

| Model A | pred Undamaged | pred Damaged |
|---|---|---|
| true Undamaged | 38 | 2 |
| true Damaged | 22 | 398 |

Per case (majority vote of the case's recordings):

| case_id | true | n_recordings | pct_correct | majority_pred | mean_p_damaged | case_correct |
|---|---|---|---|---|---|---|
| 1mm_M1 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 1mm_M2 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 1mm_M3 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 1mm_M4 | Damaged | 20 | 100.00 | Damaged | 0.99 | True |
| 1mm_M5 | Damaged | 20 | 95.00 | Damaged | 0.96 | True |
| 1mm_M6 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 1mm_M7 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M1 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M2 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M3 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M4 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M5 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 2mm_M6 | Damaged | 20 | 5.00 | Undamaged | 0.33 | False |
| 2mm_M7 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 3mm_M1 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 3mm_M2 | Damaged | 20 | 100.00 | Damaged | 0.98 | True |
| 3mm_M3 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 3mm_M4 | Damaged | 20 | 90.00 | Damaged | 0.85 | True |
| 3mm_M5 | Damaged | 20 | 100.00 | Damaged | 0.93 | True |
| 3mm_M6 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| 3mm_M7 | Damaged | 20 | 100.00 | Damaged | 1.00 | True |
| Undamaged_1st | Undamaged | 20 | 100.00 | Undamaged | 0.20 | True |
| Undamaged_2nd | Undamaged | 20 | 90.00 | Undamaged | 0.38 | True |

Decision threshold on P(damaged): **0.7** (design choice). Trade-off for the selected model:

| threshold | recall_undamaged | recall_damaged | balanced_accuracy |
|---|---|---|---|
| 0.300 | 0.600 | 0.971 | 0.786 |
| 0.400 | 0.700 | 0.967 | 0.833 |
| 0.500 | 0.850 | 0.962 | 0.906 |
| 0.600 | 0.925 | 0.957 | 0.941 |
| 0.700 | 0.950 | 0.948 | 0.949 |
| 0.800 | 0.975 | 0.938 | 0.957 |
| 0.900 | 1.000 | 0.905 | 0.952 |

Top features: dev_f2_hz (0.65), dev_f1_hz (0.29), dev_f3_hz (0.06)

**Caveat.** Only 2 healthy cases exist. Each healthy fold is trained with a single other healthy beam, so the undamaged-class numbers rest on 2 test cases and are very uncertain. Healthy and damaged recordings also come from differently named sets, so part of any "damaged vs healthy" separation may be setup or mounting differences, not damage.

## Model B: severity (1, 2, 3 mm), damaged recordings only
Selected (frequency-based sets only unless --allow-amplitude): **logreg** on feature set **freq** (3 features).

| model | feature_set | n_features | balanced_accuracy | f1_macro | mae_mm | within_1mm |
|---|---|---|---|---|---|---|
| random_forest | freq_amp | 40 | 0.886 | 0.886 | 0.169 | 0.945 |
| random_forest | freq_ratio | 5 | 0.829 | 0.829 | 0.257 | 0.914 |
| logreg | freq | 3 | 0.826 | 0.827 | 0.252 | 0.921 |
| logreg | freq_ratio | 5 | 0.826 | 0.827 | 0.252 | 0.921 |
| random_forest | freq | 3 | 0.817 | 0.818 | 0.283 | 0.900 |
| gradient_boosting | freq_amp | 40 | 0.814 | 0.814 | 0.271 | 0.914 |
| gradient_boosting | freq_ratio | 5 | 0.781 | 0.780 | 0.312 | 0.907 |
| gradient_boosting | freq | 3 | 0.760 | 0.756 | 0.350 | 0.890 |
| logreg | freq_amp | 40 | 0.698 | 0.695 | 0.402 | 0.900 |

Pooled result for the selected model: balanced accuracy 0.826, macro F1 0.827, mean absolute error 0.252 mm, within 1 mm 0.921.
Always answering the most common level would score 0.333.

| Model B | pred 1 mm | pred 2 mm | pred 3 mm |
|---|---|---|---|
| true 1 mm | 128 | 3 | 9 |
| true 2 mm | 20 | 104 | 16 |
| true 3 mm | 24 | 1 | 115 |

Per case (majority vote of the case's recordings):

| case_id | true | n_recordings | pct_correct | majority_pred | case_correct |
|---|---|---|---|---|---|
| 1mm_M1 | 1 | 20 | 100.00 | 1 | True |
| 1mm_M2 | 1 | 20 | 90.00 | 1 | True |
| 1mm_M3 | 1 | 20 | 95.00 | 1 | True |
| 1mm_M4 | 1 | 20 | 95.00 | 1 | True |
| 1mm_M5 | 1 | 20 | 60.00 | 1 | True |
| 1mm_M6 | 1 | 20 | 100.00 | 1 | True |
| 1mm_M7 | 1 | 20 | 100.00 | 1 | True |
| 2mm_M1 | 2 | 20 | 95.00 | 2 | True |
| 2mm_M2 | 2 | 20 | 100.00 | 2 | True |
| 2mm_M3 | 2 | 20 | 100.00 | 2 | True |
| 2mm_M4 | 2 | 20 | 65.00 | 2 | True |
| 2mm_M5 | 2 | 20 | 90.00 | 2 | True |
| 2mm_M6 | 2 | 20 | 0.00 | 3 | False |
| 2mm_M7 | 2 | 20 | 70.00 | 2 | True |
| 3mm_M1 | 3 | 20 | 95.00 | 3 | True |
| 3mm_M2 | 3 | 20 | 100.00 | 3 | True |
| 3mm_M3 | 3 | 20 | 100.00 | 3 | True |
| 3mm_M4 | 3 | 20 | 95.00 | 3 | True |
| 3mm_M5 | 3 | 20 | 100.00 | 3 | True |
| 3mm_M6 | 3 | 20 | 75.00 | 3 | True |
| 3mm_M7 | 3 | 20 | 10.00 | 1 | False |

Top features: f1_hz (0.48), f2_hz (0.36), f3_hz (0.16)

**Caveat.** Severity depends on crack location as well as depth. Cracks near the free end barely change the first mode, so 1 mm cracks there can look healthy or like other levels. Expect weaker results there.

## Honest limits
- Amplitude features were evaluated but not allowed into the saved model. Best Model A balanced accuracy: 0.982 with amplitude features vs 0.854 with frequency features only. Best Model B macro F1: 0.886 vs 0.829. A large gain from amplitude features is a warning sign of session leakage (gain/offset), not proof of better damage detection.
- The model was chosen among 12 combinations (Model A) and 9 (Model B) using the same leave-one-case-out scores that are reported, so the selected scores are slightly optimistic.
- Results apply to this 0.7 m cantilever beam, a single crack and a single boundary condition only.
- Do not present these figures as guarantees.
