# TRN-1 experiments (leave-one-case-out, frequency features only)

## 0. Mean frequencies per case (check the sampling-rate question)

| case_id | sampling_rate_hz | f1 | f2 | f3 | f2_std | n |
|---|---|---|---|---|---|---|
| 1mm_M1 | 303.030 | 7.654 | 39.752 | 127.759 | 0.165 | 20 |
| 1mm_M2 | 303.030 | 7.603 | 39.657 | 126.780 | 0.163 | 20 |
| 1mm_M3 | 303.030 | 7.631 | 39.734 | 128.709 | 0.205 | 20 |
| 1mm_M4 | 303.030 | 7.621 | 39.544 | 125.984 | 0.220 | 20 |
| 1mm_M5 | 303.030 | 7.584 | 39.100 | 124.732 | 0.804 | 20 |
| 1mm_M6 | 303.030 | 7.654 | 39.640 | 125.569 | 0.153 | 20 |
| 1mm_M7 | 303.030 | 7.691 | 39.620 | 125.873 | 0.102 | 20 |
| 2mm_M1 | 303.030 | 7.463 | 40.244 | 125.873 | 0.109 | 20 |
| 2mm_M2 | 303.030 | 7.490 | 40.778 | 125.686 | 0.148 | 20 |
| 2mm_M3 | 303.030 | 7.436 | 40.251 | 124.449 | 0.291 | 20 |
| 2mm_M4 | 303.030 | 7.621 | 40.346 | 125.200 | 0.370 | 20 |
| 2mm_M5 | 303.030 | 7.628 | 40.217 | 126.794 | 0.176 | 20 |
| 2mm_M6 | 303.030 | 7.538 | 38.560 | 125.606 | 0.181 | 20 |
| 2mm_M7 | 303.030 | 7.623 | 40.112 | 128.060 | 0.158 | 20 |
| 3mm_M1 | 303.030 | 7.057 | 39.125 | 125.572 | 0.106 | 20 |
| 3mm_M2 | 303.030 | 7.250 | 38.950 | 123.668 | 0.642 | 20 |
| 3mm_M3 | 303.030 | 7.323 | 38.413 | 121.900 | 1.351 | 20 |
| 3mm_M4 | 303.030 | 7.439 | 38.663 | 123.401 | 0.131 | 20 |
| 3mm_M5 | 303.030 | 7.525 | 38.971 | 122.891 | 0.105 | 20 |
| 3mm_M6 | 303.030 | 7.589 | 39.460 | 124.930 | 0.169 | 20 |
| 3mm_M7 | 303.030 | 7.654 | 39.602 | 122.614 | 0.145 | 20 |
| Undamaged_1st | 333.333 | 7.577 | 38.162 | 125.946 | 0.198 | 20 |
| Undamaged_2nd | 333.333 | 7.641 | 37.709 | 127.421 | 0.175 | 20 |

## 1. Model A: undamaged vs damaged (threshold 0.5)

| features | model | bal_acc | auc | recall_healthy | recall_damaged |
|---|---|---|---|---|---|
| base(5) | logreg | 0.854 | 0.958 | 0.750 | 0.957 |
| extended(8) | logreg | 0.854 | 0.953 | 0.750 | 0.957 |
| base(5) | extra_trees | 0.841 | 0.971 | 0.725 | 0.957 |
| base(5) | knn15 | 0.816 | 0.954 | 0.675 | 0.957 |
| extended(8) | extra_trees | 0.815 | 0.958 | 0.675 | 0.955 |
| extended(8) | knn15 | 0.774 | 0.934 | 0.600 | 0.948 |
| extended(8) | random_forest | 0.767 | 0.950 | 0.575 | 0.960 |
| base(5) | svm_rbf | 0.758 | 0.966 | 0.550 | 0.967 |
| base(5) | random_forest | 0.755 | 0.962 | 0.550 | 0.960 |
| extended(8) | svm_rbf | 0.649 | 0.956 | 0.325 | 0.974 |
| no_f2 | logreg | 0.605 | 0.561 | 0.700 | 0.510 |
| no_f2 | svm_rbf | 0.500 | 0.194 | 0.000 | 1.000 |
| no_f2 | knn15 | 0.499 | 0.291 | 0.000 | 0.998 |
| no_f2 | random_forest | 0.450 | 0.203 | 0.000 | 0.900 |
| no_f2 | extra_trees | 0.423 | 0.291 | 0.050 | 0.795 |

`no_f2` removes every f2-based feature. If it scores close to the others, the result does not depend on f2 alone.

### Threshold trade-off (logreg, extended features)

| threshold | recall_healthy | recall_damaged | bal_acc |
|---|---|---|---|
| 0.300 | 0.700 | 0.964 | 0.832 |
| 0.400 | 0.750 | 0.962 | 0.856 |
| 0.500 | 0.750 | 0.957 | 0.854 |
| 0.600 | 0.750 | 0.945 | 0.848 |
| 0.700 | 0.750 | 0.938 | 0.844 |
| 0.800 | 0.775 | 0.924 | 0.849 |

### Model A with two-sided deviation-from-healthy features

Features are |f - healthy median| / healthy median for f1-f3, with the healthy median taken from the training folds only. A drop in frequency (what a crack physically does) now counts as abnormal, as well as a rise. Last two columns: share of that case's recordings called damaged.

| model | threshold | bal_acc | auc | recall_healthy | recall_damaged | 3mm_M3 called damaged | 2mm_M6 called damaged |
|---|---|---|---|---|---|---|---|
| logreg | 0.500 | 0.906 | 0.979 | 0.850 | 0.962 | 1.000 | 0.200 |
| logreg | 0.700 | 0.949 | 0.979 | 0.950 | 0.948 | 1.000 | 0.050 |
| random_forest | 0.500 | 0.561 | 0.839 | 0.125 | 0.998 | 1.000 | 0.950 |
| random_forest | 0.700 | 0.658 | 0.839 | 0.350 | 0.967 | 1.000 | 0.400 |
| extra_trees | 0.500 | 0.668 | 0.950 | 0.375 | 0.962 | 1.000 | 0.250 |
| extra_trees | 0.700 | 0.898 | 0.950 | 0.900 | 0.895 | 1.000 | 0.000 |

## 2. Model B: severity (damaged recordings only)

| features | approach | model | bal_acc | f1_macro | mae_mm | cases_right |
|---|---|---|---|---|---|---|
| base(5) | classifier | random_forest | 0.829 | 0.829 | 0.257 | 18/21 |
| extended(8) | classifier | logreg | 0.826 | 0.827 | 0.252 | 19/21 |
| base(5) | classifier | logreg | 0.826 | 0.827 | 0.252 | 19/21 |
| base(5) | classifier | knn15 | 0.824 | 0.824 | 0.250 | 20/21 |
| extended(8) | classifier | random_forest | 0.821 | 0.822 | 0.276 | 18/21 |
| base(5) | classifier | extra_trees | 0.807 | 0.808 | 0.298 | 18/21 |
| base(5) | classifier | svm_rbf | 0.790 | 0.789 | 0.298 | 19/21 |
| extended(8) | classifier | knn15 | 0.786 | 0.784 | 0.298 | 19/21 |
| extended(8) | classifier | svm_rbf | 0.783 | 0.781 | 0.302 | 19/21 |
| extended(8) | classifier | extra_trees | 0.774 | 0.770 | 0.336 | 17/21 |
| base(5) | ordinal regression, rounded | gb_reg | 0.762 | 0.760 | 0.271 | 17/21 |
| base(5) | ordinal regression, rounded | rf_reg | 0.762 | 0.756 | 0.290 | 18/21 |
| extended(8) | ordinal regression, rounded | gb_reg | 0.757 | 0.754 | 0.271 | 17/21 |
| extended(8) | ordinal regression, rounded | rf_reg | 0.752 | 0.745 | 0.300 | 17/21 |
| extended(8) | ordinal regression, rounded | ridge | 0.455 | 0.431 | 0.562 | 10/21 |
| base(5) | ordinal regression, rounded | ridge | 0.457 | 0.421 | 0.567 | 10/21 |

Compare these rows with the saved model's scores in training_report.md (they are produced by the same folds).

Selecting the winner here uses the same folds that score it, so its score is optimistic. Report it as such.
