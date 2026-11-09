# NEW-2 and NEW-3: healthy baseline, comparison tool, comparison charts

Author: Nagarjun. Branch: `feature/new2-new3-baseline-charts`.

## NEW-2: baseline tool (`analysis/baseline.py`)

`compare_to_baseline(recording_or_feature_row)` returns Interface B:

| field | meaning |
|---|---|
| `overall_flag` | `normal`, `abnormal` or `unavailable` |
| `deviations` | ranked list: `feature, value, healthy_typical, healthy_spread, change_percent, score` (+ `direction`, `outside`, `group`) |
| `healthy_reference` | healthy f1-f3 (`frequencies_hz`) and, optionally, a healthy `spectrum` |
| `n_healthy` | `recordings`, `cases`, `case_names` |
| `summary`, `flag_rule`, `caveat` | plain-language text for the chat / UI |

* Healthy profile = median and 1.4826 x MAD per feature. It is read from `data/reference/healthy_baseline.json`
  (written by `python -m analysis.build_features_dataset`). If that file is missing it is built from the healthy
  recordings in `data/reference`.
* Flag rule: `abnormal` if any of f1_hz, f2_hz, f3_hz, f2_over_f1, f3_over_f1 is more than 3 spreads from the healthy
  median. Amplitude features (rms, variance, ...) are ranked after the frequency features and do not change the flag.
* The tool `baseline` (in `analysis/tools.py`) now calls this. Old output fields (`status`, `sensors_evaluated`,
  `deviation_ranking`, `healthy_baseline_summary`, `note`) are kept.
* NEW-4 (Kolla) can put `compare_to_baseline(df)` into `baseline_comparison`; NEW-5 reads the same fields.

## NEW-3: charts (new `chart_data` comparison modes)

* `spectrum`: healthy spectrum (blue) over the uploaded one (orange), first three peaks marked. Built in `analysis/charts.py::plot_spectrum_overlay`.
* `frequencies`: bars of f1, f2, f3 against the healthy values, percent change in each panel title, healthy range whisker. `plot_frequency_bars`.
* Chat words: "spectrum / peaks" -> spectrum chart; "frequency / frequencies / f1 / mode" + a compare word -> bar chart.
  "is this different from the healthy baseline?" -> text answer (baseline tool).

## What is needed on every machine (the folder is git-ignored)

`data/reference/` must contain `healthy_baseline.json` and at least one healthy cleaned recording (CSV) for the spectrum chart.

## Measured on the project data (460 recordings, f1-f3 from analysis.features at 300 Hz)

| set | flagged abnormal |
|---|---|
| healthy recordings (the same 40 that make the baseline) | 5 of 40 |
| all damaged recordings | 390 of 420 |
| 2 mm M6 (hard case in Kolla's model too) | 6 of 20 |

## Limits (say them when presenting)

* Baseline from only 2 healthy beams: a new healthy beam may be flagged; the 5 of 40 above is the best case (same data).
* The f2 shift is the main signal (it goes up, about +4 to +7 percent).
* The 300 Hz sampling rate is still waiting for the professor's confirmation.
* "abnormal" means "different from this baseline", not proof of damage.
