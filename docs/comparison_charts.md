# Healthy vs uploaded comparison charts (faculty requirement 5)

Three interactive Plotly charts that show how an uploaded recording differs from the
healthy (undamaged) condition. They are built by `analysis/charts.py` and returned by the
existing `chart_data` tool, so the chat shows them like any other chart (`ChatResponse.fig`).

| Chart (`comparison=`) | What it shows | Example question |
|---|---|---|
| `signals` | Healthy recording (blue) over the uploaded one (orange), one panel per sensor, both starting at t = 0 | "plot sensor 3 healthy vs uploaded" |
| `features` | Features (RMS, variance, skewness, kurtosis, peak, crest factor, FFT) as deviation from the healthy mean in standard deviations; the shaded area / dashed line is the healthy range (3 sigma); largest deviations on top | "plot features against the healthy range" |
| `heatmap` | The same deviations as a sensor x feature grid | "chart a heatmap of deviations versus healthy" |

The numbers come from Eswar's `analysis/features.py` (`build_healthy_baseline`, `rank_deviations`),
so the chart and the baseline tool agree.

## Healthy reference

Put one or more healthy recordings (`.csv`, `.xlsx`, `.xlsm`) in `data/reference/`
(or set `HEALTHY_REFERENCE_DIR`). CSV files must be cleaned recordings (`Sensor_1..5`,
`Relative_Time_Sec`); Excel files go through `analysis.sensor_pipeline.clean_sensor_bytes`.
The folder is git-ignored with the rest of `data/`. Without a reference the tool returns the
usual placeholder (`plot_json = "{}"`) and an explanation saying why. The signal overlay uses
the first healthy file; the feature baseline uses all of them.

## Tool arguments

`chart_data` gained one optional input, `comparison` (`signals` | `features` | `heatmap`).
`y_col` limits the chart to one sensor when it is a `Sensor_N` column. The output contract
(`x_col`, `y_col`, `plot_json`, `explanation`) is unchanged; a `comparison` key is added.
The dispatcher sets `comparison` when the question contains a comparison word (healthy,
baseline, compare, versus, vs, against) together with a chart word.
