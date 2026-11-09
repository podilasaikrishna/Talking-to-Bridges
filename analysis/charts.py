"""Reusable Plotly chart builders for sensor / time-series data (Week 6).

Pure functions: DataFrame in, plotly Figure out. No Streamlit import, so the same
code works in the Streamlit dashboard AND inside the backend `chart_data` tool
(which returns `fig.to_json()` as `plot_json`).
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

DEFAULT_X = "Relative_Time_Sec"
MAX_POINTS = 5000  # keep browser charts fast on 30k+ sample recordings
UNITS_NOTE = "units not yet known"
HEALTHY_COLOR = "#2a78d6"  # blue: healthy reference
TEST_COLOR = "#eb6834"  # orange: uploaded / assessed recording
Z_LIMIT = 3.0  # |z| above this is outside the healthy range


def sensor_columns(df: pd.DataFrame) -> list[str]:
    """Columns that look like sensor channels (Sensor_1 ... Sensor_5)."""
    return [c for c in df.columns if str(c).lower().startswith("sensor")]


def downsample(df: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    """Keep every n-th row so at most ~max_points rows remain."""
    if max_points and len(df) > max_points:
        return df.iloc[:: len(df) // max_points + 1]
    return df


def _check(df: pd.DataFrame, x_col: str, y_cols: Sequence[str]) -> None:
    if df is None or df.empty:
        raise ValueError("No data available to chart.")
    missing = [c for c in [x_col, *y_cols] if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found in data: {', '.join(map(str, missing))}")


def plot_time_series(
    df: pd.DataFrame,
    x_col: str = DEFAULT_X,
    y_cols: Sequence[str] | None = None,
    title: str = "Sensor values over time",
    max_points: int = MAX_POINTS,
) -> go.Figure:
    """Interactive multi-sensor line chart. Raises ValueError on bad input."""
    y_cols = list(y_cols) if y_cols else sensor_columns(df) if df is not None else []
    if not y_cols:
        raise ValueError("No sensor columns found to plot.")
    _check(df, x_col, y_cols)
    plot_df = downsample(df, max_points)
    fig = go.Figure()
    for col in y_cols:
        fig.add_trace(
            go.Scattergl(
                x=plot_df[x_col].tolist(), y=plot_df[col].tolist(), mode="lines", name=str(col)
            )
        )
    fig.update_layout(
        title=f"{title} ({UNITS_NOTE})",
        xaxis_title="Relative time (s)" if x_col == DEFAULT_X else str(x_col),
        yaxis_title="Sensor value",
        hovermode="x unified",
        legend_title_text="Sensor",
    )
    return fig


def _py(value):
    """numpy scalar -> plain Python value (keeps the figure JSON simple)."""
    return value.item() if hasattr(value, "item") else value


def _add_events(fig: go.Figure, df: pd.DataFrame, x_col: str, sensor_col: str, events) -> None:
    """Mark each anomaly event: shaded span, start/end points and the peak."""
    x, y = df[x_col], df[sensor_col]
    peak_rows, edge_rows = [], []
    for ev in events:
        start, end, peak = int(ev["start_index"]), int(ev["end_index"]), int(ev["peak_index"])
        fig.add_vrect(
            x0=_py(x.iloc[start]), x1=_py(x.iloc[end]),
            fillcolor="red", opacity=0.15, line_width=0, layer="below",
        )
        edge_rows += [start, end]
        peak_rows.append(peak)
    if edge_rows:
        fig.add_trace(
            go.Scattergl(
                x=x.iloc[edge_rows].tolist(), y=y.iloc[edge_rows].tolist(), mode="markers",
                name="Event start/end", marker={"color": "orange", "size": 6, "symbol": "circle"},
            )
        )
    if peak_rows:
        fig.add_trace(
            go.Scattergl(
                x=x.iloc[peak_rows].tolist(), y=y.iloc[peak_rows].tolist(), mode="markers",
                name="Event peak", marker={"color": "red", "size": 10, "symbol": "x"},
            )
        )


def plot_anomaly_chart(
    df: pd.DataFrame,
    x_col: str = DEFAULT_X,
    sensor_col: str = "Sensor_1",
    anomaly_col: str = "is_anomaly",
    anomaly_x: Iterable[float] | None = None,
    max_points: int = MAX_POINTS,
    events: Sequence[dict] | None = None,
    flagged_count: int | None = None,
) -> go.Figure:
    """Sensor line with anomalies marked.

    Three ways to say what is anomalous:
      * `events`: the `events` list of the anomaly-detection tool (start_index,
        end_index, peak_index per event, row positions in `df`). Each event gets a
        shaded span, start/end points and a peak marker. Use this for real
        recordings: the tool caps its per-reading lists at 50, but events cover
        every flagged reading.
      * `anomaly_col`: a boolean column in `df`.
      * `anomaly_x`: a list of x values.
    `flagged_count` only changes the title (the number of flagged readings).
    Marked points are taken from the full data, so none are lost by downsampling.
    """
    _check(df, x_col, [sensor_col])
    plot_df = downsample(df, max_points)
    fig = go.Figure(
        go.Scattergl(
            x=plot_df[x_col].tolist(),
            y=plot_df[sensor_col].tolist(),
            mode="lines",
            name=sensor_col,
            line={"width": 1},
        )
    )
    if events is not None:
        pts = df.iloc[0:0]
        _add_events(fig, df, x_col, sensor_col, events)
    elif anomaly_x is not None:
        pts = df[df[x_col].isin(list(anomaly_x))]
    elif anomaly_col in df.columns:
        pts = df[df[anomaly_col].fillna(False).astype(bool)]
    else:
        pts = df.iloc[0:0]
    if not pts.empty:
        fig.add_trace(
            go.Scattergl(
                x=pts[x_col].tolist(),
                y=pts[sensor_col].tolist(),
                mode="markers",
                name="Anomaly",
                marker={"color": "red", "size": 8, "symbol": "x"},
            )
        )
    count = flagged_count if flagged_count is not None else len(pts)
    detail = f"{count} flagged in {len(events)} event(s)" if events is not None else f"{count} flagged"
    fig.update_layout(
        title=f"Anomalies: {sensor_col} ({detail}, {UNITS_NOTE})",
        xaxis_title="Relative time (s)" if x_col == DEFAULT_X else str(x_col),
        yaxis_title="Sensor value", hovermode="x unified",
    )
    return fig


def _rebased(df: pd.DataFrame, x_col: str) -> pd.Series:
    """x values shifted so every recording starts at 0 (recordings start at different times)."""
    x = pd.to_numeric(df[x_col], errors="coerce")
    return x - x.iloc[0]


def plot_recording_comparison(
    healthy_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sensors: Sequence[str] | None = None,
    x_col: str = DEFAULT_X,
    max_points: int = MAX_POINTS,
) -> go.Figure:
    """Healthy recording (blue) overlaid on the uploaded recording (orange), one panel per sensor.

    Both recordings are shifted to start at t = 0 so they can be compared directly.
    Raises ValueError on bad input.
    """
    for name, frame in (("healthy", healthy_df), ("uploaded", test_df)):
        if frame is None or frame.empty:
            raise ValueError(f"The {name} recording is empty.")
    shared = [c for c in sensor_columns(test_df) if c in healthy_df.columns]
    sensors = [s for s in (sensors or shared) if s in shared]
    if not sensors:
        raise ValueError("No sensor columns are present in both recordings.")
    _check(healthy_df, x_col, sensors)
    _check(test_df, x_col, sensors)
    h, t = downsample(healthy_df, max_points), downsample(test_df, max_points)
    hx, tx = _rebased(h, x_col).tolist(), _rebased(t, x_col).tolist()

    fig = make_subplots(
        rows=len(sensors), cols=1, shared_xaxes=True, vertical_spacing=0.04,
        subplot_titles=[str(s) for s in sensors],
    )
    for i, col in enumerate(sensors, start=1):
        fig.add_trace(
            go.Scattergl(
                x=hx, y=h[col].tolist(), mode="lines", name="Healthy",
                line={"width": 1, "color": HEALTHY_COLOR}, legendgroup="healthy",
                showlegend=i == 1,
            ),
            row=i, col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=tx, y=t[col].tolist(), mode="lines", name="Uploaded",
                line={"width": 1, "color": TEST_COLOR}, legendgroup="test",
                showlegend=i == 1,
            ),
            row=i, col=1,
        )
    fig.update_layout(
        title=f"Healthy vs uploaded recording ({UNITS_NOTE})",
        height=max(360, 230 * len(sensors) + 90),
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 1.0, "xanchor": "right"},
    )
    fig.update_xaxes(title_text="Relative time (s)", row=len(sensors), col=1)
    fig.update_yaxes(title_text="Sensor value")
    return fig


def _split_feature(row) -> tuple[str, str]:
    """('Sensor_3', 'kurtosis') from a deviation row.

    Accepts rows where the feature is already split and rows where the full name
    ('Sensor_1_fft_peak_freq') ended up in the sensor/feature columns.
    """
    feature = str(row["feature"])
    sensor = str(row["sensor"])
    full = feature if feature.startswith("Sensor_") else f"{sensor}_{feature}"
    match = re.match(r"^(Sensor_\d+)_(.+)$", full)
    return (match.group(1), match.group(2)) if match else (sensor, feature)


def _prepare_deviations(deviations: pd.DataFrame, sensor: str | None = None) -> pd.DataFrame:
    """Tidy copy of rank_deviations() output: clean sensor/feature names, optional sensor filter."""
    if deviations is None or deviations.empty:
        raise ValueError("No feature deviations to chart.")
    needed = {"sensor", "feature", "healthy_mean", "test_mean", "z_score"}
    missing = needed - set(deviations.columns)
    if missing:
        raise ValueError(f"Deviation table is missing column(s): {', '.join(sorted(missing))}")
    out = deviations.copy()
    parts = out.apply(_split_feature, axis=1)
    out["sensor"] = [p[0] for p in parts]
    out["feature"] = [p[1] for p in parts]
    out["abs_z"] = out["z_score"].abs()
    if sensor:
        out = out[out["sensor"] == sensor]
        if out.empty:
            raise ValueError(f"No feature deviations for {sensor}.")
    return out.sort_values("abs_z", ascending=False).reset_index(drop=True)


def plot_feature_deviation(
    deviations: pd.DataFrame,
    top_n: int = 12,
    z_limit: float = Z_LIMIT,
    sensor: str | None = None,
) -> go.Figure:
    """Feature values against the healthy range.

    `deviations` is the table from `analysis.features.rank_deviations`. Each bar is how
    many healthy standard deviations a feature of the uploaded recording is from its
    healthy mean (|z|, log axis because damage can be hundreds of sigmas). The dashed
    line is the edge of the healthy range (`z_limit` sigma): bars to its right are
    outside the healthy range. Hover shows the healthy and uploaded values.
    """
    dev = _prepare_deviations(deviations, sensor).head(top_n).iloc[::-1]  # biggest on top
    labels = [f"{r.sensor} {r.feature}" for r in dev.itertuples()]
    outside = dev["abs_z"] > z_limit
    fig = go.Figure(
        go.Bar(
            x=dev["abs_z"].clip(lower=0.01).tolist(),
            y=labels,
            orientation="h",
            marker={"color": [TEST_COLOR if o else HEALTHY_COLOR for o in outside]},
            text=[f"{z:+.1f}\u03c3" for z in dev["z_score"]],
            textposition="outside",
            customdata=list(zip(dev["healthy_mean"], dev["test_mean"], dev["z_score"])),
            hovertemplate=(
                "%{y}<br>healthy mean %{customdata[0]:.4g}<br>uploaded mean %{customdata[1]:.4g}"
                "<br>deviation %{customdata[2]:+.2f}\u03c3<extra></extra>"
            ),
            showlegend=False,
        )
    )
    x_min = min(0.3, float(dev["abs_z"].min()) * 0.8, z_limit / 3)
    x_max = max(float(dev["abs_z"].max()), z_limit) * 4  # room for the value labels
    fig.add_vrect(x0=x_min, x1=z_limit, fillcolor=HEALTHY_COLOR, opacity=0.12, line_width=0, layer="below")
    fig.add_vline(x=z_limit, line_dash="dash", line_color="#333333", line_width=2)
    n_out = int((_prepare_deviations(deviations, sensor)["abs_z"] > z_limit).sum())
    scope = f" for {sensor}" if sensor else ""
    fig.update_layout(
        title=f"Features vs healthy range{scope}: {n_out} outside the range ({UNITS_NOTE})",
        xaxis={
            "type": "log",
            "range": [math.log10(x_min), math.log10(x_max)],
            "title": (
                "Deviation from healthy mean (standard deviations, log scale); "
                f"shaded area and dashed line = healthy range (up to {z_limit:g}\u03c3)"
            ),
        },
        yaxis={"automargin": True},
        height=max(320, 34 * len(dev) + 150),
        margin={"r": 60},
    )
    return fig


def plot_deviation_heatmap(deviations: pd.DataFrame, z_clip: float = 10.0) -> go.Figure:
    """Sensor x feature grid of deviations from healthy (signed z; colour clipped at +/- z_clip)."""
    dev = _prepare_deviations(deviations)
    grid = dev.pivot_table(index="sensor", columns="feature", values="z_score", aggfunc="first")
    grid = grid.sort_index()
    fig = go.Figure(
        go.Heatmap(
            z=grid.values.tolist(), x=list(grid.columns), y=list(grid.index),
            zmin=-z_clip, zmax=z_clip, zmid=0,
            colorscale=[[0, HEALTHY_COLOR], [0.5, "#f2f2f0"], [1, TEST_COLOR]],
            text=[[f"{v:+.0f}" if abs(v) >= 100 else f"{v:+.1f}" for v in row] for row in grid.values.tolist()],
            texttemplate="%{text}", colorbar={"title": "z (sigma)"},
            hovertemplate="%{y} %{x}<br>%{z:+.2f}\u03c3 from healthy<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Deviation from healthy by sensor and feature ({UNITS_NOTE})",
        yaxis={"autorange": "reversed"},
        height=max(300, 70 * len(grid.index) + 150),
    )
    return fig


def _freq_markers(fig: go.Figure, spectrum: dict, freqs: dict, color: str, name: str, ymin_floor: float) -> None:
    """Mark f1, f2, f3 on a spectrum curve (labelled points on the curve)."""
    f_axis, amp = spectrum["frequencies_hz"], spectrum["amplitude"]
    xs, ys, labels = [], [], []
    for i, key in enumerate(("f1_hz", "f2_hz", "f3_hz"), start=1):
        f = freqs.get(key)
        if not f or f <= 0:
            continue
        idx = min(range(len(f_axis)), key=lambda j: abs(f_axis[j] - f))
        xs.append(f)
        ys.append(max(float(amp[idx]), ymin_floor))
        labels.append(f"f{i} {f:.2f} Hz")
    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers+text", name=f"{name} peaks",
            text=labels, textposition="top center" if name == "Uploaded" else "bottom center",
            marker={"size": 10, "color": color, "symbol": "diamond", "line": {"width": 1, "color": "white"}},
            textfont={"color": color},
            hovertemplate="%{text}<extra>" + name + "</extra>",
        )
    )


def plot_spectrum_overlay(
    healthy_spectrum: dict,
    test_spectrum: dict,
    healthy_freqs: dict,
    test_freqs: dict,
    x_max: float = 150.0,
) -> go.Figure:
    """Healthy spectrum (blue) over the uploaded one (orange), first three peaks marked.

    Spectra are dicts with ``frequencies_hz`` and ``amplitude``; ``*_freqs`` hold f1_hz,
    f2_hz, f3_hz. Amplitude is on a log axis so the small second and third peaks stay visible.
    Raises ValueError on bad input.
    """
    for label, sp in (("healthy", healthy_spectrum), ("uploaded", test_spectrum)):
        if not sp or not sp.get("frequencies_hz") or not sp.get("amplitude"):
            raise ValueError(f"The {label} spectrum is empty.")
    floor = max(
        1e-12,
        min(min(v for v in sp["amplitude"] if v > 0) for sp in (healthy_spectrum, test_spectrum)),
    )
    fig = go.Figure()
    for sp, name, color in ((healthy_spectrum, "Healthy", HEALTHY_COLOR), (test_spectrum, "Uploaded", TEST_COLOR)):
        fig.add_trace(
            go.Scatter(
                x=sp["frequencies_hz"], y=[max(v, floor) for v in sp["amplitude"]],
                mode="lines", name=name, line={"width": 1.6, "color": color},
            )
        )
    _freq_markers(fig, healthy_spectrum, healthy_freqs, HEALTHY_COLOR, "Healthy", floor)
    _freq_markers(fig, test_spectrum, test_freqs, TEST_COLOR, "Uploaded", floor)
    fig.update_layout(
        title=f"Frequency spectrum: healthy vs uploaded ({UNITS_NOTE})",
        height=460,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.0, "yanchor": "bottom", "x": 1.0, "xanchor": "right"},
    )
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, x_max])
    # Ignore the empty first bins (below 1 Hz) when choosing the y range.
    visible = [
        v for sp in (healthy_spectrum, test_spectrum)
        for f, v in zip(sp["frequencies_hz"], sp["amplitude"]) if 1.0 <= f <= x_max and v > 0
    ]
    if visible:
        fig.update_yaxes(range=[math.log10(min(visible)) - 0.3, math.log10(max(visible)) + 0.5])
    fig.update_yaxes(title_text="Amplitude (log scale)", type="log")
    return fig


def plot_frequency_bars(deviations: Sequence[dict], spreads: float = Z_LIMIT) -> go.Figure:
    """f1, f2, f3: healthy value against the uploaded value, with the percentage change.

    ``deviations`` is the Interface B list (feature, value, healthy_typical, healthy_spread,
    change_percent, score). The black whisker on the healthy bar is the healthy range
    (+/- ``spreads`` spreads). One panel per mode, each with its own scale.
    Raises ValueError when f1..f3 are missing.
    """
    by_name = {d["feature"]: d for d in deviations}
    keys = [k for k in ("f1_hz", "f2_hz", "f3_hz") if k in by_name]
    if not keys:
        raise ValueError("No frequency values (f1-f3) are available to chart.")
    titles = []
    for k in keys:
        d = by_name[k]
        titles.append(f"Mode {k[1]}: {d['change_percent']:+.1f}%")
    fig = make_subplots(rows=1, cols=len(keys), subplot_titles=titles, horizontal_spacing=0.09)
    for i, k in enumerate(keys, start=1):
        d = by_name[k]
        outside = abs(d["score"]) > spreads
        fig.add_trace(
            go.Bar(
                x=["Healthy"], y=[d["healthy_typical"]], name="Healthy", marker_color=HEALTHY_COLOR,
                error_y={"type": "constant", "value": spreads * d["healthy_spread"], "color": "black", "thickness": 1.5},
                text=[f"{d['healthy_typical']:.2f}"], textposition="inside", insidetextanchor="middle",
                legendgroup="h", showlegend=i == 1,
                hovertemplate=f"Healthy {k}: %{{y:.3f}} Hz<extra></extra>",
            ),
            row=1, col=i,
        )
        fig.add_trace(
            go.Bar(
                x=["Uploaded"], y=[d["value"]], name="Uploaded", marker_color=TEST_COLOR,
                marker_line={"width": 3 if outside else 0, "color": "#a02020"},
                text=[f"{d['value']:.2f}"], textposition="inside", insidetextanchor="middle",
                legendgroup="t", showlegend=i == 1,
                hovertemplate=f"Uploaded {k}: %{{y:.3f}} Hz<extra></extra>",
            ),
            row=1, col=i,
        )
        top = max(d["healthy_typical"] + spreads * d["healthy_spread"], d["value"])
        fig.update_yaxes(range=[0, top * 1.12], row=1, col=i)  # bars start at zero
    fig.update_layout(
        title="First three bending frequencies: healthy vs uploaded (Hz)",
        height=450, barmode="group", showlegend=True,
        legend={"orientation": "h", "y": -0.12, "yanchor": "top", "x": 0.5, "xanchor": "center"},
    )
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    return fig


def figure_to_json(fig: go.Figure) -> str:
    """JSON string for the API (`plot_json`); the app rebuilds it with plotly.io.from_json."""
    return fig.to_json()
