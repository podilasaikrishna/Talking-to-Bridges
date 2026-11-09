import pandas as pd
import numpy as np

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None

def analyze_and_plot(df: pd.DataFrame, time_col: str, val_col: str):
    """
    Sorts data, handles missing/datetime values universally, generates an interactive 
    plot with min/max highlights, and returns a formatted textual summary.
    """
    # 1. Work on a clean copy
    data = df.copy()

    # 2. Clean and cast the value column to numeric (invalid strings convert to NaN)
    data[val_col] = pd.to_numeric(data[val_col], errors="coerce")

    # 3. Smart X-Axis Detection & Chronological Sorting
    is_datetime = False
    if data[time_col].dtype == "object":
        try:
            converted = pd.to_datetime(data[time_col], errors="coerce")
            # If at least 80% converted successfully, treat X as datetime
            if converted.notna().sum() >= 0.8 * len(data):
                data[time_col] = converted
                is_datetime = True
        except Exception:
            pass

    # Sort sequentially along X axis so lines don't zig-zag backwards
    if is_datetime or pd.api.types.is_numeric_dtype(data[time_col]):
        data = data.sort_values(by=time_col).reset_index(drop=True)

    # Drop rows where X-axis itself is missing
    data = data.dropna(subset=[time_col]).reset_index(drop=True)

    # 4. Generate Interactive Line Chart
    fig = px.line(
        data, 
        x=time_col, 
        y=val_col, 
        title=f"{val_col} Analysis over {time_col}",
        template="plotly_dark",
        markers=True if len(data) < 100 else False
    )

    # 5. Bridge gaps across missing values (NaNs in sensor/financial data)
    fig.update_traces(connectgaps=True)

    # 6. Dynamically calculate metrics and annotate Peak and Lowest readings
    valid_data = data.dropna(subset=[val_col])

    if not valid_data.empty:
        max_idx = valid_data[val_col].idxmax()
        min_idx = valid_data[val_col].idxmin()

        max_val = valid_data.loc[max_idx, val_col]
        max_time = valid_data.loc[max_idx, time_col]

        min_val = valid_data.loc[min_idx, val_col]
        min_time = valid_data.loc[min_idx, time_col]

        avg_val = valid_data[val_col].mean()

        # Format date for text display if X was parsed as datetime
        display_max_x = max_time.strftime('%Y-%m-%d') if is_datetime else str(max_time)
        display_min_x = min_time.strftime('%Y-%m-%d') if is_datetime else str(min_time)

        # Highlight Maximum Point
        fig.add_trace(go.Scatter(
            x=[max_time], y=[max_val],
            mode='markers+text',
            name='Maximum',
            text=[f'Max: {max_val:.2f}'],
            textposition="top center",
            marker=dict(color='crimson', size=12)
        ))

        # Highlight Minimum Point
        fig.add_trace(go.Scatter(
            x=[min_time], y=[min_val],
            mode='markers+text',
            name='Minimum',
            text=[f'Min: {min_val:.2f}'],
            textposition="bottom center",
            marker=dict(color='royalblue', size=12)
        ))

        # Summary text output
        explanation = (
            f"### Automated Data Summary\n"
            f"- **Metric Analyzed:** `{val_col}` over `{time_col}`\n"
            f"- **Peak Reading:** **{max_val:.2f}** at `{display_max_x}`\n"
            f"- **Lowest Reading:** **{min_val:.2f}** at `{display_min_x}`\n"
            f"- **Average Value:** **{avg_val:.2f}**"
        )
    else:
        explanation = "No valid numeric data points found to plot."

    return fig, explanation