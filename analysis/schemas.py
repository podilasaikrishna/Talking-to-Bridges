"""Pydantic schemas for analytical tool inputs and outputs (Phase 2 - Week 6).

Every tool that is registered in the ToolRegistry must have:
  - An Input schema  (validates arguments BEFORE calling the handler)
  - An Output schema (describes the structured result returned by the handler)

Design rules
------------
* All fields must be JSON-serializable (no numpy/pandas objects).
* Optional fields must carry a sensible default so callers can omit them.
* Sensor column names are grounded in the actual professor dataset schema:
    Processed by Kolla's pipeline  -> Sensor_1 .. Sensor_5, Relative_Time_Sec,
                                     Condition, Damage_Level, Specimen, Test_Type
    Uploaded fixture / CSV schema  -> sensor-specific names such as
                                     deflection, vibration, stress, temperature,
                                     crack_width, sensor_id, timestamp, condition
* The error envelope (ToolErrorOutput) is returned by the registry when a tool
  call fails; it is NOT a sub-class of the per-tool output schemas.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared error envelope
# ---------------------------------------------------------------------------

class ToolErrorOutput(BaseModel):
    """Returned by the registry / dispatcher on any tool failure."""

    tool: str
    status: str = "error"
    error_type: str = Field(
        ...,
        description="UNKNOWN_TOOL | INVALID_INPUT | DATA_UNAVAILABLE | TOOL_EXECUTION_ERROR",
    )
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1. summary_statistics
# ---------------------------------------------------------------------------

class SummaryStatisticsInput(BaseModel):
    """Input schema for the summary_statistics tool.

    Works with both the fixture CSV schema (sensor_id, metric by name) and the
    professor's processed dataset (Sensor_1 .. Sensor_5).
    """

    metric: str = Field(
        ...,
        description=(
            "Numerical measurement column name.  Examples from the fixture: "
            "'deflection', 'vibration', 'stress', 'temperature', 'crack_width'. "
            "Examples from the professor's processed dataset: 'Sensor_1' .. 'Sensor_5'."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Filter rows by sensor_id value when that column is present (e.g. 'S01').",
    )
    start_time: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 start date/time filter (e.g. '2026-01-01').",
    )
    end_time: Optional[str] = Field(
        default=None,
        description="Optional ISO-8601 end date/time filter (e.g. '2026-01-31').",
    )


class SummaryStatisticsOutput(BaseModel):
    """Structured result from the summary_statistics tool."""

    sensor_id: Optional[str] = None
    metric: str
    count: int
    min: float
    max: float
    mean: float
    median: float
    std: float
    variance: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    q25: Optional[float] = None
    q75: Optional[float] = None
    rms: Optional[float] = None   # Root-mean-square (from professor dataset)
    p2p: Optional[float] = None   # Peak-to-peak range (from professor dataset)


# ---------------------------------------------------------------------------
# 2. anomaly_detection
# ---------------------------------------------------------------------------

class AnomalyDetectionInput(BaseModel):
    """Input schema for the anomaly_detection tool.

    Uses z-score thresholding as a placeholder; Kolla's ML-based detector
    should replace the handler while keeping this schema intact.
    """

    metric: str = Field(
        ...,
        description="Numerical column to scan for anomalies (e.g. 'vibration', 'Sensor_3').",
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Filter to a single sensor before anomaly detection.",
    )
    threshold: float = Field(
        default=2.0,
        ge=0.5,
        le=10.0,
        description="Z-score magnitude at which a reading is flagged as anomalous.",
    )


class AnomalyDetectionOutput(BaseModel):
    """Structured result from the anomaly_detection tool."""

    sensor_id: Optional[str] = None
    metric: str
    threshold_z: float
    total_rows_checked: int
    anomaly_count: int
    anomaly_fraction: float
    anomalous_indices: List[int] = Field(default_factory=list)
    anomalous_timestamps: List[str] = Field(default_factory=list)
    status_flag: str   # "Normal" | "Alert"


# ---------------------------------------------------------------------------
# 3. trend_analysis
# ---------------------------------------------------------------------------

class TrendAnalysisInput(BaseModel):
    """Input schema for the trend_analysis tool.

    Fits a linear trend to the metric over time and reports slope direction
    and strength.  Works on Relative_Time_Sec (professor dataset) or any
    numeric time column (timestamp index on the fixture dataset).
    """

    metric: str = Field(
        ...,
        description="Numerical column to analyse for trend (e.g. 'deflection', 'Sensor_2').",
    )
    time_column: Optional[str] = Field(
        default=None,
        description=(
            "Name of the time / X-axis column.  Defaults to 'Relative_Time_Sec' if "
            "present, otherwise falls back to the DataFrame integer index."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor_id filter when that column is present.",
    )


class TrendAnalysisOutput(BaseModel):
    """Structured result from the trend_analysis tool."""

    metric: str
    time_column: str
    slope: float
    intercept: float
    r_squared: float
    trend_direction: str   # "increasing" | "decreasing" | "flat"
    data_points: int
    start_value: float
    end_value: float
    change_magnitude: float


# ---------------------------------------------------------------------------
# 4. correlation_analysis
# ---------------------------------------------------------------------------

class CorrelationAnalysisInput(BaseModel):
    """Input schema for the correlation_analysis tool.

    Computes Pearson correlation coefficients between pairs of numeric columns
    (e.g. Sensor_1 vs Sensor_5, deflection vs vibration).
    """

    columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of column names to correlate.  When absent, all numeric columns "
            "in the dataset are used.  Must contain at least 2 names."
        ),
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor_id filter applied before computing correlations.",
    )


class CorrelationAnalysisOutput(BaseModel):
    """Structured result from the correlation_analysis tool."""

    columns_used: List[str]
    correlation_matrix: Dict[str, Dict[str, float]]
    strongest_pair: Optional[str] = None
    weakest_pair: Optional[str] = None


# ---------------------------------------------------------------------------
# 5. model_result  (NEW-4: trained model + healthy comparison, Interface C)
# ---------------------------------------------------------------------------

class ModelResultInput(BaseModel):
    """Input schema for the model_result tool."""

    sensor_id: str = Field(
        ...,
        description="Sensor / specimen identifier for which a prediction is requested.",
    )
    target_variable: str = Field(
        default="condition",
        description="Target column/label the model predicts (e.g. 'condition', 'Damage_Level').",
    )


class ModelResultOutput(BaseModel):
    """Structured result from the model_result tool."""

    sensor_id: str
    target_variable: str
    predicted_class: str
    confidence: float
    features_used: List[str] = Field(default_factory=list)
    prediction_timestamp: str
    model_note: str = ""
    damage_class: Optional[str] = Field(
        default=None,
        description="Categorical damage classification (e.g. 'Damaged_3mm', 'Undamaged').",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity level (e.g. 'High', 'Medium', 'Low', 'None').",
    )
    contributing_parameters: List[str] = Field(
        default_factory=list,
        description="Sensors or features contributing to the predicted condition.",
    )

    # ---- NEW-4: Interface C (shared with NEW-5, the results card) ----
    model_available: bool = Field(
        default=False,
        description="True when the trained TRN-1 model produced the condition.",
    )
    model_error: str = Field(default="", description="Why the model could not run (if it could not).")
    condition: Optional[str] = Field(
        default=None, description="'Damaged' or 'Undamaged' (None when no model result)."
    )
    condition_confidence: Optional[float] = Field(
        default=None, description="Confidence in the condition, 0 to 1."
    )
    severity_mm: Optional[int] = Field(
        default=None, description="Crack depth class 1, 2 or 3 mm; None unless Damaged."
    )
    severity_confidence: Optional[float] = Field(
        default=None, description="Confidence in the severity, 0 to 1."
    )
    severity_label: Optional[str] = Field(
        default=None, description="Plain-language severity, e.g. 'medium (2 mm crack ...)'."
    )
    top_parameters: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Features that drove the decision: model importances plus the top deviations "
            "from the healthy baseline (feature, source, importance, value, healthy_typical, "
            "change_percent, score)."
        ),
    )
    baseline_comparison: Dict[str, Any] = Field(
        default_factory=dict,
        description="Interface B result for the same recording (overall_flag, deviations, ...).",
    )
    agreement: str = Field(
        default="unknown", description="'agree', 'disagree' or 'unknown' (model vs healthy baseline)."
    )
    agreement_note: str = ""
    status: str = Field(
        default="ok", description="ok | no_data | not_a_recording | features_failed"
    )
    message: str = ""
    caveats: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 6. baseline (NEW-2: healthy baseline comparison)
# ---------------------------------------------------------------------------

class BaselineInput(BaseModel):
    """Input schema for the baseline tool."""

    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor ID filter (e.g. 'Sensor_1', 'S01').",
    )
    metric: Optional[str] = Field(
        default=None,
        description="Optional metric or column name to evaluate against healthy baseline.",
    )


class BaselineOutput(BaseModel):
    """Structured result from the baseline tool (NEW-2, Interface B + older fields)."""

    overall_flag: str = Field(
        default="unavailable",
        description="'normal', 'abnormal' (different from the healthy baseline) or 'unavailable'.",
    )
    deviations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Ranked: feature, value, healthy_typical, healthy_spread, change_percent, score.",
    )
    healthy_reference: Dict[str, Any] = Field(
        default_factory=dict,
        description="Healthy f1-f3 (and optionally a reference spectrum) for the charts.",
    )
    n_healthy: Dict[str, Any] = Field(
        default_factory=dict,
        description="Number of healthy recordings and cases behind the baseline.",
    )
    summary: str = ""
    caveat: str = ""
    status: str = Field(default="Unavailable", description="'Healthy', 'Abnormal' or 'Unavailable'.")
    sensors_evaluated: List[str] = Field(default_factory=list)
    deviation_ranking: List[Dict[str, Any]] = Field(default_factory=list)
    healthy_baseline_summary: Dict[str, Any] = Field(default_factory=dict)
    note: str = ""


# ---------------------------------------------------------------------------
# 6. chart_data  (wraps Nagarjun's visualization_service)
# ---------------------------------------------------------------------------

class ChartDataInput(BaseModel):
    """Input schema for the chart_data tool."""

    y_col: str = Field(
        ...,
        description="Numerical column to plot on the Y axis (e.g. 'deflection', 'Sensor_1').",
    )
    x_col: str = Field(
        default="timestamp",
        description="Column to use for the X axis.  Defaults to 'timestamp' / 'Relative_Time_Sec'.",
    )
    show_anomalies: bool = Field(
        default=False,
        description="Mark anomalous readings on the chart (Sensor_N recordings).",
    )
    threshold: float = Field(
        default=5.0,
        gt=0,
        description="Anomaly threshold (robust z-score) used when show_anomalies is true.",
    )
    sensor_id: Optional[str] = Field(
        default=None,
        description="Optional sensor_id filter used when show_anomalies is true.",
    )
    comparison: Optional[Literal["signals", "features", "heatmap", "spectrum", "frequencies"]] = Field(
        default=None,
        description=(
            "Compare the uploaded recording with the healthy reference: 'signals' (overlay), "
            "'features' (feature deviations vs the healthy range), 'heatmap' (sensor x feature), "
            "'spectrum' (frequency spectrum, f1-f3 marked) or 'frequencies' (f1-f3 bars with percent change)."
        ),
    )


class ChartDataOutput(BaseModel):
    """Structured result from the chart_data tool."""

    x_col: str
    y_col: str
    plot_json: str
    explanation: str


# ---------------------------------------------------------------------------
# Schema registries - used by the dispatcher for pre-validation
# ---------------------------------------------------------------------------

TOOL_INPUT_SCHEMAS: Dict[str, type] = {
    "summary_statistics": SummaryStatisticsInput,
    "anomaly_detection": AnomalyDetectionInput,
    "trend_analysis": TrendAnalysisInput,
    "correlation_analysis": CorrelationAnalysisInput,
    "model_result": ModelResultInput,
    "chart_data": ChartDataInput,
    "baseline": BaselineInput,
}

TOOL_OUTPUT_SCHEMAS: Dict[str, type] = {
    "summary_statistics": SummaryStatisticsOutput,
    "anomaly_detection": AnomalyDetectionOutput,
    "trend_analysis": TrendAnalysisOutput,
    "correlation_analysis": CorrelationAnalysisOutput,
    "model_result": ModelResultOutput,
    "chart_data": ChartDataOutput,
    "baseline": BaselineOutput,
}