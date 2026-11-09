"""Sensor dataset ingestion / preprocessing pipeline (Phase 2 - Week 5, Kolla).

Turns the professor's raw ZIP files (headerless Excel workbooks, two different
layouts) into ONE standard clean table:

    DateTime | Relative_Time_Sec | Sensor_1 .. Sensor_5 | Condition | Test_Name | Source_File

Raw layouts (from docs/week5_data_analysis.md, Eswar):
    Damaged   (7 cols): Absolute time string, Relative time, 5 sensors
    Undamaged (9 cols): Hour, Minute, Second, Relative time, 5 sensors

What the pipeline does for every workbook:
    1. Reads it without headers and drops completely empty rows / columns.
    2. Detects the layout from the first columns (whole-number hours/minutes =
       Undamaged, 9 columns; clock text = Damaged, 7 columns). Stray columns after
       the data columns (e.g. a note) are ignored and counted in the report.
    3. Converts everything to numbers; rows that cannot be converted (for
       example an accidental header row) or that miss a value are dropped.
    4. Drops duplicate Relative_Time_Sec rows and sorts by Relative_Time_Sec.
    5. Builds a single DateTime column: clock of the first row + Relative_Time_Sec
       (the raw clock does not tick on every row; passing midnight is handled).
    6. Produces a quality report (rows kept / dropped, sampling rate, ...).

Nothing is dropped silently: every count is in the per-file report and in the
manifest.csv written next to the clean files. A file that cannot be processed
is recorded with status "error" and a message, and the other files continue.
Wide summary tables (thousands of columns) are recorded as "skipped".

The folder names inside the ZIP are kept as columns: Damage_Level (1mm),
Specimen (M1 / 1st), Test_Type (Displacement / Multihit / Randomhit / Singlehit)
and Hit_Group (2hit / 3hit / 4hit).

NOTE: the raw files only contain a time of day, not a calendar date. DateTime
therefore uses a fixed anchor date (default 1970-01-01). Use Relative_Time_Sec
for analysis and DateTime only for ordering / plotting.

Command line (run from the repository root):

    python -m analysis.sensor_pipeline \
        --zip damaged_bridges.zip \
        --zip "undamaged data of 2 cantilevers.zip" \
        --out data/processed/sensors
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("ttb.sensor_pipeline")

# ---------------------------------------------------------------------------
# Constants (names follow docs/week5_data_analysis.md)
# ---------------------------------------------------------------------------

SENSOR_COLUMNS: List[str] = [f"Sensor_{i}" for i in range(1, 6)]
CLEAN_COLUMNS: List[str] = [
    "DateTime",
    "Relative_Time_Sec",
    *SENSOR_COLUMNS,
    "Condition",
    "Test_Name",
    "Source_File",
    "Damage_Level",
    "Specimen",
    "Test_Type",
    "Hit_Group",
]

DAMAGED_RAW_COLUMNS: List[str] = ["Absolute_Time", "Relative_Time_Sec", *SENSOR_COLUMNS]
UNDAMAGED_RAW_COLUMNS: List[str] = ["Hour", "Minute", "Second", "Relative_Time_Sec", *SENSOR_COLUMNS]

LAYOUT_BY_COLUMN_COUNT: Dict[int, str] = {7: "Damaged", 9: "Undamaged"}

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".xlsx", ".xlsm", ".csv")

DEFAULT_ANCHOR_DATE = "1970-01-01"
SECONDS_PER_DAY = 86400.0
DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


# A workbook with more columns than this is not a raw recording (the professor's
# "1mm.xlsx" summary tables have ~3000 columns: one row per sensor, one column
# per sample). Such files are skipped on purpose, not treated as errors.
MAX_RAW_COLUMNS = 20


class SensorDataError(ValueError):
    """Raised when a sensor file or ZIP cannot be processed."""


class SkippedFileError(SensorDataError):
    """Raised for files that are deliberately not processed (not raw recordings)."""


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class FileReport:
    """Quality report for one workbook. Row counts always add up:
    rows_read = rows_kept + rows_dropped_empty + rows_dropped_invalid
                + rows_dropped_duplicate
    """

    source_file: str
    status: str = "ok"  # "ok", "skipped" (not a raw recording) or "error"
    message: str = ""
    condition: str = ""
    layout: str = ""
    rows_read: int = 0
    rows_kept: int = 0
    rows_dropped_empty: int = 0
    rows_dropped_invalid: int = 0
    rows_dropped_duplicate: int = 0
    was_out_of_order: bool = False
    extra_columns_ignored: int = 0  # stray columns after the 7/9 data columns
    extra_cells_ignored: int = 0    # filled cells inside those stray columns
    sampling_rate_hz: Optional[float] = None
    duration_sec: Optional[float] = None
    max_gap_sec: Optional[float] = None
    max_clock_mismatch_sec: Optional[float] = None
    start_datetime: str = ""
    end_datetime: str = ""
    flat_sensors: str = ""  # sensors with a constant value, e.g. "Sensor_3"
    output_file: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def detect_condition(name: str) -> Optional[str]:
    """Guess "Damaged" / "Undamaged" from a file name. Returns None if unclear.

    "undamaged" contains "damaged", so it must be checked first.
    """
    lowered = name.lower()
    if "undamaged" in lowered or "un-damaged" in lowered or "un_damaged" in lowered:
        return "Undamaged"
    if "damaged" in lowered:
        return "Damaged"
    return None


def resolve_condition(
    explicit: Optional[str], member_path: str, zip_name: str
) -> Optional[str]:
    """Decide "Damaged" / "Undamaged" for one workbook. Priority:
    1. an explicit value given by the caller,
    2. the folder names inside the ZIP ("Undamaged Data of 2 cantilever/2nd/..."),
    3. the ZIP file name ("undamaged data of 2 cantilevers.zip"),
    4. None: the caller then falls back to the detected layout.
    The layout is deliberately the LAST resort: it describes how the file is
    written, not what was measured.
    """
    return (
        _normalise_condition(explicit)
        or detect_condition(member_path)
        or detect_condition(zip_name)
    )


def _normalise_condition(condition: Optional[str]) -> Optional[str]:
    if condition is None:
        return None
    value = condition.strip().lower()
    if value == "damaged":
        return "Damaged"
    if value == "undamaged":
        return "Undamaged"
    raise ValueError("condition must be 'Damaged' or 'Undamaged' (or None)")


def _to_seconds_of_day(series: pd.Series) -> pd.Series:
    """Convert a column of clock readings to seconds since midnight (float).

    Accepts strings ("11:55:24.624000"), datetime.time objects, datetimes,
    timedeltas, and Excel day-fractions (0.5 == 12:00:00). Unreadable values
    become NaN.
    """
    if pd.api.types.is_timedelta64_dtype(series):
        return series.dt.total_seconds().astype("float64")
    if pd.api.types.is_datetime64_any_dtype(series):
        return (series - series.dt.normalize()).dt.total_seconds().astype("float64")
    if pd.api.types.is_bool_dtype(series):
        return pd.Series(np.nan, index=series.index, dtype="float64")
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64") * SECONDS_PER_DAY

    # Text such as "11:55:24.624000", "17:07:20.605 ->" or "2026-01-01 11:55:24.5":
    # take the hh:mm:ss(.fff) part and ignore anything around it.
    text = series.astype("string").str.extract(r"(\d{1,2}:\d{2}:\d{2}(?:\.\d+)?)", expand=False)
    seconds = pd.to_timedelta(text, errors="coerce").dt.total_seconds().astype("float64")
    return seconds.where(seconds < SECONDS_PER_DAY)


def _looks_like_hour_minute(raw: pd.DataFrame) -> bool:
    """True when the first two columns hold whole-number hours (0-23) and minutes (0-59)."""
    if raw.shape[1] < 2:
        return False
    hours = pd.to_numeric(raw[0], errors="coerce")
    minutes = pd.to_numeric(raw[1], errors="coerce")
    first, second = raw[0].notna(), raw[1].notna()
    if not first.any() or not second.any():
        return False
    hour_ok = (hours.between(0, 23) & (hours == hours.round()))[first].mean() >= 0.9
    minute_ok = (minutes.between(0, 59) & (minutes == minutes.round()))[second].mean() >= 0.9
    return bool(hour_ok and minute_ok)


def _read_raw(content: bytes, name: str) -> pd.DataFrame:
    """Read a headerless workbook (first sheet) or CSV into a raw DataFrame.

    Both .xlsx and .xlsm are explicitly routed through openpyxl so that
    macro-enabled workbooks (.xlsm) are handled correctly regardless of which
    engine pandas would otherwise pick as its default.  Macros are silently
    ignored by openpyxl – only the data is read.
    """
    extension = Path(name).suffix.lower()
    try:
        if extension == ".csv":
            return pd.read_csv(io.BytesIO(content), header=None)
        if extension in (".xlsx", ".xlsm"):
            # engine='openpyxl' is pinned explicitly:
            #   • .xlsx – ensures consistent behaviour across pandas versions.
            #   • .xlsm – macro-enabled workbooks; openpyxl reads data only,
            #              macros are ignored (which is exactly what we want).
            return pd.read_excel(
                io.BytesIO(content),
                header=None,
                sheet_name=0,
                engine="openpyxl",
            )
    except pd.errors.EmptyDataError as exc:
        raise SensorDataError(f"'{name}' is empty.") from exc
    except ImportError as exc:  # openpyxl not installed
        raise SensorDataError(
            f"Cannot read '{name}': openpyxl is required for .xlsx/.xlsm files "
            f"but is not installed ({exc}). Run: pip install openpyxl"
        ) from exc
    except Exception as exc:  # corrupt workbook etc.
        raise SensorDataError(f"Could not read '{name}': {exc}") from exc
    raise SensorDataError(f"Unsupported file type '{extension}' for '{name}'.")


_TEST_TYPE_PREFIXES = (
    ("disp", "Displacement"),
    ("multi", "Multihit"),
    ("random", "Randomhit"),
    ("singl", "Singlehit"),  # also matches the typo "Singlrhit" in the data
)


def parse_path_metadata(source_file: str) -> Dict[str, str]:
    """Read the experiment set-up from the folder names inside the ZIP.

    "Damaged Beams/1mm/M3/Multihit/2 Hit/D-MH 1-2-3.xlsx" gives
    Damage_Level "1mm", Specimen "M3", Test_Type "Multihit", Hit_Group "2hit".
    Anything not found stays an empty string (e.g. the undamaged ZIP has no
    damage level).
    """
    meta = {"Damage_Level": "", "Specimen": "", "Test_Type": "", "Hit_Group": ""}
    folders = [p for p in source_file.replace("\\", "/").split("/")[:-1] if p.strip()]
    for folder in folders:
        compact = re.sub(r"\s+", "", folder).lower()
        if re.fullmatch(r"\d+mm", compact):
            meta["Damage_Level"] = meta["Damage_Level"] or compact
        elif re.fullmatch(r"m\d+", compact) or re.fullmatch(r"\d+(st|nd|rd|th)", compact):
            if not meta["Specimen"]:
                meta["Specimen"] = compact.upper() if compact.startswith("m") else compact
        elif re.fullmatch(r"\d+hit", compact):
            meta["Hit_Group"] = meta["Hit_Group"] or compact
        else:
            for prefix, name in _TEST_TYPE_PREFIXES:
                if compact.startswith(prefix):
                    meta["Test_Type"] = meta["Test_Type"] or name
                    break
    return meta


def _safe_stem(source_file: str) -> str:
    """Turn 'folder/sub/file 1.xlsx' into 'folder__sub__file_1' (file-system safe)."""
    path = Path(source_file.replace("\\", "/"))
    parts = [p for p in path.with_suffix("").parts if p not in ("", "/")]
    joined = "__".join(parts) or "file"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", joined).strip("_") or "file"


# ---------------------------------------------------------------------------
# Core: clean ONE workbook
# ---------------------------------------------------------------------------

def clean_sensor_bytes(
    content: bytes,
    source_file: str,
    condition: Optional[str] = None,
    anchor_date: str = DEFAULT_ANCHOR_DATE,
) -> Tuple[pd.DataFrame, FileReport]:
    """Clean one raw workbook (given as bytes). Returns (clean_df, report).

    Raises SensorDataError if the file cannot be used at all.
    """
    condition = _normalise_condition(condition)
    anchor = pd.Timestamp(anchor_date).normalize()
    report = FileReport(source_file=source_file)

    raw = _read_raw(content, source_file)
    report.rows_read = int(len(raw))

    # 1. Drop completely empty rows. Columns are NOT dropped yet, so the position
    #    of every data column stays exactly as in the file.
    non_empty_rows = raw.dropna(how="all")
    report.rows_dropped_empty = report.rows_read - int(len(non_empty_rows))
    raw = non_empty_rows.reset_index(drop=True)
    if raw.empty:
        raise SensorDataError(f"'{source_file}' contains no data.")

    non_empty_cols = int(raw.notna().any().sum())
    if non_empty_cols > MAX_RAW_COLUMNS:
        raise SkippedFileError(
            f"'{source_file}' has {non_empty_cols} columns: it is a wide summary table "
            "(one row per sensor, one column per sample), not a raw recording. Skipped."
        )
    n_cols = int(raw.shape[1])
    raw.columns = list(range(n_cols))

    # 2. Detect the layout from the content of the first columns: in the 9-column
    #    (Undamaged) layout they are whole-number hours (0-23) and minutes (0-59);
    #    in the 7-column (Damaged) layout the first column is a clock text.
    layout = "Undamaged" if _looks_like_hour_minute(raw) else "Damaged"
    width = 9 if layout == "Undamaged" else 7
    if n_cols < width:
        raise SensorDataError(
            f"'{source_file}' has only {n_cols} columns; expected at least {width} "
            f"for the {layout} layout (Damaged = 7 columns, Undamaged = 9 columns)."
        )
    report.layout = layout

    # Columns after the data columns (typically a note such as "Single hit between
    # S1-S2", or a stray cell) are ignored, but always reported.
    extra = raw.iloc[:, width:]
    report.extra_columns_ignored = int(extra.notna().any().sum())
    report.extra_cells_ignored = int(extra.notna().sum().sum())
    if report.extra_cells_ignored:
        logger.warning(
            "%s: ignored %d filled cell(s) in %d column(s) after the %d data columns.",
            source_file, report.extra_cells_ignored, report.extra_columns_ignored, width,
        )
    raw = raw.iloc[:, :width].copy()

    if condition is not None and condition != layout:
        logger.warning(
            "%s: condition '%s' does not match the '%s' layout; "
            "parsing by layout, labelling as '%s'.",
            source_file, condition, layout, condition,
        )
    final_condition = condition or layout
    report.condition = final_condition

    # 3. Standardise columns and convert to numbers.
    if layout == "Damaged":
        df = raw.copy()
        df.columns = DAMAGED_RAW_COLUMNS
        seconds_of_day = _to_seconds_of_day(df["Absolute_Time"])
    else:
        df = raw.copy()
        df.columns = UNDAMAGED_RAW_COLUMNS
        for col in ("Hour", "Minute", "Second"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        valid_clock = (
            df["Hour"].between(0, 23)
            & df["Minute"].between(0, 59)
            & (df["Second"] >= 0)
            & (df["Second"] < 60)
        )
        seconds_of_day = (df["Hour"] * 3600.0 + df["Minute"] * 60.0 + df["Second"]).where(valid_clock)

    numeric_cols = ["Relative_Time_Sec", *SENSOR_COLUMNS]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["_seconds_of_day"] = seconds_of_day
    df = df.replace([np.inf, -np.inf], np.nan)

    # 4. Drop rows with any unreadable / missing value (e.g. a header row).
    required = ["_seconds_of_day", *numeric_cols]
    valid_mask = df[required].notna().all(axis=1)
    report.rows_dropped_invalid = int((~valid_mask).sum())
    raw_invalid_counts = df[required].isna().sum()
    df = df.loc[valid_mask].reset_index(drop=True)
    if df.empty:
        unreadable = {c: int(n) for c, n in raw_invalid_counts.items() if n}
        example = raw.iloc[0, 0] if len(raw) else ""
        raise SensorDataError(
            f"'{source_file}' has no valid data rows. Unreadable values per column: "
            f"{unreadable}. First value of the first column: {example!r}."
        )

    # 5. Sort by relative time and remove duplicate time stamps.
    report.was_out_of_order = not df["Relative_Time_Sec"].is_monotonic_increasing
    if report.was_out_of_order:
        df = df.sort_values("Relative_Time_Sec", kind="stable").reset_index(drop=True)
    duplicate_mask = df.duplicated(subset="Relative_Time_Sec", keep="first")
    report.rows_dropped_duplicate = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].reset_index(drop=True)
    report.rows_kept = int(len(df))

    # 6. Build the single DateTime column. The clock column of the raw files does
    #    not tick on every row (many rows in a row share the same clock value), so
    #    DateTime = clock of the first row + Relative_Time_Sec. That is unique,
    #    increasing and exact; passing midnight moves to the next day.
    rel_seconds = df["Relative_Time_Sec"].to_numpy(dtype="float64")
    clock = df["_seconds_of_day"].to_numpy(dtype="float64")
    elapsed = rel_seconds - rel_seconds[0]
    date_time = pd.Series(
        anchor + pd.to_timedelta(clock[0] + elapsed, unit="s").round("us"),
        index=df.index,
    )
    # Diagnostic: how far the per-row clock drifts from Relative_Time_Sec.
    day_changes = np.concatenate(([0], np.cumsum(np.diff(clock) < -SECONDS_PER_DAY / 2)))
    clock_elapsed = (clock + day_changes * SECONDS_PER_DAY) - clock[0]

    clean = pd.DataFrame(
        {
            "DateTime": date_time,
            "Relative_Time_Sec": df["Relative_Time_Sec"].astype("float64"),
        }
    )
    for col in SENSOR_COLUMNS:
        clean[col] = df[col].astype("float64")
    clean["Condition"] = final_condition
    clean["Test_Name"] = Path(source_file.replace("\\", "/")).stem
    clean["Source_File"] = source_file
    for key, value in parse_path_metadata(source_file).items():
        clean[key] = value
    clean = clean[CLEAN_COLUMNS]

    # 7. Quality metrics.
    rel = clean["Relative_Time_Sec"].to_numpy()
    report.duration_sec = float(rel[-1] - rel[0])
    if len(rel) > 1:
        steps = np.diff(rel)
        median_step = float(np.median(steps))
        report.sampling_rate_hz = round(1.0 / median_step, 3) if median_step > 0 else None
        report.max_gap_sec = float(steps.max())
        report.max_clock_mismatch_sec = float(np.abs(clock_elapsed - elapsed).max())
    report.start_datetime = clean["DateTime"].iloc[0].strftime(DATETIME_FORMAT)
    report.end_datetime = clean["DateTime"].iloc[-1].strftime(DATETIME_FORMAT)
    report.flat_sensors = ",".join(c for c in SENSOR_COLUMNS if clean[c].nunique() <= 1)

    return clean, report


def clean_zip_member(
    zip_path: Union[str, Path],
    internal_file_path: str,
    condition: Optional[str] = None,
    anchor_date: str = DEFAULT_ANCHOR_DATE,
) -> Tuple[pd.DataFrame, FileReport]:
    """Clean one workbook inside a ZIP. Drop-in upgrade of Eswar's load_sensor_data:
    the layout is detected automatically and the condition is read from the folder
    names / ZIP name when not given (see resolve_condition).
    """
    zip_path = Path(zip_path)
    condition = resolve_condition(condition, internal_file_path, zip_path.name)
    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            content = archive.read(internal_file_path)
    except FileNotFoundError as exc:
        raise SensorDataError(f"ZIP file not found: {zip_path}") from exc
    except zipfile.BadZipFile as exc:
        raise SensorDataError(f"'{zip_path}' is not a valid ZIP file.") from exc
    except KeyError as exc:
        raise SensorDataError(f"'{internal_file_path}' is not inside '{zip_path.name}'.") from exc
    return clean_sensor_bytes(content, internal_file_path, condition, anchor_date)


# ---------------------------------------------------------------------------
# Whole ZIP / whole dataset
# ---------------------------------------------------------------------------

def _is_data_member(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    posix = info.filename.replace("\\", "/")
    parts = posix.split("/")
    base = parts[-1]
    if "__MACOSX" in parts or base.startswith(("~$", ".")):
        return False
    return base.lower().endswith(SUPPORTED_EXTENSIONS)


def iter_clean_zip(
    zip_path: Union[str, Path],
    condition: Optional[str] = None,
    anchor_date: str = DEFAULT_ANCHOR_DATE,
    max_files: Optional[int] = None,
) -> Iterator[Tuple[Optional[pd.DataFrame], FileReport]]:
    """Yield (clean_df, report) for every workbook in a ZIP, one at a time
    (keeps memory low). On failure yields (None, report with status "error").
    """
    zip_path = Path(zip_path)
    explicit_condition = _normalise_condition(condition)
    try:
        archive = zipfile.ZipFile(zip_path, "r")
    except FileNotFoundError as exc:
        raise SensorDataError(f"ZIP file not found: {zip_path}") from exc
    except zipfile.BadZipFile as exc:
        raise SensorDataError(f"'{zip_path}' is not a valid ZIP file.") from exc

    with archive:
        members = sorted((i for i in archive.infolist() if _is_data_member(i)), key=lambda i: i.filename)
        if not members:
            raise SensorDataError(
                f"No {'/'.join(SUPPORTED_EXTENSIONS)} files found inside '{zip_path.name}'."
            )
        if max_files is not None:
            members = members[: max(0, max_files)]

        for info in members:
            name = info.filename
            condition = resolve_condition(explicit_condition, name, zip_path.name)
            try:
                clean, report = clean_sensor_bytes(archive.read(name), name, condition, anchor_date)
                yield clean, report
            except SkippedFileError as exc:
                logger.warning("Skipped %s: %s", name, exc)
                yield None, FileReport(
                    source_file=name,
                    status="skipped",
                    message=str(exc),
                    condition=condition or "",
                )
            except Exception as exc:  # one bad file must not stop the rest
                logger.error("Failed to process %s (%s): %s", name, zip_path.name, exc)
                yield None, FileReport(
                    source_file=name,
                    status="error",
                    message=str(exc),
                    condition=condition or "",
                )


def run_pipeline(
    zip_paths: Sequence[Union[str, Path]],
    output_dir: Union[str, Path] = "data/processed/sensors",
    anchor_date: str = DEFAULT_ANCHOR_DATE,
    max_files: Optional[int] = None,
) -> pd.DataFrame:
    """Process one or more ZIPs and write the clean tables to disk.

    Output layout:
        <output_dir>/<condition>/<file>.csv      one clean table per workbook
        <output_dir>/manifest.csv                one quality-report row per workbook

    Returns the manifest as a DataFrame. A ZIP that cannot be opened raises
    SensorDataError; failures of single workbooks are listed in the manifest.
    """
    if not zip_paths:
        raise SensorDataError("No ZIP files given.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: List[FileReport] = []
    used_paths: set = set()

    for zip_path in zip_paths:
        logger.info("Processing %s", zip_path)
        for clean, report in iter_clean_zip(zip_path, anchor_date=anchor_date, max_files=max_files):
            if clean is not None:
                folder = output_dir / (report.condition.lower() or "unknown")
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / f"{_safe_stem(report.source_file)}.csv"
                counter = 2
                while target in used_paths:
                    target = folder / f"{_safe_stem(report.source_file)}_{counter}.csv"
                    counter += 1
                used_paths.add(target)
                clean.to_csv(target, index=False, date_format=DATETIME_FORMAT)
                report.output_file = target.relative_to(output_dir).as_posix()
                logger.info(
                    "OK  %s -> %s (%d rows kept, %s Hz)",
                    report.source_file, report.output_file, report.rows_kept, report.sampling_rate_hz,
                )
            reports.append(report)

    manifest = pd.DataFrame([r.to_dict() for r in reports])
    manifest.to_csv(output_dir / "manifest.csv", index=False)

    counts = manifest["status"].value_counts()
    logger.info(
        "Finished: %d file(s): %d ok, %d skipped, %d failed. Manifest: %s",
        len(manifest), int(counts.get("ok", 0)), int(counts.get("skipped", 0)),
        int(counts.get("error", 0)), output_dir / "manifest.csv",
    )
    return manifest


# ---------------------------------------------------------------------------
# Reading the clean data back (for Nagarjun's dashboard, Krishna's EDA, tools)
# ---------------------------------------------------------------------------

def load_clean_file(path: Union[str, Path]) -> pd.DataFrame:
    """Read one clean CSV written by run_pipeline (DateTime is parsed)."""
    df = pd.read_csv(path, parse_dates=["DateTime"])
    missing = [c for c in CLEAN_COLUMNS if c not in df.columns]
    if missing:
        raise SensorDataError(f"'{path}' is not a clean sensor file; missing columns: {missing}")
    return df


def load_clean_dataset(
    output_dir: Union[str, Path] = "data/processed/sensors",
    condition: Optional[str] = None,
) -> pd.DataFrame:
    """Read all clean CSVs (optionally only 'Damaged' or 'Undamaged') into one table."""
    output_dir = Path(output_dir)
    condition = _normalise_condition(condition)
    folders = [output_dir / condition.lower()] if condition else [output_dir / "damaged", output_dir / "undamaged"]
    files = sorted(f for folder in folders if folder.is_dir() for f in folder.glob("*.csv"))
    if not files:
        raise SensorDataError(f"No clean CSV files found under '{output_dir}'. Run the pipeline first.")
    return pd.concat((load_clean_file(f) for f in files), ignore_index=True)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Clean the professor's sensor ZIP files.")
    parser.add_argument("--zip", dest="zips", action="append", required=True,
                        help="ZIP file to process (repeat for several ZIPs).")
    parser.add_argument("--out", default="data/processed/sensors", help="Output folder.")
    parser.add_argument("--anchor-date", default=DEFAULT_ANCHOR_DATE,
                        help="Calendar date used for DateTime (the raw files have no date).")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Only process the first N workbooks of each ZIP (quick test).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    try:
        manifest = run_pipeline(args.zips, args.out, args.anchor_date, args.max_files)
    except SensorDataError as exc:
        logger.error("%s", exc)
        return 2
    return 1 if (manifest["status"] == "error").any() else 0


if __name__ == "__main__":
    sys.exit(main())