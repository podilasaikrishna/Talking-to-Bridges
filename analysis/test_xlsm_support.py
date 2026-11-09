"""Quick smoke-test: verifies that sensor_pipeline._read_raw() handles .xlsm
files correctly by creating a synthetic workbook in memory and round-tripping
it through the pipeline.

Run from the repository root:
    python -m analysis.test_xlsm_support
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Make sure the repo root is on sys.path when run as a script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from openpyxl import Workbook

from analysis.sensor_pipeline import (
    SENSOR_COLUMNS,
    SensorDataError,
    _read_raw,
    clean_sensor_bytes,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_damaged_xlsm_bytes(n_rows: int = 20) -> bytes:
    """Build a minimal Damaged-layout .xlsm workbook in memory.

    Damaged layout (7 columns):
        Absolute_Time | Relative_Time_Sec | Sensor_1 .. Sensor_5
    """
    wb = Workbook()
    ws = wb.active
    ws.title = "Data"

    rng = np.random.default_rng(42)
    for i in range(n_rows):
        rel = round(i * 0.01, 6)
        abs_time = f"11:{55 + i // 60:02d}:{i % 60:02d}.000000"
        sensors = rng.uniform(0.1, 5.0, size=5).tolist()
        ws.append([abs_time, rel, *sensors])

    buf = io.BytesIO()
    # openpyxl saves as .xlsm when keep_vba=True is set; for our purposes
    # the bytes are structurally identical and pd.read_excel handles both.
    wb.save(buf)
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_read_raw_xlsm() -> None:
    """_read_raw must not raise for a .xlsm file and must return a DataFrame."""
    content = _make_damaged_xlsm_bytes()
    df = _read_raw(content, "test_file.xlsm")
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame"
    assert df.shape[1] == 7, f"Expected 7 columns, got {df.shape[1]}"
    print(f"  [PASS] _read_raw returned {df.shape[0]} rows x {df.shape[1]} cols")


def test_clean_sensor_bytes_xlsm() -> None:
    """clean_sensor_bytes must produce the standard clean columns for .xlsm."""
    content = _make_damaged_xlsm_bytes()
    clean, report = clean_sensor_bytes(content, "test_file.xlsm", condition="Damaged")

    assert report.status == "ok", f"Expected status=ok, got {report.status!r}: {report.message}"
    assert report.layout == "Damaged", f"Expected layout=Damaged, got {report.layout!r}"
    for col in SENSOR_COLUMNS:
        assert col in clean.columns, f"Missing sensor column: {col}"
    assert "DateTime" in clean.columns
    assert "Relative_Time_Sec" in clean.columns
    print(
        f"  [PASS] clean_sensor_bytes: {report.rows_kept} rows kept, "
        f"layout={report.layout}, rate={report.sampling_rate_hz} Hz"
    )


def test_openpyxl_engine_is_used() -> None:
    """Confirm openpyxl is importable (required for .xlsm support)."""
    try:
        import openpyxl  # noqa: F401
        print(f"  [PASS] openpyxl {openpyxl.__version__} is installed")
    except ImportError:
        raise AssertionError(
            "openpyxl is not installed! Run: pip install openpyxl"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_openpyxl_engine_is_used,
        test_read_raw_xlsm,
        test_clean_sensor_bytes_xlsm,
    ]
    failures = 0
    print("\n=== .xlsm support smoke-test ===\n")
    for t in tests:
        print(f"Running {t.__name__} ...")
        try:
            t()
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            failures += 1
    print(f"\n{'All tests passed' if failures == 0 else f'{failures} test(s) FAILED'}\n")
    return failures


if __name__ == "__main__":
    sys.exit(main())
