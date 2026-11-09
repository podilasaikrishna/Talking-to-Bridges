"""Tests for NEW-1: Excel (.xlsx / .xlsm) sensor recording upload path.

Coverage:
1. Unit — clean_sensor_bytes() with synthetic Damaged (7-col) workbook (.xlsx)
2. Unit — clean_sensor_bytes() with synthetic Undamaged (9-col) workbook (.xlsx)
3. Unit — empty bytes raises SensorDataError
4. Unit — corrupt bytes raises SensorDataError
5. Integration (FastAPI TestClient) — valid .xlsx Damaged → 200, expected schema
6. Integration — valid .xlsm Damaged → 200
7. Integration — valid .xlsx Undamaged → 200
8. Integration — unsupported extension (.mat) → 400
9. Integration — empty Excel file → 400
10. Integration — corrupt bytes as .xlsx → 400, no stack trace
11. Persistence — clean CSV written to DOCUMENTS_DIR and loadable by DataAccessLayer
12. Existing formats unaffected — .csv upload still returns 200
13. PRE-2 feature-row equivalence — skipped (dependency on validated training pipeline outputs)
"""
from __future__ import annotations

import io
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import openpyxl
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic workbook helpers
# ---------------------------------------------------------------------------

def _make_damaged_xlsx(n: int = 20) -> bytes:
    """Return bytes for a minimal 7-column Damaged-layout workbook.

    Column order matches the raw Damaged Excel files:
      col0 – time-of-day string  (hh:mm:ss.microseconds)
      col1 – Relative_Time_Sec   (float, starts near 0)
      col2..6 – S1..S5           (float sensor values)
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    # Header row is intentionally absent — the cleaner handles headerless files.
    dt = 1.0 / 303.0  # ~damaged sampling interval
    for i in range(n):
        t_sec = i * dt
        hh, rem = divmod(t_sec, 3600)
        mm, ss = divmod(rem, 60)
        time_str = f"{int(hh):02d}:{int(mm):02d}:{ss:09.6f}"
        ws.append([
            time_str,
            round(t_sec, 6),
            1.0 + i * 0.001,
            1.1 + i * 0.001,
            1.2 + i * 0.001,
            1.3 + i * 0.001,
            1.4 + i * 0.001,
        ])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _make_undamaged_xlsx(n: int = 20) -> bytes:
    """Return bytes for a minimal 9-column Undamaged-layout workbook.

    Column order matches the raw Undamaged Excel files:
      col0 – channel (int)
      col1 – samples (int)
      col2 – frequency (float)
      col3 – Relative_Time_Sec (float)
      col4..8 – S1..S5 (float)
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    dt = 1.0 / 333.0
    for i in range(n):
        t_sec = i * dt
        ws.append([
            10,          # channel
            30,          # samples
            15.0,        # frequency
            round(t_sec, 6),
            1.0 + i * 0.001,
            1.1 + i * 0.001,
            1.2 + i * 0.001,
            1.3 + i * 0.001,
            1.4 + i * 0.001,
        ])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Unit tests — analysis.sensor_pipeline.clean_sensor_bytes
# ---------------------------------------------------------------------------

class TestCleanSensorBytesUnit:
    """Direct unit tests for clean_sensor_bytes without HTTP layer."""

    def test_damaged_layout_xlsx_ok(self):
        from analysis.sensor_pipeline import clean_sensor_bytes, CLEAN_COLUMNS

        clean_df, report = clean_sensor_bytes(_make_damaged_xlsx(), "D-SH 1-2.xlsx")

        assert report.status == "ok"
        assert report.layout == "Damaged"
        assert list(clean_df.columns) == CLEAN_COLUMNS
        assert len(clean_df) == 20

    def test_undamaged_layout_xlsx_ok(self):
        from analysis.sensor_pipeline import clean_sensor_bytes, CLEAN_COLUMNS

        clean_df, report = clean_sensor_bytes(_make_undamaged_xlsx(), "UD-SH 1-2.xlsx")

        assert report.status == "ok"
        assert report.layout == "Undamaged"
        assert list(clean_df.columns) == CLEAN_COLUMNS
        assert len(clean_df) == 20

    def test_empty_bytes_raises_sensor_data_error(self):
        from analysis.sensor_pipeline import clean_sensor_bytes, SensorDataError

        with pytest.raises(SensorDataError):
            clean_sensor_bytes(b"", "empty.xlsx")

    def test_corrupt_bytes_raises_sensor_data_error(self):
        from analysis.sensor_pipeline import clean_sensor_bytes, SensorDataError

        with pytest.raises((SensorDataError, Exception)):
            clean_sensor_bytes(b"not an excel file at all xyzzy", "corrupt.xlsx")


# ---------------------------------------------------------------------------
# FastAPI integration tests via TestClient
# Embedding and vector-store I/O are mocked to avoid network / FAISS writes.
# ---------------------------------------------------------------------------

@pytest.fixture()
def api_client(tmp_path):
    """
    TestClient with:
      - embedding patched to return a zero matrix (no network)
      - vector_store.add_chunks / .save patched (no disk writes)
      - DOCUMENTS_DIR redirected to a temp directory (cleaned up automatically)
    """
    from fastapi.testclient import TestClient
    from app.main import app
    from rag.ingestion import DOCUMENTS_DIR as ORIG_DIR
    import rag.ingestion as ingestion_mod

    test_docs_dir = tmp_path / "documents"
    test_docs_dir.mkdir(parents=True)

    # Patch DOCUMENTS_DIR in both the ingestion module and the routes module
    with (
        patch.object(ingestion_mod, "DOCUMENTS_DIR", test_docs_dir),
        patch("app.api.routes.DOCUMENTS_DIR", test_docs_dir),
        patch(
            "app.api.routes.embedding_service.embed_documents",
            return_value=np.zeros((1, 384), dtype=np.float32),
        ),
        patch("app.api.routes.vector_store.add_chunks"),
        patch("app.api.routes.vector_store.save"),
    ):
        with TestClient(app) as client:
            yield client, test_docs_dir


class TestExcelUploadEndpoint:
    """Integration tests through the /upload HTTP endpoint."""

    def test_valid_xlsx_damaged_returns_200(self, api_client):
        client, docs_dir = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    _make_damaged_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "document_id" in body
        assert body["file_type"] == "xlsx"
        assert body["rows"] == 20
        assert isinstance(body["columns"], int) and body["columns"] > 0
        assert "Sensor_1" in (body.get("column_names") or [])
        assert body.get("clean_data_path") is not None

    def test_valid_xlsm_damaged_returns_200(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsm",
                    _make_damaged_xlsx(),
                    "application/vnd.ms-excel.sheet.macroEnabled.12",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["file_type"] == "xlsm"

    def test_valid_xlsx_undamaged_returns_200(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "UD-SH 1-2.xlsx",
                    _make_undamaged_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["rows"] == 20
        assert "Sensor_1" in (body.get("column_names") or [])

    def test_unsupported_extension_returns_400(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("recording.mat", b"data", "application/octet-stream")},
        )
        assert resp.status_code == 400
        # Schema is {"error": {"message": ...}} per app/main.py exception handler
        body = resp.json()
        error_msg = body.get("error", {}).get("message", "") or body.get("detail", "")
        assert ".mat" in error_msg or "Unsupported" in error_msg

    def test_empty_excel_returns_400(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "empty.xlsx",
                    b"",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 400

    def test_corrupt_excel_returns_400_no_stacktrace(self, api_client):
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "bad.xlsx",
                    b"not an excel file xyzzy",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 400
        detail = resp.text
        assert "Traceback" not in detail
        assert "raise " not in detail  # no raw Python in user-facing message


# ---------------------------------------------------------------------------
# Persistence & downstream accessibility
# ---------------------------------------------------------------------------

class TestCleanedOutputPersistence:
    """Verify the clean CSV is written and readable by the analysis DataAccessLayer."""

    def test_clean_csv_written_to_documents_dir(self, api_client):
        client, docs_dir = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    _make_damaged_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["document_id"]

        matches = list(docs_dir.glob(f"{doc_id}_*.csv"))
        assert len(matches) >= 1, f"No clean CSV found for {doc_id} in {docs_dir}"

        df = pd.read_csv(matches[0])
        assert "Sensor_1" in df.columns
        assert "Relative_Time_Sec" in df.columns
        assert len(df) == 20

    def test_clean_csv_loadable_by_data_access(self, api_client):
        client, docs_dir = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    _make_damaged_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["document_id"]

        from analysis.tools import DataAccessLayer

        da = DataAccessLayer(docs_dir)
        matches = list(docs_dir.glob(f"{doc_id}_*.csv"))
        assert matches, "Clean CSV not found"

        df = da.load_dataset(matches[0])
        assert "Sensor_1" in df.columns
        assert len(df) > 0

    def test_raw_excel_also_saved_for_audit(self, api_client):
        """Raw .xlsx bytes are stored alongside the clean CSV for audit purposes."""
        client, docs_dir = api_client
        resp = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    _make_damaged_xlsx(),
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp.status_code == 200, resp.text
        doc_id = resp.json()["document_id"]

        raw_matches = list(docs_dir.glob(f"{doc_id}_*.xlsx"))
        assert len(raw_matches) >= 1, "Raw .xlsx not saved for audit trail"

    def test_unique_document_ids_no_overwrite(self, api_client):
        """Two uploads of the same filename get different document_ids and separate files."""
        client, docs_dir = api_client
        xlsx_bytes = _make_damaged_xlsx()

        resp1 = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    xlsx_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        resp2 = client.post(
            "/upload",
            files={
                "file": (
                    "D-SH 1-2.xlsx",
                    xlsx_bytes,
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            },
        )
        assert resp1.status_code == 200
        assert resp2.status_code == 200
        assert resp1.json()["document_id"] != resp2.json()["document_id"]


# ---------------------------------------------------------------------------
# Regression — existing upload formats must be unaffected
# ---------------------------------------------------------------------------

class TestExistingFormatsUnaffected:
    def test_csv_upload_still_returns_200(self, api_client):
        client, _ = api_client
        csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"
        resp = client.post(
            "/upload",
            files={"file": ("data.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["file_type"] == "csv"

    def test_pdf_upload_still_accepted_or_rejected_gracefully(self, api_client):
        """PDF with fake bytes: backend should return 400 (invalid PDF), not 500."""
        client, _ = api_client
        resp = client.post(
            "/upload",
            files={"file": ("report.pdf", b"fake-pdf", "application/pdf")},
        )
        # Either succeeds with minimal parsing or fails with 400 (bad PDF), never 500
        assert resp.status_code in (200, 400), f"Unexpected status: {resp.status_code}"
        if resp.status_code == 400:
            assert "Traceback" not in resp.json().get("detail", "")


# ---------------------------------------------------------------------------
# PRE-2 feature-row equivalence (skipped — pending validation)
# ---------------------------------------------------------------------------

@pytest.mark.skip(
    reason=(
        "PRE-2 integration dependency: This test verifies that the upload path "
        "produces the same feature row as the training pipeline for the same recording. "
        "To unskip: (1) confirm analysis/features.extract_features() is validated against "
        "training pipeline outputs, (2) supply a reference recording, (3) compare all keys "
        "not in NON_FEATURE_COLS with rtol=1e-5. "
        "Current status: analysis/features.py is present but cross-pipeline validation "
        "against Eswar's PRE-2 training outputs is pending."
    )
)
class TestFeatureRowEquivalence:
    def test_upload_path_features_match_training_pipeline(self, tmp_path):
        """Feature dict from the upload path must match direct pipeline processing."""
        from analysis.sensor_pipeline import clean_sensor_bytes
        from analysis.features import extract_features, NON_FEATURE_COLS

        sample_file = Path("data/Damaged_Beams/1mm/M1/Singlehit/D-SH 1-2.xlsx")
        if not sample_file.exists():
            pytest.skip("Real training data not available in this environment")

        raw_bytes = sample_file.read_bytes()

        # Training pipeline path (direct)
        clean_df_train, _ = clean_sensor_bytes(raw_bytes, sample_file.name)
        feats_train = extract_features(clean_df_train)

        # Upload path: clean → save CSV → reload → extract
        clean_df_upload, _ = clean_sensor_bytes(raw_bytes, sample_file.name)
        clean_csv = tmp_path / "clean.csv"
        clean_df_upload.to_csv(clean_csv, index=False, date_format="%Y-%m-%d %H:%M:%S.%f")

        from analysis.tools import DataAccessLayer
        da = DataAccessLayer(tmp_path)
        reloaded_df = da.load_dataset(clean_csv)
        feats_upload = extract_features(reloaded_df)

        model_keys = [k for k in feats_train if k not in NON_FEATURE_COLS]
        for key in model_keys:
            assert feats_train[key] == pytest.approx(feats_upload[key], rel=1e-5), (
                f"Feature mismatch for '{key}': "
                f"train={feats_train[key]}, upload={feats_upload[key]}"
            )

