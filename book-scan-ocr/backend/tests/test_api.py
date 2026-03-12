"""API endpoint tests using httpx AsyncClient."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add backend dir to path so imports resolve correctly
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock PaddleOCR before importing main to avoid heavyweight initialization
paddleocr_mock = MagicMock()
paddlepaddle_mock = MagicMock()

with patch.dict("sys.modules", {"paddleocr": paddleocr_mock, "paddlepaddle": paddlepaddle_mock}):
    from main import app  # noqa: E402

client = TestClient(app)


# ──────────────────────────────────────────────
# Health check
# ──────────────────────────────────────────────


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ──────────────────────────────────────────────
# Upload
# ──────────────────────────────────────────────


def test_upload_rejects_non_pdf():
    """Non-PDF uploads must be rejected with 400."""
    fake_file = io.BytesIO(b"not a pdf")
    resp = client.post("/api/upload", files={"file": ("test.txt", fake_file, "text/plain")})
    assert resp.status_code == 400


def test_upload_accepts_pdf(tmp_path):
    """Valid PDF uploads should return a document_id and status pending."""
    # Minimal valid PDF binary
    fake_pdf = b"%PDF-1.4 fake content"
    with patch("routers.documents._run_ocr_pipeline"):
        with patch("routers.documents._pdf.convert_pdf_to_images", return_value=5):
            resp = client.post(
                "/api/upload",
                files={"file": ("test.pdf", io.BytesIO(fake_pdf), "application/pdf")},
            )
    assert resp.status_code == 200
    body = resp.json()
    assert "document_id" in body
    assert body["status"] == "pending"


# ──────────────────────────────────────────────
# Document listing
# ──────────────────────────────────────────────


def test_list_documents_returns_list():
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ──────────────────────────────────────────────
# Document status
# ──────────────────────────────────────────────


def test_status_404_unknown():
    resp = client.get("/api/documents/nonexistent-id/status")
    assert resp.status_code == 404


# ──────────────────────────────────────────────
# Settings
# ──────────────────────────────────────────────


def test_get_settings_returns_defaults():
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert "ocr_provider" in body
    assert "dpi" in body


def test_post_settings(tmp_path):
    payload = {
        "ocr_provider": "paddleocr",
        "api_key": None,
        "dpi": 300,
        "preprocessing": {
            "grayscale": True,
            "binarization": False,
            "denoise": False,
            "deskew": False,
        },
    }
    resp = client.post("/api/settings", json=payload)
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ──────────────────────────────────────────────
# Page endpoints with missing document
# ──────────────────────────────────────────────


def test_get_page_404():
    resp = client.get("/api/documents/no-such-doc/pages/1")
    assert resp.status_code == 404


def test_get_page_image_404():
    resp = client.get("/api/documents/no-such-doc/pages/1/image")
    assert resp.status_code == 404
