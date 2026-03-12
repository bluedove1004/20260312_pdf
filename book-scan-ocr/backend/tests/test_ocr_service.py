"""Unit tests for OCRService result parsing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock PaddleOCR before import
with patch.dict("sys.modules", {"paddleocr": MagicMock(), "paddlepaddle": MagicMock()}):
    from services.ocr_service import OCRService


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _make_ocr_line(text: str, confidence: float, pts=None):
    if pts is None:
        pts = [[0.0, 0.0], [100.0, 0.0], [100.0, 30.0], [0.0, 30.0]]
    return [pts, (text, confidence)]


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────


def test_parse_empty_result():
    """Empty PaddleOCR output should produce an 'empty' PageResult."""
    service = OCRService.__new__(OCRService)
    service.ocr = MagicMock()
    result = service._parse_result([], page_number=1, width=1000, height=1400)
    assert result.status == "empty"
    assert result.block_count == 0
    assert result.full_text == ""


def test_parse_single_line():
    service = OCRService.__new__(OCRService)
    service.ocr = MagicMock()
    raw = [[_make_ocr_line("안녕하세요", 0.95)]]
    result = service._parse_result(raw, page_number=1, width=800, height=1200)
    assert result.status == "completed"
    assert len(result.text_blocks) == 1
    assert result.text_blocks[0].text == "안녕하세요"
    assert result.text_blocks[0].confidence == pytest.approx(0.95, abs=1e-3)
    assert result.avg_confidence == pytest.approx(0.95, abs=1e-3)


def test_parse_multiple_lines():
    service = OCRService.__new__(OCRService)
    service.ocr = MagicMock()
    raw = [[
        _make_ocr_line("제1장", 0.98),
        _make_ocr_line("선사시대", 0.90),
    ]]
    result = service._parse_result(raw, page_number=2, width=800, height=1200)
    assert result.block_count == 2
    assert "제1장" in result.full_text
    assert "선사시대" in result.full_text
    assert result.avg_confidence == pytest.approx((0.98 + 0.90) / 2, abs=1e-3)


def test_parse_skips_empty_text():
    service = OCRService.__new__(OCRService)
    service.ocr = MagicMock()
    raw = [[
        _make_ocr_line("   ", 0.80),    # whitespace only — should be skipped
        _make_ocr_line("내용", 0.92),
    ]]
    result = service._parse_result(raw, page_number=1, width=800, height=1200)
    assert result.block_count == 1
    assert result.text_blocks[0].text == "내용"


def test_bbox_coordinates_mapped_correctly():
    pts = [[10.0, 20.0], [300.0, 20.0], [300.0, 60.0], [10.0, 60.0]]
    service = OCRService.__new__(OCRService)
    raw = [[_make_ocr_line("테스트", 0.88, pts=pts)]]
    result = service._parse_result(raw, page_number=1, width=800, height=1200)
    bbox = result.text_blocks[0].bbox
    assert bbox.top_left == [10.0, 20.0]
    assert bbox.top_right == [300.0, 20.0]
    assert bbox.bottom_right == [300.0, 60.0]
    assert bbox.bottom_left == [10.0, 60.0]
