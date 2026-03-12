"""Pydantic models for document data structures."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentStatus(str, Enum):
    """Possible states of a document during OCR processing."""

    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class BBoxCoords(BaseModel):
    """Bounding box with four corner coordinates."""

    top_left: List[float] = Field(..., description="[x, y] of top-left corner")
    top_right: List[float] = Field(..., description="[x, y] of top-right corner")
    bottom_right: List[float] = Field(..., description="[x, y] of bottom-right corner")
    bottom_left: List[float] = Field(..., description="[x, y] of bottom-left corner")


class TextBlock(BaseModel):
    """A single OCR-recognized text block (line)."""

    block_id: int
    text: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    bbox: BBoxCoords
    line_number: int


class PageResult(BaseModel):
    """OCR results for a single page."""

    page_number: int
    width: int
    height: int
    text_blocks: List[TextBlock] = []
    full_text: str = ""
    block_count: int = 0
    avg_confidence: float = 0.0
    status: str = "completed"  # "completed" | "failed" | "empty"
    error: Optional[str] = None


class DocumentResult(BaseModel):
    """Full OCR result for an entire document."""

    document_id: str
    filename: str
    total_pages: int
    ocr_engine: str = "paddleocr"
    language: str = "korean"
    created_at: datetime
    processing_time_seconds: Optional[float] = None
    pages: List[PageResult] = []


class DocumentMeta(BaseModel):
    """Lightweight document metadata stored alongside the full result."""

    document_id: str
    filename: str
    total_pages: int
    status: DocumentStatus = DocumentStatus.pending
    processed_pages: int = 0
    progress_percent: float = 0.0
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class DocumentStatusResponse(BaseModel):
    """Response body for GET /api/documents/{id}/status."""

    document_id: str
    status: DocumentStatus
    processed_pages: int
    total_pages: int
    progress_percent: float


class DocumentListItem(BaseModel):
    """Summary item for the documents list endpoint."""

    document_id: str
    filename: str
    total_pages: int
    status: DocumentStatus
    created_at: datetime
