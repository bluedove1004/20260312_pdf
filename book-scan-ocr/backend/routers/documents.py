"""Documents API router: upload, status, retrieval, and image endpoints."""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse

from config import settings
from models.document import (
    DocumentListItem,
    DocumentMeta,
    DocumentResult,
    DocumentStatus,
    DocumentStatusResponse,
)
from models.settings import PreprocessingOptions
from services.ocr_service import OCRService
from services.pdf_service import PDFService
from services.storage_service import StorageService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["documents"])

# Module-level service instances (injected at startup via app.state)
_storage = StorageService()
_pdf = PDFService()


def get_ocr_service(request: Request) -> OCRService:
    """Dependency: retrieve the app-level OCR service singleton."""
    return request.app.state.ocr_service


# ──────────────────────────────────────────────
# Background processing
# ──────────────────────────────────────────────


def _run_ocr_pipeline(
    document_id: str,
    pdf_path: Path,
    filename: str,
    dpi: int,
    preprocessing: PreprocessingOptions,
    ocr_service: OCRService,
) -> None:
    """
    Full OCR pipeline executed in background:
      1. Convert PDF pages to images
      2. Run OCR on each page
      3. Assemble and save the final result JSON
    """
    meta = _storage.load_meta(document_id)
    if meta is None:
        logger.error("Meta not found for document %s – aborting pipeline", document_id)
        return

    start_time = time.time()
    images_dir = _storage.images_dir(document_id)

    try:
        # Step 1 – PDF → images
        logger.info("[%s] Converting PDF to images (dpi=%d)", document_id, dpi)
        total_pages = _pdf.convert_pdf_to_images(pdf_path, images_dir, dpi=dpi, preprocessing=preprocessing)

        _storage.update_meta(document_id, total_pages=total_pages, status=DocumentStatus.processing)
        meta = _storage.load_meta(document_id)

        # Step 2 – OCR each page
        page_results = []
        for page_num in range(1, total_pages + 1):
            img_path = images_dir / f"page_{page_num:04d}.png"
            logger.info("[%s] OCR page %d/%d", document_id, page_num, total_pages)

            page_result = ocr_service.process_page(img_path, page_num)
            page_dict = page_result.model_dump()
            _storage.save_page_result(document_id, page_dict)
            page_results.append(page_result)

            # Update progress
            progress = round((page_num / total_pages) * 100, 1)
            _storage.update_meta(
                document_id,
                processed_pages=page_num,
                progress_percent=progress,
            )

        # Step 3 – Assemble full result
        elapsed = round(time.time() - start_time, 2)
        result = DocumentResult(
            document_id=document_id,
            filename=filename,
            total_pages=total_pages,
            created_at=meta.created_at,
            processing_time_seconds=elapsed,
            pages=page_results,
        )
        _storage.save_result(result)
        _storage.update_meta(
            document_id,
            status=DocumentStatus.completed,
            progress_percent=100.0,
            completed_at=datetime.now(tz=timezone.utc),
        )
        logger.info("[%s] Completed in %.1fs", document_id, elapsed)

    except Exception as e:
        logger.exception("[%s] Pipeline failed: %s", document_id, e)
        _storage.update_meta(document_id, status=DocumentStatus.failed, error=str(e))


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Accept a PDF upload, persist it, and start OCR in the background.

    Returns: document_id, total_pages (0 until processing starts), status.
    """
    # Validate content type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Read and size-check
    content = await file.read()
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_file_size_mb} MB",
        )

    document_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name
    pdf_path = _storage.upload_path(document_id, safe_name)

    # Save uploaded file
    async with aiofiles.open(pdf_path, "wb") as f:
        await f.write(content)

    # Create document directories and initial metadata
    _storage.create_document_dirs(document_id)
    meta = DocumentMeta(
        document_id=document_id,
        filename=safe_name,
        total_pages=0,
        status=DocumentStatus.pending,
        created_at=datetime.now(tz=timezone.utc),
    )
    _storage.save_meta(meta)

    # Load current settings for preprocessing/DPI
    ocr_service: OCRService = get_ocr_service(request)
    app_settings = _load_current_settings()
    preprocessing = PreprocessingOptions(**app_settings.get("preprocessing", {}))
    dpi = app_settings.get("dpi", settings.default_dpi)

    # Kick off background OCR
    background_tasks.add_task(
        _run_ocr_pipeline,
        document_id,
        pdf_path,
        safe_name,
        dpi,
        preprocessing,
        ocr_service,
    )

    return {"document_id": document_id, "total_pages": 0, "status": "pending"}


@router.get("/documents", response_model=List[DocumentListItem])
def list_documents() -> List[DocumentListItem]:
    """Return all uploaded documents with summary metadata."""
    return _storage.list_documents()


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def get_document_status(document_id: str) -> DocumentStatusResponse:
    """Return the current OCR processing status for a document."""
    meta = _storage.load_meta(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentStatusResponse(
        document_id=meta.document_id,
        status=meta.status,
        processed_pages=meta.processed_pages,
        total_pages=meta.total_pages,
        progress_percent=meta.progress_percent,
    )


@router.get("/documents/{document_id}")
def get_document(document_id: str) -> Any:
    """Return the full OCR result JSON for a completed document."""
    meta = _storage.load_meta(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if meta.status != DocumentStatus.completed:
        raise HTTPException(status_code=409, detail=f"Document status is '{meta.status}', not completed")
    result = _storage.load_result(document_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Result file missing")
    return result


@router.get("/documents/{document_id}/pages/{page_number}")
def get_page(document_id: str, page_number: int) -> Any:
    """Return the OCR result for a single page."""
    meta = _storage.load_meta(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    page_data = _storage.load_page_result(document_id, page_number)
    if page_data is None:
        raise HTTPException(status_code=404, detail=f"Page {page_number} not found")
    return page_data


@router.get("/documents/{document_id}/pages/{page_number}/image")
def get_page_image(document_id: str, page_number: int) -> FileResponse:
    """Return the original scanned page image (PNG)."""
    img_path = _storage.get_image_path(document_id, page_number)
    if img_path is None:
        raise HTTPException(status_code=404, detail=f"Image for page {page_number} not found")
    return FileResponse(str(img_path), media_type="image/png")


@router.get("/documents/{document_id}/download")
def download_document(document_id: str) -> FileResponse:
    """Download the full result JSON file."""
    meta = _storage.load_meta(document_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    result_path = _storage._result_path(document_id)
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not available yet")
    return FileResponse(
        str(result_path),
        media_type="application/json",
        filename=f"{document_id}_result.json",
    )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────


def _load_current_settings() -> dict:
    """Load persisted OCR settings from disk, returning defaults on failure."""
    if settings.settings_file.exists():
        try:
            return json.loads(settings.settings_file.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "dpi": settings.default_dpi,
        "preprocessing": {
            "grayscale": settings.enable_grayscale,
            "binarization": settings.enable_binarization,
            "denoise": settings.enable_denoise,
            "deskew": settings.enable_deskew,
        },
    }
