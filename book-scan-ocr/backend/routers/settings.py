"""Settings API router: persist and retrieve OCR configuration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from config import settings
from models.settings import APIKeyRequest, OCRSettings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["settings"])


@router.post("/settings")
def save_settings(payload: APIKeyRequest) -> Dict[str, str]:
    """
    Persist OCR provider, API key, DPI, and preprocessing settings to disk.

    The API key is stored as plain text in the local settings file.
    For production use, encrypt the key at rest.
    """
    data: Dict[str, Any] = {
        "ocr_provider": payload.ocr_provider.value,
        "api_key": payload.api_key,
        "dpi": payload.dpi or settings.default_dpi,
        "preprocessing": payload.preprocessing.model_dump() if payload.preprocessing else {
            "grayscale": True,
            "binarization": False,
            "denoise": False,
            "deskew": False,
        },
    }
    try:
        settings.settings_file.parent.mkdir(parents=True, exist_ok=True)
        settings.settings_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info("Settings saved: provider=%s", payload.ocr_provider.value)
        return {"status": "ok", "message": "Settings saved successfully"}
    except Exception as e:
        logger.error("Failed to save settings: %s", e)
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")


@router.get("/settings")
def get_settings() -> Dict[str, Any]:
    """Retrieve current OCR settings from disk."""
    if not settings.settings_file.exists():
        # Return defaults
        return {
            "ocr_provider": "paddleocr",
            "api_key": None,
            "dpi": settings.default_dpi,
            "preprocessing": {
                "grayscale": settings.enable_grayscale,
                "binarization": settings.enable_binarization,
                "denoise": settings.enable_denoise,
                "deskew": settings.enable_deskew,
            },
        }
    try:
        data = json.loads(settings.settings_file.read_text(encoding="utf-8"))
        # Mask API key in response
        if data.get("api_key"):
            data["api_key"] = "****" + data["api_key"][-4:]
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read settings: {e}")
