"""Pydantic models for OCR settings."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class OCRProvider(str, Enum):
    """Supported OCR providers."""

    paddleocr = "paddleocr"
    google_vision = "google_vision"
    azure = "azure"


class PreprocessingOptions(BaseModel):
    """Image preprocessing toggles."""

    grayscale: bool = True
    binarization: bool = False
    denoise: bool = False
    deskew: bool = False


class DPIOption(int, Enum):
    """Supported DPI values for PDF-to-image conversion."""

    low = 150
    medium = 200
    high = 300
    ultra = 400


class OCRSettings(BaseModel):
    """Full OCR settings persisted to disk."""

    ocr_provider: OCRProvider = OCRProvider.paddleocr
    api_key: Optional[str] = None
    dpi: DPIOption = DPIOption.high
    preprocessing: PreprocessingOptions = PreprocessingOptions()


class APIKeyRequest(BaseModel):
    """Request body for POST /api/settings."""

    ocr_provider: OCRProvider
    api_key: Optional[str] = None
    dpi: Optional[int] = 300
    preprocessing: Optional[PreprocessingOptions] = None
