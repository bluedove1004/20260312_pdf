"""OCR service: PaddleOCR-based text extraction from page images."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.document import BBoxCoords, PageResult, TextBlock

logger = logging.getLogger(__name__)


class OCRService:
    """
    Singleton-style OCR service wrapping PaddleOCR.

    Initialize once at app startup and reuse across all requests.
    PaddleOCR initialization is expensive (model loading) so we
    intentionally avoid re-instantiating on every request.
    """

    def __init__(self, lang: str = "korean", use_angle_cls: bool = True, use_gpu: bool = False) -> None:
        """
        Initialize PaddleOCR with given options.

        Args:
            lang: OCR language model to use.
            use_angle_cls: Enable text angle classification.
            use_gpu: Enable GPU inference (requires CUDA).
        """
        logger.info("Initializing PaddleOCR (lang=%s, gpu=%s)...", lang, use_gpu)
        try:
            from paddleocr import PaddleOCR

            self.ocr = PaddleOCR(
                lang=lang,
                use_angle_cls=use_angle_cls,
                use_gpu=use_gpu,
                show_log=False,
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error("Failed to initialize PaddleOCR: %s", e)
            self.ocr = None

    def process_page(self, image_path: Path, page_number: int) -> PageResult:
        """
        Run OCR on a single page image and return structured results.

        Args:
            image_path: Path to the page PNG image.
            page_number: 1-based page number.

        Returns:
            PageResult with all detected text blocks.
        """
        if self.ocr is None:
            return PageResult(
                page_number=page_number,
                width=0,
                height=0,
                status="failed",
                error="PaddleOCR not initialized",
            )

        try:
            from PIL import Image as PILImage

            with PILImage.open(image_path) as img:
                width, height = img.size

            result = self.ocr.ocr(str(image_path), cls=True)
            return self._parse_result(result, page_number, width, height)
        except Exception as e:
            logger.error("OCR failed on page %d (%s): %s", page_number, image_path, e)
            return PageResult(
                page_number=page_number,
                width=0,
                height=0,
                status="failed",
                error=str(e),
            )

    def _parse_result(
        self,
        raw_result: Any,
        page_number: int,
        width: int,
        height: int,
    ) -> PageResult:
        """
        Convert PaddleOCR's raw output into a structured PageResult.

        PaddleOCR output structure (per image):
          result = [ [line, ...], ... ]   (outer list = image batch)
          line   = [ bbox_points, (text, confidence) ]
          bbox_points = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        text_blocks: List[TextBlock] = []

        # Flatten the batch dimension (we always process one image at a time)
        lines: List[Any] = []
        if raw_result and isinstance(raw_result, list):
            for item in raw_result:
                if item:
                    lines.extend(item)

        for idx, line in enumerate(lines, start=1):
            try:
                pts, (text, confidence) = line
                if not text or not text.strip():
                    continue

                # PaddleOCR gives 4 points: TL, TR, BR, BL
                tl, tr, br, bl = pts

                block = TextBlock(
                    block_id=idx,
                    text=text.strip(),
                    confidence=round(float(confidence), 4),
                    bbox=BBoxCoords(
                        top_left=[round(tl[0], 1), round(tl[1], 1)],
                        top_right=[round(tr[0], 1), round(tr[1], 1)],
                        bottom_right=[round(br[0], 1), round(br[1], 1)],
                        bottom_left=[round(bl[0], 1), round(bl[1], 1)],
                    ),
                    line_number=idx,
                )
                text_blocks.append(block)
            except Exception as e:
                logger.warning("Skipping malformed OCR line on page %d: %s", page_number, e)

        full_text = "\n".join(b.text for b in text_blocks)
        confidences = [b.confidence for b in text_blocks]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        return PageResult(
            page_number=page_number,
            width=width,
            height=height,
            text_blocks=text_blocks,
            full_text=full_text,
            block_count=len(text_blocks),
            avg_confidence=avg_confidence,
            status="empty" if not text_blocks else "completed",
        )
