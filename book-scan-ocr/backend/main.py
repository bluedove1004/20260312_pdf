"""FastAPI application entry point."""

from __future__ import annotations

import logging
import logging.config
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import documents as doc_router
from routers import settings as settings_router
from services.ocr_service import OCRService

# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(settings.logs_dir / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Lifespan: initialise heavy resources once
# ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise PaddleOCR at startup; clean up on shutdown."""
    logger.info("Starting up — initialising OCR service...")
    app.state.ocr_service = OCRService(
        lang=settings.ocr_language,
        use_angle_cls=settings.use_angle_cls,
        use_gpu=settings.use_gpu,
    )
    logger.info("OCR service ready")
    yield
    logger.info("Shutting down")


# ──────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────

app = FastAPI(
    title="Book Scan OCR API",
    description="Upload scanned book PDFs and extract text via PaddleOCR.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server (and any localhost variant)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(doc_router.router)
app.include_router(settings_router.router)


@app.get("/health")
def health() -> dict:
    """Simple health-check endpoint."""
    return {"status": "ok", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug)
