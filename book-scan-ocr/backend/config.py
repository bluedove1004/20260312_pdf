"""Application configuration management."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # File storage
    base_dir: Path = Path(__file__).parent
    upload_dir: Path = Path(__file__).parent / "uploads"
    processed_dir: Path = Path(__file__).parent / "processed"
    logs_dir: Path = Path(__file__).parent / "logs"

    # File limits
    max_file_size_mb: int = 100

    # OCR settings
    default_dpi: int = 300
    ocr_language: str = "korean"
    use_gpu: bool = False
    use_angle_cls: bool = True

    # Preprocessing defaults
    enable_grayscale: bool = True
    enable_binarization: bool = False
    enable_denoise: bool = False
    enable_deskew: bool = False

    # Settings file
    settings_file: Path = Path(__file__).parent / "processed" / "settings.json"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [self.upload_dir, self.processed_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
