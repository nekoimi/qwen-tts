"""Optional persistence of raw uploaded reference audio."""

from __future__ import annotations

from pathlib import Path

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)


def save_raw_upload(voice_id: str, data: bytes, suffix: str = ".wav") -> Path | None:
    if not settings.SAVE_RAW_UPLOADS:
        return None
    raw_dir = settings.VOICE_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{voice_id}{suffix}"
    path.write_bytes(data)
    logger.debug("Saved raw upload to %s", path)
    return path
