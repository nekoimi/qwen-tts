"""Application configuration from environment variables."""

from __future__ import annotations

import os
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class Settings:
    MODEL_ID: str = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")
    VOICE_DIR: Path = Path(os.getenv("VOICE_DIR", "data/voices"))
    TTS_MAX_CONCURRENT: int = _env_int("TTS_MAX_CONCURRENT", 2)
    DEVICE: str = os.getenv("DEVICE", "cuda:0")
    DTYPE: str = os.getenv("DTYPE", "bfloat16")
    ATTN_IMPLEMENTATION: str = os.getenv("ATTN_IMPLEMENTATION", "sdpa")
    CHUNK_MS: float = _env_float("CHUNK_MS", 32.0)
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    MAX_UPLOAD_BYTES: int = _env_int("MAX_UPLOAD_BYTES", 20 * 1024 * 1024)
    TARGET_SAMPLE_RATE: int = _env_int("TARGET_SAMPLE_RATE", 24000)
    SAVE_RAW_UPLOADS: bool = _env_bool("SAVE_RAW_UPLOADS", False)


settings = Settings()
