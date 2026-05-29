"""File-backed cache for synthesized TTS chunks."""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import Iterator

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_LOCKS: dict[str, Lock] = {}
_LOCKS_GUARD = Lock()


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def make_cache_key(
    *,
    voice_id: str,
    voice_fingerprint: str,
    language: str,
    text: str,
    sample_rate: int,
) -> str:
    payload = {
        "version": settings.TTS_CACHE_VERSION,
        "model_id": settings.MODEL_ID,
        "voice_id": voice_id,
        "voice_fingerprint": voice_fingerprint,
        "language": language,
        "text": normalize_text(text),
        "sample_rate": sample_rate,
        "format": "f32le",
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _cache_dir_for(key: str) -> Path:
    return settings.TTS_CACHE_DIR / key[:2]


def _audio_path(key: str) -> Path:
    return _cache_dir_for(key) / f"{key}.f32le"


def _meta_path(key: str) -> Path:
    return _cache_dir_for(key) / f"{key}.json"


def _lock_for(key: str) -> Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = Lock()
            _LOCKS[key] = lock
        return lock


@contextmanager
def cache_lock(key: str) -> Iterator[None]:
    lock = _lock_for(key)
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def read_audio(key: str) -> bytes | None:
    path = _audio_path(key)
    if not path.is_file():
        return None
    try:
        now = time.time()
        os.utime(path, (now, path.stat().st_mtime))
        meta_path = _meta_path(key)
        if meta_path.is_file():
            os.utime(meta_path, (now, meta_path.stat().st_mtime))
        logger.info("TTS cache hit | key=%s | bytes=%d", key[:12], path.stat().st_size)
        return path.read_bytes()
    except OSError:
        logger.warning("Failed to read TTS cache key=%s", key[:12], exc_info=True)
        return None


def write_audio(
    key: str,
    audio: bytes,
    *,
    voice_id: str,
    language: str,
    text: str,
    sample_rate: int,
    audio_seconds: float,
    generate_seconds: float,
) -> None:
    cache_dir = _cache_dir_for(key)
    cache_dir.mkdir(parents=True, exist_ok=True)

    audio_path = _audio_path(key)
    tmp_audio_path = audio_path.with_suffix(".f32le.tmp")
    tmp_audio_path.write_bytes(audio)
    tmp_audio_path.replace(audio_path)

    now = time.time()
    meta = {
        "key": key,
        "version": settings.TTS_CACHE_VERSION,
        "model_id": settings.MODEL_ID,
        "voice_id": voice_id,
        "language": language,
        "text": normalize_text(text),
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "generate_seconds": generate_seconds,
        "bytes": len(audio),
        "created_at": now,
    }
    meta_path = _meta_path(key)
    tmp_meta_path = meta_path.with_suffix(".json.tmp")
    tmp_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_meta_path.replace(meta_path)
    logger.info("TTS cache stored | key=%s | bytes=%d", key[:12], len(audio))
    prune_cache()


def prune_cache() -> None:
    max_bytes = settings.TTS_CACHE_MAX_BYTES
    if max_bytes <= 0:
        return
    root = settings.TTS_CACHE_DIR
    if not root.is_dir():
        return

    entries: list[tuple[float, int, Path]] = []
    total = 0
    for path in root.glob("*/*.f32le"):
        try:
            stat = path.stat()
        except OSError:
            continue
        total += stat.st_size
        entries.append((stat.st_atime, stat.st_size, path))

    if total <= max_bytes:
        return

    entries.sort(key=lambda item: item[0])
    removed = 0
    for _, size, path in entries:
        if total <= max_bytes:
            break
        try:
            path.unlink(missing_ok=True)
            _meta_path(path.stem).unlink(missing_ok=True)
            total -= size
            removed += 1
        except OSError:
            logger.warning("Failed to prune TTS cache file %s", path, exc_info=True)
    if removed:
        logger.info("TTS cache pruned | removed=%d | remaining_bytes=%d", removed, total)
