"""Persist and cache voice clone prompts."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from threading import Lock
from typing import Any

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_VOICE_CACHE: dict[str, Any] = {}
_CACHE_LOCK = Lock()

_SAFE_ID = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")


def ensure_voice_dir() -> Path:
    settings.VOICE_DIR.mkdir(parents=True, exist_ok=True)
    return settings.VOICE_DIR


def validate_voice_id(voice_id: str) -> bool:
    return bool(_SAFE_ID.match(voice_id))


def _path_for(voice_id: str) -> Path:
    return ensure_voice_dir() / f"{voice_id}.pkl"


def save_voice_embedding(voice_id: str, obj: Any) -> None:
    path = _path_for(voice_id)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    with _CACHE_LOCK:
        _VOICE_CACHE[voice_id] = obj


def load_voice_embedding(voice_id: str) -> Any:
    with _CACHE_LOCK:
        if voice_id in _VOICE_CACHE:
            return _VOICE_CACHE[voice_id]
    path = _path_for(voice_id)
    if not path.is_file():
        raise FileNotFoundError(f"Unknown voice_id: {voice_id}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    with _CACHE_LOCK:
        _VOICE_CACHE[voice_id] = obj
    return obj


def has_voice(voice_id: str) -> bool:
    with _CACHE_LOCK:
        if voice_id in _VOICE_CACHE:
            return True
    return _path_for(voice_id).is_file()
