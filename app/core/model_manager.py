"""Lazy singleton for Qwen3 TTS model."""

from __future__ import annotations

import threading

import torch

from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

_model = None
_lock = threading.Lock()


def _resolve_dtype() -> torch.dtype:
    name = settings.DTYPE.lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(name, torch.bfloat16)


class ModelManager:
    """Global single loaded model instance."""

    @classmethod
    def get_model(cls):
        global _model
        if _model is None:
            with _lock:
                if _model is None:
                    from qwen_tts import Qwen3TTSModel

                    logger.info("Loading Qwen3 TTS model %s on %s", settings.MODEL_ID, settings.DEVICE)
                    kwargs = {
                        "device_map": settings.DEVICE,
                        "dtype": _resolve_dtype(),
                        "attn_implementation": settings.ATTN_IMPLEMENTATION,
                    }
                    _model = Qwen3TTSModel.from_pretrained(settings.MODEL_ID, **kwargs)
                    logger.info("Model loaded.")
        return _model
