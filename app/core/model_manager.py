"""Lazy singleton for Qwen3 TTS model."""

from __future__ import annotations

import threading
import time

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
                    backend = settings.TTS_BACKEND.strip().lower()

                    logger.info(
                        "Loading Qwen3 TTS model %s on %s (backend=%s)",
                        settings.MODEL_ID,
                        settings.DEVICE,
                        backend,
                    )
                    t0 = time.perf_counter()
                    if backend == "faster":
                        from faster_qwen3_tts import FasterQwen3TTS

                        _model = FasterQwen3TTS.from_pretrained(
                            settings.MODEL_ID,
                            device=settings.DEVICE,
                            dtype=_resolve_dtype(),
                            attn_implementation=settings.ATTN_IMPLEMENTATION,
                            max_seq_len=settings.FASTER_MAX_SEQ_LEN,
                        )
                    elif backend == "qwen":
                        from qwen_tts import Qwen3TTSModel

                        kwargs = {
                            "device_map": settings.DEVICE,
                            "dtype": _resolve_dtype(),
                            "attn_implementation": settings.ATTN_IMPLEMENTATION,
                        }
                        _model = Qwen3TTSModel.from_pretrained(settings.MODEL_ID, **kwargs)
                    else:
                        raise ValueError(f"Unsupported TTS_BACKEND={settings.TTS_BACKEND!r}")
                    logger.info("Model loaded in %.2fs.", time.perf_counter() - t0)
        return _model
