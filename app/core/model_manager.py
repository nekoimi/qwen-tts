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
                    from qwen_tts import Qwen3TTSModel

                    logger.info("Loading Qwen3 TTS model %s on %s", settings.MODEL_ID, settings.DEVICE)
                    dtype = _resolve_dtype()
                    kwargs = {
                        "device_map": settings.DEVICE,
                        "dtype": dtype,
                        "attn_implementation": settings.ATTN_IMPLEMENTATION,
                    }
                    t0 = time.perf_counter()
                    _model = Qwen3TTSModel.from_pretrained(settings.MODEL_ID, **kwargs)
                    logger.info("Model loaded in %.2fs.", time.perf_counter() - t0)

                    cls._compile_model(_model)
                    cls._warmup_cuda(_model)
        return _model

    @classmethod
    def _compile_model(cls, model: object) -> None:
        if settings.DEVICE == "cpu" or not settings.ENABLE_TORCH_COMPILE:
            return
        talker = getattr(getattr(model, "model", None), "talker", None)
        if talker is None:
            return
        try:
            logger.info("Applying torch.compile to talker (mode=reduce-overhead) ...")
            t0 = time.perf_counter()
            model.model.talker = torch.compile(talker, mode="reduce-overhead")
            logger.info("torch.compile applied in %.2fs.", time.perf_counter() - t0)
        except Exception:
            logger.warning("torch.compile failed, running without compilation", exc_info=True)

    @classmethod
    def _warmup_cuda(cls, model: object) -> None:
        if settings.DEVICE == "cpu" or not settings.ENABLE_CUDA_WARMUP:
            return
        try:
            logger.info("Running CUDA warmup ...")
            from qwen_tts import VoiceClonePromptItem

            dummy_embedding = torch.zeros(
                1024, device=settings.DEVICE, dtype=_resolve_dtype()
            )
            dummy_prompt = VoiceClonePromptItem(
                ref_code=None,
                ref_spk_embedding=dummy_embedding,
                x_vector_only_mode=True,
                icl_mode=False,
            )
            t0 = time.perf_counter()
            model.generate_voice_clone(
                text="Hi",
                language="Auto",
                voice_clone_prompt=[dummy_prompt],
                max_new_tokens=20,
            )
            torch.cuda.synchronize()
            logger.info("CUDA warmup complete in %.2fs.", time.perf_counter() - t0)
        except Exception:
            logger.warning("CUDA warmup failed, skipping", exc_info=True)
