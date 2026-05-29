"""Synthesize speech and yield float32 LE sample chunks (WAV-style IEEE float), plus empty terminator."""

from __future__ import annotations

import time
from typing import Any, Generator, Iterable

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.core.model_manager import ModelManager
from app.service.audio_utils import resample_waveform

logger = get_logger(__name__)


def _wav_to_f32le_bytes(wav: np.ndarray) -> bytes:
    """Mono float samples as little-endian float32 (same sample layout as WAV IEEE float)."""
    x = np.asarray(wav, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    f32 = np.ascontiguousarray(x, dtype="<f4")
    return f32.tobytes()


def _iter_fixed_f32le_chunks(pcm_flat: bytes, sample_rate: int, chunk_ms: float) -> Iterable[bytes]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    bytes_per_sample = 4
    chunk_samples = max(1, int(sample_rate * (chunk_ms / 1000.0)))
    chunk_bytes = chunk_samples * bytes_per_sample
    for i in range(0, len(pcm_flat), chunk_bytes):
        yield pcm_flat[i : i + chunk_bytes]


def stream_tts(
    text: str,
    voice_prompt: Any,
    *,
    language: str = "Auto",
) -> Generator[bytes, None, None]:
    """
    Run voice-clone generation and yield f32le sample chunks, then empty bytes to mark end.

    Each non-empty chunk is raw mono **float32 little-endian** bytes (4 bytes per sample),
    same encoding as WAV format IEEE float (payload only, no RIFF header). Output is resampled
    to ``TARGET_SAMPLE_RATE`` so clients can play at a fixed rate without pitch/speed mismatch.
    """
    t_start = time.perf_counter()
    model = ModelManager.get_model()
    t_generate = time.perf_counter()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
        non_streaming_mode=False,
    )
    t_after_generate = time.perf_counter()
    wav = np.asarray(wavs[0], dtype=np.float32)
    sr_i = int(sr)
    out_sr = settings.TARGET_SAMPLE_RATE
    if sr_i != out_sr:
        wav = resample_waveform(wav, sr_i, out_sr)
    pcm_bytes = _wav_to_f32le_bytes(wav)
    t_after_postprocess = time.perf_counter()
    logger.info(
        "TTS generate done | chars=%d | model_wait=%.2fs | generate=%.2fs | postprocess=%.2fs | audio=%.2fs",
        len(text),
        t_generate - t_start,
        t_after_generate - t_generate,
        t_after_postprocess - t_after_generate,
        len(pcm_bytes) / (out_sr * 4),
    )
    for chunk in _iter_fixed_f32le_chunks(pcm_bytes, out_sr, settings.CHUNK_MS):
        if chunk:
            yield chunk
    yield b""
    logger.info("TTS stream done | chars=%d | elapsed=%.2fs", len(text), time.perf_counter() - t_start)
