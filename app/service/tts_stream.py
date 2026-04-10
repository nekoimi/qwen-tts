"""Synthesize speech and yield int16 PCM chunks (plus empty terminator)."""

from __future__ import annotations

from typing import Any, Generator, Iterable

import numpy as np

from app.core.config import settings
from app.core.model_manager import ModelManager


def _wav_to_int16_pcm_bytes(wav: np.ndarray) -> bytes:
    x = np.asarray(wav, dtype=np.float64)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    return pcm.tobytes()


def _iter_fixed_pcm_chunks(pcm_flat: bytes, sample_rate: int, chunk_ms: float) -> Iterable[bytes]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    bytes_per_sample = 2
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
    Run voice-clone generation and yield PCM chunks (int16 LE), then empty bytes to mark end.
    """
    model = ModelManager.get_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
        non_streaming_mode=False,
    )
    wav = np.asarray(wavs[0])
    pcm_bytes = _wav_to_int16_pcm_bytes(wav)
    for chunk in _iter_fixed_pcm_chunks(pcm_bytes, int(sr), settings.CHUNK_MS):
        if chunk:
            yield chunk
    yield b""
