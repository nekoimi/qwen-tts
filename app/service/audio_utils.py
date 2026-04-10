"""Load and normalize reference audio for voice cloning."""

from __future__ import annotations

import io
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy import signal

from app.core.config import settings


def _to_mono(wav: np.ndarray) -> np.ndarray:
    if wav.ndim == 1:
        return wav
    return wav.mean(axis=1)


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    num = int(len(wav) * target_sr / orig_sr)
    return signal.resample(wav, num).astype(np.float32)


def load_audio_bytes(data: bytes, target_sr: int | None = None) -> Tuple[np.ndarray, int]:
    """Decode audio bytes to mono float32 waveform at ``target_sr`` (default from settings)."""
    sr = target_sr if target_sr is not None else settings.TARGET_SAMPLE_RATE
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    wav = _to_mono(wav)
    wav = _resample(wav, file_sr, sr)
    return wav, sr


def as_ref_audio_tuple(wav: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    """Format accepted by qwen_tts as ``(waveform, sample_rate)``."""
    return (wav.astype(np.float32), int(sr))
