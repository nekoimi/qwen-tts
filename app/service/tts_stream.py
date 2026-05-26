"""Synthesize speech and yield float32 LE sample chunks (WAV-style IEEE float), plus empty terminator."""

from __future__ import annotations

import re
from typing import Any, Generator, Iterable

import numpy as np

from app.core.config import settings
from app.core.model_manager import ModelManager
from app.service.audio_utils import resample_waveform

# Sentence-boundary pattern: Chinese/English punctuation + newlines
_SENT_RE = re.compile(r"(?<=[。！？；\n.!?;])\s*")

# Minimum characters per chunk to avoid too-short fragments
_MIN_CHUNK_LEN = 10
# Maximum characters before secondary split on commas
_MAX_CHUNK_LEN = 200

# Token estimation: ~13 codec tokens per second at 12.5 fps,
# ~2.5 chars per second of speech (rough heuristic for mixed CJK/Latin)
_TOKENS_PER_CHAR = 13.0 / 2.5  # ≈ 5.2 tokens per character
_DEFAULT_MAX_TOKENS = 8192
_MIN_MAX_TOKENS = 128


def _estimate_max_tokens(text: str) -> int:
    """Estimate a reasonable max_new_tokens based on text length."""
    estimated = max(_MIN_MAX_TOKENS, int(len(text) * _TOKENS_PER_CHAR * settings.TTS_MAX_TOKEN_MULTIPLIER))
    return min(estimated, _DEFAULT_MAX_TOKENS)


def _wav_to_f32le_bytes(wav: np.ndarray) -> bytes:
    """Mono float samples as little-endian float32 (same sample layout as WAV IEEE float)."""
    x = np.asarray(wav, dtype=np.float32)
    np.clip(x, -1.0, 1.0, out=x)
    return np.ascontiguousarray(x, dtype="<f4").tobytes()


def _iter_fixed_f32le_chunks(pcm_flat: bytes, sample_rate: int, chunk_ms: float) -> Iterable[bytes]:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    bytes_per_sample = 4
    chunk_samples = max(1, int(sample_rate * (chunk_ms / 1000.0)))
    chunk_bytes = chunk_samples * bytes_per_sample
    for i in range(0, len(pcm_flat), chunk_bytes):
        yield pcm_flat[i : i + chunk_bytes]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence-level chunks suitable for per-segment TTS generation."""
    text = text.strip()
    if not text:
        return []

    # Primary split on sentence-ending punctuation and newlines
    raw_parts = _SENT_RE.split(text)

    # Merge very short fragments with the next part
    merged: list[str] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        if merged and len(merged[-1]) < _MIN_CHUNK_LEN:
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)

    # Secondary split on commas for overly long chunks
    final: list[str] = []
    for chunk in merged:
        if len(chunk) <= _MAX_CHUNK_LEN:
            final.append(chunk)
            continue
        sub_parts = re.split(r"(?<=[,，、])\s*", chunk)
        buf = ""
        for sp in sub_parts:
            if not sp.strip():
                continue
            if buf and len(buf) + len(sp) > _MAX_CHUNK_LEN:
                final.append(buf)
                buf = sp
            else:
                buf = buf + sp if buf else sp
        if buf:
            final.append(buf)

    return final


def _synthesize_chunk(
    model: Any,
    text: str,
    voice_prompt: Any,
    language: str,
) -> bytes:
    """Run generation for a single text chunk and return raw f32le PCM bytes (no terminator)."""
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_prompt,
        non_streaming_mode=False,
        max_new_tokens=_estimate_max_tokens(text),
    )
    wav = np.asarray(wavs[0], dtype=np.float32)
    sr_i = int(sr)
    out_sr = settings.TARGET_SAMPLE_RATE
    if sr_i != out_sr:
        wav = resample_waveform(wav, sr_i, out_sr)
    return _wav_to_f32le_bytes(wav)


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
    model = ModelManager.get_model()
    pcm_bytes = _synthesize_chunk(model, text, voice_prompt, language)
    for chunk in _iter_fixed_f32le_chunks(pcm_bytes, settings.TARGET_SAMPLE_RATE, settings.CHUNK_MS):
        if chunk:
            yield chunk
    yield b""


def stream_tts_chunked(
    text: str,
    voice_prompt: Any,
    *,
    language: str = "Auto",
) -> Generator[bytes, None, None]:
    """
    Sentence-chunked streaming TTS: split text by sentences, generate each independently,
    and yield audio chunks immediately per sentence. Yields empty bytes at the very end.

   对外协议与 stream_tts() 完全一致：f32le PCM chunks + 空帧终止符。
    """
    chunks = _split_sentences(text)
    if not chunks:
        yield b""
        return

    model = ModelManager.get_model()
    out_sr = settings.TARGET_SAMPLE_RATE

    for sentence in chunks:
        pcm_bytes = _synthesize_chunk(model, sentence, voice_prompt, language)
        for chunk in _iter_fixed_f32le_chunks(pcm_bytes, out_sr, settings.CHUNK_MS):
            if chunk:
                yield chunk

    yield b""
