"""Synthesize speech and yield float32 LE sample chunks (WAV-style IEEE float), plus empty terminator."""

from __future__ import annotations

import re
import time
from typing import Any, Generator, Iterable

import numpy as np

from app.core.config import settings
from app.core.logger import get_logger
from app.core.model_manager import ModelManager
from app.service.audio_utils import resample_waveform
from app.storage.tts_cache import cache_lock, make_cache_key, read_audio, write_audio
from app.storage.voice_store import voice_fingerprint

logger = get_logger(__name__)

_SENTENCE_RE = re.compile(r"(?<=[。！？；\n.!?;])\s*")


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


def _split_text(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []

    parts = [part.strip() for part in _SENTENCE_RE.split(text) if part.strip()]
    if not parts:
        return [text]

    merged: list[str] = []
    for part in parts:
        if merged and len(merged[-1]) < settings.TTS_MIN_CHUNK_LEN:
            merged[-1] += part
        else:
            merged.append(part)

    chunks: list[str] = []
    max_len = max(settings.TTS_MIN_CHUNK_LEN, settings.TTS_MAX_CHUNK_LEN)
    for part in merged:
        if len(part) <= max_len:
            chunks.append(part)
            continue
        sub_parts = [sp.strip() for sp in re.split(r"(?<=[,，、])\s*", part) if sp.strip()]
        buf = ""
        for sp in sub_parts:
            if buf and len(buf) + len(sp) > max_len:
                chunks.append(buf)
                buf = sp
            else:
                buf = buf + sp if buf else sp
        if buf:
            chunks.append(buf)

    return chunks


def _synthesize_text(
    text: str,
    voice_prompt: Any,
    *,
    language: str = "Auto",
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> tuple[bytes, int]:
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
        "TTS generate done | chunk=%s/%s | chars=%d | model_wait=%.2fs | generate=%.2fs | postprocess=%.2fs | audio=%.2fs",
        chunk_index if chunk_index is not None else "-",
        chunk_count if chunk_count is not None else "-",
        len(text),
        t_generate - t_start,
        t_after_generate - t_generate,
        t_after_postprocess - t_after_generate,
        len(pcm_bytes) / (out_sr * 4),
    )
    return pcm_bytes, out_sr


def _synthesize_text_cached(
    text: str,
    voice_prompt: Any,
    *,
    voice_id: str,
    language: str = "Auto",
    chunk_index: int | None = None,
    chunk_count: int | None = None,
) -> tuple[bytes, int]:
    out_sr = settings.TARGET_SAMPLE_RATE
    key = make_cache_key(
        voice_id=voice_id,
        voice_fingerprint=voice_fingerprint(voice_id),
        language=language,
        text=text,
        sample_rate=out_sr,
    )

    cached = read_audio(key)
    if cached is not None:
        return cached, out_sr

    with cache_lock(key):
        cached = read_audio(key)
        if cached is not None:
            return cached, out_sr

        t0 = time.perf_counter()
        pcm_bytes, out_sr = _synthesize_text(
            text,
            voice_prompt,
            language=language,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
        )
        generate_seconds = time.perf_counter() - t0
        write_audio(
            key,
            pcm_bytes,
            voice_id=voice_id,
            language=language,
            text=text,
            sample_rate=out_sr,
            audio_seconds=len(pcm_bytes) / (out_sr * 4),
            generate_seconds=generate_seconds,
        )
        return pcm_bytes, out_sr


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
    pcm_bytes, out_sr = _synthesize_text(text, voice_prompt, language=language)
    for chunk in _iter_fixed_f32le_chunks(pcm_bytes, out_sr, settings.CHUNK_MS):
        if chunk:
            yield chunk
    yield b""
    logger.info("TTS stream done | chars=%d | elapsed=%.2fs", len(text), time.perf_counter() - t_start)


def stream_tts_chunked(
    text: str,
    voice_prompt: Any,
    *,
    language: str = "Auto",
    voice_id: str | None = None,
) -> Generator[bytes, None, None]:
    """
    Split longer text into sentence-sized TTS calls and yield each generated sentence immediately.
    """
    t_start = time.perf_counter()
    chunks = _split_text(text)
    if not chunks:
        yield b""
        return

    for index, chunk_text in enumerate(chunks, 1):
        if voice_id is None:
            pcm_bytes, out_sr = _synthesize_text(
                chunk_text,
                voice_prompt,
                language=language,
                chunk_index=index,
                chunk_count=len(chunks),
            )
        else:
            pcm_bytes, out_sr = _synthesize_text_cached(
                chunk_text,
                voice_prompt,
                voice_id=voice_id,
                language=language,
                chunk_index=index,
                chunk_count=len(chunks),
            )
        for chunk in _iter_fixed_f32le_chunks(pcm_bytes, out_sr, settings.CHUNK_MS):
            if chunk:
                yield chunk

    yield b""
    logger.info(
        "TTS stream done | chars=%d | chunks=%d | elapsed=%.2fs",
        len(text),
        len(chunks),
        time.perf_counter() - t_start,
    )
