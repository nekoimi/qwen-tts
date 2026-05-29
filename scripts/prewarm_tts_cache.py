#!/usr/bin/env python3
"""Pre-generate TTS cache entries for common phrases."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.core.logger import setup_logging
from app.service.tts_stream import stream_tts_chunked
from app.storage.tts_cache import normalize_text
from app.storage.voice_store import load_voice_embedding


def _load_texts(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
        if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
            raise ValueError("JSON text file must be a string array")
        texts = data
    else:
        texts = raw.splitlines()

    normalized: list[str] = []
    seen: set[str] = set()
    for text in texts:
        item = normalize_text(text)
        if not item or item.startswith("#") or item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _consume_stream(text: str, voice_prompt: object, *, voice_id: str, language: str) -> tuple[int, int]:
    chunks = 0
    total_bytes = 0
    for chunk in stream_tts_chunked(text, voice_prompt, language=language, voice_id=voice_id):
        if not chunk:
            continue
        chunks += 1
        total_bytes += len(chunk)
    return chunks, total_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description="Prewarm sentence-level Qwen TTS f32le cache.")
    parser.add_argument("--voice-id", action="append", required=True, help="Registered voice_id. Repeat for multiple voices.")
    parser.add_argument("--text-file", required=True, type=Path, help="UTF-8 text file, one phrase per line, or JSON string array.")
    parser.add_argument("--language", default="Auto", help='e.g. "Auto", "Chinese", "English"')
    args = parser.parse_args()

    setup_logging()
    texts = _load_texts(args.text_file)
    if not texts:
        raise ValueError(f"No prewarm texts found in {args.text_file}")

    print(f"prewarm voices={len(args.voice_id)} texts={len(texts)} language={args.language}")
    total_start = time.perf_counter()
    for voice_id in args.voice_id:
        prompt = load_voice_embedding(voice_id)
        voice_start = time.perf_counter()
        for index, text in enumerate(texts, 1):
            t0 = time.perf_counter()
            chunks, total_bytes = _consume_stream(text, prompt, voice_id=voice_id, language=args.language)
            print(
                f"voice={voice_id} text={index}/{len(texts)} chars={len(text)} "
                f"chunks={chunks} bytes={total_bytes} elapsed={time.perf_counter() - t0:.2f}s"
            )
        print(f"voice={voice_id} done elapsed={time.perf_counter() - voice_start:.2f}s")

    print(f"prewarm done elapsed={time.perf_counter() - total_start:.2f}s")


if __name__ == "__main__":
    main()
