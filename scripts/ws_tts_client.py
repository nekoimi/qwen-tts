#!/usr/bin/env python3
"""
最小 WebSocket 客户端：连接 ``/ws/stream``，发送 JSON，接收二进制 PCM（int16 LE），空帧表示结束。

依赖 ``websockets``，请使用::

    uv sync --extra dev

示例::

    uv run python scripts/ws_tts_client.py --voice-id my_voice_1 --content "你好" --output-wav out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json
import wave

import websockets
from websockets.exceptions import ConnectionClosed


async def run(args: argparse.Namespace) -> None:
    payload = {
        "content": args.content,
        "voice_id": args.voice_id,
        "language": args.language,
    }
    pcm_parts: list[bytes] = []

    async with websockets.connect(args.url, max_size=None) as ws:
        await ws.send(json.dumps(payload, ensure_ascii=False))
        while True:
            try:
                msg = await ws.recv()
            except ConnectionClosed as e:
                print(f"WebSocket closed: code={e.code} reason={e.reason!r}")
                return

            if isinstance(msg, str):
                print("unexpected text message:", msg)
                return

            if len(msg) == 0:
                break
            pcm_parts.append(msg)

    total = sum(len(p) for p in pcm_parts)
    print(f"chunks={len(pcm_parts)} pcm_bytes={total}")

    if args.output_wav is not None:
        data = b"".join(pcm_parts)
        with wave.open(args.output_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(args.sample_rate)
            wf.writeframes(data)
        print(f"wrote {args.output_wav!r} (mono s16le, sample_rate={args.sample_rate})")


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal client for POST-less TTS over WebSocket.")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/stream", help="WebSocket URL")
    p.add_argument("--voice-id", required=True, help="Registered voice_id")
    p.add_argument("--content", required=True, help="Text to synthesize")
    p.add_argument("--language", default="Auto", help='e.g. "Auto", "Chinese", "English"')
    p.add_argument(
        "--output-wav",
        metavar="PATH",
        help="Write concatenated PCM as a WAV file (mono s16le; SR from --sample-rate).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="WAV header sample rate; should match model output SR if you need correct playback speed.",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
