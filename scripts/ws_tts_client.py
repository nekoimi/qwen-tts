#!/usr/bin/env python3
"""
最小 WebSocket 客户端：连接 ``/ws/stream``，发送 JSON，接收二进制 **mono f32le**（与 WAV IEEE float 同布局），空帧表示结束。

依赖 ``websockets`` 与 ``soundfile``（主项目已依赖），请使用::

    uv sync --extra dev

示例::

    uv run python scripts/ws_tts_client.py --voice-id my_voice_1 --content "你好" --output-wav out.wav
    uv run python scripts/ws_tts_client.py --voice-id my_voice_1 --content-file "你好" --output-wav out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import soundfile as sf
import websockets
from websockets.exceptions import ConnectionClosed


async def run(args: argparse.Namespace) -> None:
    content = args.content if args.content else ""
    if not args.content and not args.content_file:
        raise ValueError("Must specify content or content_file")
    if args.content_file:
        with open(args.content_file, "rb", encoding="UTF8") as f:
            content = str(f.read()).replace("\r\n", "\n").replace("\n", "")

    payload = {
        "content": content,
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
    nfloats = total // 4
    print(f"chunks={len(pcm_parts)} f32_bytes={total} (~{nfloats} samples)")

    if args.output_wav is not None:
        raw = b"".join(pcm_parts)
        data = np.frombuffer(raw, dtype="<f4")
        sf.write(args.output_wav, data, args.sample_rate, subtype="FLOAT")
        print(f"wrote {args.output_wav!r} (mono float32 WAV, sample_rate={args.sample_rate})")


def main() -> None:
    p = argparse.ArgumentParser(description="Minimal client for TTS over WebSocket (f32le stream).")
    p.add_argument("--url", default="ws://127.0.0.1:8000/ws/stream", help="WebSocket URL")
    p.add_argument("--voice-id", required=True, help="Registered voice_id")
    p.add_argument("--content", required=False, help="Text to synthesize")
    p.add_argument("--content-file", required=False, help="Text file")
    p.add_argument("--language", default="Auto", help='e.g. "Auto", "Chinese", "English"')
    p.add_argument(
        "--output-wav",
        metavar="PATH",
        help="Write concatenated f32le samples as a float32 WAV file (SR from --sample-rate).",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=24000,
        help="Must match server TARGET_SAMPLE_RATE (default 24000) for correct playback length.",
    )
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
