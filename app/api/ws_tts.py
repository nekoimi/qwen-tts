"""WebSocket API: stream synthesized speech as float32 LE sample chunks (WAV-style IEEE float)."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import get_logger
from app.service.tts_stream import stream_tts_chunked
from app.storage.voice_store import load_voice_embedding
from app.worker.tts_worker import run_with_tts_limit

logger = get_logger(__name__)

router = APIRouter(tags=["tts"])

_SENTINEL = object()


@router.websocket("/ws/stream")
async def websocket_tts(websocket: WebSocket) -> None:
    await websocket.accept()
    while True:
        try:
            data = await websocket.receive_json()
        except WebSocketDisconnect:
            return
        except json.JSONDecodeError:
            await websocket.close(code=4400)
            return

        content = data.get("content")
        voice_id = data.get("voice_id")
        language = data.get("language", "Auto")

        if not isinstance(content, str) or not content.strip():
            await websocket.close(code=4400)
            return
        if not isinstance(voice_id, str) or not voice_id.strip():
            await websocket.close(code=4400)
            return
        if not isinstance(language, str) or not language.strip():
            language = "Auto"

        try:
            prompt = load_voice_embedding(voice_id)
        except FileNotFoundError:
            await websocket.close(code=4404)
            return

        try:
            async def stream_and_send():
                loop = asyncio.get_running_loop()
                gen = stream_tts_chunked(content, prompt, language=language, voice_id=voice_id)
                while True:
                    chunk = await loop.run_in_executor(None, next, gen, _SENTINEL)
                    if chunk is _SENTINEL:
                        break
                    await websocket.send_bytes(chunk)

            await run_with_tts_limit(stream_and_send)
        except Exception:
            logger.exception("websocket_tts failed voice_id=%s", voice_id)
            await websocket.close(code=1011)
            return
