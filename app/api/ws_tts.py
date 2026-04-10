"""WebSocket API: stream synthesized speech as float32 LE sample chunks (WAV-style IEEE float)."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.core.logger import get_logger
from app.service.tts_stream import stream_tts
from app.storage.voice_store import load_voice_embedding
from app.worker.tts_worker import run_with_tts_limit

logger = get_logger(__name__)

router = APIRouter(tags=["tts"])


@router.websocket("/ws/stream")
async def websocket_tts(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        data = await websocket.receive_json()
    except WebSocketDisconnect:
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

    async def run_chunks():
        def sync_list():
            return list(stream_tts(content, prompt, language=language))

        return await asyncio.to_thread(sync_list)

    try:
        chunks = await run_with_tts_limit(lambda: run_chunks())
        for chunk in chunks:
            await websocket.send_bytes(chunk)
    except Exception:
        logger.exception("websocket_tts failed voice_id=%s", voice_id)
        await websocket.close(code=1011)
        return
