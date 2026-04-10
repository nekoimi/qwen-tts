"""FastAPI entrypoint."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import http_voice, ws_tts
from app.core.logger import setup_logging
from app.storage.voice_store import ensure_voice_dir


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    ensure_voice_dir()
    yield


app = FastAPI(title="Qwen TTS Service", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(http_voice.router)
app.include_router(ws_tts.router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
