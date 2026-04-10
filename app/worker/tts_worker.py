"""Concurrency limiter around TTS inference."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from typing import TypeVar

from app.core.config import settings

T = TypeVar("T")

_tts_semaphore = asyncio.Semaphore(settings.TTS_MAX_CONCURRENT)


async def run_with_tts_limit(coro_factory: Callable[[], Coroutine[None, None, T]]) -> T:
    async with _tts_semaphore:
        return await coro_factory()
