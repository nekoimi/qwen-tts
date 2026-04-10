"""HTTP API: register reference audio as a cloneable voice."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import settings
from app.core.logger import get_logger
from app.service.audio_utils import as_ref_audio_tuple, load_audio_bytes
from app.service.voice_clone import create_prompt_from_ref_audio
from app.storage import file_store
from app.storage.voice_store import ensure_voice_dir, save_voice_embedding, validate_voice_id

logger = get_logger(__name__)

router = APIRouter(tags=["voice"])

_ALLOWED_SUFFIX = {".wav", ".mp3"}


@router.post("/add_voice_timbre")
async def add_voice_timbre(
    voice_wav: UploadFile = File(..., description="Reference audio (.wav / .mp3)"),
    voice_id: str = Form(..., description="Stable id for this voice"),
) -> bool:
    ensure_voice_dir()
    if not validate_voice_id(voice_id):
        raise HTTPException(status_code=400, detail="Invalid voice_id (use letters, digits, _ , - , max 128 chars).")
    suffix = Path(voice_wav.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIX:
        raise HTTPException(status_code=400, detail="voice_wav must be .wav or .mp3")

    raw = await voice_wav.read()
    if len(raw) > settings.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large.")

    file_store.save_raw_upload(voice_id, raw, suffix=suffix)

    try:
        wav, sr = load_audio_bytes(raw)
        ref = as_ref_audio_tuple(wav, sr)
        prompt = create_prompt_from_ref_audio(ref, x_vector_only_mode=True)
        save_voice_embedding(voice_id, prompt)
    except Exception:
        logger.exception("add_voice_timbre failed for voice_id=%s", voice_id)
        raise HTTPException(status_code=500, detail="Failed to create voice prompt.") from None

    return True
