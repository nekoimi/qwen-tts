"""Build reusable voice clone prompts."""

from __future__ import annotations

from typing import Any

from app.core.model_manager import ModelManager


def create_prompt_from_ref_audio(
    ref_audio: tuple[Any, Any],
    *,
    x_vector_only_mode: bool = True,
) -> Any:
    """
    Create voice clone prompt items for later ``generate_voice_clone(..., voice_clone_prompt=...)``.

    When ``x_vector_only_mode`` is True, ``ref_text`` is not required (lower quality than ICL).
    """
    model = ModelManager.get_model()
    return model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=None,
        x_vector_only_mode=x_vector_only_mode,
    )
