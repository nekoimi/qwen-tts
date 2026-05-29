"""Build reusable voice clone prompts."""

from __future__ import annotations

from typing import Any

from app.core.model_manager import ModelManager


def create_prompt_from_ref_audio(
    ref_audio: tuple[Any, Any],
    *,
    ref_text: str | None = None,
    x_vector_only_mode: bool | None = None,
) -> Any:
    """
    Create voice clone prompt items for later ``generate_voice_clone(..., voice_clone_prompt=...)``.

    When ``ref_text`` is provided, use the model's recommended ICL prompt path. If no
    transcript is available, fall back to x-vector-only mode.
    """
    model = ModelManager.get_model()
    use_x_vector_only = ref_text is None if x_vector_only_mode is None else x_vector_only_mode
    return model.create_voice_clone_prompt(
        ref_audio=ref_audio,
        ref_text=ref_text,
        x_vector_only_mode=use_x_vector_only,
    )
