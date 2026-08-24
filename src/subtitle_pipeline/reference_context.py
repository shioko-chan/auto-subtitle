from __future__ import annotations


_PROMPT_REFERENCE_KEYS = (
    "video",
    "franchises",
    "characters",
    "terms",
    "asr_entities",
)


def compact_reference_context(context: dict[str, object]) -> dict[str, object]:
    """Keep runtime audit payloads out of prompts that only need reference terms."""
    return {key: context[key] for key in _PROMPT_REFERENCE_KEYS if key in context}


def compact_lyrics_reference_context(
    context: dict[str, object],
) -> dict[str, object]:
    """Keep only naming and terminology references needed for lyric translation."""
    reference: dict[str, object] = {}
    franchises = context.get("franchises")
    if isinstance(franchises, list):
        names = [
            {"name": item["name"]}
            for item in franchises
            if isinstance(item, dict) and isinstance(item.get("name"), str)
        ]
        if names:
            reference["franchises"] = names
    for key in ("characters", "terms"):
        if key in context:
            reference[key] = context[key]
    return reference
