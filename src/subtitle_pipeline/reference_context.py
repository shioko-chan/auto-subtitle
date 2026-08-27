from __future__ import annotations

_PROMPT_REFERENCE_KEYS = (
    "video",
    "franchises",
    "characters",
    "terms",
    "asr_entities",
    "fan_knowledge",
)


def compact_reference_context(context: dict[str, object]) -> dict[str, object]:
    """Keep runtime audit payloads out of prompts that only need reference terms."""
    compact = {key: context[key] for key in _PROMPT_REFERENCE_KEYS if key in context}
    video = compact.get("video")
    if isinstance(video, dict):
        compact["video"] = {
            key: value for key, value in video.items() if key != "description"
        }
    return compact


def compact_translation_reference_context(
    context: dict[str, object],
    *,
    evidence_text: str,
    speakers: set[str],
) -> dict[str, object]:
    """Keep only reference entries that can help the current translation batch."""
    reference: dict[str, object] = {}
    video = context.get("video")
    if isinstance(video, dict):
        kept_video = {
            key: video[key]
            for key in (
                "title",
                "channel",
                "channel_id",
                "uploader",
                "uploader_id",
                "upload_date",
            )
            if key in video
        }
        if kept_video:
            reference["video"] = kept_video

    franchises = context.get("franchises")
    if isinstance(franchises, list) and franchises:
        reference["franchises"] = franchises

    folded = evidence_text.casefold()
    characters = context.get("characters")
    if isinstance(characters, list):
        matched_characters = [
            character
            for character in characters
            if isinstance(character, dict)
            and (
                str(character.get("id") or "") in speakers
                or any(
                    form.casefold() in folded
                    for form in _character_source_forms(character)
                )
            )
        ]
        if matched_characters:
            reference["characters"] = matched_characters

    terms = context.get("terms")
    if isinstance(terms, dict):
        matched_terms = {
            source: target
            for source, target in terms.items()
            if isinstance(source, str)
            and source
            and source.casefold() in folded
        }
        if matched_terms:
            reference["terms"] = matched_terms
    return reference


def _character_source_forms(character: dict[str, object]) -> list[str]:
    values = [character.get("source_name"), character.get("canonical")]
    aliases = character.get("aliases")
    if isinstance(aliases, list):
        values.extend(aliases)
    short_names = character.get("short_names")
    if isinstance(short_names, list):
        values.extend(
            value.get("source")
            for value in short_names
            if isinstance(value, dict)
        )
    return [value for value in values if isinstance(value, str) and value]


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
