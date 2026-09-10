"""Output contracts shared by all structured LLM requests."""
from copy import deepcopy
from pathlib import Path


def _object(**properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


def _array(items):
    return {"type": "array", "items": items}


_TEXT = {"type": "string"}
_ID = {"type": "integer"}
_NULLABLE_ID = {"type": ["integer", "null"]}
_SCHEMAS = {
    "asr-correct": _object(segments=_array(_object(segment_id=_ID, corrected_text=_TEXT))),
    "segment-translate-cues": _object(cues=_array(_object(start_id=_ID, end_id=_ID, text=_TEXT))),
    "lyrics-translate": _object(lines=_array(_object(line_id=_ID, text=_TEXT))),
    "metadata-translate": _object(title=_TEXT, description=_TEXT, content_summary=_TEXT, tags=_array(_TEXT)),
    "select-ocr-song-titles": _object(groups=_array(_object(
        group_id=_ID, song_title={"type": ["string", "null"]},
    ))),
    "extract-fan-terms": _object(terms=_array(_object(
        candidate=_TEXT, canonical_zh=_TEXT, confidence={"type": "number"},
        action={"type": "string", "enum": ["accept", "search"]}, search_query=_TEXT,
    ))),
    "review-fan-terms": _object(
        decision={"type": "string", "enum": ["accept", "reject"]}, canonical_zh=_TEXT,
    ),
    "clip-review": _object(
        worthy={"type": "boolean"}, confidence={"type": "string", "enum": ["high", "medium", "low"]},
        start_id=_NULLABLE_ID, end_id=_NULLABLE_ID, title=_TEXT, reason=_TEXT,
    ),
}


def response_format(prompt_name: str) -> dict[str, object]:
    name = Path(prompt_name).stem
    return {"type": "json_schema", "json_schema": {
        "name": name.replace('-', '_'), "strict": True, "schema": deepcopy(_SCHEMAS[name]),
    }}
