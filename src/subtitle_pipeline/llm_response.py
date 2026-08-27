from __future__ import annotations

import json


def strip_markdown_code_fence(content: object) -> str:
    if not isinstance(content, str):
        raise TypeError("LLM response content is not text")
    value = content.strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[-1]
        value = value.rsplit("```", 1)[0].strip()
    return value


def parse_json_object(content: object) -> dict[str, object]:
    parsed = json.loads(strip_markdown_code_fence(content))
    if not isinstance(parsed, dict):
        raise TypeError("LLM response must be a JSON object")
    return parsed
