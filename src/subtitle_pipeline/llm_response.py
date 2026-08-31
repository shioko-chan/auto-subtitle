from __future__ import annotations

import json
from collections.abc import Callable

from .prompt_templates import prompt_system
from .repetition import RepetitionLoopError, find_repetition_loop


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


def structured_request_body(
    *,
    model: str,
    prompt_name: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    thinking: str | None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": prompt_system(prompt_name)},
            {"role": "user", "content": prompt},
        ],
    }
    if thinking:
        body["thinking"] = {"type": thinking}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


def structured_response_content(
    response: dict[str, object],
    *,
    finish_reason: Callable[[object], str | None],
) -> object:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("LLM response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise ValueError("LLM response has no message content")
    content = message["content"]
    if isinstance(content, str):
        repetition = find_repetition_loop(content)
        if repetition is not None:
            raise RepetitionLoopError(repetition)
    reason = finish_reason(response)
    if reason not in (None, "stop"):
        raise RuntimeError(f"finish_reason={reason}")
    return content


def finish_reason(response: object) -> str | None:
    if not isinstance(response, dict):
        return None
    choices = response.get("choices")
    if (
        not isinstance(choices, list)
        or not choices
        or not isinstance(choices[0], dict)
    ):
        return None
    value = choices[0].get("finish_reason")
    return value if isinstance(value, str) else None
