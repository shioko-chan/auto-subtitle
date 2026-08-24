from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

from .prompt_templates import prompt_system, prompt_templates_digest, render_user_prompt

_PROMPT = "asr-correct.md"
_CACHE_VERSION = 1
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class ASREntity:
    surface: str
    reading: str
    aliases: tuple[str, ...] = ()


def entities_from_context(context: dict[str, object]) -> list[ASREntity]:
    values = context.get("asr_entities", [])
    if not isinstance(values, list):
        return []
    output: list[ASREntity] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            continue
        surface = str(value.get("surface") or "").strip()
        reading = str(value.get("reading") or "").strip()
        aliases_value = value.get("aliases", [])
        aliases = (
            tuple(
                alias.strip()
                for alias in aliases_value
                if isinstance(alias, str) and alias.strip()
            )
            if isinstance(aliases_value, list)
            else ()
        )
        key = _compact(surface).casefold()
        if not surface or not reading or key in seen:
            continue
        seen.add(key)
        output.append(ASREntity(surface, reading, aliases))
    return output


def correct_asr_windows(
    records: list[dict[str, object]],
    *,
    entities: list[ASREntity],
    request: Callable[[dict[str, object]], dict[str, object]],
    model: str,
    cache_path: Path,
    audit_path: Path,
    batch_windows: int = 6,
    batch_chars: int = 3000,
) -> list[dict[str, object]]:
    if not records:
        return []
    cache = _load_cache(cache_path)
    lock = threading.Lock()
    corrected: list[dict[str, object] | None] = [None] * len(records)
    pending: list[tuple[int, dict[str, object], str, list[ASREntity]]] = []
    for index, record in enumerate(records):
        text = str(record.get("text") or "")
        deterministic = _replace_exact_aliases(text, entities)
        candidates = _candidate_entities(deterministic, entities)
        signature = _record_signature(
            deterministic, str(record.get("language") or ""), candidates, model
        )
        cached = cache["records"].get(signature)
        if isinstance(cached, str):
            corrected[index] = _corrected_record(record, cached, "cache")
        else:
            pending.append((index, record, deterministic, candidates))

    for batch in _pending_batches(pending, batch_windows, batch_chars):
        entity_union = _unique_entities(
            entity for _, _, _, candidates in batch for entity in candidates
        )
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_system(_PROMPT)},
                {
                    "role": "user",
                    "content": render_user_prompt(
                        _PROMPT,
                        ENTITY_REFERENCE=_format_entities(entity_union),
                        TARGET="\n".join(
                            f"<{position} candidates="
                            f"{'｜'.join(entity.surface for entity in candidates) or '(none)'}>"
                            f"{deterministic}"
                            for position, (
                                _,
                                _,
                                deterministic,
                                candidates,
                            ) in enumerate(batch)
                        ),
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        response: dict[str, object] | None = None
        batch_error: str | None = None
        try:
            response = request(body)
            parsed = _parse_response(response, len(batch))
        except Exception as exc:  # noqa: BLE001 - API and validation failures fail open.
            logging.warning(
                "ASR correction downgraded %d windows to rule-normalized text: %s",
                len(batch),
                exc,
            )
            parsed = None
            batch_error = f"{type(exc).__name__}: {exc}"
        _append_audit(
            audit_path,
            {
                "event": "asr_correction_batch",
                "window_ids": [
                    record.get("window_id", index) for index, record, _, _ in batch
                ],
                "request": body,
                "response": response,
                "error": batch_error,
            },
        )
        for position, (index, record, deterministic, candidates) in enumerate(batch):
            if parsed is None:
                value = deterministic
                method = "rule_fallback"
                error = "request_or_validation_failed"
            else:
                value = parsed[position]
                if deterministic and not value:
                    value = deterministic
                    method = "rule_fallback"
                    error = "empty_corrected_text"
                else:
                    method = "llm" if value != deterministic else "unchanged"
                    error = None
            corrected[index] = _corrected_record(record, value, method)
            signature = _record_signature(
                deterministic,
                str(record.get("language") or ""),
                candidates,
                model,
            )
            with lock:
                cache["records"][signature] = value
                _write_json(cache_path, cache)
                _append_audit(
                    audit_path,
                    {
                        "window_id": record.get("window_id", index),
                        "start": record.get("core_start"),
                        "end": record.get("core_end"),
                        "language": record.get("language"),
                        "original_text": record.get("text", ""),
                        "rule_text": deterministic,
                        "corrected_text": value,
                        "method": method,
                        "error": error,
                        "candidates": [entity.surface for entity in candidates],
                    },
                )
    return [value for value in corrected if value is not None]


def _pending_batches(
    pending: list[tuple[int, dict[str, object], str, list[ASREntity]]],
    maximum_windows: int,
    maximum_chars: int,
) -> list[list[tuple[int, dict[str, object], str, list[ASREntity]]]]:
    if maximum_windows < 1 or maximum_chars < 1:
        raise ValueError("ASR correction batch limits must be positive")
    batches: list[list[tuple[int, dict[str, object], str, list[ASREntity]]]] = []
    current: list[tuple[int, dict[str, object], str, list[ASREntity]]] = []
    characters = 0
    for item in pending:
        item_characters = len(item[2])
        if current and (
            len(current) >= maximum_windows
            or characters + item_characters > maximum_chars
        ):
            batches.append(current)
            current = []
            characters = 0
        current.append(item)
        characters += item_characters
    if current:
        batches.append(current)
    return batches


def _replace_exact_aliases(text: str, entities: list[ASREntity]) -> str:
    replacements: list[tuple[str, str]] = []
    for entity in entities:
        for alias in entity.aliases:
            if alias != entity.surface:
                replacements.append((alias, entity.surface))
    output = text
    for alias, surface in sorted(
        replacements, key=lambda item: len(item[0]), reverse=True
    ):
        output = output.replace(alias, surface)
    return output


def _candidate_entities(text: str, entities: list[ASREntity]) -> list[ASREntity]:
    compact = _compact(text)
    normalized = _normalize_kana(compact)
    reading = _reading_for_text(text)
    output: list[ASREntity] = []
    for entity in entities:
        forms = [entity.surface, entity.reading, *entity.aliases]
        normalized_forms = [_normalize_kana(_compact(value)) for value in forms]
        if any(value and value in normalized for value in normalized_forms):
            output.append(entity)
            continue
        if not _JAPANESE_RE.search(text):
            continue
        target = _normalize_kana(_compact(entity.reading))
        if (
            target
            and max(
                _best_window_ratio(normalized, target),
                _best_window_ratio(reading, target),
            )
            >= 0.72
        ):
            output.append(entity)
    return output


def _best_window_ratio(text: str, target: str) -> float:
    if not text or not target:
        return 0.0
    minimum = max(1, len(target) - 3)
    maximum = min(len(text), len(target) + 4)
    if len(text) <= maximum:
        return SequenceMatcher(None, text, target).ratio()
    return max(
        SequenceMatcher(None, text[start : start + size], target).ratio()
        for size in range(minimum, maximum + 1)
        for start in range(len(text) - size + 1)
    )


def _parse_response(response: dict[str, object], expected: int) -> list[str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("ASR correction response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("ASR correction response has no message content")
    value = json.loads(message["content"])
    windows = value.get("windows") if isinstance(value, dict) else None
    if not isinstance(windows, list) or len(windows) != expected:
        raise ValueError("ASR correction response window count mismatch")
    output: list[str] = []
    for expected_id, window in enumerate(windows):
        if not isinstance(window, dict):
            raise ValueError("ASR correction window is not an object")
        try:
            window_id = int(window.get("window_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("ASR correction window_id is not an integer") from exc
        corrected_text = window.get("corrected_text")
        if window_id != expected_id or not isinstance(corrected_text, str):
            raise ValueError("ASR correction response IDs or text are invalid")
        output.append(corrected_text.strip())
    return output


def _corrected_record(
    record: dict[str, object], text: str, method: str
) -> dict[str, object]:
    return {
        **record,
        "original_text": record.get("text", ""),
        "text": text,
        "correction_method": method,
    }


def _format_entities(entities: list[ASREntity]) -> str:
    if not entities:
        return "(none)"
    return "\n".join(
        f"<{entity.surface}> reading={entity.reading} aliases={'｜'.join(entity.aliases) or '(none)'}"
        for entity in entities
    )


def _unique_entities(values: object) -> list[ASREntity]:
    output: list[ASREntity] = []
    seen: set[str] = set()
    for value in values:
        key = _compact(value.surface).casefold()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


def _record_signature(
    text: str, language: str, entities: list[ASREntity], model: str
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "text": text,
        "language": language,
        "entities": [(item.surface, item.reading, item.aliases) for item in entities],
        "model": model,
        "prompt": prompt_templates_digest(_PROMPT),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _compact(value: str) -> str:
    return _SPACE_RE.sub("", unicodedata.normalize("NFKC", value))


def _normalize_kana(value: str) -> str:
    return "".join(
        chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in value
    ).casefold()


@cache
def _sudachi_tokenizer() -> object:
    from sudachipy import dictionary

    return dictionary.Dictionary().create()


def _reading_for_text(text: str) -> str:
    from sudachipy import tokenizer

    morphemes = _sudachi_tokenizer().tokenize(text, tokenizer.Tokenizer.SplitMode.A)
    return _normalize_kana("".join(morpheme.reading_form() for morpheme in morphemes))


def _load_cache(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        value = {}
    if value.get("version") != _CACHE_VERSION or not isinstance(
        value.get("records"), dict
    ):
        return {"version": _CACHE_VERSION, "records": {}}
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _append_audit(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
