from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import cache
from pathlib import Path

from .cache import CacheStore
from .fan_knowledge import KnowledgeHit
from .llm_response import (
    finish_reason,
    parse_json_object,
    structured_request_body,
    structured_response_content,
)
from .prompt_templates import render_user_prompt
from .prompt_budget import batch_requests, fit_optional_text

logger = logging.getLogger(__name__)

_PROMPT = "asr-correct.md"
_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
_RETRIEVAL_FRAGMENT_CHARS = 120
_SENTENCE_END_RE = re.compile(r"(?<=[。！？!?])")


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
    window_chars: int = 3000,
    max_tokens: int = 8192,
    validate_request: Callable[[dict[str, object]], None],
    retrieve_knowledge: Callable[[list[tuple[dict[str, object], str]]], list[list[KnowledgeHit]]]
    | None = None,
) -> list[dict[str, object]]:
    if not records:
        return []
    stage = CacheStore(cache_path).stage("asr_correction", lambda: {
        "records": records, "entities": [asdict(value) for value in entities],
        "model": model, "window_chars": window_chars, "max_tokens": max_tokens,
    })
    records = stage.plan["records"]
    entities = [ASREntity(**value) for value in stage.plan["entities"]]
    model, window_chars, max_tokens = (stage.plan[key] for key in ("model", "window_chars", "max_tokens"))
    cached_result = stage.get("__result__")
    if cached_result is not None:
        return cached_result
    corrected: list[dict[str, object] | None] = [None] * len(records)
    prepared: list[tuple[int, dict[str, object], str, list[ASREntity]]] = []
    for index, record in enumerate(records):
        text = str(record.get("text") or "")
        deterministic = _replace_exact_aliases(text, entities)
        candidates = _candidate_entities(deterministic, entities)
        prepared.append((index, record, deterministic, candidates))

    def prepare_requests():
        retrieval_items = [
            (index, record, fragment)
            for index, record, text, _ in prepared
            for fragment in _retrieval_fragments(text)
        ]
        logger.info("ASR correction preparing retrieval: records=%d fragments=%d", len(prepared), len(retrieval_items))
        retrieved = (
            retrieve_knowledge([(record, fragment) for _, record, fragment in retrieval_items])
            if retrieve_knowledge is not None else [[] for _ in retrieval_items]
        )
        fragments_by_index = {index: [] for index, *_ in prepared}
        for (index, _, fragment), hits in zip(retrieval_items, retrieved, strict=True):
            fragments_by_index[index].append((fragment, hits))

        def render(window):
            return structured_request_body(
                model=model, prompt_name=_PROMPT,
                prompt=render_user_prompt(
                    _PROMPT,
                    ENTITY_REFERENCE=_format_entities(_unique_entities(
                        entity for _, _, _, candidates in window for entity in candidates
                    )),
                    TARGET=_format_correction_window(window, fragments_by_index),
                ),
                max_tokens=max_tokens, temperature=0, thinking=None,
            )

        fitted = []
        trimmed = {}
        for index, record, text, candidates in prepared:
            original = str(record.get('chat_text') or '')
            def render_chat(chat):
                return render([(index, {**record, 'chat_text': chat}, text, candidates)])
            chat = fit_optional_text(original, render_request=render_chat, validate_request=validate_request)
            fitted.append((index, {**record, 'chat_text': chat}, text, candidates))
            if len(chat) < len(original):
                trimmed[str(index)] = len(original) - len(chat)
                logger.info("ASR correction trimmed optional chat: window_id=%s removed_chars=%d", record.get('window_id', index), len(original) - len(chat))
        batches = [
            batch
            for window in _correction_windows(fitted, window_chars)
            for batch in batch_requests(window, render_request=render, validate_request=validate_request)
        ]
        logger.info("ASR correction budgeted %d records into %d requests", len(prepared), len(batches))
        return [
            {
                'indices': [index for index, *_ in window],
                'body': render(window),
                'knowledge_ids': [hit.record_id for hit in _unique_knowledge(
                    hit for index, *_ in window for _, hits in fragments_by_index[index] for hit in hits
                )],
                'fragments': {
                    str(index): [{'text': fragment, 'knowledge_ids': [hit.record_id for hit in hits]}
                                 for fragment, hits in fragments_by_index[index]]
                    for index, *_ in window
                },
                'trimmed_chat_chars': {str(index): trimmed[str(index)] for index, *_ in window if str(index) in trimmed},
            }
            for window in batches
        ]

    requests = stage.remember('requests', prepare_requests)
    logger.info("ASR correction retrieval completed; starting LLM correction")
    for window_number, prepared_request in enumerate(requests):
        unit_id = str(window_number)
        window = [prepared[index] for index in prepared_request['indices']]
        cached = stage.get(unit_id)
        if cached is not None:
            for (index, *_), value in zip(window, cached, strict=True):
                corrected[index] = value
            continue
        logger.info("ASR correction LLM window %d/%d", window_number + 1, len(requests))
        body = prepared_request['body']
        # Check cached requests too, before the downgrade-on-generation-error path.
        validate_request(body)
        response: dict[str, object] | None = None
        batch_error: str | None = None
        try:
            response = request(body)
            content = structured_response_content(
                response, finish_reason=finish_reason
            )
            parsed = _parse_response(content, len(window))
        except Exception as exc:  # noqa: BLE001 - API and validation failures fail open.
            logger.warning(
                "ASR correction downgraded %d windows to rule-normalized text: %s",
                len(window),
                exc,
            )
            parsed = None
            batch_error = f"{type(exc).__name__}: {exc}"
        _append_audit(
            audit_path,
            {
                "event": "asr_correction_window",
                "window_ids": [
                    record.get("window_id", index) for index, record, _, _ in window
                ],
                "request": body,
                "trimmed_chat_chars": prepared_request["trimmed_chat_chars"],
                "response": response,
                "error": batch_error,
                "knowledge_ids": prepared_request["knowledge_ids"],
            },
        )
        cached_values: list[dict[str, object]] = []
        downgrade_reasons: list[str] = []
        for position, (index, record, deterministic, candidates) in enumerate(window):
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
            cached_values.append(corrected[index])
            if error:
                downgrade_reasons.append(error)
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
                    "knowledge_ids": list(dict.fromkeys(
                        record_id
                        for fragment in prepared_request["fragments"][str(index)]
                        for record_id in fragment["knowledge_ids"]
                    )),
                    "retrieval_fragments": prepared_request["fragments"][str(index)],
                },
            )
        stage.put(unit_id, cached_values, source="asr_correction", reason="; ".join(sorted(set(downgrade_reasons))) or None)
    result = [value for value in corrected if value is not None]
    stage.finish(result)
    return result


def _correction_windows(
    pending: list[tuple[int, dict[str, object], str, list[ASREntity]]],
    maximum_chars: int,
) -> list[list[tuple[int, dict[str, object], str, list[ASREntity]]]]:
    if maximum_chars < 1:
        raise ValueError("ASR correction window limit must be positive")
    windows: list[list[tuple[int, dict[str, object], str, list[ASREntity]]]] = []
    current: list[tuple[int, dict[str, object], str, list[ASREntity]]] = []
    characters = 0
    for item in pending:
        item_characters = len(item[2])
        if current and (
            characters + item_characters > maximum_chars
        ):
            windows.append(current)
            current = []
            characters = 0
        current.append(item)
        characters += item_characters
    if current:
        windows.append(current)
    return windows


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


def _parse_response(content: object, expected: int) -> list[str]:
    value = parse_json_object(content)
    segments = value.get("segments") if isinstance(value, dict) else None
    if not isinstance(segments, list) or len(segments) != expected:
        raise ValueError("ASR correction response segment count mismatch")
    output: list[str] = []
    for expected_id, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise TypeError("ASR correction segment is not an object")
        try:
            segment_id = int(segment.get("segment_id"))
        except (TypeError, ValueError) as exc:
            raise ValueError("ASR correction segment_id is not an integer") from exc
        corrected_text = segment.get("corrected_text")
        if segment_id != expected_id or not isinstance(corrected_text, str):
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


def _format_correction_window(
    window: list[tuple[int, dict[str, object], str, list[ASREntity]]],
    fragments_by_index: dict[int, list[tuple[str, list[KnowledgeHit]]]],
) -> str:
    values: list[str] = []
    for position, (index, record, _text, candidates) in enumerate(window):
        chat_text = str(record.get("chat_text") or "").strip() or "(none)"
        candidate_text = "｜".join(entity.surface for entity in candidates) or "(none)"
        fragments: list[str] = []
        for fragment_id, (fragment, knowledge) in enumerate(
            fragments_by_index[index]
        ):
            knowledge_text = (
                "\n".join(
                    f"- <{hit.kind}:{hit.title}> {hit.body[:180]}"
                    for hit in knowledge[:3]
                )
                or "(none)"
            )
            fragments.append(
                f'<RETRIEVAL_FRAGMENT id="{fragment_id}">\n'
                f"TEXT:\n{fragment}\n"
                f"FAN_KNOWLEDGE:\n{knowledge_text}\n"
                "</RETRIEVAL_FRAGMENT>"
            )
        fragment_text = "\n".join(fragments)
        values.append(
            f'<SEGMENT id="{position}" candidates="{candidate_text}">\n'
            f"CURRENT_VIDEO_CHAT:\n{chat_text}\n"
            f"LOCAL_RETRIEVAL:\n{fragment_text}\n"
            f"ASR_TEXT:\n{_text}\n"
            "</SEGMENT>"
        )
    return "<CORRECTION_WINDOW>\n" + "\n\n".join(values) + "\n</CORRECTION_WINDOW>"


def _retrieval_fragments(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return [""]
    fragments: list[str] = []
    current = ""
    for sentence in _SENTENCE_END_RE.split(text):
        while len(sentence) > _RETRIEVAL_FRAGMENT_CHARS:
            if current:
                fragments.append(current)
                current = ""
            fragments.append(sentence[:_RETRIEVAL_FRAGMENT_CHARS])
            sentence = sentence[_RETRIEVAL_FRAGMENT_CHARS:]
        if current and len(current) + len(sentence) > _RETRIEVAL_FRAGMENT_CHARS:
            fragments.append(current)
            current = ""
        current += sentence
    if current:
        fragments.append(current)
    return fragments


def _unique_knowledge(values: object) -> list[KnowledgeHit]:
    output: list[KnowledgeHit] = []
    seen: set[str] = set()
    for value in values:
        if value.record_id not in seen:
            seen.add(value.record_id)
            output.append(value)
    return output


def _unique_entities(values: object) -> list[ASREntity]:
    output: list[ASREntity] = []
    seen: set[str] = set()
    for value in values:
        key = _compact(value.surface).casefold()
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


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


def _append_audit(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
