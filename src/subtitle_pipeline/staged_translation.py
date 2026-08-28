from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .config import LLMConfig, SegmentationConfig, TranslationConfig
from .fan_knowledge import KnowledgeHit
from .llm_response import (
    parse_json_object,
    structured_request_body,
    structured_response_content,
)
from .local_segmentation import LocalUnit, SpeakerTrack
from .prompt_templates import prompt_templates_digest, render_user_prompt
from .reference_context import compact_translation_reference_context
from .source_language import (
    combine_source_languages,
    join_source_fragments,
    language_for_text,
)
from .subtitles import Cue, cue_from_mapping, text_display_width
from .telemetry import stage_metrics
from .translation_support import (
    best_split,
    contains_kana,
    dialogue_context,
    escape_prompt_text,
    machine_translate_with_protected_terms,
    normalize_residual_japanese,
    reference_replacements,
    reference_text,
    source_timed_units,
    window_ranges,
    window_source_language,
)

logger = logging.getLogger(__name__)
_CACHE_VERSION = 4
_SEGMENT_PROMPT = "segment-source-cues.md"
_TRANSLATE_PROMPT = "translate-fixed-cues.md"
_REVIEW_PROMPT = "review-fixed-translations.md"
_CONTENT_ATTEMPTS = 2
_TOPIC_MAX_CUES = 16
_TOPIC_MAX_SECONDS = 90.0
_TOPIC_GAP_SECONDS = 5.0
_TOPIC_KNOWLEDGE_HITS = 6
_TOPIC_KNOWLEDGE_CHARS = 1800
_LOCAL_OUTPUT_RESERVE_TOKENS = 4096
_TOPIC_SHIFT = re.compile(
    r"^(?:そういえば|ところで|話(?:は|を)?変(?:わ|え)|次(?:に|は)|別の話)"
)


@dataclass(frozen=True)
class SegmentRecord:
    track: str
    start_id: int
    end_id: int


@dataclass(frozen=True)
class TranslationOutcome:
    text: str
    source: str
    llm_text: str | None = None
    downgrade_reason: str | None = None
    request_id: str | None = None


class PromptBudgetExceeded(RuntimeError):
    pass


def run_segmentation(
    *,
    tracks: list[SpeakerTrack],
    source_cues: list[Cue],
    segmentation: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    source_maximum_units: float,
    cache_path: Path | None,
    sudachi_versions: dict[str, str],
) -> list[Cue]:
    signature = _signature(
        "segmentation",
        {
            "tracks": [asdict(track) for track in tracks],
            "config": asdict(segmentation),
            "model": llm.model,
            "source_maximum_units": source_maximum_units,
            "sudachi": sudachi_versions,
            "prompt": prompt_templates_digest(_SEGMENT_PROMPT),
        },
    )
    cached = _load_records(cache_path, signature, SegmentRecord)
    if cached is not None:
        records = [SegmentRecord(**value) for value in cached]
        _validate_final_segments(records, tracks, source_maximum_units)
        return _records_to_source_cues(source_cues, tracks, records)

    all_units = [unit for track in tracks for unit in track.units]
    windows = [
        (track, start, end)
        for track in tracks
        for start, end in window_ranges(track.units, segmentation)
    ]

    def process(track: SpeakerTrack, start: int, end: int) -> list[SegmentRecord]:
        if end - start == 1:
            unit = track.units[start]
            if text_display_width(unit.text) > source_maximum_units:
                logger.warning(
                    "single local source unit exceeds Japanese guidance "
                    "track=%s unit=%d width=%.3f limit=%.3f text=%r",
                    track.key,
                    unit.local_id,
                    text_display_width(unit.text),
                    source_maximum_units,
                    unit.text,
                )
            return [SegmentRecord(track.key, unit.local_id, unit.local_id + 1)]
        return _segment_resilient(
            track,
            start,
            end,
            all_units,
            segmentation,
            llm,
            request,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            source_maximum_units,
        )

    records: list[SegmentRecord] = []
    if windows:
        logger.info(
            "segmenting %d source windows with concurrency=%d Japanese_limit=%.3f",
            len(windows),
            llm.max_concurrency,
            source_maximum_units,
        )
        with (
            stage_metrics("llm.source_segmentation"),
            ThreadPoolExecutor(
                max_workers=min(llm.max_concurrency, len(windows)),
                thread_name_prefix="source-segment",
            ) as executor,
        ):
            futures = {executor.submit(process, *item): item for item in windows}
            for future in as_completed(futures):
                records.extend(future.result())
    records.sort(key=lambda value: (value.track, value.start_id))
    _validate_final_segments(records, tracks, source_maximum_units)
    _write_records(cache_path, signature, records)
    return _records_to_source_cues(source_cues, tracks, records)


def run_fixed_translation(
    *,
    source_cues: list[Cue],
    llm: LLMConfig,
    translation: TranslationConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    local_translate: Callable[[str], str],
    translation_context: dict[str, object],
    honorific_rules: str,
    cache_path: Path | None,
    retrieve_knowledge: Callable[[list[Cue], str], list[KnowledgeHit]] | None = None,
    retrieve_chat: Callable[[list[Cue]], str] | None = None,
    audit_path: Path | None = None,
) -> list[Cue]:
    signature = _signature(
        "translation",
        {
            "source": [asdict(cue) for cue in source_cues],
            "model": llm.model,
            "target_language": translation.target_language,
            "translation": asdict(translation),
            "context": translation_context,
            "prompt": prompt_templates_digest(_TRANSLATE_PROMPT),
        },
    )
    cached = _load_records(cache_path, signature, Cue)
    if cached is not None:
        result = [cue_from_mapping(value) for value in cached]
        _write_translation_audit(
            audit_path,
            [
                _translation_audit_entry(
                    index,
                    source_cues[index],
                    cue.text,
                    "cache",
                )
                for index, cue in enumerate(result)
            ],
        )
        return result

    translations: dict[int, TranslationOutcome] = {}
    pending: list[int] = []
    for cue_id, cue in enumerate(source_cues):
        if cue.preferred_translation:
            translations[cue_id] = TranslationOutcome(
                cue.preferred_translation.strip(), "preferred_translation"
            )
        else:
            pending.append(cue_id)
    groups = _topic_request_groups(source_cues, pending, translation)

    replacements = reference_replacements(translation_context)
    audit_lock = threading.Lock()

    def audit_outcomes(values: dict[int, TranslationOutcome]) -> None:
        _write_translation_audit(
            audit_path,
            [
                _translation_audit_entry(
                    cue_id,
                    source_cues[cue_id],
                    outcome.text,
                    outcome.source,
                    llm_text=outcome.llm_text,
                    downgrade_reason=outcome.downgrade_reason,
                    request_id=outcome.request_id,
                )
                for cue_id, outcome in sorted(values.items())
            ],
            lock=audit_lock,
        )

    audit_outcomes(translations)

    def process(ids: list[int]) -> dict[int, TranslationOutcome]:
        return _translate_resilient(
            ids,
            source_cues,
            llm,
            request,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            local_translate,
            translation_context,
            replacements,
            honorific_rules,
            translation,
            retrieve_knowledge,
            retrieve_chat,
        )

    if groups:
        logger.info(
            "translating %d fixed source cues in %d request group(s) "
            "with concurrency=%d",
            len(pending),
            len(groups),
            llm.max_concurrency,
        )
        lock = threading.Lock()
        with (
            stage_metrics("llm.fixed_cue_translation"),
            ThreadPoolExecutor(
                max_workers=min(llm.max_concurrency, len(groups)),
                thread_name_prefix="fixed-translate",
            ) as executor,
        ):
            futures = [executor.submit(process, group) for group in groups]
            for future in as_completed(futures):
                completed = future.result()
                with lock:
                    translations.update(completed)
                audit_outcomes(completed)

    if set(translations) != set(range(len(source_cues))):
        raise RuntimeError("fixed translation did not cover every source cue")
    result = [
        replace(cue, text=translations[index].text, source_text=cue.text)
        for index, cue in enumerate(source_cues)
    ]
    _write_records(cache_path, signature, result)
    return result


def run_translation_review(
    *,
    source_cues: list[Cue],
    translated_cues: list[Cue],
    llm: LLMConfig,
    translation: TranslationConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    finish_reason: Callable[[object], str | None],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    translation_context: dict[str, object],
    maximum_units: float,
    honorific_rules: str,
    cache_path: Path | None,
    retrieve_knowledge: Callable[[list[Cue], str], list[KnowledgeHit]] | None = None,
    retrieve_chat: Callable[[list[Cue]], str] | None = None,
    audit_path: Path | None = None,
) -> list[Cue]:
    if len(source_cues) != len(translated_cues):
        raise ValueError(
            "translation review requires aligned source and translated cues"
        )
    signature = _signature(
        "translation_review",
        {
            "source": [asdict(cue) for cue in source_cues],
            "draft": [asdict(cue) for cue in translated_cues],
            "model": llm.model,
            "target_language": translation.target_language,
            "maximum_units": maximum_units,
            "context": translation_context,
            "prompt": prompt_templates_digest(_REVIEW_PROMPT),
        },
    )
    cached = _load_records(cache_path, signature, Cue)
    if cached is not None:
        result = [cue_from_mapping(value) for value in cached]
        _write_translation_audit(
            audit_path,
            [
                _translation_review_audit_entry(
                    cue_id,
                    source_cues[cue_id],
                    translated_cues[cue_id].text,
                    cue.text,
                    maximum_units,
                    "cache",
                )
                for cue_id, cue in enumerate(result)
                if source_cues[cue_id].kind != "singing"
            ],
        )
        return result

    result = list(translated_cues)
    pending = [
        cue_id for cue_id, cue in enumerate(source_cues) if cue.kind != "singing"
    ]
    groups = _topic_request_groups(source_cues, pending, translation)
    audit_lock = threading.Lock()

    def process(ids: list[int]) -> tuple[dict[int, Cue], list[dict[str, object]]]:
        selected = [source_cues[cue_id] for cue_id in ids]
        chat_evidence = retrieve_chat(selected) if retrieve_chat else ""
        knowledge = _topic_knowledge(
            retrieve_knowledge(selected, chat_evidence)
            if retrieve_knowledge is not None
            else []
        )
        start = min(cue.start for cue in selected)
        end = max(cue.end for cue in selected)
        dialogue = _translation_dialogue_context(
            source_cues, selected, start, end, translation
        )
        source_index = {id(cue): cue_id for cue_id, cue in enumerate(source_cues)}
        evidence_text = "\n".join(
            [
                *(cue.text for cue in selected),
                *(translated_cues[cue_id].text for cue_id in ids),
                *(cue.text for cue in dialogue),
                chat_evidence,
                *(hit.body for hit in knowledge),
            ]
        )
        reference = compact_translation_reference_context(
            translation_context,
            evidence_text=evidence_text,
            speakers={cue.speaker for cue in selected if cue.speaker},
        )
        prompt = render_user_prompt(
            _REVIEW_PROMPT,
            MAXIMUM_UNITS=f"{maximum_units:.3f}",
            HONORIFIC_TRANSLATION_RULES=honorific_rules,
            REFERENCE_TEXT=reference_text(reference),
            CHAT_EVIDENCE=chat_evidence or "(none)",
            DIALOGUE_CONTEXT="\n".join(
                _format_review_context_cue(
                    cue,
                    translated_cues[source_index[id(cue)]],
                )
                for cue in dialogue
            )
            or "(none)",
            CUE_TEXT=_format_review_cues(
                ids,
                source_cues,
                translated_cues,
                knowledge,
                maximum_units,
            ),
        )
        body = structured_request_body(
            model=llm.model,
            prompt_name=_REVIEW_PROMPT,
            prompt=prompt,
            max_tokens=translation.max_tokens,
            temperature=0.1,
            json_mode=llm.json_mode,
            thinking=llm.thinking,
        )
        response: object = None
        content: object = None
        reviewed = {cue_id: translated_cues[cue_id] for cue_id in ids}
        request_id: str | None = None
        method = "unchanged"
        try:
            _validate_prompt_budget(prompt, llm, translation)
            response = request(body)
            content = structured_response_content(
                response, finish_reason=finish_reason
            )
            corrections = parse_json_object(content).get("corrections")
            if not isinstance(corrections, list):
                raise TypeError("translation review requires a corrections array")
            local_to_global = dict(enumerate(ids))
            seen: set[int] = set()
            for position, value in enumerate(corrections):
                if not isinstance(value, dict) or set(value) != {"cue_id", "text"}:
                    raise RuntimeError(
                        f"translation review correction {position} "
                        "has unexpected fields"
                    )
                local_id = _integer(value["cue_id"], "cue_id", position)
                if local_id not in local_to_global or local_id in seen:
                    raise RuntimeError(
                        f"invalid or duplicate review cue_id={local_id}"
                    )
                text = value["text"]
                if not isinstance(text, str) or not text.strip():
                    raise TypeError(
                        f"translation review cue {position} text is not non-empty text"
                    )
                text = text.strip()
                if contains_kana(text, translation.target_language):
                    raise RuntimeError(
                        f"translation review cue {position} contains residual Japanese"
                    )
                seen.add(local_id)
                global_id = local_to_global[local_id]
                reviewed[global_id] = replace(translated_cues[global_id], text=text)
            request_id_value = response.get("_audit_request_id")
            request_id = (
                request_id_value if isinstance(request_id_value, str) else None
            )
            method = "llm_review"
        except Exception as exc:  # noqa: BLE001 - single-attempt review fails open
            log_invalid_response(
                "fixed translation review", exc, content, body, response
            )
            logger.warning(
                "translation review failed without retry; preserving "
                "%d draft cue(s): %s",
                len(ids),
                exc,
            )
            method = f"draft_preserved_{type(exc).__name__}"

        audit_entries: list[dict[str, object]] = []
        for cue_id in ids:
            final_text = reviewed[cue_id].text
            width = text_display_width(final_text)
            if width > maximum_units:
                logger.warning(
                    "accepting overwide reviewed cue cue_id=%d width=%.3f "
                    "limit=%.3f text=%r",
                    cue_id,
                    width,
                    maximum_units,
                    final_text,
                )
            audit_entries.append(
                _translation_review_audit_entry(
                    cue_id,
                    source_cues[cue_id],
                    translated_cues[cue_id].text,
                    final_text,
                    maximum_units,
                    method,
                    request_id=request_id,
                )
            )
        return reviewed, audit_entries

    if groups:
        logger.info(
            "reviewing %d translated cues in %d request group(s) "
            "with concurrency=%d soft_Chinese_limit=%.3f",
            len(pending),
            len(groups),
            llm.max_concurrency,
            maximum_units,
        )
        with (
            stage_metrics("llm.fixed_translation_review"),
            ThreadPoolExecutor(
                max_workers=min(llm.max_concurrency, len(groups)),
                thread_name_prefix="fixed-review",
            ) as executor,
        ):
            futures = [executor.submit(process, group) for group in groups]
            for future in as_completed(futures):
                reviewed, audit_entries = future.result()
                for cue_id, cue in reviewed.items():
                    result[cue_id] = cue
                _write_translation_audit(
                    audit_path, audit_entries, lock=audit_lock
                )

    _write_records(cache_path, signature, result)
    return result


def _segment_resilient(
    track: SpeakerTrack,
    start: int,
    end: int,
    all_units: list[LocalUnit],
    config: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    maximum_units: float,
) -> list[SegmentRecord]:
    try:
        return _request_segmentation(
            track,
            start,
            end,
            all_units,
            config,
            llm,
            request,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            maximum_units,
        )
    except Exception as exc:
        if is_nontransient(exc) or retry_delay(exc, 1) is not None:
            raise
        if end - start <= 1:
            unit = track.units[start]
            return [SegmentRecord(track.key, unit.local_id, unit.local_id + 1)]
        split = best_split(track.units, start, end)
        logger.warning(
            "shrinking failed source-segmentation window %s:%d-%d at %d: %s",
            track.key,
            start,
            end,
            split,
            exc,
        )
        return [
            *_segment_resilient(
                track,
                start,
                split,
                all_units,
                config,
                llm,
                request,
                parse_content,
                finish_reason,
                retry_delay,
                is_nontransient,
                log_invalid_response,
                maximum_units,
            ),
            *_segment_resilient(
                track,
                split,
                end,
                all_units,
                config,
                llm,
                request,
                parse_content,
                finish_reason,
                retry_delay,
                is_nontransient,
                log_invalid_response,
                maximum_units,
            ),
        ]


def _request_segmentation(
    track: SpeakerTrack,
    start: int,
    end: int,
    all_units: list[LocalUnit],
    config: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    maximum_units: float,
) -> list[SegmentRecord]:
    previous_error: Exception | None = None
    transient_attempts = 0
    for content_attempt in range(_CONTENT_ATTEMPTS):
        selected = track.units[start:end]
        prompt = render_user_prompt(
            _SEGMENT_PROMPT,
            SOURCE_LANGUAGE=window_source_language(selected),
            SOURCE_MAXIMUM_UNITS=f"{maximum_units:.3f}",
            DIALOGUE_CONTEXT=dialogue_context(all_units, selected, config) or "(none)",
            TARGET_TEXT="\n".join(
                f"<{i}>{escape_prompt_text(unit.text)}" for i, unit in enumerate(selected)
            ),
            RETRY_SECTION=""
            if previous_error is None
            else f"\n\nPREVIOUS_RESPONSE_ERROR: {previous_error}",
        )
        body = structured_request_body(
            model=llm.model,
            prompt_name=_SEGMENT_PROMPT,
            prompt=prompt,
            max_tokens=config.max_tokens,
            temperature=0.1,
            json_mode=llm.json_mode,
            thinking=llm.thinking,
        )
        content: object = None
        response: object = None
        try:
            response = request(body)
            content = structured_response_content(
                response, finish_reason=finish_reason
            )
            return _validate_segments(
                parse_content(content), track, start, end, maximum_units
            )
        except Exception as exc:
            log_invalid_response(
                "source cue segmentation", exc, content, body, response
            )
            if is_nontransient(exc):
                raise
            delay = retry_delay(exc, transient_attempts + 1)
            if delay is not None:
                transient_attempts += 1
                if transient_attempts >= llm.max_retries:
                    raise
                time.sleep(delay)
                continue
            previous_error = exc
            if content_attempt + 1 >= min(_CONTENT_ATTEMPTS, llm.max_retries):
                raise
    raise RuntimeError("source segmentation exhausted retries")


def _validate_segments(
    values: list[object],
    track: SpeakerTrack,
    start: int,
    end: int,
    maximum_units: float,
) -> list[SegmentRecord]:
    if not values:
        raise RuntimeError("segmentation response contains no cues")
    expected = 0
    records: list[SegmentRecord] = []
    for position, value in enumerate(values):
        if not isinstance(value, dict) or set(value) != {"start_id", "end_id"}:
            raise RuntimeError(f"segmentation cue {position} has unexpected fields")
        relative_start = _integer(value["start_id"], "start_id", position)
        relative_end = _integer(value["end_id"], "end_id", position) + 1
        if (
            relative_start != expected
            or relative_end <= relative_start
            or relative_end > end - start
        ):
            raise RuntimeError(f"segmentation coverage failed at cue {position}")
        selected = track.units[start + relative_start : start + relative_end]
        source = join_source_fragments((unit.text, unit.language) for unit in selected)
        width = text_display_width(source)
        if width > maximum_units:
            raise RuntimeError(
                f"source cue width={width:.3f} exceeds Japanese limit={maximum_units:.3f}"
            )
        records.append(
            SegmentRecord(
                track.key,
                track.units[start + relative_start].local_id,
                track.units[start + relative_end - 1].local_id + 1,
            )
        )
        expected = relative_end
    if expected != end - start:
        raise RuntimeError("segmentation response did not cover the final unit")
    return records


def _translate_resilient(
    ids: list[int],
    cues: list[Cue],
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    local_translate: Callable[[str], str],
    base_context: dict[str, object],
    replacements: tuple[tuple[str, str], ...],
    honorific_rules: str,
    translation: TranslationConfig,
    retrieve_knowledge: Callable[[list[Cue], str], list[KnowledgeHit]] | None,
    retrieve_chat: Callable[[list[Cue]], str] | None,
    context_excluded_ids: frozenset[int] | None = None,
) -> dict[int, TranslationOutcome]:
    excluded_ids = context_excluded_ids or frozenset(ids)
    selected = [cues[cue_id] for cue_id in ids]
    chat_evidence = retrieve_chat(selected) if retrieve_chat else ""
    knowledge = _topic_knowledge(
        retrieve_knowledge(selected, chat_evidence)
        if retrieve_knowledge is not None
        else []
    )
    try:
        return _request_translation(
            ids,
            cues,
            llm,
            request,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            local_translate,
            base_context,
            knowledge,
            replacements,
            honorific_rules,
            translation,
            chat_evidence,
            excluded_ids,
        )
    except Exception as exc:
        if isinstance(exc, PromptBudgetExceeded):
            if len(ids) == 1:
                cue_id = ids[0]
                logger.warning(
                    "translation prompt exceeds local context for one cue; "
                    "using local machine translation cue_id=%d",
                    cue_id,
                )
                return {
                    cue_id: TranslationOutcome(
                        _fallback(cues[cue_id], replacements, local_translate),
                        "local_mt",
                        downgrade_reason="prompt_budget_exceeded",
                    )
                }
            middle = len(ids) // 2
            logger.warning(
                "splitting fixed-translation topic size=%d to satisfy token budget",
                len(ids),
            )
            return {
                **_translate_resilient(
                    ids[:middle], cues, llm, request, parse_content, finish_reason,
                    retry_delay, is_nontransient, log_invalid_response,
                    local_translate, base_context, replacements, honorific_rules,
                    translation, retrieve_knowledge, retrieve_chat, excluded_ids,
                ),
                **_translate_resilient(
                    ids[middle:], cues, llm, request, parse_content, finish_reason,
                    retry_delay, is_nontransient, log_invalid_response,
                    local_translate, base_context, replacements, honorific_rules,
                    translation, retrieve_knowledge, retrieve_chat, excluded_ids,
                ),
            }
        if is_nontransient(exc) or retry_delay(exc, 1) is not None:
            raise
        if len(ids) == 1:
            cue_id = ids[0]
            logger.warning(
                "LLM translation downgrade reason=%s fallback=local_machine_translation cue_id=%d original_text=%r",
                type(exc).__name__,
                cue_id,
                cues[cue_id].text,
            )
            return {
                cue_id: TranslationOutcome(
                    _fallback(cues[cue_id], replacements, local_translate),
                    "local_mt",
                    downgrade_reason=type(exc).__name__,
                )
            }
        middle = len(ids) // 2
        logger.warning(
            "shrinking failed fixed-translation batch size=%d: %s", len(ids), exc
        )
        return {
            **_translate_resilient(
                ids[:middle],
                cues,
                llm,
                request,
                parse_content,
                finish_reason,
                retry_delay,
                is_nontransient,
                log_invalid_response,
                local_translate,
                base_context,
                replacements,
                honorific_rules,
                translation,
                retrieve_knowledge,
                retrieve_chat,
                excluded_ids,
            ),
            **_translate_resilient(
                ids[middle:],
                cues,
                llm,
                request,
                parse_content,
                finish_reason,
                retry_delay,
                is_nontransient,
                log_invalid_response,
                local_translate,
                base_context,
                replacements,
                honorific_rules,
                translation,
                retrieve_knowledge,
                retrieve_chat,
                excluded_ids,
            ),
        }


def _request_translation(
    ids: list[int],
    cues: list[Cue],
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[
        [str, Exception, object, dict[str, object] | None, object], None
    ],
    local_translate: Callable[[str], str],
    context: dict[str, object],
    knowledge: list[KnowledgeHit],
    replacements: tuple[tuple[str, str], ...],
    honorific_rules: str,
    translation: TranslationConfig,
    chat_evidence: str,
    context_excluded_ids: frozenset[int],
) -> dict[int, TranslationOutcome]:
    previous_error: Exception | None = None
    transient_attempts = 0
    for content_attempt in range(_CONTENT_ATTEMPTS):
        local_to_global = dict(enumerate(ids))
        selected = [cues[value] for value in ids]
        start = min(cue.start for cue in selected)
        end = max(cue.end for cue in selected)
        dialogue = _translation_dialogue_context(
            cues,
            [cues[cue_id] for cue_id in context_excluded_ids],
            start,
            end,
            translation,
        )
        prompt = _fit_translation_prompt(
            ids=ids,
            cues=cues,
            selected=selected,
            dialogue=dialogue,
            knowledge=knowledge,
            chat_evidence=chat_evidence,
            context=context,
            llm=llm,
            translation=translation,
            honorific_rules=honorific_rules,
            previous_error=previous_error,
        )
        body = structured_request_body(
            model=llm.model,
            prompt_name=_TRANSLATE_PROMPT,
            prompt=prompt,
            max_tokens=translation.max_tokens,
            temperature=0.1,
            json_mode=llm.json_mode,
            thinking=llm.thinking,
        )
        content: object = None
        response: object = None
        try:
            response = request(body)
            content = structured_response_content(
                response, finish_reason=finish_reason
            )
            values = parse_content(content)
            request_id = response.get("_audit_request_id")
            result: dict[int, TranslationOutcome] = {}
            seen_local_ids: set[int] = set()
            for position, value in enumerate(values):
                if not isinstance(value, dict) or set(value) != {"cue_id", "text"}:
                    raise RuntimeError(
                        f"translation cue {position} has unexpected fields"
                    )
                local_id = _integer(value["cue_id"], "cue_id", position)
                if local_id not in local_to_global or local_id in seen_local_ids:
                    raise RuntimeError(
                        f"invalid or duplicate translation cue_id={local_id}"
                    )
                seen_local_ids.add(local_id)
                text = value["text"]
                if not isinstance(text, str):
                    raise TypeError(f"translation cue {position} text is not a string")
                global_id = local_to_global[local_id]
                cue = cues[global_id]
                text = text.strip()
                if not text:
                    logger.warning(
                        "LLM translation downgrade reason=empty_translation fallback=local_machine_translation cue_id=%d original_text=%r",
                        global_id,
                        cue.text,
                    )
                    result[global_id] = TranslationOutcome(
                        _fallback(cue, replacements, local_translate),
                        "local_mt",
                        llm_text=text,
                        downgrade_reason="empty_translation",
                        request_id=(request_id if isinstance(request_id, str) else None),
                    )
                elif contains_kana(text, translation.target_language):
                    logger.warning(
                        "LLM translation downgrade reason=residual_japanese fallback=protected_machine_translation cue_id=%d text=%r",
                        global_id,
                        text,
                    )
                    result[global_id] = TranslationOutcome(
                        normalize_residual_japanese(
                            text, replacements, local_translate, cue.language
                        ),
                        "protected_local_mt",
                        llm_text=text,
                        downgrade_reason="residual_japanese",
                        request_id=(request_id if isinstance(request_id, str) else None),
                    )
                else:
                    result[global_id] = TranslationOutcome(
                        text,
                        "llm",
                        llm_text=text,
                        request_id=(request_id if isinstance(request_id, str) else None),
                    )
            if set(result) != set(ids):
                raise RuntimeError("translation response did not cover every cue_id")
            return result
        except Exception as exc:
            log_invalid_response("fixed cue translation", exc, content, body, response)
            if is_nontransient(exc):
                raise
            delay = retry_delay(exc, transient_attempts + 1)
            if delay is not None:
                transient_attempts += 1
                if transient_attempts >= llm.max_retries:
                    raise
                time.sleep(delay)
                continue
            previous_error = exc
            if content_attempt + 1 >= min(_CONTENT_ATTEMPTS, llm.max_retries):
                raise
    raise RuntimeError("fixed translation exhausted retries")


def _format_translation_cues(
    ids: list[int],
    cues: list[Cue],
    knowledge: list[KnowledgeHit],
) -> str:
    knowledge_text = _format_topic_knowledge(knowledge)
    values = ["<TOPIC_BLOCK>", f"FAN_KNOWLEDGE:\n{knowledge_text}", "CUES:"]
    for local_id, cue_id in enumerate(ids):
        cue = cues[cue_id]
        values.append(
            f'<CUE id="{local_id}" '
            f'speaker="{escape_prompt_text(cue.speaker or "unknown")}" '
            f'language="{language_for_text(cue.text, cue.language)}">\n'
            f"ASR_TEXT:\n{escape_prompt_text(cue.text)}\n"
            "</CUE>"
        )
    values.append("</TOPIC_BLOCK>")
    return "\n\n".join(values)


def _format_review_cues(
    ids: list[int],
    source_cues: list[Cue],
    translated_cues: list[Cue],
    knowledge: list[KnowledgeHit],
    maximum_units: float,
) -> str:
    knowledge_text = _format_topic_knowledge(knowledge)
    values = ["<TOPIC_BLOCK>", f"FAN_KNOWLEDGE:\n{knowledge_text}", "CUES:"]
    for local_id, cue_id in enumerate(ids):
        source = source_cues[cue_id]
        draft = translated_cues[cue_id]
        values.append(
            f'<CUE id="{local_id}" width="{text_display_width(draft.text):.3f}" '
            f'limit="{maximum_units:.3f}" '
            f'speaker="{escape_prompt_text(source.speaker or "unknown")}" '
            f'language="{language_for_text(source.text, source.language)}">\n'
            f"SOURCE_TEXT:\n{escape_prompt_text(source.text)}\n"
            f"DRAFT_TRANSLATION:\n{escape_prompt_text(draft.text)}\n"
            "</CUE>"
        )
    values.append("</TOPIC_BLOCK>")
    return "\n\n".join(values)


def _format_review_context_cue(source: Cue, translated: Cue) -> str:
    return (
        f"<{source.speaker or 'unknown'}>"
        f"{escape_prompt_text(source.text)} => {escape_prompt_text(translated.text)}"
    )


def _fit_translation_prompt(
    *,
    ids: list[int],
    cues: list[Cue],
    selected: list[Cue],
    dialogue: list[Cue],
    knowledge: list[KnowledgeHit],
    chat_evidence: str,
    context: dict[str, object],
    llm: LLMConfig,
    translation: TranslationConfig,
    honorific_rules: str,
    previous_error: Exception | None,
) -> str:
    active_knowledge = list(knowledge)
    active_chat = chat_evidence.splitlines()
    active_dialogue = list(dialogue)
    original_counts = (
        len(active_knowledge),
        len(active_chat),
        len(active_dialogue),
    )

    while True:
        chat_text = "\n".join(active_chat)
        evidence_text = "\n".join(
            [
                *(cue.text for cue in selected),
                *(cue.text for cue in active_dialogue),
                chat_text,
                *(hit.body for hit in active_knowledge),
            ]
        )
        reference = compact_translation_reference_context(
            context,
            evidence_text=evidence_text,
            speakers={cue.speaker for cue in selected if cue.speaker},
        )
        prompt = render_user_prompt(
            _TRANSLATE_PROMPT,
            TARGET_LANGUAGE=translation.target_language,
            HONORIFIC_TRANSLATION_RULES=honorific_rules,
            REFERENCE_TEXT=reference_text(reference),
            CHAT_EVIDENCE=chat_text or "(none)",
            DIALOGUE_CONTEXT="\n".join(
                f"<{cue.speaker or 'unknown'}>{escape_prompt_text(cue.text)}"
                for cue in active_dialogue
            )
            or "(none)",
            SOURCE_TEXT=_format_translation_cues(ids, cues, active_knowledge),
            RETRY_SECTION=(
                ""
                if previous_error is None
                else f"\n\nPREVIOUS_RESPONSE_ERROR: {previous_error}"
            ),
        )
        if _prompt_within_budget(prompt, llm, translation):
            reduced_counts = (
                len(active_knowledge),
                len(active_chat),
                len(active_dialogue),
            )
            if reduced_counts != original_counts:
                logger.info(
                    "reduced translation evidence to fit local context: "
                    "knowledge=%d/%d chat=%d/%d dialogue=%d/%d",
                    reduced_counts[0],
                    original_counts[0],
                    reduced_counts[1],
                    original_counts[1],
                    reduced_counts[2],
                    original_counts[2],
                )
            return prompt

        knowledge_chars = sum(min(180, len(hit.body)) for hit in active_knowledge)
        chat_chars = sum(len(line) for line in active_chat)
        if active_chat and chat_chars >= knowledge_chars:
            active_chat = _reduce_chat_lines(active_chat)
        elif active_knowledge:
            active_knowledge.pop()
        elif active_chat:
            active_chat = _reduce_chat_lines(active_chat)
        elif active_dialogue:
            active_dialogue.pop(_farthest_dialogue_index(active_dialogue, selected))
        else:
            _validate_prompt_budget(prompt, llm, translation)
            return prompt


def _reduce_chat_lines(lines: list[str]) -> list[str]:
    if len(lines) <= 1:
        return []
    target = len(lines) // 2
    paid = [index for index, line in enumerate(lines) if line.startswith("[SC ")]
    selected = paid[:target]
    if len(selected) < target:
        regular = [index for index in range(len(lines)) if index not in set(paid)]
        slots = target - len(selected)
        selected.extend(_evenly_spaced_indices(regular, slots))
    return [lines[index] for index in sorted(selected)]


def _evenly_spaced_indices(indices: list[int], count: int) -> list[int]:
    if count <= 0 or not indices:
        return []
    if count >= len(indices):
        return indices
    if count == 1:
        return [indices[len(indices) // 2]]
    return [
        indices[round(position * (len(indices) - 1) / (count - 1))]
        for position in range(count)
    ]


def _farthest_dialogue_index(dialogue: list[Cue], selected: list[Cue]) -> int:
    start = min(cue.start for cue in selected)
    end = max(cue.end for cue in selected)

    def distance(cue: Cue) -> float:
        if cue.end <= start:
            return start - cue.end
        if cue.start >= end:
            return cue.start - end
        return 0.0

    return max(range(len(dialogue)), key=lambda index: distance(dialogue[index]))


def _topic_request_groups(
    cues: list[Cue], ids: list[int], translation: TranslationConfig
) -> list[list[int]]:
    groups: list[list[int]] = []
    current: list[int] = []
    characters = 0
    for cue_id in ids:
        cue_chars = len(cues[cue_id].text)
        previous_id = current[-1] if current else None
        previous = cues[current[-1]] if current else None
        first = cues[current[0]] if current else None
        if current and (
            len(current) >= min(translation.batch_cues, _TOPIC_MAX_CUES)
            or characters + cue_chars > translation.batch_chars
            or (previous is not None and cues[cue_id].start - previous.end > _TOPIC_GAP_SECONDS)
            or (first is not None and cues[cue_id].end - first.start > _TOPIC_MAX_SECONDS)
            or (
                previous_id is not None
                and any(
                    value.kind == "singing"
                    for value in cues[previous_id + 1 : cue_id]
                )
            )
            or (
                previous is not None
                and (previous.kind == "singing") != (cues[cue_id].kind == "singing")
            )
            or (len(current) >= 2 and _TOPIC_SHIFT.match(cues[cue_id].text.strip()))
        ):
            groups.append(current)
            current = []
            characters = 0
        current.append(cue_id)
        characters += cue_chars
    if current:
        groups.append(current)
    return groups


def _topic_knowledge(hits: list[KnowledgeHit]) -> list[KnowledgeHit]:
    selected: list[KnowledgeHit] = []
    seen: set[str] = set()
    used = 0
    for hit in hits:
        if hit.record_id in seen:
            continue
        body_chars = min(180, len(hit.body))
        item_chars = len(hit.kind) + len(hit.title) + body_chars + 6
        if selected and used + item_chars > _TOPIC_KNOWLEDGE_CHARS:
            continue
        selected.append(hit)
        seen.add(hit.record_id)
        used += item_chars
        if len(selected) >= _TOPIC_KNOWLEDGE_HITS:
            break
    return selected


def _format_topic_knowledge(knowledge: list[KnowledgeHit]) -> str:
    return (
        "\n".join(
            f"- <{hit.kind}:{hit.title}> {hit.body[:180]}" for hit in knowledge
        )
        or "(none)"
    )


def _estimate_prompt_tokens(text: str) -> int:
    cjk = sum(
        "\u3040" <= character <= "\u30ff"
        or "\u3400" <= character <= "\u9fff"
        for character in text
    )
    return cjk + math.ceil((len(text) - cjk) / 4)


def _validate_prompt_budget(
    prompt: str, llm: LLMConfig, translation: TranslationConfig
) -> None:
    if not llm.local_server_enabled:
        return
    reserve = min(_LOCAL_OUTPUT_RESERVE_TOKENS, translation.max_tokens)
    budget = llm.local_server_context_size - reserve
    estimated = _estimate_prompt_tokens(prompt)
    if estimated > budget:
        raise PromptBudgetExceeded(
            f"estimated prompt tokens={estimated} exceed budget={budget} "
            f"(context={llm.local_server_context_size}, output_reserve={reserve})"
        )


def _prompt_within_budget(
    prompt: str, llm: LLMConfig, translation: TranslationConfig
) -> bool:
    try:
        _validate_prompt_budget(prompt, llm, translation)
    except PromptBudgetExceeded:
        return False
    return True


def _records_to_source_cues(
    source: list[Cue], tracks: list[SpeakerTrack], records: list[SegmentRecord]
) -> list[Cue]:
    track_map = {track.key: track for track in tracks}
    result: list[Cue] = []
    for record in records:
        track = track_map[record.track]
        units = track.units[record.start_id : record.end_id]
        source_values = [
            source[index] for unit in units for index in unit.source_indices
        ]
        kinds = {cue.kind for cue in source_values}
        result.append(
            Cue(
                min(cue.start for cue in source_values),
                max(cue.end for cue in source_values),
                join_source_fragments(
                    (cue.text, cue.language) for cue in source_values
                ),
                track.speaker,
                next(iter(kinds)) if len(kinds) == 1 else "speech",
                preferred_translation=(
                    source_values[0].preferred_translation
                    if len(source_values) == 1
                    else None
                ),
                source_units=source_timed_units(source_values),
                language=combine_source_languages(
                    language_for_text(cue.text, cue.language) for cue in source_values
                ),
            )
        )
    result.sort(key=lambda cue: (cue.start, cue.end, cue.speaker or ""))
    return result


def _validate_final_segments(
    records: list[SegmentRecord], tracks: list[SpeakerTrack], maximum_units: float
) -> None:
    by_track: dict[str, list[SegmentRecord]] = {}
    for record in records:
        by_track.setdefault(record.track, []).append(record)
    for track in tracks:
        expected = 0
        for record in sorted(
            by_track.get(track.key, []), key=lambda value: value.start_id
        ):
            if record.start_id != expected or record.end_id <= record.start_id:
                raise RuntimeError(
                    f"segmentation cache track {track.key} is not contiguous"
                )
            source = join_source_fragments(
                (unit.text, unit.language)
                for unit in track.units[record.start_id : record.end_id]
            )
            if (
                len(track.units[record.start_id : record.end_id]) > 1
                and text_display_width(source) > maximum_units
            ):
                raise RuntimeError(
                    f"segmentation cache track {track.key} contains an overwide merged cue"
                )
            expected = record.end_id
        if expected != len(track.units):
            raise RuntimeError(f"segmentation cache track {track.key} is incomplete")


def _fallback(
    cue: Cue,
    replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str],
) -> str:
    return machine_translate_with_protected_terms(
        cue.text,
        replacements,
        local_translate,
        force=True,
        source_language=cue.language,
    )


def _translation_audit_entry(
    cue_id: int,
    cue: Cue,
    final_text: str,
    source: str,
    *,
    llm_text: str | None = None,
    downgrade_reason: str | None = None,
    request_id: str | None = None,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "translation_result",
        "request_id": request_id,
        "cue_id": cue_id,
        "start": cue.start,
        "end": cue.end,
        "speaker": cue.speaker,
        "kind": cue.kind,
        "source_text": cue.text,
        "llm_text": llm_text,
        "final_text": final_text,
        "translation_source": source,
        "downgrade_reason": downgrade_reason,
    }


def _translation_review_audit_entry(
    cue_id: int,
    source: Cue,
    draft_text: str,
    final_text: str,
    maximum_units: float,
    method: str,
    *,
    request_id: str | None = None,
) -> dict[str, object]:
    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "event": "translation_review_result",
        "request_id": request_id,
        "cue_id": cue_id,
        "start": source.start,
        "end": source.end,
        "speaker": source.speaker,
        "kind": source.kind,
        "source_text": source.text,
        "draft_text": draft_text,
        "final_text": final_text,
        "changed": final_text != draft_text,
        "final_width": text_display_width(final_text),
        "soft_width_limit": maximum_units,
        "overwide": text_display_width(final_text) > maximum_units,
        "review_source": method,
    }


def _write_translation_audit(
    path: Path | None,
    entries: list[dict[str, object]],
    *,
    lock: threading.Lock | None = None,
) -> None:
    if path is None or not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)

    def append() -> None:
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False, default=repr) + "\n")

    if lock is None:
        append()
    else:
        with lock:
            append()


def _integer(value: object, field: str, position: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"cue {position} {field} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value)
    raise TypeError(f"cue {position} {field} is not an integer")


def _translation_dialogue_context(
    cues: list[Cue],
    selected: list[Cue],
    start: float,
    end: float,
    config: TranslationConfig,
) -> list[Cue]:
    selected_ids = {id(cue) for cue in selected}
    candidates = [
        cue
        for cue in cues
        if cue.end >= start - config.context_before_seconds
        and cue.start <= end + config.context_after_seconds
        and id(cue) not in selected_ids
    ]
    center = (start + end) / 2
    candidates.sort(key=lambda cue: abs((cue.start + cue.end) / 2 - center))
    kept: list[Cue] = []
    used = 0
    for cue in candidates:
        line_chars = len(cue.text) + len(cue.speaker or "unknown") + 3
        if used + line_chars > config.context_max_chars:
            continue
        kept.append(cue)
        used += line_chars
    kept.sort(key=lambda cue: (cue.start, cue.end, cue.speaker or ""))
    return kept


def _signature(kind: str, payload: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            {"version": _CACHE_VERSION, "kind": kind, **payload},
            ensure_ascii=False,
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _load_records(
    path: Path | None, signature: str, _kind: type[object]
) -> list[dict[str, object]] | None:
    if path is None or not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("version") != _CACHE_VERSION
            or payload.get("signature") != signature
        ):
            return None
        records = payload.get("records")
        return records if isinstance(records, list) else None
    except (OSError, ValueError, TypeError):
        return None


def _write_records(path: Path | None, signature: str, records: list[object]) -> None:
    if path is None:
        return
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "records": [asdict(record) for record in records],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
