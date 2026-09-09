from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .cache import CacheStore, StageCache, config_snapshot, restore_config
from .config import LLMConfig, SegmentationConfig, TranslationConfig
from .fan_knowledge import KnowledgeHit
from .llm_response import (
    structured_request_body,
    structured_response_content,
)
from .local_segmentation import LocalUnit, SpeakerTrack
from .prompt_budget import (
    PromptBudgetExceeded,
    estimate_prompt_tokens,
    validate_prompt_budget,
)
from .prompt_templates import render_user_prompt
from .reference_context import compact_translation_reference_context
from .repetition import RepetitionLoopError
from .source_language import (
    combine_source_languages,
    join_source_fragments,
    language_for_text,
)
from .subtitles import Cue, cue_from_mapping, text_display_width
from .telemetry import stage_metrics
from .translation_support import (
    best_split,
    dialogue_context,
    escape_prompt_text,
    machine_translate_with_protected_terms,
    reference_replacements,
    reference_text,
    source_timed_units,
    window_ranges,
    window_source_language,
)

logger = logging.getLogger(__name__)
_SEGMENT_PROMPT = "segment-source-cues.md"
_TRANSLATE_PROMPT = "translate-fixed-cues.md"
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


@dataclass(frozen=True)
class _TopicEvidence:
    term_references: list[KnowledgeHit]
    knowledge: list[KnowledgeHit]
    chat: str


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
    stage = None
    if cache_path is not None:
        stage = CacheStore(cache_path).stage("segmentation", lambda: {
            "tracks": [asdict(track) for track in tracks],
            "source": [asdict(cue) for cue in source_cues],
            "segmentation": config_snapshot(segmentation),
            "llm": config_snapshot(llm), "maximum_units": source_maximum_units,
        })
        plan = stage.plan
        source_cues = [cue_from_mapping(value) for value in plan["source"]]
        tracks = [SpeakerTrack(value["key"], value["speaker"], tuple(
            LocalUnit(**unit) for unit in value["units"])) for value in plan["tracks"]]
        segmentation = restore_config(segmentation, plan["segmentation"])
        llm = restore_config(llm, plan["llm"])
        source_maximum_units = plan["maximum_units"]
        cached = stage.get("__result__")
        if cached is not None:
            return [cue_from_mapping(value) for value in cached]

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
            cache=stage,
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
    _validate_final_segments(records, tracks)
    result = _records_to_source_cues(source_cues, tracks, records)
    if stage is not None:
        stage.finish(result)
    return result


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
    stage = None
    if cache_path is not None:
        stage = CacheStore(cache_path).stage("translation", lambda: {
            "source": [asdict(cue) for cue in source_cues],
            "translation": config_snapshot(translation), "llm": config_snapshot(llm),
            "context": translation_context, "honorific_rules": honorific_rules,
        })
        plan = stage.plan
        source_cues = [cue_from_mapping(value) for value in plan["source"]]
        translation = restore_config(translation, plan["translation"])
        llm = restore_config(llm, plan["llm"])
        translation_context, honorific_rules = plan["context"], plan["honorific_rules"]
        cached = stage.get("__result__")
        if cached is not None:
            outcomes = stage.get("__outcomes__") or {}
            entries = []
            for index, value in outcomes.items():
                outcome = TranslationOutcome(**value)
                entries.append({**_translation_audit_entry(
                    int(index), source_cues[int(index)], outcome.text, outcome.source,
                    llm_text=outcome.llm_text, downgrade_reason=outcome.downgrade_reason,
                    request_id=outcome.request_id), "cache_hit": True})
            _write_translation_audit(audit_path, entries)
            return [cue_from_mapping(value) for value in cached]

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

    def prepare(ids: list[int]):
        key = ",".join(map(str, ids))
        cached = stage.get("translate:" + key) if stage is not None else None
        if cached is not None:
            return None

        def prepare_evidence():
            evidence = _prepare_topic_evidence(
                [ids], source_cues, retrieve_knowledge, retrieve_chat,
                metric_name="rag.fixed_translation",
            )[tuple(ids)]
            return asdict(evidence)

        if stage is None:
            evidence = _prepare_topic_evidence(
                [ids], source_cues, retrieve_knowledge, retrieve_chat,
                metric_name="rag.fixed_translation",
            )[tuple(ids)]
        else:
            from .fan_knowledge import KnowledgeScore
            snapshot = stage.remember("evidence:" + key, prepare_evidence)
            def hits(values):
                return [KnowledgeHit(**{**value, "score": KnowledgeScore(**value["score"])}) for value in values]
            evidence = _TopicEvidence(hits(snapshot["term_references"]), hits(snapshot["knowledge"]), snapshot["chat"])
        return evidence

    evidence_by_group = {tuple(ids): prepare(ids) for ids in groups}

    def process(ids: list[int]) -> dict[int, TranslationOutcome]:
        cached = stage.get("translate:" + ",".join(map(str, ids))) if stage is not None else None
        if cached is not None:
            return {int(index): TranslationOutcome(**value) for index, value in cached.items()}
        evidence = evidence_by_group[tuple(ids)]
        return _translate_resilient(
            ids, source_cues, llm, request, parse_content, finish_reason,
            retry_delay, is_nontransient, log_invalid_response, local_translate,
            translation_context, replacements, honorific_rules, translation, evidence,
            cache=stage,
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
    if stage is not None:
        stage.put("__outcomes__", {index: asdict(value) for index, value in translations.items()}, kind="aggregate")
        stage.finish(result)
    return result


def _segment_resilient(*args, cache: StageCache | None = None, **kwargs):
    if cache is None:
        return _segment_uncached(*args, **kwargs)
    track, start, end = args[:3]
    key = f"{track.key}:{start}:{end}"
    saved = cache.get("segment:" + key)
    if saved is not None:
        return [SegmentRecord(**value) for value in saved]
    split = cache.get("split:" + key)
    try:
        if split is not None:
            result = [*_segment_resilient(track, start, split, *args[3:], cache=cache),
                      *_segment_resilient(track, split, end, *args[3:], cache=cache)]
        else:
            result = _segment_uncached(*args, cache=cache, **kwargs)
        cache.put("segment:" + key, result, kind="aggregate" if cache.get("split:" + key) is not None else "result")
        return result
    except Exception as exc:
        cache.failed("segment:" + key, exc)
        raise


def _segment_uncached(
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
    cache: StageCache | None = None,
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
        if cache is not None:
            cache.put(f"split:{track.key}:{start}:{end}", split, kind="plan")
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
                cache=cache,
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
                cache=cache,
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
            return _validate_segments(parse_content(content), track, start, end)
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


def _translate_resilient(*args, cache: StageCache | None = None, **kwargs):
    if cache is None:
        return _translate_uncached(*args, **kwargs)
    ids = args[0]
    key = ",".join(map(str, ids))
    saved = cache.get("translate:" + key)
    if saved is not None:
        return {int(index): TranslationOutcome(**value) for index, value in saved.items()}
    middle = cache.get("split:" + key)
    try:
        if middle is not None:
            tail = args[1:] if len(args) > 15 else (*args[1:], frozenset(ids))
            result = {**_translate_resilient(ids[:middle], *tail, cache=cache),
                      **_translate_resilient(ids[middle:], *tail, cache=cache)}
        else:
            result = _translate_uncached(*args, cache=cache, **kwargs)
        reasons = sorted({value.downgrade_reason for value in result.values() if value.downgrade_reason})
        cache.put("translate:" + key, {index: asdict(value) for index, value in result.items()},
                  source="translation", reason="; ".join(reasons) or None,
                  kind="aggregate" if cache.get("split:" + key) is not None else "result")
        return result
    except Exception as exc:
        cache.failed("translate:" + key, exc)
        raise


def _translate_uncached(
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
    evidence: _TopicEvidence,
    context_excluded_ids: frozenset[int] | None = None,
    cache: StageCache | None = None,
) -> dict[int, TranslationOutcome]:
    excluded_ids = context_excluded_ids or frozenset(ids)
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
            evidence,
            replacements,
            honorific_rules,
            translation,
            excluded_ids,
        )
    except Exception as exc:
        if isinstance(exc, PromptBudgetExceeded):
            if len(ids) == 1:
                raise
            middle = len(ids) // 2
            if cache is not None:
                cache.put("split:" + ",".join(map(str, ids)), middle, kind="plan")
            logger.warning(
                "splitting fixed-translation topic size=%d to satisfy token budget",
                len(ids),
            )
            return {
                **_translate_resilient(
                    ids[:middle], cues, llm, request, parse_content, finish_reason,
                    retry_delay, is_nontransient, log_invalid_response,
                    local_translate, base_context, replacements, honorific_rules,
                    translation, evidence, excluded_ids, cache=cache,
                ),
                **_translate_resilient(
                    ids[middle:], cues, llm, request, parse_content, finish_reason,
                    retry_delay, is_nontransient, log_invalid_response,
                    local_translate, base_context, replacements, honorific_rules,
                    translation, evidence, excluded_ids, cache=cache,
                ),
            }
        if is_nontransient(exc) or retry_delay(exc, 1) is not None:
            raise
        if len(ids) == 1:
            cue_id = ids[0]
            if isinstance(exc, RepetitionLoopError):
                no_evidence, no_evidence_error = _probe_repetition_translation(
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
                    _TopicEvidence([], [], ""),
                    replacements,
                    honorific_rules,
                    translation,
                    excluded_ids,
                )
                trigger = "target"
                recovered = no_evidence
                if no_evidence is not None:
                    trigger = (
                        "chat"
                        if evidence.chat and not (
                            evidence.term_references or evidence.knowledge
                        )
                        else "knowledge"
                        if (evidence.term_references or evidence.knowledge)
                        and not evidence.chat
                        else "chat_or_knowledge"
                    )
                    if evidence.term_references or evidence.knowledge:
                        knowledge_only, knowledge_error = (
                            _probe_repetition_translation(
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
                                _TopicEvidence(
                                    evidence.term_references,
                                    evidence.knowledge,
                                    "",
                                ),
                                replacements,
                                honorific_rules,
                                translation,
                                excluded_ids,
                            )
                        )
                        if isinstance(knowledge_error, RepetitionLoopError):
                            trigger = "knowledge"
                        elif knowledge_only is not None:
                            trigger = "chat" if evidence.chat else "interaction"
                            recovered = knowledge_only
                        else:
                            trigger = "evidence_interaction"
                elif not isinstance(no_evidence_error, RepetitionLoopError):
                    trigger = "undetermined"
                diagnosis = {
                    "trigger": trigger,
                    "cue_ids": ids,
                    "source_texts": [cues[value].text for value in ids],
                    "knowledge_ids": [
                        hit.record_id
                        for hit in [
                            *evidence.term_references,
                            *evidence.knowledge,
                        ]
                    ],
                    "chat_lines": len(evidence.chat.splitlines()),
                    "pattern": exc.match.pattern[:200],
                    "repeats": exc.match.repeats,
                    "minimal_reproducer": trigger == "target",
                }
                log_invalid_response(
                    "repetition minimizer",
                    exc,
                    diagnosis,
                    None,
                    None,
                )
                if recovered is not None:
                    return recovered
                logger.warning(
                    "LLM translation downgrade reason=repetition_loop "
                    "fallback=local_machine_translation cue_id=%d original_text=%r",
                    cue_id,
                    cues[cue_id].text,
                )
                return {
                    cue_id: TranslationOutcome(
                        _fallback(cues[cue_id], replacements, local_translate),
                        "local_mt",
                        downgrade_reason="repetition_loop",
                    )
                }
            raise
        middle = len(ids) // 2
        if cache is not None:
            cache.put("split:" + ",".join(map(str, ids)), middle, kind="plan")
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
                evidence,
                excluded_ids,
                cache=cache,
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
                evidence,
                excluded_ids,
                cache=cache,
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
    evidence: _TopicEvidence,
    replacements: tuple[tuple[str, str], ...],
    honorific_rules: str,
    translation: TranslationConfig,
    context_excluded_ids: frozenset[int],
    content_attempts: int = _CONTENT_ATTEMPTS,
) -> dict[int, TranslationOutcome]:
    previous_error: Exception | None = None
    transient_attempts = 0
    for content_attempt in range(content_attempts):
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
            evidence=evidence,
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
            if isinstance(exc, RepetitionLoopError):
                raise
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
            if content_attempt + 1 >= min(content_attempts, llm.max_retries):
                raise
    raise RuntimeError("fixed translation exhausted retries")


def _probe_repetition_translation(
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
    evidence: _TopicEvidence,
    replacements: tuple[tuple[str, str], ...],
    honorific_rules: str,
    translation: TranslationConfig,
    context_excluded_ids: frozenset[int],
) -> tuple[dict[int, TranslationOutcome] | None, Exception | None]:
    try:
        return (
            _request_translation(
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
                context,
                evidence,
                replacements,
                honorific_rules,
                translation,
                context_excluded_ids,
                content_attempts=1,
            ),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - probes classify every request failure
        return None, exc


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


def _fit_translation_prompt(
    *,
    ids: list[int],
    cues: list[Cue],
    selected: list[Cue],
    dialogue: list[Cue],
    evidence: _TopicEvidence,
    context: dict[str, object],
    llm: LLMConfig,
    translation: TranslationConfig,
    honorific_rules: str,
    previous_error: Exception | None,
) -> str:
    active_knowledge = list(evidence.knowledge)
    active_chat = evidence.chat.splitlines()
    active_dialogue = list(dialogue)
    original_counts = (
        len(active_knowledge),
        len(active_chat),
        len(active_dialogue),
    )

    while True:
        chat_text = "\n".join(active_chat)
        reference = compact_translation_reference_context(context)
        prompt = render_user_prompt(
            _TRANSLATE_PROMPT,
            TARGET_LANGUAGE=translation.target_language,
            HONORIFIC_TRANSLATION_RULES=honorific_rules,
            REFERENCE_TEXT=reference_text(reference),
            TERM_REFERENCE=_format_term_references(evidence.term_references),
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


def _prepare_topic_evidence(
    groups: list[list[int]],
    cues: list[Cue],
    retrieve_knowledge: Callable[[list[Cue], str], list[KnowledgeHit]] | None,
    retrieve_chat: Callable[[list[Cue]], str] | None,
    *,
    metric_name: str,
) -> dict[tuple[int, ...], _TopicEvidence]:
    prepared: dict[tuple[int, ...], _TopicEvidence] = {}
    if not groups:
        return prepared
    with stage_metrics(metric_name):
        for ids in groups:
            selected = [cues[cue_id] for cue_id in ids]
            try:
                chat = retrieve_chat(selected) if retrieve_chat else ""
            except Exception as exc:  # noqa: BLE001 - optional evidence fails open.
                logger.warning("topic chat retrieval failed ids=%s: %s", ids, exc)
                chat = ""
            try:
                hits = (
                    retrieve_knowledge(selected, chat)
                    if retrieve_knowledge is not None
                    else []
                )
                term_references = [
                    hit for hit in hits if "term_reference" in hit.retrieval_ranks
                ]
                knowledge = _topic_knowledge(
                    [
                        hit
                        for hit in hits
                        if "term_reference" not in hit.retrieval_ranks
                    ]
                )
            except Exception as exc:  # noqa: BLE001 - optional evidence fails open.
                logger.warning(
                    "topic knowledge retrieval failed ids=%s: %s", ids, exc
                )
                term_references = []
                knowledge = []
            prepared[tuple(ids)] = _TopicEvidence(
                term_references, knowledge, chat
            )
    return prepared


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


def _format_term_references(references: list[KnowledgeHit]) -> str:
    return (
        "\n".join(
            f"- <{hit.kind}:{hit.title}> {hit.body}" for hit in references
        )
        or "(none)"
    )


def _validate_prompt_budget(
    prompt: str, llm: LLMConfig, translation: TranslationConfig
) -> None:
    if not llm.local_server_enabled:
        return
    reserve = min(_LOCAL_OUTPUT_RESERVE_TOKENS, translation.max_tokens)
    validate_prompt_budget(
        prompt,
        context_size=llm.local_server_context_size,
        max_output_tokens=reserve,
        estimate_tokens=estimate_prompt_tokens,
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
    records: list[SegmentRecord], tracks: list[SpeakerTrack]
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


