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

from .cache import CacheStore, config_snapshot, restore_config
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
from .prompt_templates import prompt_system, render_user_prompt
from .reference_context import compact_translation_reference_context
from .source_language import (
    combine_source_languages,
    join_source_fragments,
    language_for_text,
)
from .subtitles import Cue, cue_from_mapping, text_display_width
from .telemetry import stage_metrics
from .translation_support import (
    escape_prompt_text,
    machine_translate_with_protected_terms,
    reference_replacements,
    reference_text,
    source_timed_units,
    window_ranges,
)

logger = logging.getLogger(__name__)
_TRANSLATE_PROMPT = "segment-translate-cues.md"
_CONTENT_ATTEMPTS = 2
_TOPIC_MAX_CUES = 16
_TOPIC_MAX_SECONDS = 90.0
_TOPIC_GAP_SECONDS = 5.0
_TOPIC_KNOWLEDGE_HITS = 6
_TOPIC_KNOWLEDGE_CHARS = 1800
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
    maximum_units: float,
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
            MAXIMUM_UNITS=maximum_units,
            SOURCE_MAXIMUM_UNITS=maximum_units * 1.25,
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
    reserve = translation.max_tokens
    validate_prompt_budget(
        prompt_system(_TRANSLATE_PROMPT) + "\n" + prompt,
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


def run_joint_translation(
    *, tracks: list[SpeakerTrack], source_cues: list[Cue],
    segmentation: SegmentationConfig, llm: LLMConfig,
    translation: TranslationConfig, request: Callable,
    parse_content: Callable, finish_reason: Callable, retry_delay: Callable,
    is_nontransient: Callable, log_invalid_response: Callable,
    local_translate: Callable[[str], str], translation_context: dict[str, object],
    honorific_rules: str, maximum_units: float, cache_path: Path | None,
    retrieve_knowledge: Callable | None = None,
    retrieve_chat: Callable | None = None, audit_path: Path | None = None,
) -> tuple[list[Cue], list[Cue]]:
    stage = None
    if cache_path is not None:
        stage = CacheStore(cache_path).stage("translation", lambda: {
            "tracks": [asdict(track) for track in tracks],
            "source": [asdict(cue) for cue in source_cues],
            "segmentation": config_snapshot(segmentation),
            "translation": config_snapshot(translation), "llm": config_snapshot(llm),
            "context": translation_context, "honorific_rules": honorific_rules,
            "maximum_units": maximum_units,
        })
        plan = stage.plan
        source_cues = [cue_from_mapping(value) for value in plan["source"]]
        tracks = [SpeakerTrack(value["key"], value["speaker"], tuple(
            LocalUnit(**unit) for unit in value["units"])) for value in plan["tracks"]]
        segmentation = restore_config(segmentation, plan["segmentation"])
        translation = restore_config(translation, plan["translation"])
        llm = restore_config(llm, plan["llm"])
        translation_context, honorific_rules = plan["context"], plan["honorific_rules"]
        maximum_units = plan["maximum_units"]
        cached = stage.get("__result__")
        if cached is not None:
            _write_translation_audit(audit_path, [
                {**entry, "cache_hit": True} for entry in cached["audit"]])
            return ([cue_from_mapping(v) for v in cached["source"]],
                    [cue_from_mapping(v) for v in cached["translated"]])

    # Flatten in track order so adjacent IDs can only refer to adjacent local units.
    units = [(track, unit) for track in tracks for unit in track.units]
    cues = [_records_to_source_cues(source_cues, [track], [
        SegmentRecord(track.key, unit.local_id, unit.local_id + 1)])[0]
        for track, unit in units]
    groups = []
    offset = 0
    for track in tracks:
        for start, end in window_ranges(track.units, segmentation):
            # Keep pretranslated lyrics out of the model and never merge across them.
            pending = []
            for index in range(offset + start, offset + end):
                if cues[index].preferred_translation:
                    if pending:
                        groups.extend(_topic_request_groups(cues, pending, translation))
                        pending = []
                else:
                    pending.append(index)
            if pending:
                groups.extend(_topic_request_groups(cues, pending, translation))
        offset += len(track.units)

    def key(ids):
        return "joint:" + ",".join(map(str, ids))

    # Complete retrieval before starting model generation, persisting each snapshot.
    evidence_by_group = {}
    for ids in groups:
        if stage is not None and stage.get(key(ids)) is not None:
            continue
        def prepare():
            return asdict(_prepare_topic_evidence(
                [ids], cues, retrieve_knowledge, retrieve_chat,
                metric_name="rag.joint_translation")[tuple(ids)])
        snapshot = stage.remember("evidence:" + key(ids), prepare) if stage else prepare()
        from .fan_knowledge import KnowledgeScore
        def hits(values):
            return [KnowledgeHit(**{**v, "score": KnowledgeScore(**v["score"])}) for v in values]
        evidence_by_group[tuple(ids)] = _TopicEvidence(
            hits(snapshot["term_references"]), hits(snapshot["knowledge"]), snapshot["chat"])

    replacements = reference_replacements(translation_context)

    def ask(ids, evidence):
        previous_error = None
        content_attempts = transient_attempts = 0
        while content_attempts < min(_CONTENT_ATTEMPTS, llm.max_retries):
            selected = [cues[i] for i in ids]
            dialogue = _translation_dialogue_context(
                cues, selected, min(c.start for c in selected),
                max(c.end for c in selected), translation)
            prompt = _fit_translation_prompt(
                ids=ids, cues=cues, selected=selected, dialogue=dialogue,
                evidence=evidence, context=translation_context, llm=llm,
                translation=translation, honorific_rules=honorific_rules,
                previous_error=previous_error, maximum_units=maximum_units)
            body = structured_request_body(
                model=llm.model, prompt_name=_TRANSLATE_PROMPT, prompt=prompt,
                max_tokens=translation.max_tokens, temperature=0.1, thinking=llm.thinking)
            response = content = None
            try:
                response = request(body)
                content = structured_response_content(response, finish_reason=finish_reason)
                values = _validate_joint_response(parse_content(content), len(ids), maximum_units)
                return [{"start": ids[v["start_id"]], "end": ids[v["end_id"]] + 1,
                         "outcome": asdict(TranslationOutcome(v["text"], "llm", v["text"],
                             request_id=response.get("_audit_request_id")))} for v in values]
            except Exception as exc:
                log_invalid_response("joint segmentation and translation", exc, content, body, response)
                if isinstance(exc, PromptBudgetExceeded) or is_nontransient(exc):
                    raise
                delay = retry_delay(exc, transient_attempts + 1)
                if delay is not None:
                    transient_attempts += 1
                    if transient_attempts >= llm.max_retries:
                        raise
                    time.sleep(delay)
                    continue
                previous_error = exc
                content_attempts += 1
        raise previous_error or RuntimeError("joint translation exhausted retries")

    def process(ids, evidence):
        try:
            return process_uncached(ids, evidence)
        except BaseException as exc:
            if stage:
                stage.failed(key(ids), exc)
            raise

    def process_uncached(ids, evidence):
        cached = stage.get(key(ids)) if stage else None
        if cached is not None:
            return cached
        split = stage.get("split:" + key(ids)) if stage else None
        if split is None:
            try:
                result = ask(ids, evidence)
            except Exception as exc:
                if is_nontransient(exc) or retry_delay(exc, 1) is not None:
                    raise
                if len(ids) == 1:
                    if isinstance(exc, PromptBudgetExceeded):
                        raise
                    text = _fallback(cues[ids[0]], replacements, local_translate)
                    _validate_joint_response([{"start_id": 0, "end_id": 0, "text": text}], 1, maximum_units)
                    logger.warning("joint translation local MT fallback unit=%d reason=%s", ids[0], exc)
                    result = [{"start": ids[0], "end": ids[0] + 1, "outcome": asdict(
                        TranslationOutcome(text, "local_mt", downgrade_reason=str(exc)))}]
                else:
                    split = len(ids) // 2
                    if stage:
                        stage.put("split:" + key(ids), split, kind="plan")
                    logger.warning("splitting joint translation window units=%d reason=%s", len(ids), exc)
        if split is not None:
            result = process(ids[:split], evidence) + process(ids[split:], evidence)
        if stage:
            reasons = sorted({v["outcome"]["downgrade_reason"] for v in result
                              if v["outcome"]["downgrade_reason"]})
            stage.put(key(ids), result, source="translation", reason="; ".join(reasons) or None,
                      kind="aggregate" if split is not None else "result")
        return result

    results = [{"start": i, "end": i + 1, "outcome": asdict(
        TranslationOutcome(c.preferred_translation.strip(), "preferred_translation"))}
        for i, c in enumerate(cues) if c.preferred_translation]
    if groups:
        logger.info("joint segmentation and translation: %d local units in %d windows, concurrency=%d",
                    len(cues), len(groups), llm.max_concurrency)
        with stage_metrics("llm.joint_cue_translation"), ThreadPoolExecutor(
            max_workers=min(llm.max_concurrency, len(groups)), thread_name_prefix="joint-translate"
        ) as executor:
            futures = [executor.submit(process, ids, evidence_by_group.get(tuple(ids))) for ids in groups]
            for future in as_completed(futures):
                results.extend(future.result())
    results.sort(key=lambda v: v["start"])
    expected = 0
    pairs = []
    for value in results:
        start, end = value["start"], value["end"]
        if start != expected or not start < end <= len(units):
            raise RuntimeError("joint translation result has invalid unit coverage")
        track, first = units[start]
        last_track, last = units[end - 1]
        if track.key != last_track.key:
            raise RuntimeError("joint translation crossed speaker tracks")
        cue = _records_to_source_cues(source_cues, [track], [
            SegmentRecord(track.key, first.local_id, last.local_id + 1)])[0]
        outcome = TranslationOutcome(**value["outcome"])
        pairs.append((cue, replace(cue, text=outcome.text, source_text=cue.text), outcome))
        expected = end
    if expected != len(units):
        raise RuntimeError("joint translation did not cover all local units")
    pairs.sort(key=lambda p: (p[0].start, p[0].end, p[0].speaker or ""))
    sources = [p[0] for p in pairs]
    translated = [p[1] for p in pairs]
    audit = [_translation_audit_entry(i, cue, out.text, out.source, llm_text=out.llm_text,
             downgrade_reason=out.downgrade_reason, request_id=out.request_id)
             for i, (cue, _, out) in enumerate(pairs)]
    _write_translation_audit(audit_path, audit)
    if stage:
        stage.finish({"source": [asdict(c) for c in sources],
                      "translated": [asdict(c) for c in translated], "audit": audit})
    return sources, translated


def _validate_joint_response(values: list[object], count: int, maximum_units: float) -> list[dict]:
    expected = 0
    result = []
    for value in values:
        if not isinstance(value, dict) or set(value) != {"start_id", "end_id", "text"}:
            raise RuntimeError("joint cue must contain start_id, end_id and text")
        start, end, text = value["start_id"], value["end_id"], value["text"]
        if type(start) is not int or type(end) is not int or start != expected or not start <= end < count:
            raise RuntimeError(f"invalid unit range; expected start_id={expected}, end_id<{count}")
        if not isinstance(text, str) or not text.strip():
            raise RuntimeError("joint cue translation must be non-empty")
        text = text.strip()
        if text_display_width(text) > maximum_units * 2:
            raise RuntimeError(f"translation exceeds two-line capacity ({maximum_units * 2}): {text}")
        result.append({"start_id": start, "end_id": end, "text": text})
        expected = end + 1
    if expected != count:
        raise RuntimeError(f"joint response covers {expected}/{count} local units")
    return result
