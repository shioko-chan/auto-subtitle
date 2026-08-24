from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import LLMConfig, SegmentationConfig
from .joint_translation import (
    _best_split,
    _contains_kana,
    _dialogue_context,
    _escape,
    _machine_translate_with_protected_terms,
    _reference_replacements,
    _reference_text,
    _source_timed_units,
    _window_ranges,
    _window_source_language,
    normalize_residual_japanese,
)
from .local_segmentation import LocalUnit, SpeakerTrack
from .prompt_templates import prompt_system, prompt_templates_digest, render_user_prompt
from .source_language import (
    combine_source_languages,
    join_source_fragments,
    language_for_text,
)
from .subtitles import Cue, cue_from_mapping, text_display_width
from .telemetry import stage_metrics

logger = logging.getLogger(__name__)
_CACHE_VERSION = 1
_SEGMENT_PROMPT = "segment-source-cues.md"
_TRANSLATE_PROMPT = "translate-fixed-cues.md"
_CONTENT_ATTEMPTS = 2


@dataclass(frozen=True)
class SegmentRecord:
    track: str
    start_id: int
    end_id: int


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
        for start, end in _window_ranges(track.units, segmentation)
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
    segmentation: SegmentationConfig,
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
    maximum_units: float,
    honorific_rules: str,
    cache_path: Path | None,
) -> list[Cue]:
    signature = _signature(
        "translation",
        {
            "source": [asdict(cue) for cue in source_cues],
            "model": llm.model,
            "target_language": llm.target_language,
            "maximum_units": maximum_units,
            "context": translation_context,
            "prompt": prompt_templates_digest(_TRANSLATE_PROMPT),
        },
    )
    cached = _load_records(cache_path, signature, Cue)
    if cached is not None:
        return [cue_from_mapping(value) for value in cached]

    translations: dict[int, str] = {}
    pending: list[int] = []
    for cue_id, cue in enumerate(source_cues):
        if cue.preferred_translation:
            translations[cue_id] = cue.preferred_translation.strip()
        else:
            pending.append(cue_id)
    groups: list[list[int]] = []
    current: list[int] = []
    characters = 0
    for cue_id in pending:
        cue_chars = len(source_cues[cue_id].text)
        if current and (
            len(current) >= segmentation.request_batch_windows
            or characters + cue_chars > segmentation.request_batch_chars
        ):
            groups.append(current)
            current = []
            characters = 0
        current.append(cue_id)
        characters += cue_chars
    if current:
        groups.append(current)

    replacements = _reference_replacements(translation_context)

    def process(ids: list[int]) -> dict[int, str]:
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
            maximum_units,
            honorific_rules,
        )

    if groups:
        logger.info(
            "translating %d fixed source cues in %d request group(s) with concurrency=%d Chinese_limit=%.3f",
            len(pending),
            len(groups),
            llm.max_concurrency,
            maximum_units,
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
                with lock:
                    translations.update(future.result())

    if set(translations) != set(range(len(source_cues))):
        raise RuntimeError("fixed translation did not cover every source cue")
    result = [
        replace(cue, text=translations[index], source_text=cue.text)
        for index, cue in enumerate(source_cues)
    ]
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
        split = _best_split(track.units, start, end)
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
            SOURCE_LANGUAGE=_window_source_language(selected),
            SOURCE_MAXIMUM_UNITS=f"{maximum_units:.3f}",
            DIALOGUE_CONTEXT=_dialogue_context(all_units, selected, config) or "(none)",
            TARGET_TEXT="\n".join(
                f"<{i}>{_escape(unit.text)}" for i, unit in enumerate(selected)
            ),
            RETRY_SECTION=""
            if previous_error is None
            else f"\n\nPREVIOUS_RESPONSE_ERROR: {previous_error}",
        )
        body = _body(llm, _SEGMENT_PROMPT, prompt)
        content: object = None
        response: object = None
        try:
            response = request(body)
            content = response["choices"][0]["message"]["content"]
            reason = finish_reason(response)
            if reason not in (None, "stop"):
                raise RuntimeError(f"finish_reason={reason}")
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
    context: dict[str, object],
    replacements: tuple[tuple[str, str], ...],
    maximum_units: float,
    honorific_rules: str,
) -> dict[int, str]:
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
            context,
            replacements,
            maximum_units,
            honorific_rules,
        )
    except Exception as exc:
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
            return {cue_id: _fallback(cues[cue_id], replacements, local_translate)}
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
                context,
                replacements,
                maximum_units,
                honorific_rules,
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
                context,
                replacements,
                maximum_units,
                honorific_rules,
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
    replacements: tuple[tuple[str, str], ...],
    maximum_units: float,
    honorific_rules: str,
) -> dict[int, str]:
    previous_error: Exception | None = None
    transient_attempts = 0
    for content_attempt in range(_CONTENT_ATTEMPTS):
        local_to_global = dict(enumerate(ids))
        selected = [cues[value] for value in ids]
        start = min(cue.start for cue in selected)
        end = max(cue.end for cue in selected)
        dialogue = [
            cue
            for cue in cues
            if cue.end >= start - 5.0 and cue.start <= end + 5.0 and cue not in selected
        ]
        prompt = render_user_prompt(
            _TRANSLATE_PROMPT,
            TARGET_LANGUAGE=llm.target_language,
            MAXIMUM_UNITS=f"{maximum_units:.3f}",
            HONORIFIC_TRANSLATION_RULES=honorific_rules,
            REFERENCE_TEXT=_reference_text(context),
            DIALOGUE_CONTEXT="\n".join(
                f"<{cue.speaker or 'unknown'}>{_escape(cue.text)}" for cue in dialogue
            )
            or "(none)",
            SOURCE_TEXT="\n".join(
                f"<{local_id}>[{language_for_text(cue.text, cue.language)}] {_escape(cue.text)}"
                for local_id, cue in enumerate(selected)
            ),
            RETRY_SECTION=""
            if previous_error is None
            else f"\n\nPREVIOUS_RESPONSE_ERROR: {previous_error}",
        )
        body = _body(llm, _TRANSLATE_PROMPT, prompt)
        content: object = None
        response: object = None
        try:
            response = request(body)
            content = response["choices"][0]["message"]["content"]
            reason = finish_reason(response)
            if reason not in (None, "stop"):
                raise RuntimeError(f"finish_reason={reason}")
            values = parse_content(content)
            result: dict[int, str] = {}
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
                    text = _fallback(cue, replacements, local_translate)
                elif _contains_kana(text, llm.target_language):
                    logger.warning(
                        "LLM translation downgrade reason=residual_japanese fallback=protected_machine_translation cue_id=%d text=%r",
                        global_id,
                        text,
                    )
                    text = normalize_residual_japanese(
                        text, replacements, local_translate, cue.language
                    )
                if text_display_width(text) > maximum_units:
                    logger.warning(
                        "accepting overwide translated cue cue_id=%d width=%.3f limit=%.3f text=%r",
                        global_id,
                        text_display_width(text),
                        maximum_units,
                        text,
                    )
                result[global_id] = text
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
                source_units=_source_timed_units(source_values),
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
    return _machine_translate_with_protected_terms(
        cue.text,
        replacements,
        local_translate,
        force=True,
        source_language=cue.language,
    )


def _integer(value: object, field: str, position: int) -> int:
    if isinstance(value, bool):
        raise TypeError(f"cue {position} {field} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("+-").isdigit():
        return int(value)
    raise TypeError(f"cue {position} {field} is not an integer")


def _body(llm: LLMConfig, prompt_name: str, prompt: str) -> dict[str, object]:
    body: dict[str, object] = {
        "model": llm.model,
        "temperature": 0.1,
        "max_tokens": llm.max_tokens,
        "messages": [
            {"role": "system", "content": prompt_system(prompt_name)},
            {"role": "user", "content": prompt},
        ],
    }
    if llm.thinking:
        body["thinking"] = {"type": llm.thinking}
    if llm.json_mode:
        body["response_format"] = {"type": "json_object"}
    return body


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
