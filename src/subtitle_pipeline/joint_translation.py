from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import urllib.error
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from .config import LLMConfig, SegmentationConfig
from .local_segmentation import LocalUnit, SpeakerTrack
from .prompt_templates import (
    prompt_system,
    prompt_templates_digest,
    render_user_prompt,
)
from .subtitles import Cue, text_display_width
from .telemetry import stage_metrics

_CACHE_VERSION = 4
_CONTENT_ATTEMPTS = 2
_PROMPT_NAME = "joint-segment-translate.md"
_KANA_FRAGMENT_RE = re.compile(r"[\u3040-\u30ff]+")
_DEPENDENT_PARTICLE_PREFIXES = (
    "を",
    "が",
    "に",
    "へ",
    "と",
    "で",
    "から",
    "まで",
    "より",
    "の",
)
logger = logging.getLogger(__name__)
@dataclass(frozen=True)
class JointRecord:
    track: str
    start_id: int
    end_id: int
    text: str


@dataclass(frozen=True)
class JointResult:
    source_cues: list[Cue]
    translated_cues: list[Cue]


@dataclass(frozen=True)
class _RelativeRecord:
    start_id: int
    end_id: int
    text: str


class CoverageValidationError(RuntimeError):
    def __init__(
        self,
        message: str,
        patch_start: int,
        patch_end: int,
        preserved: list[JointRecord],
    ):
        self.patch_start = patch_start
        self.patch_end = patch_end
        self.preserved = preserved
        super().__init__(message)


class LocalFallbackError(RuntimeError):
    pass


def run_joint_translation(
    *,
    tracks: list[SpeakerTrack],
    source_cues: list[Cue],
    segmentation: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    translation_context: dict[str, object],
    maximum_units: float,
    validation_maximum_units: float,
    cache_path: Path | None,
    sudachi_versions: dict[str, str],
    honorific_rules: str,
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[[str, Exception, object], None],
    local_translate: Callable[[str], str],
) -> JointResult:
    ranges = [
        (track.key, start, end)
        for track in tracks
        for start, end in _window_ranges(track.units, segmentation)
    ]
    signature = _signature(
        tracks,
        segmentation,
        llm,
        translation_context,
        maximum_units,
        validation_maximum_units,
        sudachi_versions,
    )
    windows, final = _load_cache(cache_path, signature, tracks)
    reference_replacements = _reference_replacements(translation_context)
    if final is not None:
        try:
            _validate_final_records(
                final, tracks, llm.target_language, validation_maximum_units
            )
        except RuntimeError as exc:
            logger.warning("ignoring invalid final joint cache records: %s", exc)
            final = None
        else:
            return _records_to_result(source_cues, tracks, final)

    track_map = {track.key: track for track in tracks}
    for track_key, start, end in ranges:
        key = _range_key(track_key, start, end)
        cached = windows.get(key)
        if cached is None:
            continue
        base_id = track_map[track_key].units[start].local_id
        try:
            _validate_records(
                [
                    {
                        "start_id": record.start_id - base_id,
                        "end_id": record.end_id - base_id,
                        "text": record.text,
                    }
                    for record in cached
                ],
                track_map[track_key],
                start,
                end,
                validation_maximum_units,
                llm.target_language,
                validate_language=False,
                reference_replacements=reference_replacements,
                local_translate=local_translate,
            )
        except (RuntimeError, TypeError) as exc:
            logger.warning("discarding invalid cached joint window %s: %s", key, exc)
            windows.pop(key, None)
    all_units = [unit for track in tracks for unit in track.units]
    cache_lock = threading.Lock()

    def write_cache() -> None:
        with cache_lock:
            _write_cache(cache_path, signature, windows, None)

    def process(track_key: str, start: int, end: int) -> list[JointRecord]:
        track = track_map[track_key]
        return _request_resilient(
            track,
            start,
            end,
            all_units,
            segmentation,
            llm,
            request,
            translation_context,
            maximum_units,
            validation_maximum_units,
            honorific_rules,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            local_translate,
        )

    missing = [item for item in ranges if _range_key(*item) not in windows]
    if missing:
        logger.info(
            "jointly segmenting and translating %d/%d speaker-track windows "
            "with concurrency=%d",
            len(missing),
            len(ranges),
            llm.max_concurrency,
        )
        errors: list[Exception] = []
        with (
            stage_metrics("llm.joint_segment_translate"),
            ThreadPoolExecutor(
                max_workers=min(llm.max_concurrency, len(missing)),
                thread_name_prefix="joint-subtitle",
            ) as executor,
        ):
            futures: dict[Future[list[JointRecord]], tuple[str, int, int]] = {
                executor.submit(process, *item): item for item in missing
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    records = future.result()
                except Exception as exc:  # noqa: BLE001 - report sibling failures too
                    errors.append(exc)
                    logger.error("joint subtitle window %s failed: %s", item, exc)
                else:
                    windows[_range_key(*item)] = records
                    write_cache()
        if errors:
            raise errors[0]

    records = [record for item in ranges for record in windows[_range_key(*item)]]
    _validate_final_records(
        records, tracks, llm.target_language, validation_maximum_units
    )
    _write_cache(cache_path, signature, windows, records)
    return _records_to_result(source_cues, tracks, records)


def _request_resilient(
    track: SpeakerTrack,
    start: int,
    end: int,
    all_units: list[LocalUnit],
    segmentation: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    translation_context: dict[str, object],
    maximum_units: float,
    validation_maximum_units: float,
    honorific_rules: str,
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[[str, Exception, object], None],
    local_translate: Callable[[str], str],
    coverage_patch_attempted: bool = False,
) -> list[JointRecord]:
    try:
        return _request_window(
            track,
            start,
            end,
            all_units,
            segmentation,
            llm,
            request,
            translation_context,
            maximum_units,
            validation_maximum_units,
            honorific_rules,
            parse_content,
            finish_reason,
            retry_delay,
            is_nontransient,
            log_invalid_response,
            local_translate,
        )
    except LocalFallbackError:
        raise
    except CoverageValidationError as exc:
        if not coverage_patch_attempted:
            logger.warning(
                "patching invalid joint coverage track=%s original=%d-%d "
                "patch=%d-%d",
                track.key,
                start,
                end,
                exc.patch_start,
                exc.patch_end,
            )
            replacement = _request_resilient(
                track,
                exc.patch_start,
                exc.patch_end,
                all_units,
                segmentation,
                llm,
                request,
                translation_context,
                maximum_units,
                validation_maximum_units,
                honorific_rules,
                parse_content,
                finish_reason,
                retry_delay,
                is_nontransient,
                log_invalid_response,
                local_translate,
                True,
            )
            return sorted(
                [*exc.preserved, *replacement], key=lambda record: record.start_id
            )
        failure: Exception = exc
    except Exception as exc:  # noqa: BLE001 - classify transport and validation failures
        failure = exc
    if (
        is_nontransient(failure)
        or retry_delay(failure, 1) is not None
        or end - start <= 1
    ):
        raise failure
    split = _best_split(track.units, start, end)
    logger.warning(
        "shrinking failed joint window %s:%d-%d at %d",
        track.key,
        start,
        end,
        split,
    )
    left = _request_resilient(
        track, start, split, all_units, segmentation, llm, request,
        translation_context, maximum_units, validation_maximum_units,
        honorific_rules, parse_content, finish_reason, retry_delay,
        is_nontransient, log_invalid_response, local_translate,
    )
    right = _request_resilient(
        track, split, end, all_units, segmentation, llm, request,
        translation_context, maximum_units, validation_maximum_units,
        honorific_rules, parse_content, finish_reason, retry_delay,
        is_nontransient, log_invalid_response, local_translate,
    )
    return [*left, *right]


def _request_window(
    track: SpeakerTrack,
    start: int,
    end: int,
    all_units: list[LocalUnit],
    segmentation: SegmentationConfig,
    llm: LLMConfig,
    request: Callable[[dict[str, object]], dict[str, object]],
    translation_context: dict[str, object],
    maximum_units: float,
    validation_maximum_units: float,
    honorific_rules: str,
    parse_content: Callable[[object], list[object]],
    finish_reason: Callable[[object], str | None],
    retry_delay: Callable[[Exception, int], float | None],
    is_nontransient: Callable[[Exception], bool],
    log_invalid_response: Callable[[str, Exception, object], None],
    local_translate: Callable[[str], str],
) -> list[JointRecord]:
    prompt_error: Exception | None = None
    content_attempts = 0
    transient_attempts = 0
    while True:
        prompt = _prompt(
            track,
            start,
            end,
            all_units,
            segmentation,
            translation_context,
            maximum_units,
            llm.target_language,
            honorific_rules,
            prompt_error,
        )
        body: dict[str, object] = {
            "model": llm.model,
            "temperature": 0.1,
            "max_tokens": llm.max_tokens,
            "messages": [
                {"role": "system", "content": prompt_system(_PROMPT_NAME)},
                {"role": "user", "content": prompt},
            ],
        }
        if llm.thinking:
            body["thinking"] = {"type": llm.thinking}
        if llm.json_mode:
            body["response_format"] = {"type": "json_object"}
        content: object = None
        try:
            response = request(body)
            content = response["choices"][0]["message"]["content"]
            reason = finish_reason(response)
            if reason not in (None, "stop"):
                raise RuntimeError(f"finish_reason={reason}")
            records = _validate_records(
                parse_content(content),
                track,
                start,
                end,
                validation_maximum_units,
                llm.target_language,
                validate_language=False,
                reference_replacements=_reference_replacements(
                    translation_context
                ),
                local_translate=local_translate,
            )
            logger.info(
                "joint subtitle response track=%s global_range=%d-%d "
                "request_range=0-%d cues=%d",
                track.key,
                start,
                end - 1,
                end - start - 1,
                len(records),
            )
            return records
        except (CoverageValidationError, LocalFallbackError):
            raise
        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            urllib.error.URLError,
            TimeoutError,
            RuntimeError,
        ) as exc:
            log_invalid_response("joint segmentation and translation", exc, content)
            if is_nontransient(exc):
                raise
            delay = retry_delay(exc, transient_attempts + 1)
            if delay is not None:
                transient_attempts += 1
                if transient_attempts >= llm.max_retries:
                    raise
                time.sleep(delay)
                continue
            content_attempts += 1
            prompt_error = exc
            if content_attempts >= min(_CONTENT_ATTEMPTS, llm.max_retries):
                raise


def _prompt(
    track: SpeakerTrack,
    start: int,
    end: int,
    all_units: list[LocalUnit],
    config: SegmentationConfig,
    translation_context: dict[str, object],
    maximum_units: float,
    target_language: str,
    honorific_rules: str,
    previous_error: Exception | None,
) -> str:
    selected = track.units[start:end]
    target = "\n".join(
        f"<{request_id}>{_escape(unit.text)}"
        for request_id, unit in enumerate(selected)
    )
    target = f"<{track.key}>\n{target}"
    context = _dialogue_context(all_units, selected, config)
    retry = "" if previous_error is None else f"\n\nPREVIOUS_RESPONSE_ERROR: {previous_error}"
    return render_user_prompt(
        _PROMPT_NAME,
        TARGET_LANGUAGE=target_language,
        HONORIFIC_TRANSLATION_RULES=honorific_rules,
        REFERENCE_TEXT=_reference_text(translation_context),
        MAXIMUM_UNITS=f"{maximum_units:.3f}",
        DIALOGUE_CONTEXT=context or "(none)",
        TARGET_TEXT=target,
        RETRY_SECTION=retry,
    )


def _dialogue_context(
    all_units: list[LocalUnit],
    selected: tuple[LocalUnit, ...],
    config: SegmentationConfig,
) -> str:
    lower = selected[0].start - config.dialogue_context_seconds
    upper = selected[-1].end + config.dialogue_context_seconds
    selected_keys = {(unit.track, unit.local_id) for unit in selected}
    candidates = [
        unit
        for unit in all_units
        if unit.end > lower
        and unit.start < upper
        and (unit.track, unit.local_id) not in selected_keys
    ]
    center = (selected[0].start + selected[-1].end) / 2
    candidates.sort(key=lambda unit: abs((unit.start + unit.end) / 2 - center))
    kept: list[LocalUnit] = []
    used = 0
    for unit in candidates:
        line = f"<{unit.track}>{_escape(unit.text)}"
        if used + len(line) + 1 > config.dialogue_context_max_chars:
            continue
        kept.append(unit)
        used += len(line) + 1
    kept.sort(key=lambda unit: (unit.start, unit.end, unit.track))
    return "\n".join(f"<{unit.track}>{_escape(unit.text)}" for unit in kept)


def _window_ranges(
    units: tuple[LocalUnit, ...], config: SegmentationConfig
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < len(units):
        if units[start].kind in {"singing", "conditioned_speech"}:
            ranges.append((start, start + 1))
            start += 1
            continue
        end = start
        characters = 0
        limit_hit = False
        while end < len(units):
            if end > start and units[end].kind in {"singing", "conditioned_speech"}:
                break
            next_chars = characters + len(units[end].text)
            if end > start and (
                end - start >= config.model_window_units
                or next_chars > config.model_window_chars
            ):
                limit_hit = True
                break
            characters = next_chars
            end += 1
        if limit_hit and end - start > 1:
            search_start = max(start + 1, end - min(20, end - start))
            candidates = list(range(search_start, end + 1))
            safe_candidates = [
                value
                for value in candidates
                if value >= len(units)
                or not _starts_dependent_particle(units[value].text)
            ]
            split = max(
                safe_candidates or candidates,
                key=lambda value: (
                    units[value - 1].boundary_score_after
                    if units[value - 1].boundary_score_after is not None
                    else -10_000,
                    value,
                ),
            )
            end = split
        ranges.append((start, max(start + 1, end)))
        start = max(start + 1, end)
    return ranges


def _best_split(units: tuple[LocalUnit, ...], start: int, end: int) -> int:
    candidates = list(range(start + 1, end))
    safe_candidates = [
        value
        for value in candidates
        if not _starts_dependent_particle(units[value].text)
    ]
    return max(
        safe_candidates or candidates,
        key=lambda value: (
            units[value - 1].boundary_score_after
            if units[value - 1].boundary_score_after is not None
            else -10_000,
            -abs(value - (start + end) / 2),
        ),
    )


def _validate_records(
    values: list[object],
    track: SpeakerTrack,
    start: int,
    end: int,
    maximum_units: float,
    target_language: str,
    *,
    validate_language: bool,
    reference_replacements: tuple[tuple[str, str], ...] = (),
    local_translate: Callable[[str], str] | None = None,
) -> list[JointRecord]:
    if not values:
        raise RuntimeError("joint response contains no cues")
    final = end - start
    relative: list[_RelativeRecord] = []
    for position, value in enumerate(values):
        if not isinstance(value, dict):
            raise TypeError(f"joint cue {position} is not an object")
        if set(value) != {"start_id", "end_id", "text"}:
            raise RuntimeError(f"joint cue {position} has unexpected fields")
        start_id = _coerce_integer_id(value["start_id"], position, "start_id")
        end_id = _coerce_integer_id(value["end_id"], position, "end_id")
        text = value["text"]
        if not isinstance(text, str):
            raise TypeError(f"joint cue {position} text is not a string")
        relative.append(_RelativeRecord(start_id, end_id, text.strip()))

    expected = 0
    for record in relative:
        if (
            record.start_id != expected
            or record.end_id <= record.start_id
            or record.end_id > final
        ):
            raise _coverage_error(
                relative,
                track,
                start,
                end,
                maximum_units,
                target_language,
                validate_language,
                reference_replacements,
                local_translate,
                expected,
            )
        expected = record.end_id
    if expected != final:
        raise _coverage_error(
            relative,
            track,
            start,
            end,
            maximum_units,
            target_language,
            validate_language,
            reference_replacements,
            local_translate,
            expected,
        )
    return [
        _finalize_relative_record(
            record,
            track,
            start,
            maximum_units,
            target_language,
            validate_language,
            reference_replacements,
            local_translate,
        )
        for record in relative
    ]


def _coerce_integer_id(value: object, position: int, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"joint cue {position} {field} is not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
        return int(value)
    raise TypeError(f"joint cue {position} {field} is not an integer")


def _finalize_relative_record(
    record: _RelativeRecord,
    track: SpeakerTrack,
    window_start: int,
    maximum_units: float,
    target_language: str,
    validate_language: bool,
    reference_replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str] | None,
) -> JointRecord:
    source = "".join(
        unit.text
        for unit in track.units[
            window_start + record.start_id : window_start + record.end_id
        ]
    )
    global_start = track.units[window_start + record.start_id].local_id
    global_end = track.units[window_start + record.end_id - 1].local_id + 1
    text = record.text
    if not text:
        _log_translation_downgrade(
            "empty_translation",
            track.key,
            global_start,
            global_end,
            source,
            reference_replacements,
        )
        text = _machine_translate_with_protected_terms(
            source, reference_replacements, local_translate, force=True
        )
    elif _KANA_FRAGMENT_RE.search(text):
        _log_translation_downgrade(
            "residual_japanese",
            track.key,
            global_start,
            global_end,
            text,
            reference_replacements,
        )
        text = normalize_residual_japanese(
            text, reference_replacements, local_translate
        )
    width = text_display_width(text)
    if width > maximum_units:
        logger.warning(
            "accepting overwide translated cue track=%s range=%d-%d "
            "width=%.3f limit=%.3f text=%r",
            track.key,
            window_start + record.start_id,
            window_start + record.end_id,
            width,
            maximum_units,
            text,
        )
    if validate_language and _contains_kana(text, target_language):
        logger.warning(
            "local machine translation retained Japanese kana track=%s "
            "range=%d-%d text=%r",
            track.key,
            window_start + record.start_id,
            window_start + record.end_id,
            text,
        )
    return JointRecord(track.key, global_start, global_end, text)


def _log_translation_downgrade(
    reason: str,
    track: str,
    start_id: int,
    end_id: int,
    original_text: str,
    replacements: tuple[tuple[str, str], ...],
) -> None:
    protected_terms = sum(
        original_text.count(source) for source, _target in replacements
    )
    logger.warning(
        "LLM translation downgrade reason=%s fallback=local_machine_translation "
        "track=%s range=%d-%d protected_terms=%d original_text=%r",
        reason,
        track,
        start_id,
        end_id,
        protected_terms,
        original_text,
    )


def _coverage_error(
    records: list[_RelativeRecord],
    track: SpeakerTrack,
    start: int,
    end: int,
    maximum_units: float,
    target_language: str,
    validate_language: bool,
    reference_replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str] | None,
    expected: int,
) -> CoverageValidationError:
    final = end - start
    prefix: list[_RelativeRecord] = []
    cursor = 0
    for record in records:
        if (
            record.start_id != cursor
            or not 0 <= record.start_id < record.end_id <= final
        ):
            break
        prefix.append(record)
        cursor = record.end_id

    suffix: list[_RelativeRecord] = []
    cursor = final
    for record in reversed(records):
        if (
            record.end_id != cursor
            or not 0 <= record.start_id < record.end_id <= final
        ):
            break
        suffix.insert(0, record)
        cursor = record.start_id

    prefix_end = prefix[-1].end_id if prefix else 0
    while suffix and suffix[0].start_id < prefix_end:
        suffix.pop(0)

    patch_start = prefix[-1].start_id if prefix else 0
    patch_end = suffix[0].end_id if suffix else final
    if patch_end <= patch_start:
        patch_start, patch_end = 0, final
        prefix = []
        suffix = []

    preserved = [
        _finalize_relative_record(
            record,
            track,
            start,
            maximum_units,
            target_language,
            validate_language,
            reference_replacements,
            local_translate,
        )
        for record in [*prefix[:-1], *suffix[1:]]
    ]
    return CoverageValidationError(
        f"joint response coverage failed near {expected}; "
        f"patching relative range [{patch_start},{patch_end})",
        start + patch_start,
        start + patch_end,
        preserved,
    )


def _validate_final_records(
    records: list[JointRecord],
    tracks: list[SpeakerTrack],
    _target_language: str,
    _maximum_units: float,
) -> None:
    by_track: dict[str, list[JointRecord]] = {}
    for record in records:
        by_track.setdefault(record.track, []).append(record)
    for track in tracks:
        values = sorted(by_track.get(track.key, []), key=lambda item: item.start_id)
        expected = 0
        for value in values:
            if value.start_id != expected or value.end_id <= value.start_id:
                raise RuntimeError(f"cached track {track.key} is not contiguous")
            expected = value.end_id
        if expected != len(track.units):
            raise RuntimeError(f"cached track {track.key} is incomplete")


def _records_to_result(
    source: list[Cue], tracks: list[SpeakerTrack], records: list[JointRecord]
) -> JointResult:
    track_map = {track.key: track for track in tracks}
    pairs: list[tuple[Cue, Cue]] = []
    for record in records:
        units = track_map[record.track].units[record.start_id : record.end_id]
        source_indices = [index for unit in units for index in unit.source_indices]
        source_values = [source[index] for index in source_indices]
        start = min(cue.start for cue in source_values)
        end = max(cue.end for cue in source_values)
        speaker = track_map[record.track].speaker
        kinds = {cue.kind for cue in source_values}
        kind = next(iter(kinds)) if len(kinds) == 1 else "speech"
        source_cue = Cue(
            start,
            end,
            "".join(cue.text for cue in source_values),
            speaker,
            kind,
        )
        pairs.append(
            (
                source_cue,
                replace(source_cue, text=record.text, source_text=source_cue.text),
            )
        )
    pairs.sort(key=lambda pair: (pair[0].start, pair[0].end, pair[0].speaker or ""))
    return JointResult(
        [pair[0] for pair in pairs],
        [pair[1] for pair in pairs],
    )


def _signature(
    tracks: list[SpeakerTrack],
    segmentation: SegmentationConfig,
    llm: LLMConfig,
    context: dict[str, object],
    maximum_units: float,
    validation_maximum_units: float,
    sudachi_versions: dict[str, str],
) -> str:
    payload = {
        "version": _CACHE_VERSION,
        "tracks": [
            {"key": track.key, "speaker": track.speaker, "units": [asdict(unit) for unit in track.units]}
            for track in tracks
        ],
        "segmentation": asdict(segmentation),
        "llm": {
            "base_url": llm.base_url,
            "api_style": llm.api_style,
            "model": llm.model,
            "target_language": llm.target_language,
            "thinking": llm.thinking,
            "local_translation_model": llm.local_translation_model,
            "local_translation_device": llm.local_translation_device,
        },
        "context": context,
        "maximum_units": maximum_units,
        "validation_maximum_units": validation_maximum_units,
        "sudachi": sudachi_versions,
        "prompt": prompt_templates_digest(_PROMPT_NAME),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _load_cache(
    path: Path | None, signature: str, tracks: list[SpeakerTrack]
) -> tuple[dict[str, list[JointRecord]], list[JointRecord] | None]:
    if path is None or not path.is_file():
        return {}, None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("version") != _CACHE_VERSION or value.get("signature") != signature:
            logger.info("joint subtitle cache signature changed; starting fresh")
            return {}, None
        windows = {
            key: [_decode_record(item) for item in items]
            for key, items in value.get("windows", {}).items()
        }
        final_value = value.get("final")
        final = None if final_value is None else [_decode_record(item) for item in final_value]
        return windows, final
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError) as exc:
        logger.warning("ignoring unreadable joint subtitle cache %s: %s", path, exc)
        return {}, None


def _write_cache(
    path: Path | None,
    signature: str,
    windows: dict[str, list[JointRecord]],
    final: list[JointRecord] | None,
) -> None:
    if path is None:
        return
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "windows": {key: [asdict(item) for item in values] for key, values in windows.items()},
        "final": None if final is None else [asdict(item) for item in final],
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _decode_record(value: object) -> JointRecord:
    if not isinstance(value, dict):
        raise TypeError("cached joint record is not an object")
    return JointRecord(
        str(value["track"]), int(value["start_id"]), int(value["end_id"]), str(value["text"])
    )


def _range_key(track: str, start: int, end: int) -> str:
    return f"{track}:{start}:{end}"


def _reference_text(context: dict[str, object]) -> str:
    if not context:
        return "(none)"
    compact = {key: value for key, value in context.items() if key != "asr_evidence"}
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def _escape(text: str) -> str:
    return text.replace("<", "＜").replace(">", "＞")


def _contains_kana(text: str, target_language: str) -> bool:
    return target_language in {"简体中文", "繁體中文", "Chinese"} and any(
        "\u3040" <= character <= "\u30ff" for character in text
    )


def normalize_residual_japanese(
    text: str,
    replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str] | None = None,
) -> str:
    """Protect known references and machine-translate remaining Japanese text."""
    if not _KANA_FRAGMENT_RE.search(text):
        return text
    return _machine_translate_with_protected_terms(
        text, replacements, local_translate, force=False
    )


def _machine_translate_with_protected_terms(
    text: str,
    replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str] | None,
    *,
    force: bool,
) -> str:
    if not replacements:
        return _machine_translate(text, local_translate)

    targets = dict(replacements)
    pattern = re.compile("|".join(re.escape(source) for source, _ in replacements))
    pieces: list[str] = []
    offset = 0
    for match in pattern.finditer(text):
        unprotected = text[offset : match.start()]
        pieces.append(
            _machine_translate(unprotected, local_translate)
            if force or _KANA_FRAGMENT_RE.search(unprotected)
            else unprotected
        )
        pieces.append(targets[match.group(0)])
        offset = match.end()
    remainder = text[offset:]
    pieces.append(
        _machine_translate(remainder, local_translate)
        if force or _KANA_FRAGMENT_RE.search(remainder)
        else remainder
    )
    return "".join(pieces)


def _machine_translate(
    text: str, local_translate: Callable[[str], str] | None
) -> str:
    if not text.strip():
        return text
    if local_translate is None:
        raise LocalFallbackError("local machine translator is not configured")
    try:
        translated = local_translate(text).strip()
    except Exception as exc:
        raise LocalFallbackError(
            f"local machine translation failed for {text!r}: {exc}"
        ) from exc
    if not translated:
        raise LocalFallbackError(
            f"local machine translation returned empty text for {text!r}"
        )
    return translated


def _reference_replacements(
    context: dict[str, object],
) -> tuple[tuple[str, str], ...]:
    replacements: dict[str, str] = {}

    def add(source: object, target: object) -> None:
        if not isinstance(source, str) or not isinstance(target, str):
            return
        source = source.strip()
        target = target.strip()
        if source and target:
            replacements[source] = target

    terms = context.get("terms", {})
    if isinstance(terms, dict):
        for source, target in terms.items():
            add(source, target)

    characters = context.get("characters", [])
    if isinstance(characters, list):
        for character in characters:
            if not isinstance(character, dict):
                continue
            canonical = character.get("canonical")
            add(character.get("source_name"), canonical)
            aliases = character.get("aliases", [])
            if isinstance(aliases, list):
                for alias in aliases:
                    add(alias, canonical)
            short_names = character.get("short_names", [])
            if isinstance(short_names, list):
                for short_name in short_names:
                    if isinstance(short_name, dict):
                        add(short_name.get("source"), short_name.get("target"))

    return tuple(
        sorted(replacements.items(), key=lambda item: (-len(item[0]), item[0]))
    )


def _starts_dependent_particle(text: str) -> bool:
    return text.startswith(_DEPENDENT_PARTICLE_PREFIXES)
