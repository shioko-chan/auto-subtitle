from __future__ import annotations

import inspect
import json
import re
from collections.abc import Callable

from .config import SegmentationConfig
from .local_segmentation import LocalUnit
from .reference_context import compact_reference_context
from .source_language import (
    language_for_text,
    source_separator,
)
from .subtitles import Cue, TimedTextUnit, timed_text_units

_KANA_FRAGMENT_RE = re.compile(r"[\u3040-\u30ff]+")
_PUNCTUATION_UNIT_RE = re.compile(r"^[\s、。！？…・,.!?;:「」『』（）()【】]+$")
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


class LocalFallbackError(RuntimeError):
    pass


def escape_prompt_text(text: str) -> str:
    return text.replace("<", "＜").replace(">", "＞")


def machine_translate_with_protected_terms(
    text: str,
    replacements: tuple[tuple[str, str], ...],
    local_translate: Callable[[str], str] | None,
    *,
    force: bool,
    source_language: str | None = None,
) -> str:
    if not replacements:
        return _machine_translate(text, local_translate, source_language)

    targets = dict(replacements)
    pattern = re.compile("|".join(re.escape(source) for source, _ in replacements))
    pieces: list[str] = []
    offset = 0
    for match in pattern.finditer(text):
        unprotected = text[offset : match.start()]
        pieces.append(
            _machine_translate(unprotected, local_translate, source_language)
            if force or _KANA_FRAGMENT_RE.search(unprotected)
            else unprotected
        )
        pieces.append(targets[match.group(0)])
        offset = match.end()
    remainder = text[offset:]
    pieces.append(
        _machine_translate(remainder, local_translate, source_language)
        if force or _KANA_FRAGMENT_RE.search(remainder)
        else remainder
    )
    return "".join(pieces)


def reference_replacements(
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


def reference_text(context: dict[str, object]) -> str:
    if not context:
        return "(none)"
    compact = compact_reference_context(context)
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def source_timed_units(source: list[Cue]) -> tuple[TimedTextUnit, ...]:
    values: list[TimedTextUnit] = []
    for cue in source:
        aligned = timed_text_units(cue) or (
            TimedTextUnit(cue.text, cue.start, cue.end),
        )
        for unit in aligned:
            if _PUNCTUATION_UNIT_RE.fullmatch(unit.text) and values:
                previous = values[-1]
                values[-1] = TimedTextUnit(
                    previous.text + unit.text,
                    previous.start,
                    max(previous.end, unit.end),
                )
            else:
                text = unit.text
                if values:
                    text = (
                        source_separator(
                            values[-1].text,
                            text,
                            language_for_text(values[-1].text),
                            language_for_text(text, cue.language),
                        )
                        + text
                    )
                values.append(TimedTextUnit(text, unit.start, unit.end))
    return tuple(values)


def window_ranges(
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
            if (
                end > start
                and units[end].start - units[end - 1].end
                >= config.speaker_episode_gap_seconds
            ):
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
            end = max(
                safe_candidates or candidates,
                key=lambda value: (
                    units[value - 1].boundary_score_after
                    if units[value - 1].boundary_score_after is not None
                    else -10_000,
                    value,
                ),
            )
        ranges.append((start, max(start + 1, end)))
        start = max(start + 1, end)
    return ranges


def _machine_translate(
    text: str,
    local_translate: Callable[[str], str] | None,
    source_language: str | None,
) -> str:
    if not text.strip():
        return text
    if local_translate is None:
        raise LocalFallbackError("local machine translator is not configured")
    try:
        translated = _invoke_local_translate(
            local_translate, text, source_language
        ).strip()
    except Exception as exc:
        raise LocalFallbackError(
            f"local machine translation failed for {text!r}: {exc}"
        ) from exc
    if not translated:
        raise LocalFallbackError(
            f"local machine translation returned empty text for {text!r}"
        )
    return translated


def _invoke_local_translate(
    local_translate: Callable[..., str], text: str, source_language: str | None
) -> str:
    try:
        parameters = inspect.signature(local_translate).parameters.values()
    except (TypeError, ValueError):
        parameters = ()
    accepts_language = any(
        parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}
        or parameter.name == "source_language"
        for parameter in parameters
    )
    if accepts_language:
        return local_translate(text, source_language=source_language)
    return local_translate(text)


def _starts_dependent_particle(text: str) -> bool:
    return text.startswith(_DEPENDENT_PARTICLE_PREFIXES)
