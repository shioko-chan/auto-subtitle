from __future__ import annotations

import re
from collections.abc import Iterable

_JAPANESE_RE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_NO_SPACE_BEFORE = frozenset(",.!?;:%)]}、。！？；：％）］｝'\"")
_NO_SPACE_AFTER = frozenset("([{（［｛'\"")


def normalize_source_language(value: str | None) -> str | None:
    if value is None:
        return None
    compact = value.strip().lower().replace("_", "-")
    if not compact:
        return None
    if compact in {"ja", "jpn", "japanese", "日本語"}:
        return "Japanese"
    if compact in {"en", "eng", "english", "英語"}:
        return "English"
    if compact in {"mixed", "multilingual", "multiple"}:
        return "mixed"
    return value.strip()


def language_for_text(text: str, hint: str | None = None) -> str | None:
    has_japanese = bool(_JAPANESE_RE.search(text))
    has_latin = bool(_LATIN_RE.search(text))
    if has_japanese and has_latin:
        return "mixed"
    if has_japanese:
        return "Japanese"
    if has_latin:
        return "English"
    return normalize_source_language(hint)


def combine_source_languages(values: Iterable[str | None]) -> str | None:
    normalized = {
        value
        for item in values
        if (value := normalize_source_language(item)) is not None
    }
    if not normalized:
        return None
    if len(normalized) == 1:
        return next(iter(normalized))
    return "mixed"


def join_source_fragments(
    values: Iterable[tuple[str, str | None]],
) -> str:
    result = ""
    previous_language: str | None = None
    for raw_text, raw_language in values:
        text = raw_text.strip()
        if not text:
            continue
        language = language_for_text(text, raw_language)
        if result:
            result += source_separator(result, text, previous_language, language)
        result += text
        previous_language = combine_source_languages((previous_language, language))
    return result


def source_separator(
    left: str,
    right: str,
    left_language: str | None,
    right_language: str | None,
) -> str:
    if right[0] in _NO_SPACE_BEFORE or left[-1] in _NO_SPACE_AFTER:
        return ""
    if left[-1] in "-‐‑–—/" or right[0] in "-‐‑–—/":
        return ""
    left_latin = bool(_LATIN_RE.match(left[-1]))
    right_latin = bool(_LATIN_RE.match(right[0]))
    if left_latin and right_latin:
        return " "
    if right_latin and left[-1] in ".,!?;:":
        return " "
    return " " if left_language == right_language == "English" else ""
