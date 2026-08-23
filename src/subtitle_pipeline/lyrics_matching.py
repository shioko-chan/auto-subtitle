from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

from sudachipy import dictionary, tokenizer

from .lyrics_library import LibrarySong

_SMALL_KANA = frozenset("ぁぃぅぇぉゃゅょゎァィゥェォャュョヮ")


@dataclass(frozen=True)
class LyricAnchor:
    cue_index: int
    line_start: int
    line_end: int
    score: float


@dataclass(frozen=True)
class SongMatch:
    song: LibrarySong
    anchors: tuple[LyricAnchor, ...]
    score: float


class JapaneseNormalizer:
    def __init__(self) -> None:
        self._tokenizer = dictionary.Dictionary().create()

    def __call__(self, text: str) -> str:
        values: list[str] = []
        for word in self._tokenizer.tokenize(
            unicodedata.normalize("NFKC", text), tokenizer.Tokenizer.SplitMode.A
        ):
            if _is_nonpronounced_word(word):
                continue
            reading = word.reading_form()
            values.append(reading if reading and reading != "*" else word.surface())
        return self._clean_reading("".join(values))

    def display_units(
        self, text: str, supplied_reading: str | None = None
    ) -> list[tuple[str, str]]:
        normalized = unicodedata.normalize("NFKC", text)
        if supplied_reading:
            units = self._split_surface_reading(
                normalized, self._clean_reading(supplied_reading)
            )
            return units if "".join(value[0] for value in units) == text else []
        units: list[tuple[str, str]] = []
        prefix = ""
        for word in self._tokenizer.tokenize(
            normalized, tokenizer.Tokenizer.SplitMode.A
        ):
            surface = word.surface()
            if _is_nonpronounced_word(word):
                value = ""
            else:
                reading = word.reading_form()
                value = self._clean_reading(
                    reading if reading and reading != "*" else surface
                )
            if value:
                split = self._split_surface_reading(surface, value)
                if not split:
                    return []
                first_text, first_reading = split[0]
                split[0] = (prefix + first_text, first_reading)
                units.extend(split)
                prefix = ""
            elif units:
                previous_text, previous_reading = units[-1]
                units[-1] = (previous_text + surface, previous_reading)
            else:
                prefix += surface
        if prefix and units:
            previous_text, previous_reading = units[-1]
            units[-1] = (previous_text + prefix, previous_reading)
        if not units or "".join(value[0] for value in units) != text:
            return []
        return units

    def _split_surface_reading(
        self, surface: str, reading: str
    ) -> list[tuple[str, str]]:
        clusters = _display_clusters(surface)
        if not clusters or not reading:
            return []
        known = [
            self._clean_reading(value) if _is_kana_cluster(value) else None
            for value in clusters
        ]
        anchors: list[tuple[int, int, int]] = []
        cursor = 0
        for index, value in enumerate(known):
            if not value:
                continue
            position = reading.find(value, cursor)
            if position < 0:
                return _fallback_character_units(clusters, reading)
            anchors.append((index, position, position + len(value)))
            cursor = position + len(value)

        assigned = ["" for _value in clusters]
        for index, start, end in anchors:
            assigned[index] = reading[start:end]
        boundaries = [(-1, 0, 0), *anchors, (len(clusters), len(reading), len(reading))]
        for left, right in zip(boundaries, boundaries[1:]):
            left_index, _left_start, left_end = left
            right_index, right_start, _right_end = right
            indices = list(range(left_index + 1, right_index))
            value = reading[left_end:right_start]
            if not indices:
                if value:
                    return _fallback_character_units(clusters, reading)
                continue
            parts = _split_reading(value, len(indices))
            if parts is None:
                return _fallback_character_units(clusters, reading)
            for index, part in zip(indices, parts):
                assigned[index] = part
        if any(not value for value in assigned) or "".join(assigned) != reading:
            return _fallback_character_units(clusters, reading)
        return list(zip(clusters, assigned))

    @staticmethod
    def _clean_reading(value: str) -> str:
        value = value.casefold()
        value = "".join(
            chr(ord(char) - 0x60) if "ァ" <= char <= "ヶ" else char for char in value
        )
        return re.sub(r"[^0-9a-zぁ-ゖー]", "", value)


def _is_nonpronounced_word(word: object) -> bool:
    part_of_speech = word.part_of_speech()  # type: ignore[attr-defined]
    return bool(part_of_speech) and part_of_speech[0] in {
        "空白",
        "補助記号",
        "記号",
    }


def _display_clusters(value: str) -> list[str]:
    clusters: list[str] = []
    for char in value:
        category = unicodedata.category(char)
        if clusters and (
            char in _SMALL_KANA
            or char == "ー"
            or category.startswith("M")
            or category.startswith("P")
            or category.startswith("Z")
        ):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _is_kana_cluster(value: str) -> bool:
    return all(
        char == "ー"
        or char in _SMALL_KANA
        or "ぁ" <= char <= "ゖ"
        or "ァ" <= char <= "ヶ"
        for char in value
    )


def _reading_clusters(value: str) -> list[str]:
    clusters: list[str] = []
    for char in value:
        if clusters and (char in _SMALL_KANA or char == "ー"):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _split_reading(value: str, count: int) -> list[str] | None:
    if count == 0:
        return [] if not value else None
    clusters = _reading_clusters(value)
    if len(clusters) < count:
        return None
    base, extra = divmod(len(clusters), count)
    output: list[str] = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < extra else 0)
        output.append("".join(clusters[cursor : cursor + width]))
        cursor += width
    return output


def _fallback_character_units(
    text_clusters: list[str], reading: str
) -> list[tuple[str, str]]:
    reading_values = _reading_clusters(reading)
    if not reading_values:
        return []
    count = min(len(text_clusters), len(reading_values))
    text_parts = _partition_values(text_clusters, count)
    reading_parts = _partition_values(reading_values, count)
    return [
        ("".join(text_part), "".join(reading_part))
        for text_part, reading_part in zip(text_parts, reading_parts)
    ]


def _partition_values(values: list[str], count: int) -> list[list[str]]:
    base, extra = divmod(len(values), count)
    output: list[list[str]] = []
    cursor = 0
    for index in range(count):
        width = base + (1 if index < extra else 0)
        output.append(values[cursor : cursor + width])
        cursor += width
    return output


def match_song(
    asr_lines: list[str],
    songs: list[LibrarySong],
    *,
    anchor_threshold: float = 0.48,
    minimum_anchors: int = 3,
    minimum_score: float = 0.48,
    minimum_margin: float = 0.05,
    normalizer: JapaneseNormalizer | None = None,
) -> SongMatch | None:
    normalizer = normalizer or JapaneseNormalizer()
    normalized_asr = [normalizer(value) for value in asr_lines]
    candidates: list[SongMatch] = []
    for song in songs:
        normalized_lyrics = [
            normalizer(line.reading or line.text) for line in song.lines
        ]
        choices: list[LyricAnchor] = []
        for cue_index, hypothesis in enumerate(normalized_asr):
            if len(hypothesis) < 3:
                continue
            for line_start in range(len(normalized_lyrics)):
                combined = ""
                for line_end in range(
                    line_start + 1, min(len(normalized_lyrics), line_start + 4) + 1
                ):
                    combined += normalized_lyrics[line_end - 1]
                    score = SequenceMatcher(
                        None, hypothesis, combined, autojunk=False
                    ).ratio()
                    if score >= anchor_threshold:
                        choices.append(
                            LyricAnchor(cue_index, line_start, line_end, score)
                        )
        anchors = _semiglobal_anchor_path(choices)
        if len(anchors) >= minimum_anchors:
            score = sum(anchor.score for anchor in anchors) / len(anchors)
            coverage = min(1.0, len(anchors) / max(1, len(normalized_asr)))
            candidates.append(
                SongMatch(song, tuple(anchors), score * (0.65 + 0.35 * coverage))
            )
    if not candidates:
        return None
    candidates.sort(key=lambda value: value.score, reverse=True)
    if candidates[0].score < minimum_score:
        return None
    if (
        len(candidates) > 1
        and candidates[0].score - candidates[1].score < minimum_margin
    ):
        return None
    return candidates[0]


def _semiglobal_anchor_path(
    choices: list[LyricAnchor],
) -> list[LyricAnchor]:
    """Choose an ordered ASR path while leaving lyric prefix/suffix unpenalized."""
    ordered = sorted(
        choices,
        key=lambda value: (value.cue_index, value.line_start, value.line_end),
    )
    if not ordered:
        return []
    scores = [value.score for value in ordered]
    previous: list[int | None] = [None] * len(ordered)
    lengths = [1] * len(ordered)
    for current_index, current in enumerate(ordered):
        for prior_index in range(current_index):
            prior = ordered[prior_index]
            if (
                prior.cue_index >= current.cue_index
                or prior.line_end > current.line_start
            ):
                continue
            skipped_cues = current.cue_index - prior.cue_index - 1
            skipped_lines = current.line_start - prior.line_end
            transition = (
                scores[prior_index]
                + current.score
                - 0.18 * skipped_cues
                - 0.06 * skipped_lines
            )
            candidate_length = lengths[prior_index] + 1
            if (transition, candidate_length) > (
                scores[current_index],
                lengths[current_index],
            ):
                scores[current_index] = transition
                lengths[current_index] = candidate_length
                previous[current_index] = prior_index
    best = max(
        range(len(ordered)),
        key=lambda index: (lengths[index] >= 2, scores[index], lengths[index]),
    )
    path: list[LyricAnchor] = []
    cursor: int | None = best
    while cursor is not None:
        path.append(ordered[cursor])
        cursor = previous[cursor]
    return list(reversed(path))
