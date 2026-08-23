from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from sudachipy import dictionary, tokenizer

from .config import SegmentationConfig
from .subtitles import Cue

_ATOMIC_KINDS = frozenset({"singing", "conditioned_speech"})
_TERMINAL_POS = frozenset({"動詞", "形容詞", "形状詞", "助動詞"})
_DANGLING_PARTICLE_TYPES = frozenset({"格助詞", "接続助詞", "準体助詞"})
_STRONG_RIGHT_POS = frozenset({"助詞", "助動詞", "接尾辞"})
_SENTENCE_ENDINGS = (
    "ね",
    "よ",
    "な",
    "ぞ",
    "ぜ",
    "わ",
    "さ",
    "かな",
    "かしら",
    "んだ",
    "のだ",
    "です",
    "ます",
    "でした",
    "ました",
    "でしょう",
    "だろう",
    "じゃん",
)
_DISCOURSE_STARTERS = frozenset(
    {
        "はい",
        "うん",
        "いや",
        "でも",
        "じゃあ",
        "さて",
        "ところで",
        "えっと",
        "あの",
        "まず",
        "次に",
        "ちなみに",
        "つまり",
        "だから",
        "そして",
        "あと",
        "まあ",
    }
)
_UNKNOWN_BRIDGE_MAX_GAP_SECONDS = 0.5
_UNKNOWN_GRAMMAR_MAX_GAP_SECONDS = 0.75
_UNKNOWN_NEAREST_MAX_DISTANCE_SECONDS = 0.5


@dataclass(frozen=True)
class Morphology:
    surface: str
    start: int
    end: int
    pos: tuple[str, ...]
    conjugation_type: str
    conjugation_form: str


@dataclass(frozen=True)
class BoundaryScore:
    left_source_index: int
    right_source_index: int
    gap_ms: int
    block_seconds: float
    score: int
    factors: tuple[str, ...]
    left_morpheme: Morphology | None
    right_morpheme: Morphology | None


@dataclass(frozen=True)
class LocalUnit:
    track: str
    local_id: int
    source_indices: tuple[int, ...]
    start: float
    end: float
    text: str
    speaker: str | None
    kind: str
    boundary_score_after: int | None = None
    source_pos: tuple[str | None, ...] = ()
    preferred_translation: str | None = None


@dataclass(frozen=True)
class SpeakerTrack:
    key: str
    speaker: str | None
    units: tuple[LocalUnit, ...]


class SudachiAnalyzer:
    def __init__(self) -> None:
        self._tokenizer = dictionary.Dictionary().create()
        self._dependency_nlp = None

    @property
    def versions(self) -> dict[str, str]:
        return {
            name: _package_version(name)
            for name in ("SudachiPy", "SudachiDict-core", "ginza", "ja-ginza", "spacy")
        }

    def analyze(self, text: str) -> list[Morphology]:
        result: list[Morphology] = []
        cursor = 0
        for value in self._tokenizer.tokenize(text, tokenizer.Tokenizer.SplitMode.A):
            surface = value.surface()
            start = text.find(surface, cursor)
            if start < 0:
                raise RuntimeError("Sudachi token could not be mapped to source text")
            end = start + len(surface)
            pos = tuple(str(item) for item in value.part_of_speech())
            result.append(
                Morphology(
                    surface=surface,
                    start=start,
                    end=end,
                    pos=pos,
                    conjugation_type=pos[4] if len(pos) > 4 else "*",
                    conjugation_form=pos[5] if len(pos) > 5 else "*",
                )
            )
            cursor = end
        return result

    def dependency_boundary_strengths(
        self, left: str, middle: str, right: str
    ) -> tuple[int, int]:
        if self._dependency_nlp is None:
            import spacy

            # GiNZA 5.2's optional compound splitter has an invalid null setting
            # under current spaCy. Dependency and bunsetsu parsing do not need it.
            self._dependency_nlp = spacy.load("ja_ginza", exclude=["compound_splitter"])
        text = left + middle + right
        document = self._dependency_nlp(text)
        return (
            _dependency_boundary_strength(document, len(left)),
            _dependency_boundary_strength(document, len(left) + len(middle)),
        )


def build_speaker_tracks(
    cues: list[Cue],
    config: SegmentationConfig,
    *,
    analyzer: SudachiAnalyzer | None = None,
    audit_path: Path | None = None,
) -> tuple[list[SpeakerTrack], dict[str, str]]:
    analyzer = analyzer or SudachiAnalyzer()
    cues, speaker_reattributions = _reattribute_unknown_speakers(cues, analyzer)
    cues, final_assignments = _resolve_remaining_unknown_speakers(cues)
    speaker_reattributions.extend(final_assignments)
    track_indices: dict[str, list[int]] = {}
    for index, cue in enumerate(cues):
        if cue.kind == "speech" and cue.speaker_assignment == "discarded":
            continue
        track_indices.setdefault(_track_key(cue.speaker), []).append(index)

    audits: list[dict[str, object]] = []
    tracks: list[SpeakerTrack] = []
    for key, indices in track_indices.items():
        episodes = _track_episodes(cues, indices, config.speaker_episode_gap_seconds)
        units: list[LocalUnit] = []
        for episode in episodes:
            episode_units, episode_audit = _segment_episode(
                cues, episode, key, config, analyzer, len(units)
            )
            units.extend(episode_units)
            audits.append(episode_audit)
        speaker = cues[indices[0]].speaker if indices else None
        tracks.append(SpeakerTrack(key, speaker, tuple(units)))

    tracks.sort(key=lambda track: track.units[0].start if track.units else float("inf"))
    versions = analyzer.versions
    if audit_path is not None:
        payload = {
            "version": 1,
            "sudachi": versions,
            "config": asdict(config),
            "speaker_reattributions": speaker_reattributions,
            "speaker_assignments": [
                {
                    "source_index": index,
                    "start": cue.start,
                    "end": cue.end,
                    "text": cue.text,
                    "speaker": cue.speaker,
                    "reason": cue.speaker_assignment,
                    "fallback_speaker": cue.speaker_fallback,
                    "fallback_distance_seconds": cue.speaker_fallback_distance,
                }
                for index, cue in enumerate(cues)
                if cue.kind == "speech"
            ],
            "episodes": audits,
            "tracks": [
                {
                    "key": track.key,
                    "speaker": track.speaker,
                    "units": [asdict(unit) for unit in track.units],
                }
                for track in tracks
            ],
        }
        temporary = audit_path.with_suffix(audit_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(audit_path)
    return tracks, versions


def _reattribute_unknown_speakers(
    cues: list[Cue], analyzer: SudachiAnalyzer
) -> tuple[list[Cue], list[dict[str, object]]]:
    """Bridge short unknown runs when timing and Japanese syntax support one speaker."""
    resolved = list(cues)
    audit: list[dict[str, object]] = []
    index = 0
    while index < len(resolved):
        if resolved[index].speaker is not None or resolved[index].kind != "speech":
            index += 1
            continue
        start = index
        while (
            index + 1 < len(resolved)
            and resolved[index + 1].speaker is None
            and resolved[index + 1].kind == "speech"
        ):
            index += 1
        end = index + 1
        previous = next(
            (
                resolved[position]
                for position in range(start - 1, -1, -1)
                if resolved[position].speaker is not None
                and resolved[position].kind == "speech"
            ),
            None,
        )
        following = next(
            (
                resolved[position]
                for position in range(end, len(resolved))
                if resolved[position].speaker is not None
                and resolved[position].kind == "speech"
            ),
            None,
        )
        (
            replacement,
            reason,
            left_score,
            right_score,
            left_dependency,
            right_dependency,
        ) = _unknown_run_candidate(
            resolved,
            start,
            end,
            previous,
            following,
            analyzer,
        )
        if replacement is not None and not _has_competing_activity(
            resolved,
            resolved[start].start,
            resolved[end - 1].end,
            replacement,
            start,
            end,
        ):
            for position in range(start, end):
                resolved[position] = replace(
                    resolved[position],
                    speaker=replacement,
                    speaker_assignment=reason,
                )
            audit.append(
                {
                    "source_indices": list(range(start, end)),
                    "speaker": replacement,
                    "reason": reason,
                    "left_boundary_score": left_score,
                    "right_boundary_score": right_score,
                    "left_dependency_strength": left_dependency,
                    "right_dependency_strength": right_dependency,
                }
            )
        index = end
    return resolved, audit


def _unknown_run_candidate(
    cues: list[Cue],
    start: int,
    end: int,
    previous: Cue | None,
    following: Cue | None,
    analyzer: SudachiAnalyzer,
) -> tuple[str | None, str | None, int | None, int | None, int | None, int | None]:
    first = cues[start]
    last = cues[end - 1]
    left_gap = (
        float("inf") if previous is None else max(0.0, first.start - previous.end)
    )
    right_gap = (
        float("inf") if following is None else max(0.0, following.start - last.end)
    )
    left_score = (
        None
        if previous is None
        else _syntactic_boundary_score(previous, first, analyzer)
    )
    right_score = (
        None
        if following is None
        else _syntactic_boundary_score(last, following, analyzer)
    )
    left_dependency: int | None = None
    right_dependency: int | None = None

    if (
        previous is not None
        and following is not None
        and previous.speaker == following.speaker
    ):
        close = max(left_gap, right_gap) <= _UNKNOWN_BRIDGE_MAX_GAP_SECONDS
        grammatical = (
            max(left_gap, right_gap) <= _UNKNOWN_GRAMMAR_MAX_GAP_SECONDS
            and min(left_score or 0, right_score or 0) < 0
        )
        if close or grammatical:
            return (
                previous.speaker,
                "same_speaker_bridge",
                left_score,
                right_score,
                None,
                None,
            )

    if (
        previous is not None
        and following is not None
        and previous.speaker != following.speaker
    ):
        middle = "".join(cue.text for cue in cues[start:end])
        left_dependency, right_dependency = analyzer.dependency_boundary_strengths(
            previous.text, middle, following.text
        )

    if previous is not None and left_gap <= _UNKNOWN_BRIDGE_MAX_GAP_SECONDS:
        competing_score = right_score if right_score is not None else 2
        if (
            left_score is not None
            and left_score < 0
            and competing_score - left_score >= 2
            and left_dependency is not None
            and right_dependency is not None
            and left_dependency > right_dependency
        ):
            return (
                previous.speaker,
                "grammar_left",
                left_score,
                right_score,
                left_dependency,
                right_dependency,
            )
    if following is not None and right_gap <= _UNKNOWN_BRIDGE_MAX_GAP_SECONDS:
        competing_score = left_score if left_score is not None else 2
        if (
            right_score is not None
            and right_score < 0
            and competing_score - right_score >= 2
            and left_dependency is not None
            and right_dependency is not None
            and right_dependency > left_dependency
        ):
            return (
                following.speaker,
                "grammar_right",
                left_score,
                right_score,
                left_dependency,
                right_dependency,
            )
    return None, None, left_score, right_score, left_dependency, right_dependency


def _resolve_remaining_unknown_speakers(
    cues: list[Cue],
) -> tuple[list[Cue], list[dict[str, object]]]:
    resolved = list(cues)
    audit: list[dict[str, object]] = []
    for index, cue in enumerate(resolved):
        if cue.speaker is not None or cue.kind != "speech":
            continue
        distance = cue.speaker_fallback_distance
        if (
            cue.speaker_fallback is not None
            and distance is not None
            and distance <= _UNKNOWN_NEAREST_MAX_DISTANCE_SECONDS
        ):
            resolved[index] = replace(
                cue,
                speaker=cue.speaker_fallback,
                speaker_assignment="nearest_fallback",
            )
            audit.append(
                {
                    "source_indices": [index],
                    "speaker": cue.speaker_fallback,
                    "reason": "nearest_fallback",
                    "distance_seconds": round(distance, 6),
                }
            )
            continue
        resolved[index] = replace(cue, speaker_assignment="discarded")
        audit.append(
            {
                "source_indices": [index],
                "speaker": None,
                "reason": "discarded",
                "distance_seconds": (
                    round(distance, 6) if distance is not None else None
                ),
                "start": cue.start,
                "end": cue.end,
                "text": cue.text,
            }
        )
    return resolved, audit


def _dependency_boundary_strength(document: object, boundary: int) -> int:
    import ginza

    if boundary <= 0 or boundary >= len(document.text):
        return 0
    for span in ginza.bunsetu_spans(document):
        if span.start_char < boundary < span.end_char:
            return 3
    strength = 0
    for token in document:
        if token.head is token:
            continue
        token_position = token.idx + len(token.text) / 2
        head_position = token.head.idx + len(token.head.text) / 2
        if (
            min(token_position, head_position)
            < boundary
            < max(token_position, head_position)
        ):
            strength = max(strength, 2 if abs(token.i - token.head.i) <= 1 else 1)
    return strength


def _syntactic_boundary_score(left: Cue, right: Cue, analyzer: SudachiAnalyzer) -> int:
    morphology = analyzer.analyze(left.text + right.text)
    left_morpheme, right_morpheme, inside = _morphemes_at_boundary(
        morphology, len(left.text)
    )
    score = 0
    if _is_terminal_predicate(left_morpheme):
        score += 2
    if _is_sentence_ending(left.text, left_morpheme):
        score += 1
    if _is_new_utterance(right.text, right_morpheme):
        score += 1
    if _is_dangling(left_morpheme):
        score -= 2
    if inside or _is_strong_connection(right_morpheme):
        score -= 3
    return score


def _has_competing_activity(
    cues: list[Cue],
    start: float,
    end: float,
    speaker: str,
    run_start: int,
    run_end: int,
) -> bool:
    return any(
        position < run_start or position >= run_end
        for position, cue in enumerate(cues)
        if cue.speaker is not None
        and cue.speaker != speaker
        and cue.end > start
        and cue.start < end
    )


def _track_key(speaker: str | None) -> str:
    return speaker or "unknown"


def _track_episodes(
    cues: list[Cue], indices: list[int], hard_gap_seconds: float
) -> list[list[int]]:
    episodes: list[list[int]] = []
    current: list[int] = []
    previous: int | None = None
    for index in indices:
        cue = cues[index]
        split = (
            previous is not None and cue.start - cues[previous].end >= hard_gap_seconds
        )
        if previous is not None and cue.speaker is None:
            split = split or any(
                other.speaker is not None
                and other.end > cues[previous].end
                and other.start < cue.start
                for other in cues
            )
        if previous is not None:
            split = (
                split
                or cues[previous].kind in _ATOMIC_KINDS
                or cue.kind in _ATOMIC_KINDS
            )
        if split and current:
            episodes.append(current)
            current = []
        current.append(index)
        previous = index
    if current:
        episodes.append(current)
    return episodes


def _segment_episode(
    cues: list[Cue],
    indices: list[int],
    track: str,
    config: SegmentationConfig,
    analyzer: SudachiAnalyzer,
    id_offset: int,
) -> tuple[list[LocalUnit], dict[str, object]]:
    if len(indices) == 1 or cues[indices[0]].kind in _ATOMIC_KINDS:
        unit = _make_unit(cues, indices, track, id_offset, None)
        return [unit], {"track": track, "source_indices": indices, "boundaries": []}

    text = "".join(cues[index].text for index in indices)
    morphology = analyzer.analyze(text)
    cue_offsets: list[int] = []
    cursor = 0
    for index in indices:
        cursor += len(cues[index].text)
        cue_offsets.append(cursor)

    cuts, boundaries = _choose_cuts_and_scores(
        cues, indices, cue_offsets, morphology, config
    )
    units: list[LocalUnit] = []
    start = 0
    for local_offset, end in enumerate([*cuts, len(indices)]):
        source_indices = indices[start:end]
        score_after = boundaries[end - 1].score if end < len(indices) else None
        units.append(
            _make_unit(
                cues,
                source_indices,
                track,
                id_offset + local_offset,
                score_after,
            )
        )
        start = end
    audit = {
        "track": track,
        "source_indices": indices,
        "morphology": [asdict(item) for item in morphology],
        "boundaries": [asdict(item) for item in boundaries],
        "cuts": cuts,
    }
    return units, audit


def _score_boundary(
    cues: list[Cue],
    indices: list[int],
    position: int,
    block_start: int,
    character_offset: int,
    morphology: list[Morphology],
) -> BoundaryScore:
    left = cues[indices[position]]
    right = cues[indices[position + 1]]
    gap_ms = max(0, round((right.start - left.end) * 1000))
    block_seconds = left.end - cues[indices[block_start]].start
    score = 0
    factors: list[str] = []

    gap_score = (
        4
        if gap_ms >= 600
        else 3
        if gap_ms >= 400
        else 2
        if gap_ms >= 250
        else 1
        if gap_ms >= 120
        else 0
    )
    if gap_score:
        score += gap_score
        factors.append(f"gap:{gap_score:+d}")
    if block_seconds >= 2:
        score += 1
        factors.append("duration>=2s:+1")
    if block_seconds >= 4:
        score += 2
        factors.append("duration>=4s:+2")

    left_morpheme, right_morpheme, inside = _morphemes_at_boundary(
        morphology, character_offset
    )
    if _is_terminal_predicate(left_morpheme):
        score += 2
        factors.append("terminal_predicate:+2")
    if _is_sentence_ending(left.text, left_morpheme):
        score += 1
        factors.append("sentence_ending:+1")
    if _is_new_utterance(right.text, right_morpheme):
        score += 1
        factors.append("new_utterance:+1")
    if _is_dangling(left_morpheme):
        score -= 2
        factors.append("dangling:-2")
    if inside or _is_strong_connection(right_morpheme):
        score -= 3
        factors.append("strong_connection:-3")

    return BoundaryScore(
        left_source_index=indices[position],
        right_source_index=indices[position + 1],
        gap_ms=gap_ms,
        block_seconds=round(block_seconds, 3),
        score=score,
        factors=tuple(factors),
        left_morpheme=left_morpheme,
        right_morpheme=right_morpheme,
    )


def _morphemes_at_boundary(
    values: Iterable[Morphology], offset: int
) -> tuple[Morphology | None, Morphology | None, bool]:
    left: Morphology | None = None
    right: Morphology | None = None
    inside = False
    for value in values:
        if value.start < offset < value.end:
            return value, value, True
        if value.end <= offset:
            left = value
        if right is None and value.start >= offset:
            right = value
    return left, right, inside


def _is_terminal_predicate(value: Morphology | None) -> bool:
    if value is None or not value.pos or value.pos[0] not in _TERMINAL_POS:
        return False
    form = value.conjugation_form
    return "終止形" in form or "命令形" in form


def _is_sentence_ending(text: str, value: Morphology | None) -> bool:
    return (
        value is not None and len(value.pos) > 1 and value.pos[1] == "終助詞"
    ) or any(text.endswith(ending) for ending in _SENTENCE_ENDINGS)


def _is_new_utterance(text: str, value: Morphology | None) -> bool:
    if any(text.startswith(item) for item in _DISCOURSE_STARTERS):
        return True
    return value is not None and value.pos and value.pos[0] in {"接続詞", "感動詞"}


def _is_dangling(value: Morphology | None) -> bool:
    if value is None or not value.pos:
        return False
    if (
        value.pos[0] == "助詞"
        and len(value.pos) > 1
        and value.pos[1] in _DANGLING_PARTICLE_TYPES
    ):
        return True
    return (
        value.pos[0] in _TERMINAL_POS
        and value.conjugation_form not in {"*", ""}
        and not (
            "終止形" in value.conjugation_form or "命令形" in value.conjugation_form
        )
    )


def _is_strong_connection(value: Morphology | None) -> bool:
    return value is not None and value.pos and value.pos[0] in _STRONG_RIGHT_POS


def _choose_cuts_and_scores(
    cues: list[Cue],
    indices: list[int],
    cue_offsets: list[int],
    morphology: list[Morphology],
    config: SegmentationConfig,
) -> tuple[list[int], list[BoundaryScore]]:
    cuts: list[int] = []
    final_scores: dict[int, BoundaryScore] = {}
    start = 0
    position = start
    while position < len(indices) - 1:
        boundary = _score_boundary(
            cues,
            indices,
            position,
            start,
            cue_offsets[position],
            morphology,
        )
        final_scores[position] = boundary
        if boundary.score >= config.boundary_score_threshold:
            cuts.append(position + 1)
            start = position + 1
            position = start
            continue
        projected_duration = (
            cues[indices[position + 1]].end - cues[indices[start]].start
        )
        if projected_duration >= config.local_unit_max_seconds:
            candidates = [
                candidate
                for candidate in range(start, position + 1)
                if cues[indices[candidate]].end - cues[indices[start]].start
                >= config.local_unit_min_fallback_seconds
            ]
            if candidates:
                scored_candidates = []
                for candidate in candidates:
                    score = _score_boundary(
                        cues,
                        indices,
                        candidate,
                        start,
                        cue_offsets[candidate],
                        morphology,
                    )
                    final_scores[candidate] = score
                    scored_candidates.append((candidate, score))
                best = max(
                    scored_candidates,
                    key=lambda item: (item[1].score, item[0]),
                )[0]
                cut = best + 1
                cuts.append(cut)
                start = cut
                position = cut
                continue
        position += 1
    for position in range(len(indices) - 1):
        final_scores.setdefault(
            position,
            _score_boundary(
                cues,
                indices,
                position,
                0,
                cue_offsets[position],
                morphology,
            ),
        )
    return cuts, [final_scores[position] for position in range(len(indices) - 1)]


def _make_unit(
    cues: list[Cue],
    indices: list[int],
    track: str,
    local_id: int,
    score_after: int | None,
) -> LocalUnit:
    first = cues[indices[0]]
    last = cues[indices[-1]]
    kinds = {cues[index].kind for index in indices}
    return LocalUnit(
        track=track,
        local_id=local_id,
        source_indices=tuple(indices),
        start=first.start,
        end=last.end,
        text="".join(cues[index].text for index in indices),
        speaker=first.speaker,
        kind=first.kind if len(kinds) == 1 else "speech",
        boundary_score_after=score_after,
        source_pos=tuple(cues[index].pos for index in indices),
        preferred_translation=(
            first.preferred_translation if len(indices) == 1 else None
        ),
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError as exc:
        raise RuntimeError(
            f"required segmentation dependency is missing: {name}"
        ) from exc
