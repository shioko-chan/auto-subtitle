from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import shutil
import socket
import subprocess
import sys
import unicodedata
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from .commands import require_command
from .config import SongIdentificationConfig
from .lyrics_library import LibrarySong, LyricLine, LyricsLibrary
from .lyrics_matching import JapaneseNormalizer, LyricAnchor, SongMatch, match_song
from .prompt_templates import prompt_templates_digest
from .source_language import language_for_text
from .subtitles import Cue, TimedTextUnit, cue_from_mapping, text_display_width

_CACHE_VERSION = 22
_PROMPT_VERSION = 8
_RELAXED_SEARCH_TITLE_KEYWORDS = (
    "歌枠",
    "弾き語り",
    "カラオケ",
    "歌ってみた",
    "セトリ",
)
_RELAXED_SEARCH_MINIMUM_SECONDS = 10.0
_STANDARD_SEARCH_MINIMUM_SECONDS = 15.0
_MAX_WEB_SEARCH_QUERIES = 2
_MAX_LYRIC_FETCHES_PER_QUERY = 3
_SONG_ALIGNMENT_SUPPORT_GAP_SECONDS = 90.0
_STABLE_METADATA_KEYS = (
    "id",
    "display_id",
    "title",
    "channel_id",
    "channel",
    "uploader_id",
    "uploader",
    "timestamp",
    "release_timestamp",
    "duration",
    "live_status",
)


@dataclass(frozen=True)
class SongSearchGroup:
    start: float
    end: float
    cue_ids: tuple[int, ...]


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    score: float
    frames: int
    first_time: float
    last_time: float


@dataclass(frozen=True)
class VerifiedLyricSpan:
    start: float
    end: float
    cue_id: int
    song_id: str
    lyric_line_ids: tuple[int, ...]
    likelihood_per_frame: float


@dataclass(frozen=True)
class SongIdentificationResult:
    corrected_cues: list[Cue]
    reports: list[dict[str, object]]
    verified_lyric_spans: tuple[VerifiedLyricSpan, ...] = ()


def identify_and_align_songs(
    video: Path,
    cues: list[Cue],
    metadata: dict[str, object],
    job_dir: Path,
    config: SongIdentificationConfig,
    *,
    source_maximum_units: float | None = None,
) -> SongIdentificationResult:
    groups = group_song_search_groups(cues, config.song_search_group_gap_seconds)
    if not config.enabled or not groups:
        return SongIdentificationResult(cues, [])

    cache_path = job_dir / "song-identification-cache.json"
    signature = _signature(
        video,
        cues,
        metadata,
        config,
        source_maximum_units=source_maximum_units,
    )
    cached = _load_cache(cache_path, signature, cues)
    if cached is not None:
        return cached

    ocr_cache_path = job_dir / "song-ocr-cache.json"
    ocr_signature = _ocr_signature(video, groups, metadata, config)
    candidate_sets = _load_ocr_cache(ocr_cache_path, ocr_signature, len(groups))
    if candidate_sets is None:
        candidate_sets = []
        ocr_succeeded = False
        if not _title_uses_music_mode(metadata):
            logging.info(
                "skipping song OCR because the video title is not an explicit "
                "music format"
            )
            candidate_sets = [[] for _ in groups]
            ocr_succeeded = True
        else:
            try:
                logging.info("starting EasyOCR song-title worker")
                ocr = _EasyOCR(config)
            except Exception as exc:
                logging.warning(
                    "song OCR worker failed to start; continuing without OCR: %s",
                    exc,
                )
                candidate_sets = [[] for _ in groups]
            else:
                try:
                    logging.info("EasyOCR song-title worker is ready")
                    logging.info(
                        "running one-frame song OCR for %d search groups", len(groups)
                    )
                    for index, group in enumerate(groups):
                        try:
                            candidates = collect_ocr_candidates(
                                video,
                                group,
                                job_dir / "song-ocr-frames" / f"{index:03d}",
                                config,
                                ocr,
                            )
                        except Exception as exc:
                            logging.warning(
                                "song OCR failed for %.3f-%.3fs: %s",
                                group.start,
                                group.end,
                                exc,
                            )
                            candidates = []
                        candidate_sets.append(candidates)
                        logging.info(
                            "song OCR group %d/%d timestamp=%.3fs candidates=%d",
                            index + 1,
                            len(groups),
                            group.start,
                            len(candidates),
                        )
                    ocr_succeeded = True
                finally:
                    ocr.close()
        if ocr_succeeded:
            _write_ocr_cache(ocr_cache_path, ocr_signature, candidate_sets)

    reports: list[dict[str, object]] = []
    verified_spans: list[VerifiedLyricSpan] = []
    lyric_replacements: dict[int, list[Cue]] = {}
    discarded_song_region_cues: set[int] = set()
    confirmed_song_names: set[str] = set()
    library = LyricsLibrary(Path(config.lyrics_library_path).resolve())
    try:
        for index, group in enumerate(groups):
            candidates = candidate_sets[index]
            singing_ids = [
                cue_id for cue_id in group.cue_ids if cues[cue_id].kind == "singing"
            ]
            route_cues = {
                "alt_cue_ids": singing_ids,
                "speech_cue_ids": _routed_speech_cue_ids(group, cues, config),
            }
            hypotheses = [cues[cue_id].text for cue_id in singing_ids]
            library_songs = library.songs()
            logging.info(
                "song search group %d/%d range=%.3f-%.3fs alt_cues=%d "
                "ocr_candidates=%d",
                index + 1,
                len(groups),
                group.start,
                group.end,
                len(singing_ids),
                len(candidates),
            )
            match = _match_candidates(hypotheses, library_songs, config)
            provenance = "local_library"
            web_search_audit: dict[str, object] | None = None
            if match is None:
                queries = _build_lyric_search_queries(
                    hypotheses,
                    candidates,
                    library_songs,
                    confirmed_song_names,
                )
                policy = _web_search_policy(
                    [cues[cue_id] for cue_id in singing_ids],
                    metadata,
                    has_trusted_ocr=any(
                        item.get("source") == "ocr" for item in queries
                    ),
                )
                if policy["eligible"]:
                    logging.info(
                        "song search group %d/%d starting web search mode=%s "
                        "alt_coverage=%.3fs queries=%d",
                        index + 1,
                        len(groups),
                        policy["mode"],
                        policy["alt_coverage_seconds"],
                        len(queries),
                    )
                    fetched, search_audit = _search_canonical_lyrics(
                        hypotheses,
                        queries,
                        config,
                    )
                    web_search_audit = {**policy, **search_audit}
                    match = _match_candidates(hypotheses, fetched, config)
                    logging.info(
                        "song search group %d/%d web search completed "
                        "fetches=%d confirmed=%s",
                        index + 1,
                        len(groups),
                        len(search_audit.get("fetches", [])),
                        match is not None,
                    )
                    provenance = "web"
                    if match is not None:
                        match = SongMatch(
                            library.store_canonical_song(
                                title=match.song.title,
                                artist=match.song.artist,
                                aliases=list(match.song.aliases),
                                source_url=match.song.source_url,
                                lines=[
                                    (line.text, line.reading)
                                    for line in match.song.lines
                                ],
                            ),
                            match.anchors,
                            match.score,
                        )
                else:
                    logging.info(
                        "song search group %d/%d skipped web search reason=%s "
                        "alt_coverage=%.3fs",
                        index + 1,
                        len(groups),
                        policy["decision_reason"],
                        policy["alt_coverage_seconds"],
                    )
                    web_search_audit = {
                        **policy,
                        "queries": [],
                        "fetches": [],
                        "worker_errors": [],
                    }
            if match is None:
                discarded_song_region_cues.update(singing_ids)
                reports.append(
                    {
                        "song": None,
                        "artist": None,
                        "confidence": "low",
                        "evidence": ["no_continuous_canonical_lyric_match"],
                        "sources": [],
                        "alignments": [],
                        "search_group": asdict(group),
                        "route_cues": route_cues,
                        "ocr_candidates": [asdict(item) for item in candidates],
                        "web_search": web_search_audit,
                    }
                )
                continue
            logging.info(
                "song search group %d/%d matched title=%r artist=%r source=%s "
                "score=%.3f",
                index + 1,
                len(groups),
                match.song.title,
                match.song.artist,
                provenance,
                match.score,
            )
            song = match.song
            confirmed_song_names.update(
                normalized
                for name in (song.title, *song.aliases)
                if (normalized := _normalize_identity_text(name))
            )
            alignment_ids, refined_match = _refine_match_with_speech_support(
                cues,
                singing_ids,
                route_cues["speech_cue_ids"],
                match,
                config,
            )
            if refined_match is not None:
                match = refined_match
            else:
                alignment_ids = singing_ids
                match = SongMatch(song, match.anchors, match.score)
            timing, pyshiro_audit = _align_match_with_pyshiro(
                job_dir, video, cues, alignment_ids, match, config
            )
            replacements, alignments = _apply_local_match(
                cues,
                alignment_ids,
                match,
                timing,
                pyshiro_audit=pyshiro_audit,
                source_maximum_units=source_maximum_units,
            )
            recovered, recovered_alignments, gap_audit = _recover_lyric_gaps(
                job_dir,
                cues,
                alignment_ids,
                match,
                config,
                video=video,
            )
            for cue_id, values in recovered.items():
                replacements.setdefault(cue_id, []).extend(values)
                replacements[cue_id].sort(key=lambda value: (value.start, value.end))
            alignments.extend(recovered_alignments)
            pyshiro_audit.extend(gap_audit)
            lyric_replacements.update(replacements)
            likelihood_by_cue = {
                int(item["cue_id"]): float(item.get("likelihood_per_frame", -1e9))
                for item in pyshiro_audit
                if isinstance(item, dict)
                and item.get("status")
                in {"aligned", "neighbor_recovered", "gap_recovered"}
                and isinstance(item.get("cue_id"), int)
            }
            for cue_id, values in replacements.items():
                for value in values:
                    line_ids = tuple(
                        line_id
                        for item in alignments
                        if item.get("asr_cue_ids") == [cue_id]
                        and float(item.get("start", value.start)) <= value.start + 1e-3
                        and float(item.get("end", value.end)) >= value.end - 1e-3
                        for line_id in item.get("lyric_line_ids", [])
                        if isinstance(line_id, int)
                    )
                    verified_spans.append(
                        VerifiedLyricSpan(
                            value.start,
                            value.end,
                            cue_id,
                            song.song_id,
                            line_ids,
                            likelihood_by_cue.get(cue_id, -1e9),
                        )
                    )
            matched_ids = {item["asr_cue_ids"][0] for item in alignments}
            for cue_id in singing_ids:
                if cue_id in matched_ids:
                    continue
                discarded_song_region_cues.add(cue_id)
            report_group = _expanded_report_group(group, alignment_ids, match, cues)
            reports.append(
                {
                    "song_id": song.song_id,
                    "song": song.title,
                    "artist": song.artist,
                    "confidence": "high" if match.score >= 0.68 else "medium",
                    "evidence": [
                        provenance,
                        "continuous_character_anchors",
                        *(
                            ["multiple_nonoverlapping_song_takes"]
                            if len({anchor.take_index for anchor in match.anchors}) > 1
                            else []
                        ),
                    ],
                    "sources": [song.source_url],
                    "score": round(match.score, 6),
                    "alignments": alignments,
                    "pyshiro": pyshiro_audit,
                    "search_group": report_group,
                    "route_cues": route_cues,
                    "ocr_candidates": [asdict(item) for item in candidates],
                    "web_search": web_search_audit,
                }
            )
    finally:
        library.close()
    if discarded_song_region_cues:
        logging.info(
            "discarded %d song-region cues without verified lyric alignment",
            len(discarded_song_region_cues),
        )
    corrected: list[Cue] = []
    for cue_id, cue in enumerate(cues):
        if cue_id in lyric_replacements:
            corrected.extend(lyric_replacements[cue_id])
            continue
        if cue_id in discarded_song_region_cues:
            continue
        corrected.append(cue)
    corrected, arbitration = arbitrate_verified_lyrics(corrected, verified_spans)
    for report in reports:
        group = report.get("search_group")
        if not isinstance(group, dict):
            report["arbitration"] = []
            continue
        start = float(group.get("start", -1e9))
        end = float(group.get("end", 1e9))
        report["arbitration"] = [
            item
            for item in arbitration
            if min(end, float(item["end"])) - max(start, float(item["start"])) > 0
        ]
    # Song discovery may add canonical lyrics to the library.
    # Sign the cache against the resulting library state, not its state at startup.
    signature = _signature(
        video,
        cues,
        metadata,
        config,
        source_maximum_units=source_maximum_units,
    )
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "reports": reports,
        "corrected_cues": [asdict(cue) for cue in corrected],
        "verified_lyric_spans": [asdict(span) for span in verified_spans],
    }
    if not any(report.get("error") or report.get("tool_errors") for report in reports):
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(cache_path)
    return SongIdentificationResult(corrected, reports, tuple(verified_spans))


def arbitrate_verified_lyrics(
    cues: list[Cue], verified_spans: list[VerifiedLyricSpan]
) -> tuple[list[Cue], list[dict[str, object]]]:
    spans = sorted(verified_spans, key=lambda value: (value.start, value.end))
    if not spans:
        return cues, []
    result: list[Cue] = []
    audit: list[dict[str, object]] = []
    for cue in cues:
        if cue.kind != "speech":
            result.append(cue)
            continue
        overlapping = [
            span
            for span in spans
            if min(cue.end, span.end) - max(cue.start, span.start) > 0
        ]
        if not overlapping:
            result.append(cue)
            continue
        units = tuple(cue.source_units)
        if not units:
            covered = _covered_duration(cue.start, cue.end, overlapping)
            ratio = covered / max(1e-6, cue.end - cue.start)
            action = "discarded_without_units" if ratio >= 0.8 else "kept_conflict"
            audit.append(
                {
                    "start": cue.start,
                    "end": cue.end,
                    "action": action,
                    "lyric_coverage": round(ratio, 6),
                }
            )
            if ratio < 0.8:
                result.append(cue)
            continue
        retained = [
            unit
            for unit in units
            if _covered_duration(unit.start, unit.end, overlapping)
            / max(1e-6, unit.end - unit.start)
            < 0.8
        ]
        if not retained:
            audit.append(
                {
                    "start": cue.start,
                    "end": cue.end,
                    "action": "discarded_covered_units",
                    "retained_units": 0,
                }
            )
            continue
        runs: list[list[TimedTextUnit]] = [[retained[0]]]
        for unit in retained[1:]:
            if unit.start - runs[-1][-1].end <= 0.25:
                runs[-1].append(unit)
            else:
                runs.append([unit])
        for run in runs:
            text = "".join(unit.text for unit in run).strip()
            if not text:
                continue
            result.append(
                replace(
                    cue,
                    start=run[0].start,
                    end=run[-1].end,
                    text=text,
                    source_units=tuple(run),
                )
            )
        audit.append(
            {
                "start": cue.start,
                "end": cue.end,
                "action": "trimmed_at_aligner_units",
                "retained_units": len(retained),
                "removed_units": len(units) - len(retained),
            }
        )
    result.sort(key=lambda cue: (cue.start, cue.end, cue.kind, cue.speaker or ""))
    return result, audit


def _covered_duration(
    start: float, end: float, spans: list[VerifiedLyricSpan]
) -> float:
    intersections = sorted(
        (max(start, span.start), min(end, span.end))
        for span in spans
        if min(end, span.end) - max(start, span.start) > 0
    )
    merged: list[list[float]] = []
    for left, right in intersections:
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return sum(right - left for left, right in merged)


def translate_aligned_song_lyrics(
    result: SongIdentificationResult,
    config: SongIdentificationConfig,
    translate_lyrics: Callable[..., object],
    translation_context: dict[str, object] | None = None,
    lyrics_translation_model: str | None = None,
) -> SongIdentificationResult:
    if not result.reports:
        return result
    corrected = list(result.corrected_cues)
    library = LyricsLibrary(Path(config.lyrics_library_path).resolve())
    try:
        for report in result.reports:
            song_id = report.get("song_id")
            if not isinstance(song_id, str):
                continue
            song = library.get(song_id)
            if song is None:
                logging.warning(
                    "identified song is missing from lyrics library: %s", song_id
                )
                continue
            song = _ensure_song_translations(
                library,
                song,
                translate_lyrics,
                translation_context or {},
                lyrics_translation_model,
            )
            translations: dict[str, str] = {}
            alignments = report.get("alignments")
            if isinstance(alignments, list):
                for alignment in alignments:
                    if not isinstance(alignment, dict):
                        continue
                    source = alignment.get("corrected_text")
                    line_ids = alignment.get("lyric_line_ids")
                    if not isinstance(source, str) or not isinstance(line_ids, list):
                        continue
                    lines = [
                        song.lines[line_id]
                        for line_id in line_ids
                        if isinstance(line_id, int) and 0 <= line_id < len(song.lines)
                    ]
                    if len(lines) != len(line_ids):
                        continue
                    translated = "".join(line.translation or "" for line in lines)
                    fragment_count = alignment.get("fragment_count")
                    fragment_index = alignment.get("fragment_index")
                    if (
                        translated
                        and isinstance(fragment_count, int)
                        and fragment_count > 1
                        and isinstance(fragment_index, int)
                    ):
                        siblings = [
                            item
                            for item in alignments
                            if isinstance(item, dict)
                            and item.get("lyric_line_ids") == line_ids
                            and item.get("fragment_count") == fragment_count
                            and item.get("take_index") == alignment.get("take_index")
                            and item.get("asr_cue_ids") == alignment.get("asr_cue_ids")
                        ]
                        siblings.sort(
                            key=lambda item: int(item.get("fragment_index", -1))
                        )
                        if len(siblings) == fragment_count:
                            pieces = _split_text_by_weights(
                                translated,
                                [
                                    float(item.get("fragment_weight", 1.0))
                                    for item in siblings
                                ],
                            )
                            if 0 <= fragment_index < len(pieces):
                                translated = pieces[fragment_index]
                    if translated:
                        translations[source] = translated
            search_group = report.get("search_group")
            if not isinstance(search_group, dict):
                continue
            start = float(search_group.get("start", float("-inf")))
            end = float(search_group.get("end", float("inf")))
            for index, cue in enumerate(corrected):
                translated = translations.get(cue.text)
                if (
                    translated
                    and cue.kind == "singing"
                    and cue.start >= start - 1e-3
                    and cue.end <= end + 1e-3
                ):
                    corrected[index] = replace(cue, preferred_translation=translated)
    finally:
        library.close()
    return SongIdentificationResult(
        corrected, result.reports, result.verified_lyric_spans
    )


def split_aligned_song_cues(
    result: SongIdentificationResult, source_maximum_units: float
) -> SongIdentificationResult:
    corrected: list[Cue] = []
    for cue in result.corrected_cues:
        if (
            cue.kind != "singing"
            or text_display_width(cue.text) <= source_maximum_units
            or not cue.source_units
        ):
            corrected.append(cue)
            continue
        fragments = _split_aligned_lyric_line(
            cue.text,
            cue.source_units,
            cue.start,
            cue.end,
            source_maximum_units,
        )
        translations = _split_text_by_weights(
            cue.preferred_translation or "",
            [text_display_width(text) for text, _units in fragments],
        )
        for (text, units), translated in zip(fragments, translations):
            corrected.append(
                replace(
                    cue,
                    start=units[0].start,
                    end=units[-1].end,
                    text=text,
                    preferred_translation=translated or None,
                    source_units=units,
                )
            )
    return SongIdentificationResult(
        corrected, result.reports, result.verified_lyric_spans
    )


def _match_candidates(
    hypotheses: list[str], songs: list[LibrarySong], config: SongIdentificationConfig
) -> SongMatch | None:
    return match_song(
        hypotheses,
        songs,
        anchor_threshold=config.match_anchor_threshold,
        minimum_anchors=config.match_minimum_anchors,
        minimum_score=config.match_minimum_score,
    )


def _web_search_policy(
    singing_cues: list[Cue],
    metadata: dict[str, object],
    *,
    has_trusted_ocr: bool,
) -> dict[str, object]:
    title = unicodedata.normalize("NFKC", str(metadata.get("title") or ""))
    mode = "relaxed" if _title_uses_music_mode(metadata) else "standard"
    minimum_seconds = (
        _RELAXED_SEARCH_MINIMUM_SECONDS
        if mode == "relaxed"
        else _STANDARD_SEARCH_MINIMUM_SECONDS
    )
    coverage_seconds = _merged_cue_duration(singing_cues)
    eligible = has_trusted_ocr or coverage_seconds >= minimum_seconds
    reason = (
        "trusted_ocr"
        if has_trusted_ocr
        else "alt_coverage_threshold_met"
        if eligible
        else "insufficient_alt_coverage"
    )
    return {
        "mode": mode,
        "title": title,
        "alt_coverage_seconds": round(coverage_seconds, 3),
        "minimum_alt_coverage_seconds": minimum_seconds,
        "trusted_ocr": has_trusted_ocr,
        "eligible": eligible,
        "decision_reason": reason,
    }


def _title_uses_music_mode(metadata: dict[str, object]) -> bool:
    title = unicodedata.normalize("NFKC", str(metadata.get("title") or ""))
    return any(keyword in title for keyword in _RELAXED_SEARCH_TITLE_KEYWORDS)


def _merged_cue_duration(cues: list[Cue]) -> float:
    intervals = sorted(
        (float(cue.start), float(cue.end)) for cue in cues if cue.end > cue.start
    )
    if not intervals:
        return 0.0
    total = 0.0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    return total + current_end - current_start


def _search_text_coverage(needle: str, haystack: str) -> float:
    if not needle or not haystack:
        return 0.0
    match = SequenceMatcher(None, needle, haystack, autojunk=False).find_longest_match()
    return match.size / len(needle)


def _rank_lyric_search_results(
    results: list[object],
    *,
    phrase: str,
    source: str,
    normalizer: JapaneseNormalizer,
) -> list[dict[str, object]]:
    phrase_text = _normalize_identity_text(phrase)
    phrase_reading = normalizer(phrase)
    phrase_parts = [
        value
        for value in (
            _normalize_identity_text(item) for item in re.split(r"[/／｜|]", phrase)
        )
        if len(value) >= 2
    ]
    ranked: list[tuple[tuple[float, float, float, int], dict[str, object]]] = []
    for original_rank, raw in enumerate(results):
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "")
        if not url or not _supported_lyrics_url(url):
            continue
        title = str(raw.get("title") or "")
        snippet = str(raw.get("snippet") or "")
        normalized_title = _normalize_identity_text(title)
        normalized_summary = _normalize_identity_text(f"{title} {snippet}")
        exact_title_match = float(
            source == "ocr" and any(part in normalized_title for part in phrase_parts)
        )
        lexical_coverage = _search_text_coverage(phrase_text, normalized_summary)
        summary_reading = normalizer(f"{title} {snippet}"[:1000])
        kana_coverage = _search_text_coverage(phrase_reading, summary_reading)
        item = {
            **raw,
            "original_rank": original_rank,
            "ranking": {
                "ocr_title_match": bool(exact_title_match),
                "lexical_coverage": round(lexical_coverage, 6),
                "kana_coverage": round(kana_coverage, 6),
            },
        }
        ranked.append(
            (
                (
                    exact_title_match,
                    lexical_coverage,
                    kana_coverage,
                    -original_rank,
                ),
                item,
            )
        )
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [
        {**item, "summary_rank": rank} for rank, (_score, item) in enumerate(ranked)
    ]


def _search_canonical_lyrics(
    hypotheses: list[str],
    queries: list[dict[str, object]],
    config: SongIdentificationConfig,
) -> tuple[list[LibrarySong], dict[str, object]]:
    tools = _WebTools(config)
    search_audit: list[dict[str, object]] = []
    fetch_audit: list[dict[str, object]] = []
    songs: dict[str, LibrarySong] = {}
    fetched_urls: set[str] = set()
    normalizer = JapaneseNormalizer()
    confirmed_after_query: int | None = None
    for query_index, item in enumerate(queries[:_MAX_WEB_SEARCH_QUERIES]):
        query = str(item["query"])
        record = dict(item)
        ranked_results: list[dict[str, object]] = []
        try:
            raw_results = tools.search(query)
            parsed_results = json.loads(raw_results)
            if isinstance(parsed_results, list):
                ranked_results = _rank_lyric_search_results(
                    parsed_results,
                    phrase=str(item["phrase"]),
                    source=str(item["source"]),
                    normalizer=normalizer,
                )
                record["results"] = ranked_results
            else:
                record["results"] = []
                if isinstance(parsed_results, dict) and parsed_results.get("error"):
                    record["error"] = str(parsed_results["error"])[:500]
        except Exception as exc:
            logging.warning("canonical lyric search failed for %r: %s", query, exc)
            record["error"] = str(exc)[:500]
        selected_urls: list[str] = []
        for result in ranked_results:
            url = str(result.get("url") or "")
            if not url or url in fetched_urls:
                continue
            selected_urls.append(url)
            if len(selected_urls) >= _MAX_LYRIC_FETCHES_PER_QUERY:
                break
        record["selected_urls"] = selected_urls
        search_audit.append(record)
        for url in selected_urls:
            fetched_urls.add(url)
            fetch_record: dict[str, object] = {
                "url": url,
                "query_index": query_index,
            }
            try:
                value = tools.fetch_lyrics(url)
            except Exception as exc:
                logging.warning("canonical lyric fetch failed for %s: %s", url, exc)
                fetch_record.update(status="error", error=str(exc)[:500])
                fetch_audit.append(fetch_record)
                continue
            if value is None:
                fetch_record["status"] = "not_lyrics"
                fetch_audit.append(fetch_record)
                continue
            title, artist, lines = value
            fetch_record.update(
                status="fetched",
                title=title,
                artist=artist,
                lyric_line_count=len(lines),
            )
            fetch_audit.append(fetch_record)
            identity = hashlib.sha256(f"{title}\0{artist}".encode()).hexdigest()[:24]
            source_hash = hashlib.sha256("\n".join(lines).encode()).hexdigest()
            songs[identity] = LibrarySong(
                identity,
                title,
                artist,
                (),
                url,
                source_hash,
                tuple(LyricLine(index, text) for index, text in enumerate(lines)),
            )
        if _match_candidates(hypotheses, list(songs.values()), config) is not None:
            confirmed_after_query = query_index
            break
    audit: dict[str, object] = {
        "queries": search_audit,
        "fetches": fetch_audit,
        "worker_errors": tools.errors,
        "confirmed_after_query": confirmed_after_query,
    }
    return list(songs.values()), audit


def _build_lyric_search_queries(
    hypotheses: list[str],
    ocr: list[OCRCandidate],
    library_songs: list[LibrarySong],
    confirmed_song_names: set[str],
) -> list[dict[str, object]]:
    seen: set[str] = set()

    def make_query(
        source: str, phrase: str, reason: str, *, minimum_length: int
    ) -> dict[str, object] | None:
        compact = re.sub(r"\s+", " ", phrase).strip(' "')
        if len(compact) < minimum_length:
            return None
        query = f"{compact} 歌詞"
        if query in seen:
            return None
        seen.add(query)
        return {"source": source, "phrase": phrase, "reason": reason, "query": query}

    queries: list[dict[str, object]] = []

    fragments = _asr_lyric_search_fragments(hypotheses)[:2]
    if fragments:
        combined = " ".join(fragments)
        query = make_query(
            "asr",
            combined,
            "combined_high_information_asr_fragments",
            minimum_length=8,
        )
        if query is not None:
            queries.append(query)

    for candidate in sorted(
        ocr,
        key=lambda item: (-item.score, -item.frames, item.first_time, item.text),
    ):
        normalized_candidate = _normalize_identity_text(candidate.text)
        if any(name in normalized_candidate for name in confirmed_song_names):
            continue
        qualified = _qualified_ocr_song_evidence(candidate.text, library_songs)
        if qualified is None:
            continue
        phrase, reason = qualified
        query = make_query("ocr", phrase, reason, minimum_length=3)
        if query is not None:
            queries.append(query)
            break
    return queries[:_MAX_WEB_SEARCH_QUERIES]


def _asr_lyric_search_fragments(hypotheses: list[str]) -> list[str]:
    candidates: dict[str, float] = {}
    for hypothesis in hypotheses:
        normalized = unicodedata.normalize("NFKC", hypothesis)
        clauses = re.split(r"[。！？!?、,;；\n]+", normalized)
        for clause in clauses:
            clause = re.sub(r"\s+", " ", clause).strip()
            if not clause:
                continue
            japanese = bool(re.search(r"[\u3040-\u30ff\u3400-\u9fff]", clause))
            fragments = clause.split() if japanese else [clause]
            for fragment in fragments:
                fragment = fragment.strip(" '‘’\"“”()（）[]【】")
                compact = re.sub(r"\s+", "", fragment)
                if len(compact) < 8 or _looks_like_non_lyric_asr(fragment):
                    continue
                excerpt = fragment.strip()
                unique_ratio = len(set(compact.casefold())) / len(compact)
                score = min(len(compact), 20) + unique_ratio * 8
                if 12 <= len(compact) <= 24:
                    score += 4
                if re.search(
                    r"(?:なんか|っていう|今日は|ですけど|けども|だからさ)",
                    fragment,
                ):
                    score -= 8
                candidates[excerpt] = max(candidates.get(excerpt, float("-inf")), score)
    ranked = sorted(candidates.items(), key=lambda item: (-item[1], item[0]))
    return [item[0] for item in ranked]


def _looks_like_non_lyric_asr(text: str) -> bool:
    compact = re.sub(r"[^0-9A-Za-z\u3040-\u30ff\u3400-\u9fff]", "", text)
    if not compact:
        return True
    if re.fullmatch(r"(?i)(?:h+m+|m+h+|la+|na+|oh+|ah+)+", compact):
        return True
    return bool(re.search(r"(.{1,4})\1{3,}", compact))


def _qualified_ocr_song_evidence(
    text: str, library_songs: list[LibrarySong]
) -> tuple[str, str] | None:
    normalized = _normalize_identity_text(text)
    if not normalized or _looks_like_ocr_ui_noise(text):
        return None
    for song in library_songs:
        names = (song.title, song.artist, *song.aliases)
        if any(
            len(name_normalized := _normalize_identity_text(name)) >= 3
            and name_normalized in normalized
            for name in names
        ):
            return text, "local_song_or_artist_name"
    labeled = re.search(
        r"(?:曲名|楽曲|歌名|song|title)\s*[:：]\s*(.{2,80})",
        text,
        flags=re.IGNORECASE,
    )
    if labeled:
        return labeled.group(1).strip(), "explicit_song_title_label"
    decorated = re.fullmatch(r"\s*[♪♫♬♩]+\s*(.{2,80}?)\s*[♪♫♬♩]+\s*", text)
    if decorated:
        return decorated.group(1).strip(), "music_note_title_decoration"
    title_artist = re.fullmatch(r"\s*(.{2,50}?)\s*(?:/|／|｜|\|)\s*(.{2,50}?)\s*", text)
    if title_artist:
        return text.strip(), "title_artist_separator"
    return None


def _normalize_identity_text(text: str) -> str:
    return re.sub(
        r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", text).casefold(),
    )


def _looks_like_ocr_ui_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
    return bool(
        re.search(r"(?i)(?:date)?\d{1,4}[\-/@]\d{1,2}", compact)
        or re.search(r"(?i)\b(?:mon|tue|wed|thu|fri|sat|sun)\b", text)
        or re.fullmatch(r"(?:\d{1,4}[/.:\-]){1,3}\d{1,4}(?:[A-Za-z]+)?", compact)
        or re.fullmatch(r"\d+/\d+", compact)
        or re.fullmatch(
            r"(?:REC|LIVE|ON AIR|TUE|WED|THU|FRI|SAT|SUN)",
            compact,
            re.IGNORECASE,
        )
    )


def _ensure_song_translations(
    library: LyricsLibrary,
    song: LibrarySong,
    translate_lyrics: Callable[..., object],
    translation_context: dict[str, object],
    model: str | None,
) -> LibrarySong:
    if all(line.translation for line in song.lines):
        return song
    if not all(line.translation for line in song.lines):
        translated = translate_lyrics(
            song.title,
            song.artist,
            [line.text for line in song.lines],
            translation_context=translation_context,
        )
        if (
            isinstance(translated, tuple)
            and len(translated) == 2
            and isinstance(translated[0], dict)
            and translated[1] == "llm"
        ):
            translations, source = translated
        elif isinstance(translated, dict):
            translations, source = translated, "llm"
        else:
            raise TypeError("lyrics translator returned an invalid result")
        library.store_translations(
            song.song_id,
            translations,
            source=source,
            model=model,
            prompt_hash=prompt_templates_digest("lyrics-translate.md"),
        )
        refreshed = library.get(song.song_id)
        assert refreshed is not None
        song = refreshed

    refreshed = library.get(song.song_id)
    assert refreshed is not None
    return refreshed


def _split_aligned_lyric_line(
    text: str,
    source_units: tuple[TimedTextUnit, ...],
    start: float,
    end: float,
    maximum_units: float | None,
) -> list[tuple[str, tuple[TimedTextUnit, ...]]]:
    if maximum_units is None or text_display_width(text) <= maximum_units:
        return [(text, source_units)]

    units = source_units
    if not units or "".join(unit.text for unit in units) != text:
        units = (TimedTextUnit(text, start, end),)

    characters: list[TimedTextUnit] = []
    for unit in units:
        if len(unit.text) <= 1:
            characters.append(unit)
            continue
        duration = max(0.0, unit.end - unit.start)
        for index, character in enumerate(unit.text):
            character_start = unit.start + duration * index / len(unit.text)
            character_end = unit.start + duration * (index + 1) / len(unit.text)
            characters.append(TimedTextUnit(character, character_start, character_end))

    fragments: list[tuple[str, tuple[TimedTextUnit, ...]]] = []
    current: list[TimedTextUnit] = []
    current_width = 0.0
    for unit in characters:
        width = text_display_width(unit.text)
        if current and current_width + width > maximum_units:
            fragments.append(("".join(item.text for item in current), tuple(current)))
            current = []
            current_width = 0.0
        current.append(unit)
        current_width += width
    if current:
        fragments.append(("".join(item.text for item in current), tuple(current)))
    return fragments


def _split_text_by_weights(text: str, weights: list[float]) -> list[str]:
    if not weights:
        return []
    if len(weights) == 1:
        return [text]
    if not text:
        return ["" for _weight in weights]
    total = sum(max(0.0, weight) for weight in weights) or float(len(weights))
    result: list[str] = []
    start = 0
    cumulative = 0.0
    for weight in weights[:-1]:
        cumulative += max(0.0, weight)
        target = round(len(text) * cumulative / total)
        target = min(len(text), max(start, target))
        result.append(text[start:target])
        start = target
    result.append(text[start:])
    return result


def _apply_local_match(
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    timing: dict[
        int,
        tuple[tuple[int, float, float] | tuple[int, float, float, object], ...],
    ],
    *,
    pyshiro_audit: list[dict[str, object]] | None = None,
    source_maximum_units: float | None = None,
) -> tuple[dict[int, list[Cue]], list[dict[str, object]]]:
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    for anchor in match.anchors:
        cue_id = singing_ids[anchor.cue_index]
        line_ids = list(range(anchor.line_start, anchor.line_end))
        lines = [match.song.lines[line_id] for line_id in line_ids]
        line_timing = timing.get(cue_id)
        if (
            line_timing is not None
            and [item[0] for item in line_timing] == line_ids
            and all(item[2] > item[1] for item in line_timing)
        ):
            replacement: list[Cue] = []
            for line, timing_value in zip(lines, line_timing):
                line_id, start, end = timing_value[:3]
                source_units = tuple(timing_value[3]) if len(timing_value) == 4 else ()
                fragments = _split_aligned_lyric_line(
                    line.text,
                    source_units,
                    start,
                    end,
                    source_maximum_units,
                )
                translated_fragments = _split_text_by_weights(
                    line.translation or "",
                    [text_display_width(text) for text, _units in fragments],
                )
                for fragment_index, ((text, units), translated) in enumerate(
                    zip(fragments, translated_fragments)
                ):
                    fragment_start = units[0].start if units else start
                    fragment_end = units[-1].end if units else end
                    replacement.append(
                        Cue(
                            **{
                                **asdict(cues[cue_id]),
                                "start": fragment_start,
                                "end": fragment_end,
                                "text": text,
                                "kind": "singing",
                                "language": language_for_text(text, "Japanese"),
                                "preferred_translation": translated or None,
                                "source_units": units,
                            }
                        )
                    )
                    alignments.append(
                        {
                            "asr_cue_ids": [cue_id],
                            "lyric_line_ids": [line_id],
                            "match": "lyrics",
                            "corrected_text": text,
                            "source_line_text": line.text,
                            "fragment_index": fragment_index,
                            "fragment_count": len(fragments),
                            "fragment_weight": text_display_width(text),
                            "start": fragment_start,
                            "end": fragment_end,
                            "score": round(anchor.score, 6),
                            "take_index": anchor.take_index,
                        }
                    )
            replacements[cue_id] = replacement
            continue

        audit = next(
            (
                item
                for item in pyshiro_audit or []
                if item.get("cue_id") == cue_id
                and item.get("take_index") == anchor.take_index
            ),
            {},
        )
        logging.warning(
            "canonical lyric anchor rejected by pySHIRO cue_id=%d lines=%s "
            "reason=%s likelihood=%s competing=%s",
            cue_id,
            line_ids,
            audit.get("reason", "timing_unavailable"),
            audit.get("likelihood_per_frame"),
            audit.get("best_competing_likelihood_per_frame"),
        )
    return replacements, alignments


def _recover_lyric_gaps(
    job_dir: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
    *,
    video: Path | None = None,
) -> tuple[
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not match.anchors:
        return {}, [], []
    manifest_path = job_dir / "vocal-candidates" / "manifest.json"
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if uv is None or not worker.is_file():
        return {}, [], []
    raw_manifests = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else []
    )
    manifests = [value for value in raw_manifests if isinstance(value, dict)]
    probes: list[dict[str, object]] = []
    ordered = sorted(
        match.anchors, key=lambda value: (value.take_index, value.cue_index)
    )
    for left, right in pairwise(ordered):
        if (
            right.take_index != left.take_index
            or right.cue_index <= left.cue_index
            or right.line_start <= left.line_end
            or min(left.score, right.score) < config.match_anchor_threshold
        ):
            continue
        left_cue_id = singing_ids[left.cue_index]
        right_cue_id = singing_ids[right.cue_index]
        center = (cues[left_cue_id].end + cues[right_cue_id].start) / 2
        half = config.lyric_gap_recheck_seconds / 2
        start = max(0.0, center - half)
        end = start + config.lyric_gap_recheck_seconds
        manifest = next(
            (
                value
                for value in manifests
                if float(value.get("start", -1)) <= start
                and float(value.get("end", -1)) >= end
            ),
            None,
        )
        if manifest is None and video is None:
            continue
        if end - start < 2:
            continue
        probes.append(
            {
                "gap_id": len(probes),
                "left": left,
                "right": right,
                "left_cue_id": left_cue_id,
                "right_cue_id": right_cue_id,
                "start": start,
                "end": end,
                "manifest": manifest,
            }
        )
    if video is not None:
        _materialize_gap_vocal_stems(
            video,
            probes,
            manifests,
            manifest_path,
            config.vocal_separation_device,
        )
    normalizer = JapaneseNormalizer()
    output_dir = job_dir / "song-alignment" / match.song.song_id / "gap-recheck"
    output_dir.mkdir(parents=True, exist_ok=True)
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for probe in probes:
        gap_id = int(probe["gap_id"])
        left = probe["left"]
        right = probe["right"]
        assert isinstance(left, LyricAnchor) and isinstance(right, LyricAnchor)
        missing_ids = list(range(left.line_end, right.line_start))
        missing_lines = [match.song.lines[index] for index in missing_ids]
        audit: dict[str, object] = {
            "status": "gap_rejected",
            "gap_id": gap_id,
            "cue_id": int(probe["left_cue_id"]),
            "take_index": left.take_index,
            "lyric_line_ids": missing_ids,
            "start": probe["start"],
            "end": probe["end"],
        }
        manifest = probe["manifest"]
        if not isinstance(manifest, dict):
            audit["reason"] = "verified_gap_vocal_stem_unavailable"
            if probe.get("stem_error"):
                audit["stem_error"] = probe["stem_error"]
            audits.append(audit)
            continue
        source = manifest_path.parent / str(manifest["path"])
        wav = output_dir / f"gap-{gap_id:04d}.wav"
        if not _extract_vocal_window(
            source,
            wav,
            float(probe["start"]) - float(manifest["start"]),
            float(probe["end"]) - float(probe["start"]),
        ):
            audit["reason"] = "vocal_window_extraction_failed"
            audits.append(audit)
            continue
        active_ratio = _vocal_active_ratio(wav)
        audit["vocal_active_ratio"] = round(active_ratio, 6)
        if active_ratio < config.lyric_gap_vocal_active_ratio:
            audit["reason"] = "insufficient_vocal_activity"
            audits.append(audit)
            continue
        adjacent = [
            match.song.lines[left.line_end - 1],
            match.song.lines[right.line_start],
        ]
        expanded = [adjacent[0], *missing_lines, adjacent[1]]
        baseline_failure: dict[str, object] = {}
        aligned_failure: dict[str, object] = {}
        baseline = _run_pyshiro_lines(
            uv,
            worker,
            wav,
            adjacent,
            normalizer,
            failure_audit=baseline_failure,
        )
        aligned = _run_pyshiro_lines(
            uv,
            worker,
            wav,
            expanded,
            normalizer,
            failure_audit=aligned_failure,
        )
        if baseline is None or aligned is None:
            audit["reason"] = "pyshiro_recheck_failed"
            audit["pyshiro_failures"] = {
                "baseline": baseline_failure if baseline is None else None,
                "expanded": aligned_failure if aligned is None else None,
            }
            audits.append(audit)
            continue
        competitors = [baseline]
        competitor_failures: list[dict[str, object]] = []
        for alternative in _competing_line_sets(
            match.song, missing_ids, normalizer=normalizer
        ):
            failure: dict[str, object] = {}
            response = _run_pyshiro_lines(
                uv,
                worker,
                wav,
                [adjacent[0], *alternative, adjacent[1]],
                normalizer,
                failure_audit=failure,
            )
            if response is not None:
                competitors.append(response)
            else:
                competitor_failures.append(failure)
        if competitor_failures:
            audit["competitor_failures"] = competitor_failures
        accepted, aligned_score, best_competing = _pyshiro_likelihood_wins(
            aligned,
            competitors,
            config,
            margin=config.pyshiro_gap_likelihood_margin,
        )
        baseline_score = float(baseline.get("likelihood_per_frame", -1e9))
        audit["baseline_likelihood_per_frame"] = baseline_score
        audit["expanded_likelihood_per_frame"] = aligned_score
        audit["best_competing_likelihood_per_frame"] = best_competing
        audit["validation_policy"] = "between_anchors_continuous_block"
        audit["required_likelihood_margin"] = config.pyshiro_gap_likelihood_margin
        if not accepted:
            audit["reason"] = "pyshiro_candidate_not_preferred"
            audits.append(audit)
            continue
        ranges = aligned.get("lines")
        units_by_line = aligned.get("units")
        if (
            not isinstance(ranges, list)
            or not isinstance(units_by_line, list)
            or len(ranges) != len(expanded)
            or len(units_by_line) != len(expanded)
        ):
            audit["reason"] = "pyshiro_line_count_mismatch"
            audits.append(audit)
            continue
        recovered_cues: list[Cue] = []
        previous_end = float(probe["start"])
        valid = True
        for line_id, line, value, raw_units in zip(
            missing_ids, missing_lines, ranges[1:-1], units_by_line[1:-1]
        ):
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not isinstance(raw_units, list)
            ):
                valid = False
                break
            start = float(probe["start"]) + float(value[0])
            end = float(probe["start"]) + float(value[1])
            units = tuple(
                TimedTextUnit(
                    str(unit["text"]),
                    float(probe["start"]) + float(unit["start"]),
                    float(probe["start"]) + float(unit["end"]),
                )
                for unit in raw_units
                if isinstance(unit, dict)
            )
            if start < previous_end - 1e-3 or end <= start or not units:
                valid = False
                break
            if any(
                unit.end <= unit.start or unit.end - unit.start > 5 for unit in units
            ):
                valid = False
                break
            recovered_cues.append(
                Cue(
                    start,
                    end,
                    line.text,
                    cues[int(probe["left_cue_id"])].speaker,
                    "singing",
                    preferred_translation=line.translation,
                    source_units=units,
                    language=language_for_text(line.text, "Japanese"),
                )
            )
            alignments.append(
                {
                    "asr_cue_ids": [int(probe["left_cue_id"])],
                    "lyric_line_ids": [line_id],
                    "match": "lyrics_gap_recovered",
                    "corrected_text": line.text,
                    "start": start,
                    "end": end,
                    "score": round(aligned_score, 6),
                    "take_index": left.take_index,
                }
            )
            previous_end = end
        if not valid:
            if recovered_cues:
                del alignments[-len(recovered_cues) :]
            audit["reason"] = "pyshiro_timeline_rejected"
            audits.append(audit)
            continue
        replacements.setdefault(int(probe["left_cue_id"]), []).extend(recovered_cues)
        audit["status"] = "gap_recovered"
        audit["reason"] = "vocal_and_pyshiro_confirmed"
        audits.append(audit)

    neighbor_replacements, neighbor_alignments, neighbor_audits = (
        _recover_take_neighbor_lyrics(
            cues,
            singing_ids,
            match,
            config,
            manifests,
            manifest_path,
            uv,
            worker,
            normalizer,
            output_dir,
        )
    )
    for cue_id, values in neighbor_replacements.items():
        replacements.setdefault(cue_id, []).extend(values)
    alignments.extend(neighbor_alignments)
    audits.extend(neighbor_audits)
    phrase_replacements, phrase_alignments, phrase_audits = (
        _recover_acoustic_phrase_neighbors(
            job_dir,
            cues,
            singing_ids,
            match,
            config,
            manifests,
            manifest_path,
            uv,
            worker,
            normalizer,
            output_dir,
        )
    )
    for cue_id, values in phrase_replacements.items():
        replacements.setdefault(cue_id, []).extend(values)
    alignments.extend(phrase_alignments)
    audits.extend(phrase_audits)
    return replacements, alignments, audits


def _materialize_gap_vocal_stems(
    video: Path,
    probes: list[dict[str, object]],
    manifests: list[dict[str, object]],
    manifest_path: Path,
    device: str,
) -> None:
    missing: dict[tuple[float, float], dict[str, object]] = {}
    for probe in probes:
        if isinstance(probe.get("manifest"), dict):
            continue
        start = round(float(probe["start"]), 3)
        end = round(float(probe["end"]), 3)
        key = (start, end)
        if key in missing:
            continue
        digest = hashlib.sha256(f"{start:.3f}:{end:.3f}".encode()).hexdigest()[:12]
        missing[key] = {
            "start": start,
            "end": end,
            "path": f"verified-gap-{digest}.vocals.wav",
            "source": "verified_lyric_gap",
        }
    if not missing:
        return

    from .audio_analysis import separate_vocal_ranges

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        separate_vocal_ranges(
            video,
            [
                (
                    start,
                    end,
                    manifest_path.parent / str(entry["path"]),
                )
                for (start, end), entry in missing.items()
            ],
            device,
        )
    except Exception as exc:
        logging.warning("verified lyric gap vocal separation failed: %s", exc)
        for probe in probes:
            if not isinstance(probe.get("manifest"), dict):
                probe["stem_error"] = str(exc)[:500]
        return

    manifests.extend(missing.values())
    manifests.sort(key=lambda value: (float(value["start"]), float(value["end"])))
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    for probe in probes:
        if isinstance(probe.get("manifest"), dict):
            continue
        key = (round(float(probe["start"]), 3), round(float(probe["end"]), 3))
        probe["manifest"] = missing.get(key)


def _recover_take_neighbor_lyrics(
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
    manifests: list[dict[str, object]],
    manifest_path: Path,
    uv: str,
    worker: Path,
    normalizer: JapaneseNormalizer,
    output_dir: Path,
) -> tuple[
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    by_take: dict[int, list[LyricAnchor]] = {}
    for anchor in match.anchors:
        by_take.setdefault(anchor.take_index, []).append(anchor)
    takes = [
        sorted(values, key=lambda value: value.cue_index)
        for _take, values in sorted(by_take.items())
    ]
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    candidates: list[tuple[int, list[int], str, int]] = []
    if takes:
        first = takes[0][0]
        if first.cue_index > 0 and first.line_start > 0:
            position = first.cue_index - 1
            start = max(0, first.line_start - config.lyric_neighbor_max_lines)
            candidates.append(
                (
                    position,
                    list(range(start, first.line_start)),
                    "performance_prefix",
                    first.take_index,
                )
            )
        last = takes[-1][-1]
        if last.cue_index + 1 < len(singing_ids):
            position = last.cue_index + 1
            cue = cues[singing_ids[position]]
            line_ids = _select_suffix_neighbor_lines(
                match.song,
                last.line_end,
                cue.text,
                cue.end - cue.start,
                config.lyric_neighbor_max_lines,
                normalizer,
            )
            if line_ids:
                candidates.append(
                    (position, line_ids, "performance_suffix", last.take_index)
                )
    for left_take, right_take in pairwise(takes):
        left = left_take[-1]
        right = right_take[0]
        positions = list(range(left.cue_index + 1, right.cue_index))
        if not positions:
            continue
        suffix_position = positions[0]
        suffix_cue = cues[singing_ids[suffix_position]]
        suffix_ids = _select_suffix_neighbor_lines(
            match.song,
            left.line_end,
            suffix_cue.text,
            suffix_cue.end - suffix_cue.start,
            config.lyric_neighbor_max_lines,
            normalizer,
        )
        if suffix_ids:
            candidates.append(
                (suffix_position, suffix_ids, "take_suffix", left.take_index)
            )

        prefix_ids = list(
            range(0, min(right.line_start, config.lyric_neighbor_max_lines))
        )
        if prefix_ids:
            prefix_position = positions[-1]
            if prefix_position == suffix_position and suffix_ids:
                candidates[-1] = (
                    suffix_position,
                    [*suffix_ids, *prefix_ids],
                    "take_suffix_and_prefix",
                    left.take_index,
                )
            else:
                candidates.append(
                    (prefix_position, prefix_ids, "take_prefix", right.take_index)
                )

    for position, line_ids, route, take_index in candidates:
        cue_id = singing_ids[position]
        recovered, recovered_alignments, audit = _align_acoustic_neighbor(
            cues[cue_id],
            cue_id,
            line_ids,
            route,
            take_index,
            match,
            config,
            manifests,
            manifest_path,
            uv,
            worker,
            normalizer,
            output_dir,
        )
        audits.append(audit)
        if not recovered:
            continue
        replacements.setdefault(cue_id, []).extend(recovered)
        alignments.extend(recovered_alignments)
    return replacements, alignments, audits


def _select_suffix_neighbor_lines(
    song: LibrarySong,
    line_start: int,
    asr_text: str,
    duration: float,
    maximum_lines: int,
    normalizer: JapaneseNormalizer,
) -> list[int]:
    if line_start >= len(song.lines) or duration <= 0:
        return []
    hypothesis = normalizer(asr_text)
    candidates: list[tuple[float, int, list[int]]] = []
    for line_end in range(
        line_start + 1, min(len(song.lines), line_start + maximum_lines) + 1
    ):
        lines = song.lines[line_start:line_end]
        display_units = [
            normalizer.display_units(line.text, line.reading) for line in lines
        ]
        if any(not values for values in display_units):
            continue
        expected = normalizer("".join(line.reading or line.text for line in lines))
        text_score = (
            SequenceMatcher(None, hypothesis, expected, autojunk=False).ratio()
            if hypothesis and expected
            else 0.0
        )
        unit_count = sum(len(values) for values in display_units)
        seconds_per_unit = duration / max(1, unit_count)
        duration_score = 1.0 / (1.0 + abs(seconds_per_unit - 0.24))
        score = 0.35 * text_score + 0.65 * duration_score
        candidates.append((score, line_end, list(range(line_start, line_end))))
    if not candidates:
        return []
    return max(candidates, key=lambda value: value[:2])[2]


def _recover_acoustic_phrase_neighbors(
    job_dir: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
    manifests: list[dict[str, object]],
    manifest_path: Path,
    uv: str,
    worker: Path,
    normalizer: JapaneseNormalizer,
    output_dir: Path,
) -> tuple[
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    analysis_path = job_dir / "audio-analysis.json"
    if not analysis_path.is_file() or not match.anchors:
        return {}, [], []
    try:
        payload = json.loads(analysis_path.read_text(encoding="utf-8"))
        raw_phrases = payload["acoustic_phrases"]
        phrases = [
            value
            for value in raw_phrases
            if isinstance(value, dict) and value.get("route_alt") is False
        ]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return {}, [], []
    if not phrases:
        return {}, [], []

    ordered = sorted(
        match.anchors, key=lambda value: (value.take_index, value.cue_index)
    )
    first = ordered[0]
    last = ordered[-1]
    first_cue_id = singing_ids[first.cue_index]
    last_cue_id = singing_ids[last.cue_index]
    first_cue = cues[first_cue_id]
    last_cue = cues[last_cue_id]
    candidates: list[tuple[dict[str, object], int, list[int], str, int]] = []
    before = [
        value
        for value in phrases
        if 0
        <= first_cue.start - float(value.get("end", -1))
        <= config.song_search_group_gap_seconds
    ]
    if before and first.line_start > 0:
        phrase = max(before, key=lambda value: float(value["end"]))
        line_start = max(0, first.line_start - config.lyric_neighbor_max_lines)
        candidates.append(
            (
                phrase,
                first_cue_id,
                list(range(line_start, first.line_start)),
                "acoustic_phrase_prefix",
                first.take_index,
            )
        )
    after = [
        value
        for value in phrases
        if 0
        <= float(value.get("start", -1)) - last_cue.end
        <= config.song_search_group_gap_seconds
    ]
    if after and last.line_end < len(match.song.lines):
        phrase = min(after, key=lambda value: float(value["start"]))
        line_ids = _select_suffix_neighbor_lines(
            match.song,
            last.line_end,
            "",
            float(phrase["end"]) - float(phrase["start"]),
            config.lyric_neighbor_max_lines,
            normalizer,
        )
        if line_ids:
            candidates.append(
                (
                    phrase,
                    last_cue_id,
                    line_ids,
                    "acoustic_phrase_suffix",
                    last.take_index,
                )
            )

    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for phrase, cue_id, line_ids, route, take_index in candidates:
        probe = Cue(
            float(phrase["start"]),
            float(phrase["end"]),
            "",
            cues[cue_id].speaker,
            "singing",
        )
        recovered, recovered_alignments, audit = _align_acoustic_neighbor(
            probe,
            cue_id,
            line_ids,
            route,
            take_index,
            match,
            config,
            manifests,
            manifest_path,
            uv,
            worker,
            normalizer,
            output_dir,
        )
        audits.append(audit)
        if not recovered:
            continue
        replacements.setdefault(cue_id, []).extend(recovered)
        alignments.extend(recovered_alignments)
    return replacements, alignments, audits


def _lyric_unit_timeline_errors(
    units: tuple[TimedTextUnit, ...],
    *,
    line_id: int,
    lyric_text: str,
) -> list[dict[str, object]]:
    errors: list[dict[str, object]] = []
    for unit_index, unit in enumerate(units):
        duration = unit.end - unit.start
        if duration > 0:
            continue
        errors.append(
            {
                "issue": "non_positive_unit_duration",
                "lyric_line_id": line_id,
                "lyric_text": lyric_text,
                "unit_index": unit_index,
                "unit_text": unit.text,
                "start": round(unit.start, 6),
                "end": round(unit.end, 6),
                "duration_seconds": round(duration, 6),
                "description": "display unit ends at or before it starts",
            }
        )
    return errors


def _align_acoustic_neighbor(
    cue: Cue,
    cue_id: int,
    line_ids: list[int],
    route: str,
    take_index: int,
    match: SongMatch,
    config: SongIdentificationConfig,
    manifests: list[dict[str, object]],
    manifest_path: Path,
    uv: str,
    worker: Path,
    normalizer: JapaneseNormalizer,
    output_dir: Path,
) -> tuple[list[Cue], list[dict[str, object]], dict[str, object]]:
    audit: dict[str, object] = {
        "status": "neighbor_rejected",
        "route": route,
        "cue_id": cue_id,
        "take_index": take_index,
        "lyric_line_ids": line_ids,
        "start": cue.start,
        "end": cue.end,
    }
    if cue.end <= cue.start or cue.end - cue.start > config.pyshiro_max_window_seconds:
        audit["reason"] = "duration_outside_pyshiro_limit"
        return [], [], audit
    manifest = next(
        (
            value
            for value in manifests
            if float(value.get("start", -1)) <= cue.start
            and float(value.get("end", -1)) >= cue.end
        ),
        None,
    )
    if manifest is None:
        audit["reason"] = "no_covering_vocal_stem"
        return [], [], audit
    source = manifest_path.parent / str(manifest["path"])
    wav = output_dir / f"neighbor-{take_index:02d}-{route}-{cue_id:04d}.wav"
    if not _extract_vocal_window(
        source,
        wav,
        cue.start - float(manifest["start"]),
        cue.end - cue.start,
    ):
        audit["reason"] = "vocal_window_extraction_failed"
        return [], [], audit
    active_ratio = _vocal_active_ratio(wav)
    audit["vocal_active_ratio"] = round(active_ratio, 6)
    if active_ratio < config.lyric_gap_vocal_active_ratio:
        audit["reason"] = "insufficient_vocal_activity"
        return [], [], audit
    lines = [match.song.lines[line_id] for line_id in line_ids]
    candidate_failure: dict[str, object] = {}
    response = _run_pyshiro_lines(
        uv,
        worker,
        wav,
        lines,
        normalizer,
        failure_audit=candidate_failure,
    )
    if response is None:
        audit["reason"] = "pyshiro_neighbor_failed"
        audit["pyshiro_failure"] = candidate_failure
        return [], [], audit
    alternatives: list[dict[str, object]] = []
    alternative_failures: list[dict[str, object]] = []
    for lines_value in _competing_line_sets(
        match.song, line_ids, normalizer=normalizer
    ):
        failure: dict[str, object] = {}
        value = _run_pyshiro_lines(
            uv,
            worker,
            wav,
            lines_value,
            normalizer,
            failure_audit=failure,
        )
        if value is not None:
            alternatives.append(value)
        else:
            alternative_failures.append(failure)
    if alternative_failures:
        audit["competitor_failures"] = alternative_failures
    accepted, likelihood, best_alternative = _pyshiro_likelihood_wins(
        response, alternatives, config
    )
    audit["likelihood_per_frame"] = likelihood
    audit["best_competing_likelihood_per_frame"] = best_alternative
    audit["validation_policy"] = "one_sided_neighbor_extension"
    audit["required_likelihood_margin"] = config.pyshiro_likelihood_margin
    if not accepted:
        audit["reason"] = "pyshiro_candidate_not_preferred"
        return [], [], audit
    ranges = response.get("lines")
    units_by_line = response.get("units")
    if (
        not isinstance(ranges, list)
        or not isinstance(units_by_line, list)
        or len(ranges) != len(line_ids)
        or len(units_by_line) != len(line_ids)
    ):
        audit["reason"] = "pyshiro_line_count_mismatch"
        return [], [], audit
    recovered: list[Cue] = []
    alignments: list[dict[str, object]] = []
    previous_end = cue.start
    for line_id, line, value, raw_units in zip(line_ids, lines, ranges, units_by_line):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(raw_units, list)
        ):
            audit["reason"] = "pyshiro_timeline_rejected"
            return [], [], audit
        start = cue.start + float(value[0])
        end = cue.start + float(value[1])
        units = tuple(
            TimedTextUnit(
                str(unit["text"]),
                cue.start + float(unit["start"]),
                cue.start + float(unit["end"]),
            )
            for unit in raw_units
            if isinstance(unit, dict)
        )
        if start < previous_end - 1e-3 or end <= start or not units:
            audit["reason"] = "pyshiro_timeline_rejected"
            return [], [], audit
        unit_errors = _lyric_unit_timeline_errors(
            units,
            line_id=line_id,
            lyric_text=line.text,
        )
        if unit_errors:
            audit["reason"] = "pyshiro_unit_duration_rejected"
            audit["validation_errors"] = unit_errors
            return [], [], audit
        recovered.append(
            Cue(
                start,
                end,
                line.text,
                cue.speaker,
                "singing",
                preferred_translation=line.translation,
                source_units=units,
                language=language_for_text(line.text, "Japanese"),
            )
        )
        alignments.append(
            {
                "asr_cue_ids": [cue_id],
                "lyric_line_ids": [line_id],
                "match": "lyrics_acoustic_neighbor",
                "corrected_text": line.text,
                "start": start,
                "end": end,
                "score": round(likelihood, 6),
                "take_index": take_index,
            }
        )
        previous_end = end
    coverage = (recovered[-1].end - recovered[0].start) / (cue.end - cue.start)
    audit["timeline_coverage"] = round(coverage, 6)
    if coverage < config.lyric_neighbor_min_coverage:
        audit["reason"] = "insufficient_timeline_coverage"
        return [], [], audit
    audit["status"] = "neighbor_recovered"
    audit["reason"] = "vocal_and_pyshiro_confirmed_without_asr_gate"
    return recovered, alignments, audit


def _extract_vocal_window(
    source: Path, destination: Path, start: float, duration: float
) -> bool:
    completed = subprocess.run(
        [
            require_command("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{duration:.3f}",
            "-i",
            str(source),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def _vocal_active_ratio(wav: Path) -> float:
    import numpy as np
    import soundfile as sf

    samples, sample_rate = sf.read(wav, dtype="float32", always_2d=False)
    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if not len(values):
        return 0.0
    frame = max(1, round(0.1 * sample_rate))
    starts = np.arange(0, len(values), frame)
    ends = np.minimum(starts + frame, len(values))
    prefix = np.concatenate(([0.0], np.cumsum(np.square(values), dtype=np.float64)))
    rms = np.sqrt((prefix[ends] - prefix[starts]) / (ends - starts))
    peak = float(rms.max()) if len(rms) else 0.0
    if peak <= 1e-7:
        return 0.0
    return float(np.mean(rms >= max(peak * 0.05, 1e-4)))


def _run_pyshiro_lines(
    uv: str,
    worker: Path,
    wav: Path,
    lines: list[LyricLine],
    normalizer: JapaneseNormalizer,
    *,
    failure_audit: dict[str, object] | None = None,
) -> dict[str, object] | None:
    display_units = [
        normalizer.display_units(line.text, line.reading) for line in lines
    ]
    request_summary = {
        "wav": str(wav.resolve()),
        "worker": str(worker.resolve()),
        "lyric_line_count": len(lines),
        "lyrics": [
            {
                "text": line.text,
                "reading": line.reading,
                "display_unit_count": len(units),
            }
            for line, units in zip(lines, display_units)
        ],
    }
    if any(not values for values in display_units):
        if failure_audit is not None:
            failure_audit.update(
                {
                    "reason": "lyric_display_units_empty",
                    "failure_stage": "request_normalization",
                    "request": request_summary,
                }
            )
        return None
    request = {
        "wav": str(wav.resolve()),
        "readings": [
            "".join(reading for _text, reading in values) for values in display_units
        ],
        "display_units": [
            [{"text": text, "reading": reading} for text, reading in values]
            for values in display_units
        ],
    }
    try:
        result = subprocess.run(
            [uv, "run", "--project", str(worker.parent), "python", str(worker)],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        if failure_audit is not None:
            failure_audit.update(
                {
                    "reason": "pyshiro_worker_timeout",
                    "failure_stage": "worker_execution",
                    "timeout_seconds": 120,
                    "stdout_tail": _stream_tail(exc.stdout),
                    "stderr_tail": _stream_tail(exc.stderr),
                    "request": request_summary,
                }
            )
        return None
    except OSError as exc:
        if failure_audit is not None:
            failure_audit.update(
                {
                    "reason": "pyshiro_worker_start_failed",
                    "failure_stage": "worker_start",
                    "exception_type": type(exc).__name__,
                    "error": str(exc),
                    "request": request_summary,
                }
            )
        return None
    try:
        response = _parse_worker_json_output(result.stdout)
    except ValueError as exc:
        if failure_audit is not None:
            failure_audit.update(
                {
                    "reason": "pyshiro_worker_output_invalid",
                    "failure_stage": "worker_response_parse",
                    "worker_returncode": result.returncode,
                    "error": str(exc),
                    "stdout_tail": _stream_tail(result.stdout),
                    "stderr_tail": _stream_tail(result.stderr),
                    "request": request_summary,
                }
            )
        return None
    if result.returncode or response.get("ok") is not True:
        if failure_audit is not None:
            failure_audit.update(
                {
                    "reason": "pyshiro_worker_failed",
                    "failure_stage": "worker_execution",
                    "worker_returncode": result.returncode,
                    "worker_error_type": response.get("error_type"),
                    "worker_error": response.get("error"),
                    "worker_traceback": response.get("traceback"),
                    "stdout_tail": _stream_tail(result.stdout),
                    "stderr_tail": _stream_tail(result.stderr),
                    "request": request_summary,
                }
            )
        return None
    return response


def _stream_tail(value: object, limit: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode("utf-8", errors="replace")
    else:
        text = str(value)
    return text[-limit:]


def _pyshiro_likelihood_wins(
    candidate: dict[str, object],
    alternatives: list[dict[str, object]],
    config: SongIdentificationConfig,
    *,
    margin: float | None = None,
    require_preference: bool = True,
) -> tuple[bool, float, float | None]:
    required_margin = config.pyshiro_likelihood_margin if margin is None else margin
    score = float(candidate.get("likelihood_per_frame", -1e9))
    alternative_scores = [
        float(value.get("likelihood_per_frame", -1e9)) for value in alternatives
    ]
    best_alternative = max(alternative_scores) if alternative_scores else None
    accepted = score >= config.pyshiro_likelihood_floor and (
        not require_preference
        or best_alternative is None
        or score >= best_alternative + required_margin
    )
    return accepted, score, best_alternative


def _competing_line_sets(
    song: LibrarySong,
    line_ids: list[int],
    *,
    normalizer: JapaneseNormalizer,
    minimum: int = 2,
) -> list[list[LyricLine]]:
    width = len(line_ids)
    if width <= 0:
        return []
    candidates: list[tuple[int, list[LyricLine]]] = []
    target_start = line_ids[0]
    target_key = _lyric_lines_reading_key(
        [song.lines[index] for index in line_ids], normalizer
    )
    seen = {target_key}
    for start in range(0, len(song.lines) - width + 1):
        ids = list(range(start, start + width))
        if ids == line_ids:
            continue
        lines = list(song.lines[start : start + width])
        key = _lyric_lines_reading_key(lines, normalizer)
        if key in seen:
            continue
        seen.add(key)
        candidates.append((abs(start - target_start), lines))
    candidates.sort(key=lambda item: item[0])
    return [lines for _distance, lines in candidates[:minimum]]


def _lyric_lines_reading_key(
    lines: list[LyricLine], normalizer: JapaneseNormalizer
) -> str:
    return normalizer("".join(line.reading or line.text for line in lines))


def _anchor_has_continuous_support(
    anchor: LyricAnchor, anchors: tuple[LyricAnchor, ...]
) -> bool:
    if anchor.line_end - anchor.line_start >= 2:
        return True
    return any(
        other.take_index == anchor.take_index
        and (
            (
                other.cue_index == anchor.cue_index - 1
                and other.line_end == anchor.line_start
            )
            or (
                other.cue_index == anchor.cue_index + 1
                and anchor.line_end == other.line_start
            )
        )
        for other in anchors
        if other != anchor
    )


def _parse_worker_json_output(output: str) -> dict[str, object]:
    candidates = [output.strip()]
    candidates.extend(line.strip() for line in reversed(output.splitlines()))
    candidates.extend(
        output[position:].strip()
        for position, character in reversed(list(enumerate(output)))
        if character == "{"
    )
    for candidate in candidates:
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError(
        f"pySHIRO worker produced no JSON object; stdout_tail={output[-500:]!r}"
    )


def _align_match_with_pyshiro(
    job_dir: Path,
    video: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
) -> tuple[
    dict[int, tuple[tuple[int, float, float, tuple[TimedTextUnit, ...]], ...]],
    list[dict[str, object]],
]:
    manifest_path = job_dir / "vocal-candidates" / "manifest.json"
    if not manifest_path.is_file():
        return {}, [{"status": "skipped", "reason": "vocal_manifest_missing"}]
    manifests = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifests, list):
        return {}, [{"status": "skipped", "reason": "vocal_manifest_invalid"}]
    support_stem_error = _materialize_anchor_vocal_stems(
        video,
        cues,
        singing_ids,
        match,
        manifests,
        manifest_path,
        config.device,
    )
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if uv is None or not worker.is_file():
        return {}, [{"status": "skipped", "reason": "pyshiro_worker_unavailable"}]
    output_dir = job_dir / "song-alignment" / match.song.song_id
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = JapaneseNormalizer()
    timings: dict[
        int, tuple[tuple[int, float, float, tuple[TimedTextUnit, ...]], ...]
    ] = {}
    audits: list[dict[str, object]] = []
    for anchor_index, anchor in enumerate(match.anchors):
        cue_id = singing_ids[anchor.cue_index]
        cue = cues[cue_id]
        duration = cue.end - cue.start
        if duration <= 0 or duration > config.pyshiro_max_window_seconds:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "skipped",
                    "reason": "duration_outside_pyshiro_limit",
                    "duration": duration,
                }
            )
            continue
        manifest = next(
            (
                item
                for item in manifests
                if isinstance(item, dict)
                and float(item.get("start", -1)) <= cue.start
                and float(item.get("end", -1)) >= cue.end
            ),
            None,
        )
        if manifest is None:
            audit = {
                "cue_id": cue_id,
                "take_index": anchor.take_index,
                "status": "skipped",
                "reason": "no_covering_vocal_stem",
            }
            if support_stem_error is not None:
                audit["stem_error"] = support_stem_error
            audits.append(audit)
            continue
        source = manifest_path.parent / str(manifest["path"])
        wav = output_dir / f"anchor-{anchor_index:04d}.wav"
        completed = subprocess.run(
            [
                require_command("ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{cue.start - float(manifest['start']):.3f}",
                "-t",
                f"{duration:.3f}",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-y",
                str(wav),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": completed.stderr[-300:],
                }
            )
            continue
        active_ratio = _vocal_active_ratio(wav)
        if active_ratio < config.lyric_gap_vocal_active_ratio:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "rejected",
                    "reason": "insufficient_vocal_activity",
                    "vocal_active_ratio": round(active_ratio, 6),
                }
            )
            continue
        line_ids = list(range(anchor.line_start, anchor.line_end))
        lines = [match.song.lines[line_id] for line_id in line_ids]
        display_units = [
            normalizer.display_units(line.text, line.reading) for line in lines
        ]
        failure_audit: dict[str, object] = {}
        response = _run_pyshiro_lines(
            uv,
            worker,
            wav,
            lines,
            normalizer,
            failure_audit=failure_audit,
        )
        if response is None:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    **failure_audit,
                }
            )
            continue
        strong_anchor = _anchor_has_continuous_support(anchor, match.anchors)
        competing: list[dict[str, object]] = []
        competing_failures: list[dict[str, object]] = []
        if not strong_anchor:
            for lines_value in _competing_line_sets(
                match.song, line_ids, normalizer=normalizer
            ):
                competing_failure: dict[str, object] = {}
                value = _run_pyshiro_lines(
                    uv,
                    worker,
                    wav,
                    lines_value,
                    normalizer,
                    failure_audit=competing_failure,
                )
                if value is not None:
                    competing.append(value)
                else:
                    competing_failures.append(competing_failure)
        required_margin = 0.0 if strong_anchor else config.pyshiro_likelihood_margin
        accepted, likelihood, best_competing = _pyshiro_likelihood_wins(
            response,
            competing,
            config,
            margin=required_margin,
            require_preference=not strong_anchor,
        )
        if not accepted:
            audit = {
                "cue_id": cue_id,
                "take_index": anchor.take_index,
                "status": "rejected",
                "reason": "pyshiro_candidate_not_preferred",
                "likelihood_per_frame": likelihood,
                "best_competing_likelihood_per_frame": best_competing,
                "validation_policy": (
                    "strong_continuous_alt_anchor"
                    if strong_anchor
                    else "isolated_single_line_alt_anchor"
                ),
                "required_likelihood_margin": required_margin,
                "requires_competitor_preference": not strong_anchor,
            }
            if competing_failures:
                audit["competitor_failures"] = competing_failures
            audits.append(audit)
            continue
        ranges = response.get("lines")
        owned_units = response.get("units")
        if not isinstance(ranges, list) or not ranges:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": "missing_line_ranges",
                }
            )
            continue
        if (
            len(ranges) != len(line_ids)
            or not all(isinstance(value, list) and len(value) == 2 for value in ranges)
            or not isinstance(owned_units, list)
            or len(owned_units) != len(line_ids)
        ):
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": "line_range_count_mismatch",
                }
            )
            continue
        line_ranges_values = []
        validation_errors: list[dict[str, object]] = []
        for line_id, line, value, unit_values in zip(
            line_ids, lines, ranges, owned_units
        ):
            if not isinstance(unit_values, list) or not unit_values:
                validation_errors.append(
                    {
                        "issue": "empty_display_units",
                        "lyric_line_id": line_id,
                        "lyric_text": line.text,
                        "description": "pySHIRO returned no display units for the lyric line",
                    }
                )
                continue
            units = tuple(
                TimedTextUnit(
                    str(unit["text"]),
                    cue.start + float(unit["start"]),
                    cue.start + float(unit["end"]),
                )
                for unit in unit_values
                if isinstance(unit, dict)
            )
            if not units:
                validation_errors.append(
                    {
                        "issue": "empty_display_units",
                        "lyric_line_id": line_id,
                        "lyric_text": line.text,
                        "description": "pySHIRO display units contained no valid objects",
                    }
                )
                continue
            unit_errors = _lyric_unit_timeline_errors(
                units,
                line_id=line_id,
                lyric_text=line.text,
            )
            if unit_errors:
                validation_errors.extend(unit_errors)
                continue
            if line_ranges_values and units[0].start < line_ranges_values[-1][2] - 1e-3:
                previous_end = line_ranges_values[-1][2]
                validation_errors.append(
                    {
                        "issue": "line_timeline_overlap",
                        "lyric_line_id": line_id,
                        "lyric_text": line.text,
                        "line_start": round(units[0].start, 6),
                        "previous_line_end": round(previous_end, 6),
                        "overlap_seconds": round(previous_end - units[0].start, 6),
                        "description": "lyric line starts before the previous aligned line ends",
                    }
                )
                continue
            line_ranges_values.append(
                (
                    line_id,
                    cue.start + float(value[0]),
                    cue.start + float(value[1]),
                    units,
                )
            )
        if validation_errors or len(line_ranges_values) != len(line_ids):
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": "lyric_unit_ranges_invalid",
                    "lyric_line_ids": line_ids,
                    "lyric_texts": [line.text for line in lines],
                    "validation_errors": validation_errors,
                }
            )
            continue
        line_ranges = tuple(line_ranges_values)
        timings[cue_id] = line_ranges
        audits.append(
            {
                "cue_id": cue_id,
                "take_index": anchor.take_index,
                "status": "aligned",
                "start": line_ranges[0][1],
                "end": line_ranges[-1][2],
                "lyric_line_ids": line_ids,
                "lyric_languages": [
                    language_for_text(line.text, "Japanese") for line in lines
                ],
                "alignment_readings": [
                    "".join(reading for _text, reading in units)
                    for units in display_units
                ],
                "line_ranges": [
                    {
                        "lyric_line_id": line_id,
                        "start": start,
                        "end": end,
                        "units": [asdict(unit) for unit in units],
                    }
                    for line_id, start, end, units in line_ranges
                ],
                "likelihood_per_frame": likelihood,
                "best_competing_likelihood_per_frame": best_competing,
                "validation_policy": (
                    "strong_continuous_alt_anchor"
                    if strong_anchor
                    else "isolated_single_line_alt_anchor"
                ),
                "required_likelihood_margin": required_margin,
                "requires_competitor_preference": not strong_anchor,
                "vocal_active_ratio": round(active_ratio, 6),
                "phonemes": response.get("phonemes"),
            }
        )
    audit_path = output_dir / "alignment.json"
    audit_path.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return timings, audits


def _materialize_anchor_vocal_stems(
    video: Path,
    cues: list[Cue],
    cue_ids: list[int],
    match: SongMatch,
    manifests: list[dict[str, object]],
    manifest_path: Path,
    device: str,
) -> str | None:
    missing: dict[tuple[float, float], dict[str, object]] = {}
    for anchor in match.anchors:
        cue = cues[cue_ids[anchor.cue_index]]
        if not (cue.speaker_assignment or "").startswith("song_alignment_support:"):
            continue
        if any(
            float(item.get("start", -1)) <= cue.start
            and float(item.get("end", -1)) >= cue.end
            for item in manifests
            if isinstance(item, dict)
        ):
            continue
        start = round(cue.start, 3)
        end = round(cue.end, 3)
        digest = hashlib.sha256(f"{start:.3f}:{end:.3f}".encode()).hexdigest()[:12]
        missing[(start, end)] = {
            "start": start,
            "end": end,
            "path": f"speech-support-{digest}.vocals.wav",
            "source": "speech_asr_song_support",
        }
    if not missing:
        return None

    from .audio_analysis import separate_vocal_ranges

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        separate_vocal_ranges(
            video,
            [
                (start, end, manifest_path.parent / str(entry["path"]))
                for (start, end), entry in missing.items()
            ],
            device,
        )
    except Exception as exc:
        logging.warning("song alignment support vocal separation failed: %s", exc)
        return str(exc)[:500]

    manifests.extend(missing.values())
    manifests.sort(key=lambda value: (float(value["start"]), float(value["end"])))
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifests, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return None


def group_song_search_groups(
    cues: list[Cue], maximum_gap: float
) -> list[SongSearchGroup]:
    singing = [(index, cue) for index, cue in enumerate(cues) if cue.kind == "singing"]
    if not singing:
        return []
    groups: list[list[tuple[int, Cue]]] = [[singing[0]]]
    for item in singing[1:]:
        if item[1].start - groups[-1][-1][1].end <= maximum_gap:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [
        SongSearchGroup(
            group[0][1].start,
            group[-1][1].end,
            tuple(item[0] for item in group),
        )
        for group in groups
    ]


def _expanded_report_group(
    group: SongSearchGroup,
    alignment_ids: list[int],
    match: SongMatch,
    cues: list[Cue],
) -> dict[str, object]:
    cue_ids = list(
        dict.fromkeys(alignment_ids[anchor.cue_index] for anchor in match.anchors)
    )
    if not cue_ids:
        return asdict(group)
    return {
        "start": min(group.start, *(cues[cue_id].start for cue_id in cue_ids)),
        "end": max(group.end, *(cues[cue_id].end for cue_id in cue_ids)),
        "cue_ids": list(group.cue_ids),
    }


def _routed_speech_cue_ids(
    group: SongSearchGroup,
    cues: list[Cue],
    config: SongIdentificationConfig,
) -> list[int]:
    result: list[int] = []
    for cue_id, cue in enumerate(cues):
        if cue.kind != "speech":
            continue
        gap = (
            _SONG_ALIGNMENT_SUPPORT_GAP_SECONDS
            if (cue.speaker_assignment or "").startswith("song_alignment_support:")
            else config.song_search_group_gap_seconds
        )
        if cue.end >= group.start - gap and cue.start <= group.end + gap:
            result.append(cue_id)
    return result


def _refine_match_with_speech_support(
    cues: list[Cue],
    singing_ids: list[int],
    speech_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
) -> tuple[list[int], SongMatch | None]:
    support_ids = [
        cue_id
        for cue_id in speech_ids
        if (cues[cue_id].speaker_assignment or "").startswith("song_alignment_support:")
    ]
    groups_by_window: dict[str, list[int]] = {}
    for cue_id in sorted(support_ids, key=lambda value: cues[value].start):
        assignment = cues[cue_id].speaker_assignment or ""
        groups_by_window.setdefault(assignment, []).append(cue_id)
    supported_ids: list[int] = []
    for group_ids in groups_by_window.values():
        if (
            _match_candidates(
                [cues[cue_id].text for cue_id in group_ids], [match.song], config
            )
            is not None
        ):
            supported_ids.extend(group_ids)
    if not supported_ids:
        return singing_ids, None

    anchored_singing_ids = [singing_ids[anchor.cue_index] for anchor in match.anchors]
    selected_ids = sorted(
        set([*anchored_singing_ids, *supported_ids]),
        key=lambda cue_id: (cues[cue_id].start, cues[cue_id].end),
    )
    refined = _match_candidates(
        [cues[cue_id].text for cue_id in selected_ids], [match.song], config
    )
    if refined is None:
        return singing_ids, None

    alignment_ids = sorted(
        set([*singing_ids, *speech_ids]),
        key=lambda cue_id: (cues[cue_id].start, cues[cue_id].end),
    )
    alignment_positions = {cue_id: index for index, cue_id in enumerate(alignment_ids)}
    anchors = tuple(
        replace(
            anchor,
            cue_index=alignment_positions[selected_ids[anchor.cue_index]],
        )
        for anchor in refined.anchors
    )
    return alignment_ids, SongMatch(refined.song, anchors, refined.score)


def collect_ocr_candidates(
    video: Path,
    group: SongSearchGroup,
    frame_dir: Path,
    config: SongIdentificationConfig,
    ocr: Any,
) -> list[OCRCandidate]:
    timestamp = max(0.0, group.start)
    frame = _extract_frame(video, frame_dir, timestamp)
    observations: list[tuple[str, float, int, float]] = []
    for text, score in ocr.read(frame):
        normalized = _normalize_ocr_text(text)
        if normalized and score >= config.minimum_ocr_score:
            observations.append((normalized, score, 0, timestamp))
    return aggregate_ocr_observations(observations, 1)


def aggregate_ocr_observations(
    observations: list[tuple[str, float, int, float]], minimum_frames: int
) -> list[OCRCandidate]:
    clusters: list[list[tuple[str, float, int, float]]] = []
    for observation in observations:
        match = next(
            (
                cluster
                for cluster in clusters
                if SequenceMatcher(None, cluster[0][0], observation[0]).ratio() >= 0.86
            ),
            None,
        )
        if match is None:
            clusters.append([observation])
        else:
            match.append(observation)
    candidates: list[OCRCandidate] = []
    for cluster in clusters:
        frame_ids = {item[2] for item in cluster}
        if len(frame_ids) < minimum_frames:
            continue
        best = max(cluster, key=lambda item: (len(item[0]), item[1]))
        candidates.append(
            OCRCandidate(
                best[0],
                sum(item[1] for item in cluster) / len(cluster),
                len(frame_ids),
                min(item[3] for item in cluster),
                max(item[3] for item in cluster),
            )
        )
    return sorted(candidates, key=lambda item: (item.frames, item.score), reverse=True)


def apply_lyric_corrections(
    cues: list[Cue], reports: list[dict[str, object]]
) -> list[Cue]:
    replacements: dict[int, tuple[int, str]] = {}
    consumed: set[int] = set()
    for report in reports:
        if report.get("confidence") not in {"high", "medium"}:
            continue
        alignments = report.get("alignments", [])
        search_group = report.get("search_group", {})
        raw_allowed = (
            search_group.get("cue_ids", []) if isinstance(search_group, dict) else []
        )
        allowed_ids = (
            set(raw_allowed)
            if isinstance(raw_allowed, (list, tuple))
            and all(isinstance(value, int) for value in raw_allowed)
            else set()
        )
        if not isinstance(alignments, list):
            continue
        last_id = -1
        for item in alignments:
            if not isinstance(item, dict) or item.get("match") != "lyrics":
                continue
            ids = item.get("asr_cue_ids")
            text = item.get("corrected_text")
            if (
                not isinstance(ids, list)
                or not ids
                or not all(isinstance(value, int) for value in ids)
                or not isinstance(text, str)
                or not text.strip()
            ):
                continue
            if ids != list(range(ids[0], ids[-1] + 1)) or ids[0] <= last_id:
                continue
            if ids[0] < 0 or ids[-1] >= len(cues) or any(i in consumed for i in ids):
                continue
            if not set(ids) <= allowed_ids:
                continue
            replacements[ids[0]] = (ids[-1], " ".join(text.split()))
            consumed.update(ids)
            last_id = ids[-1]
    output: list[Cue] = []
    index = 0
    while index < len(cues):
        replacement = replacements.get(index)
        if replacement is None:
            output.append(cues[index])
            index += 1
            continue
        end_id, text = replacement
        first, last = cues[index], cues[end_id]
        output.append(
            Cue(
                first.start,
                last.end,
                text,
                first.speaker,
                "singing",
                language=language_for_text(text, "Japanese"),
            )
        )
        index = end_id + 1
    if replacements:
        logging.info(
            "applied verified lyrics to %d ASR cue groups (%d source cues)",
            len(replacements),
            len(consumed),
        )
    return output


class _EasyOCR:
    def __init__(self, config: SongIdentificationConfig):
        self.process: subprocess.Popen[str] | None = None
        worker = Path(__file__).resolve().parents[2] / "tools/song_ocr/worker.py"
        if not worker.is_file():
            raise RuntimeError(f"song OCR worker is unavailable at {worker}")
        self.process = subprocess.Popen(
            [sys.executable, str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._exchange(
            {
                "device": config.device,
            }
        )

    def read(self, path: Path) -> list[tuple[str, float]]:
        response = self._exchange({"path": str(path.resolve())})
        values = response.get("values", [])
        if not isinstance(values, list):
            raise RuntimeError("song OCR worker returned malformed values")
        return [
            (str(item[0]), float(item[1]))
            for item in values
            if isinstance(item, list) and len(item) == 2
        ]

    def _exchange(self, payload: dict[str, object]) -> dict[str, object]:
        if (
            self.process is None
            or self.process.stdin is None
            or self.process.stdout is None
        ):
            raise RuntimeError("song OCR worker pipes are unavailable")
        self.process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            code = self.process.poll()
            raise RuntimeError(f"song OCR worker exited unexpectedly ({code})")
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(str(response.get("error", "song OCR worker failed")))
        return response

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __del__(self) -> None:
        self.close()


def _extract_frame(video: Path, directory: Path, timestamp: float) -> Path:
    cache_key = hashlib.sha256(f"single:{timestamp:.3f}".encode()).hexdigest()[:12]
    directory = directory / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "frame.jpg"
    if output.is_file():
        return output
    ffmpeg = require_command("ffmpeg")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-q:v",
            "3",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"song OCR frame extraction failed: {completed.stderr[-500:]}"
        )
    if not output.is_file():
        raise RuntimeError("song OCR frame extraction produced no frame")
    return output


class _WebTools:
    def __init__(self, config: SongIdentificationConfig):
        self.config = config
        self.worker_project = Path(config.search_worker_project).resolve()
        self.allowed_urls: set[str] = set()
        self.queries: list[str] = []
        self.errors: list[str] = []

    def search(self, query: str) -> str:
        if not query.strip() or len(query) > 300:
            return json.dumps({"error": "invalid query"})
        self.queries.append(query)
        response = self._worker(
            {
                "action": "search",
                "query": query,
                "limit": self.config.max_search_results,
            }
        )
        results = response.get("results", [])
        for error in response.get("errors", []):
            self.errors.append(str(error)[:500])
        if not results:
            return json.dumps({"error": "all search backends failed"})
        compact = []
        for item in results:
            url = str(item.get("href") or item.get("url") or "")
            if _supported_lyrics_url(url):
                self.allowed_urls.add(url)
                compact.append(
                    {
                        "title": item.get("title"),
                        "url": url,
                        "snippet": item.get("body"),
                    }
                )
        return json.dumps(compact, ensure_ascii=False)

    def fetch_lyrics(self, url: str) -> tuple[str, str, list[str]] | None:
        if url not in self.allowed_urls or not _supported_lyrics_url(url):
            return None
        response = self._worker({"action": "fetch_lyrics", "url": url})
        if response.get("error"):
            raise RuntimeError(str(response["error"])[:500])
        title = response.get("title")
        artist = response.get("artist")
        lines = response.get("lines")
        if (
            not isinstance(title, str)
            or not isinstance(artist, str)
            or not isinstance(lines, list)
            or len(lines) < 3
            or not all(isinstance(line, str) and line.strip() for line in lines)
        ):
            return None
        return title.strip(), artist.strip(), [line.strip() for line in lines]

    def _worker(self, payload: dict[str, object]) -> dict[str, object]:
        uv = shutil.which("uv")
        worker = self.worker_project / "worker.py"
        if uv is None or not worker.is_file():
            raise RuntimeError(f"song search worker is unavailable at {worker}")
        completed = subprocess.run(
            [uv, "run", "--project", str(self.worker_project), "python", str(worker)],
            input=json.dumps(payload, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr[-500:] or "song search worker failed")
        response = json.loads(completed.stdout)
        if not isinstance(response, dict):
            raise RuntimeError("song search worker returned malformed JSON")
        return response


def _supported_lyrics_url(url: str) -> bool:
    if not _public_http_url(url):
        return False
    parsed = urlsplit(url)
    hostname = (parsed.hostname or "").casefold()
    path = parsed.path
    if hostname == "utaten.com" or hostname.endswith(".utaten.com"):
        return re.fullmatch(r"/lyric/[^/]+/?", path) is not None
    if hostname == "uta-net.com" or hostname.endswith(".uta-net.com"):
        return re.fullmatch(r"/(?:movie|song)/\d+/?", path) is not None
    if hostname == "oricon.co.jp" or hostname.endswith(".oricon.co.jp"):
        return bool(
            re.fullmatch(r"/prof/\d+/lyrics/\d+/?", path)
            or (
                path == "/php/lyrics/LyricsDisp.php"
                and re.search(r"(?:^|&)music=\d+(?:&|$)", parsed.query)
            )
        )
    if hostname == "awa.fm" or hostname.endswith(".awa.fm"):
        return re.fullmatch(r"/track/[^/]+/?", path) is not None
    return False


def _public_http_url(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    try:
        literal = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        literal = None
    if literal is not None:
        return _public_address(literal)
    try:
        addresses = socket.getaddrinfo(parsed.hostname, parsed.port or 443)
    except socket.gaierror:
        return False
    return all(
        _public_address(address) or address in ipaddress.ip_network("198.18.0.0/15")
        for address in (ipaddress.ip_address(item[4][0]) for item in addresses)
    )


def _public_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return not (
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_reserved
    )


def _normalize_ocr_text(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _signature(
    video: Path,
    cues: list[Cue],
    metadata: dict[str, object],
    config: SongIdentificationConfig,
    *,
    source_maximum_units: float | None = None,
) -> str:
    stat = video.stat()
    library_path = Path(config.lyrics_library_path).resolve()
    library = LyricsLibrary(library_path)
    try:
        library_digest = library.canonical_digest()
    finally:
        library.close()
    payload = {
        "version": _PROMPT_VERSION,
        "video": [stat.st_size, stat.st_mtime_ns],
        "cues": [asdict(cue) for cue in cues],
        "metadata": {
            key: metadata[key] for key in _STABLE_METADATA_KEYS if key in metadata
        },
        "config": asdict(config),
        "lyrics_library": library_digest,
        "source_maximum_units": source_maximum_units,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode()
    ).hexdigest()


def _ocr_signature(
    video: Path,
    groups: list[SongSearchGroup],
    metadata: dict[str, object],
    config: SongIdentificationConfig,
) -> str:
    stat = video.stat()
    payload = {
        "version": _CACHE_VERSION,
        "video": [stat.st_size, stat.st_mtime_ns],
        "search_groups": [asdict(group) for group in groups],
        "music_title_mode": _title_uses_music_mode(metadata),
        "ocr": {
            key: value
            for key, value in asdict(config).items()
            if key
            in {
                "device",
                "minimum_ocr_score",
            }
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _load_ocr_cache(
    path: Path, signature: str, group_count: int
) -> list[list[OCRCandidate]] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("version") != _CACHE_VERSION
            or value.get("signature") != signature
        ):
            return None
        groups = value["candidate_sets"]
        if not isinstance(groups, list) or len(groups) != group_count:
            return None
        candidates = [
            [OCRCandidate(**item) for item in group]
            for group in groups
            if isinstance(group, list)
        ]
        if len(candidates) != group_count:
            return None
        logging.info("using song OCR cache with %d search groups", len(candidates))
        return candidates
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning("ignoring unreadable song OCR cache %s: %s", path, exc)
        return None


def _write_ocr_cache(
    path: Path, signature: str, candidate_sets: list[list[OCRCandidate]]
) -> None:
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "candidate_sets": [
            [asdict(candidate) for candidate in candidates]
            for candidates in candidate_sets
        ],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_cache(
    path: Path, signature: str, cues: list[Cue]
) -> SongIdentificationResult | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("version") != _CACHE_VERSION
            or value.get("signature") != signature
        ):
            return None
        raw_corrected = value["corrected_cues"]
        reports = value["reports"]
        if not isinstance(raw_corrected, list) or not isinstance(reports, list):
            return None
        corrected = [cue_from_mapping(item) for item in raw_corrected]
        spans = tuple(
            VerifiedLyricSpan(
                **{
                    **item,
                    "lyric_line_ids": tuple(item.get("lyric_line_ids", ())),
                }
            )
            for item in value.get("verified_lyric_spans", [])
            if isinstance(item, dict)
        )
        logging.info("using song identification cache with %d reports", len(reports))
        return SongIdentificationResult(corrected, reports, spans)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning(
            "ignoring unreadable song identification cache %s: %s", path, exc
        )
        return None
