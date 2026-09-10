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
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from itertools import pairwise
from math import ceil
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .cache import StageCache, CacheStore, config_snapshot, restore_config
from .commands import require_command
from .config import SongIdentificationConfig
from .llm_response import (
    finish_reason,
    parse_json_object,
    structured_request_body,
    structured_response_content,
)
from .lyrics_library import LibrarySong, LyricLine, LyricsLibrary
from .lyrics_matching import JapaneseNormalizer, LyricAnchor, SongMatch, match_song
from .prompt_budget import batch_prompt_items
from .prompt_templates import prompt_system, render_user_prompt
from .source_language import language_for_text
from .subtitles import Cue, TimedTextUnit, cue_from_mapping, text_display_width

_PROMPT_VERSION = 8
_OCR_TITLE_PROMPT = "select-ocr-song-titles.md"
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
    bounds: tuple[float, float, float, float]


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
    select_ocr_titles: Callable[[list[list[OCRCandidate]]], list[str | None]]
    | None = None,
) -> SongIdentificationResult:
    stage = CacheStore(job_dir / "cache.sqlite3").stage("song_identification", lambda: {
        "config": config_snapshot(config), "cues": [asdict(cue) for cue in cues],
        "metadata": metadata, "source_maximum_units": source_maximum_units,
    })
    config = restore_config(config, stage.plan["config"])
    cues = [cue_from_mapping(value) for value in stage.plan["cues"]]
    metadata = stage.plan["metadata"]
    source_maximum_units = stage.plan["source_maximum_units"]
    cached = stage.get("__result__")
    if cached is not None:
        return decode_song_result(cached)
    groups = group_song_search_groups(cues, config.song_search_group_gap_seconds)
    if not config.enabled or not groups:
        result = SongIdentificationResult(cues, [])
        stage.finish(result)
        return result
    candidate_sets = []
    ocr_failures = []
    ocr = None
    worker_error = None
    try:
        for index, group in enumerate(groups):
            key = f"ocr:{index}"
            saved = stage.get(key)
            if saved is None:
                reason = None
                candidates = []
                if _title_uses_music_mode(metadata):
                    try:
                        if worker_error is not None:
                            raise RuntimeError(worker_error)
                        if ocr is None:
                            try:
                                ocr = _EasyOCR(config)
                            except Exception as exc:
                                worker_error = f"{type(exc).__name__}: {exc}"
                                raise
                        candidates = collect_ocr_candidates(
                            video, group, job_dir / "song-ocr-frames" / f"{index:03d}", config, ocr,
                        )
                    except Exception as exc:
                        reason = f"{type(exc).__name__}: {exc}"
                        logging.warning("song OCR unavailable for group %d: %s", index, reason)
                saved = {"candidates": [asdict(candidate) for candidate in candidates], "error": reason}
                stage.put(key, saved, source="ocr", reason=reason)
            candidate_sets.append([OCRCandidate(**item) for item in saved["candidates"]])
            ocr_failures.append(saved["error"])
    finally:
        if ocr is not None:
            ocr.close()

    selection = stage.get("ocr_titles")
    if selection is None:
        reason = None
        titles = [None for _ in groups]
        if select_ocr_titles is not None and any(candidate_sets):
            try:
                titles = select_ocr_titles(candidate_sets)
                if len(titles) != len(groups):
                    raise ValueError("OCR title selector changed the search-group count")
            except Exception as exc:
                reason = f"{type(exc).__name__}: {exc}"
                logging.warning("song-title selection downgraded: %s", reason)
                titles = [None for _ in groups]
        selection = {"titles": titles, "error": reason}
        stage.put("ocr_titles", selection, source="llm", reason=reason)
    ocr_titles = selection["titles"]
    ocr_title_selection_failed = selection["error"] is not None

    reports: list[dict[str, object]] = []
    verified_spans: list[VerifiedLyricSpan] = []
    lyric_replacements: dict[int, list[Cue]] = {}
    discarded_song_region_cues: set[int] = set()
    confirmed_song_names: set[str] = set()
    library = LyricsLibrary(Path(config.lyrics_library_path).resolve())
    try:
        for index, group in enumerate(groups):
            saved = stage.get(f"group:{index}")
            if saved is not None:
                reports.extend(saved["reports"])
                verified_spans.extend(VerifiedLyricSpan(**value) for value in saved["spans"])
                lyric_replacements.update({int(key): [cue_from_mapping(item) for item in values]
                                           for key, values in saved["replacements"].items()})
                discarded_song_region_cues.update(saved["discarded"])
                confirmed_song_names.update(saved["names"])
                continue
            with stage.attempt(f"group:{index}"):
                report_start, span_start = len(reports), len(verified_spans)
                prior_replacements = set(lyric_replacements)
                prior_discarded = set(discarded_song_region_cues)
                candidates = candidate_sets[index]
                ocr_title = ocr_titles[index]
                group_singing_ids = [
                    cue_id for cue_id in group.cue_ids if cues[cue_id].kind == "singing"
                ]
                speech_ids = _routed_speech_cue_ids(group, cues, config)
                library_songs = library.songs()
                logging.info(
                    "song search group %d/%d range=%.3f-%.3fs alt_cues=%d "
                    "ocr_candidates=%d",
                    index + 1,
                    len(groups),
                    group.start,
                    group.end,
                    len(group_singing_ids),
                    len(candidates),
                )
                pending_ids = set(group_singing_ids)
                while pending_ids:
                    singing_ids = sorted(pending_ids)
                    hypotheses = [cues[cue_id].text for cue_id in singing_ids]
                    route_cues = {
                        "alt_cue_ids": singing_ids,
                        "speech_cue_ids": speech_ids,
                    }
                    queries = _build_lyric_search_queries(
                        hypotheses,
                        ocr_title,
                        confirmed_song_names,
                    )
                    policy = _web_search_policy(
                        [cues[cue_id] for cue_id in singing_ids],
                        metadata,
                        has_trusted_ocr=any(
                            item.get("source") == "ocr" for item in queries
                        ),
                    )
                    web_search_audit: dict[str, object] = {
                        **policy,
                        "queries": [],
                        "fetches": [],
                        "worker_errors": [],
                    }
                    resolved: (
                        tuple[
                            str,
                            SongMatch,
                            list[int],
                            dict[int, list[Cue]],
                            list[dict[str, object]],
                            list[dict[str, object]],
                        ]
                        | None
                    ) = None
                    attempted_song_ids: set[str] = set()
                    for song in _songs_matching_ocr_title(library_songs, ocr_title):
                        validated = _validate_ocr_title_song(
                            job_dir,
                            video,
                            cues,
                            singing_ids,
                            song,
                            config,
                        )
                        if validated[2]:
                            resolved = ("local_library", *validated)
                            break
                    if resolved is None:
                        local_matches = _rank_candidate_matches(
                            hypotheses, library_songs, config
                        )
                    else:
                        local_matches = []
                    for candidate_match in local_matches:
                        attempted_song_ids.add(candidate_match.song.song_id)
                        validated = _validate_song_match(
                            job_dir,
                            video,
                            cues,
                            singing_ids,
                            candidate_match,
                            config,
                            source_maximum_units,
                        )
                        (
                            candidate_match,
                            alignment_ids,
                            replacements,
                            alignments,
                            audits,
                        ) = validated
                        if replacements:
                            resolved = (
                                "local_library",
                                candidate_match,
                                alignment_ids,
                                replacements,
                                alignments,
                                audits,
                            )
                            break

                    if resolved is None and policy["eligible"]:
                        logging.info(
                            "song search group %d/%d starting web search mode=%s "
                            "alt_coverage=%.3fs queries=%d",
                            index + 1,
                            len(groups),
                            policy["mode"],
                            policy["alt_coverage_seconds"],
                            len(queries),
                        )
                        for query in queries[:_MAX_WEB_SEARCH_QUERIES]:
                            fetched, audit = _search_canonical_lyrics(
                                hypotheses, [query], config
                            )
                            for key in ("queries", "fetches", "worker_errors"):
                                values = web_search_audit[key]
                                assert isinstance(values, list)
                                values.extend(audit.get(key, []))
                            for song in _songs_matching_ocr_title(fetched, ocr_title):
                                validated = _validate_ocr_title_song(
                                    job_dir,
                                    video,
                                    cues,
                                    singing_ids,
                                    song,
                                    config,
                                )
                                if validated[2]:
                                    resolved = ("web", *validated)
                                    break
                            if resolved is not None:
                                break
                            fetched_matches = _rank_candidate_matches(
                                hypotheses, fetched, config
                            )
                            for candidate_match in fetched_matches:
                                if candidate_match.song.song_id in attempted_song_ids:
                                    continue
                                attempted_song_ids.add(candidate_match.song.song_id)
                                validated = _validate_song_match(
                                    job_dir,
                                    video,
                                    cues,
                                    singing_ids,
                                    candidate_match,
                                    config,
                                    source_maximum_units,
                                )
                                (
                                    candidate_match,
                                    alignment_ids,
                                    replacements,
                                    alignments,
                                    audits,
                                ) = validated
                                if replacements:
                                    resolved = (
                                        "web",
                                        candidate_match,
                                        alignment_ids,
                                        replacements,
                                        alignments,
                                        audits,
                                    )
                                    break
                            if resolved is not None:
                                break
                    elif resolved is None:
                        logging.info(
                            "song search group %d/%d skipped web search reason=%s "
                            "alt_coverage=%.3fs",
                            index + 1,
                            len(groups),
                            policy["decision_reason"],
                            policy["alt_coverage_seconds"],
                        )
                    if resolved is None:
                        discarded_song_region_cues.update(singing_ids)
                        reports.append(
                            {
                                "song": None,
                                "artist": None,
                                "confidence": "low",
                                "evidence": ["no_acoustically_verified_lyric_match"],
                                "sources": [],
                                "alignments": [],
                                "search_group": asdict(group),
                                "route_cues": route_cues,
                                "ocr_candidates": [asdict(item) for item in candidates],
                                "ocr_song_title": ocr_title,
                                "web_search": web_search_audit,
                            }
                        )
                        break

                    (
                        provenance,
                        match,
                        alignment_ids,
                        replacements,
                        alignments,
                        pyshiro_audit,
                    ) = resolved
                    if provenance == "web":
                        stored = library.store_canonical_song(
                            title=match.song.title,
                            artist=match.song.artist,
                            aliases=list(match.song.aliases),
                            source_url=match.song.source_url,
                            lines=[(line.text, line.reading) for line in match.song.lines],
                        )
                        match = SongMatch(stored, match.anchors, match.score)
                        library_songs.append(stored)
                    song = match.song
                    confirmed_song_names.update(
                        normalized
                        for name in (song.title, *song.aliases)
                        if (normalized := _normalize_identity_text(name))
                    )
                    logging.info(
                        "song search group %d/%d verified title=%r artist=%r "
                        "source=%s score=%.3f",
                        index + 1,
                        len(groups),
                        song.title,
                        song.artist,
                        provenance,
                        match.score,
                    )
                    for cue_id, values in replacements.items():
                        lyric_replacements.setdefault(cue_id, []).extend(values)
                        lyric_replacements[cue_id].sort(
                            key=lambda value: (value.start, value.end)
                        )
                    verified_spans.extend(
                        _verified_spans_for_match(
                            match, replacements, alignments, pyshiro_audit
                        )
                    )
                    consumed_ids = {
                        alignment_ids[anchor.cue_index]
                        for anchor in match.anchors
                        if 0 <= anchor.cue_index < len(alignment_ids)
                    }
                    consumed_ids.update(
                        cue_id for cue_id in replacements if cue_id in pending_ids
                    )
                    # Do not rematch holes inside the selected continuous episode
                    # as independent paths over the same audio.
                    if consumed_ids:
                        start = min(cues[cue_id].start for cue_id in consumed_ids)
                        end = max(cues[cue_id].end for cue_id in consumed_ids)
                        consumed_ids.update(
                            cue_id for cue_id in singing_ids
                            if start <= cues[cue_id].start < end
                        )
                    discarded_song_region_cues.update(consumed_ids - replacements.keys())
                    pending_ids.difference_update(consumed_ids)
                    report_group = _expanded_report_group(group, alignment_ids, match, cues)
                    reports.append(
                        {
                            "song_id": song.song_id,
                            "song": song.title,
                            "artist": song.artist,
                            "confidence": "high" if match.score >= 0.68 else "medium",
                            "evidence": [provenance, "acoustically_verified_lyrics"],
                            "sources": [song.source_url],
                            "score": round(match.score, 6),
                            "alignments": alignments,
                            "pyshiro": pyshiro_audit,
                            "search_group": report_group,
                            "route_cues": route_cues,
                            "ocr_candidates": [asdict(item) for item in candidates],
                            "ocr_song_title": ocr_title,
                            "web_search": web_search_audit,
                        }
                    )
                group_reports = reports[report_start:]
                errors = _song_errors(group_reports)
                if ocr_failures[index]:
                    errors.append(ocr_failures[index])
                if ocr_title_selection_failed:
                    errors.append("ocr_title_selection_failed")
                stage.put(f"group:{index}", {
                    "reports": group_reports, "spans": [asdict(value) for value in verified_spans[span_start:]],
                    "replacements": {key: [asdict(cue) for cue in values] for key, values in lyric_replacements.items()
                                     if key not in prior_replacements},
                    "discarded": sorted(discarded_song_region_cues - prior_discarded),
                    "names": sorted(confirmed_song_names),
                }, source="song_identification", reason="; ".join(errors) or None)
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
    result = SongIdentificationResult(corrected, reports, tuple(verified_spans))
    stage.finish(result)
    (job_dir / "song-identification.json").write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _song_errors(value: object) -> list[str]:
    errors = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"error", "tool_errors", "worker_errors"} and child:
                errors.append(str(child))
            elif isinstance(child, (dict, list)):
                errors.extend(_song_errors(child))
    elif isinstance(value, list):
        for child in value:
            errors.extend(_song_errors(child))
    return errors


def decode_song_result(value: dict) -> SongIdentificationResult:
    return SongIdentificationResult(
        [cue_from_mapping(item) for item in value["corrected_cues"]], value["reports"],
        tuple(VerifiedLyricSpan(**item) for item in value.get("verified_lyric_spans", [])),
    )


def arbitrate_verified_lyrics(
    cues: list[Cue], verified_spans: list[VerifiedLyricSpan]
) -> tuple[list[Cue], list[dict[str, object]]]:
    spans: list[VerifiedLyricSpan] = []
    for span in sorted(verified_spans, key=lambda value: (value.start, value.end)):
        if span.end <= span.start:
            continue
        previous = spans[-1] if spans else None
        if (
            previous is not None
            and previous.song_id == span.song_id
            and span.start - previous.end <= 2.0
            and previous.lyric_line_ids
            and span.lyric_line_ids
            and min(span.lyric_line_ids) <= max(previous.lyric_line_ids) + 1
            and max(span.lyric_line_ids) >= max(previous.lyric_line_ids)
        ):
            spans[-1] = replace(
                previous,
                end=max(previous.end, span.end),
                lyric_line_ids=tuple(sorted({*previous.lyric_line_ids, *span.lyric_line_ids})),
            )
        else:
            spans.append(span)
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
            audit.append(
                {
                    "start": cue.start,
                    "end": cue.end,
                    "action": "discarded_without_units",
                }
            )
            continue
        runs: list[list[TimedTextUnit]] = []
        current: list[TimedTextUnit] = []
        for unit in units:
            blocked = any(
                unit.start < span.end and unit.end > span.start
                or unit.start == unit.end and span.start <= unit.start < span.end
                for span in overlapping
            )
            if blocked:
                if current:
                    runs.append(current)
                    current = []
                continue
            if current and unit.start - current[-1].end > 0.25:
                runs.append(current)
                current = []
            current.append(unit)
        if current:
            runs.append(current)
        retained_count = sum(len(run) for run in runs)
        if not runs:
            audit.append(
                {
                    "start": cue.start,
                    "end": cue.end,
                    "action": "discarded_covered_units",
                    "retained_units": 0,
                }
            )
            continue
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
                "retained_units": retained_count,
                "removed_units": len(units) - retained_count,
            }
        )
    result.sort(key=lambda cue: (cue.start, cue.end, cue.kind, cue.speaker or ""))
    return result, audit


def translate_aligned_song_lyrics(
    result: SongIdentificationResult,
    config: SongIdentificationConfig,
    translate_lyrics: Callable[..., object],
    translation_context: dict[str, object] | None = None,
    lyrics_translation_model: str | None = None,
    *, cache_path: Path,
) -> SongIdentificationResult:
    if not result.reports:
        return result
    stage = CacheStore(cache_path).stage("lyrics_translation", lambda: {
        "result": asdict(result), "config": config_snapshot(config),
        "context": translation_context or {}, "model": lyrics_translation_model,
    })
    result = decode_song_result(stage.plan["result"])
    config = restore_config(config, stage.plan["config"])
    translation_context, lyrics_translation_model = stage.plan["context"], stage.plan["model"]
    saved = stage.get("__result__")
    if saved is not None:
        return decode_song_result(saved)
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
                cache=stage,
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
    translated_result = SongIdentificationResult(corrected, result.reports, result.verified_lyric_spans)
    if stage is not None:
        stage.finish(translated_result)
    return translated_result


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


def _rank_candidate_matches(
    hypotheses: list[str], songs: list[LibrarySong], config: SongIdentificationConfig
) -> list[SongMatch]:
    matches = [
        match
        for song in songs
        if (match := _match_candidates(hypotheses, [song], config)) is not None
    ]
    return sorted(matches, key=lambda value: value.score, reverse=True)


def _songs_matching_ocr_title(
    songs: list[LibrarySong], ocr_title: str | None
) -> list[LibrarySong]:
    normalized_title = _normalize_identity_text(ocr_title or "")
    if not normalized_title:
        return []
    matches: list[LibrarySong] = []
    for song in songs:
        names = {
            normalized
            for value in (song.title, *song.aliases)
            if (normalized := _normalize_identity_text(value))
        }
        if not any(
            normalized_title == name
            or (
                min(len(normalized_title), len(name)) >= 4
                and (normalized_title in name or name in normalized_title)
            )
            for name in names
        ):
            continue
        matches.append(song)
    return matches


def _validate_ocr_title_song(
    job_dir: Path,
    video: Path,
    cues: list[Cue],
    singing_ids: list[int],
    song: LibrarySong,
    config: SongIdentificationConfig,
) -> tuple[
    SongMatch,
    list[int],
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    empty_match = SongMatch(song, (), 1.0)
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if uv is None or not worker.is_file() or not song.lines:
        return (
            empty_match,
            singing_ids,
            {},
            [],
            [
                {
                    "status": "skipped",
                    "route": "trusted_ocr_title_direct",
                    "reason": "pyshiro_worker_unavailable"
                    if uv is None or not worker.is_file()
                    else "canonical_lyrics_empty",
                }
            ],
        )

    probes: list[dict[str, object]] = []
    for cue_index, cue_id in enumerate(singing_ids):
        cue = cues[cue_id]
        duration = cue.end - cue.start
        if duration < 2:
            continue
        part_count = max(1, ceil(duration / config.pyshiro_max_window_seconds))
        for part in range(part_count):
            probes.append(
                {
                    "start": cue.start + duration * part / part_count,
                    "end": cue.start + duration * (part + 1) / part_count,
                    "cue_id": cue_id,
                    "cue_index": cue_index,
                    "take_index": 0,
                    "part": part,
                    "part_count": part_count,
                    "anchor_sides": 0,
                    "require_preference": False,
                    "required_margin": 0.0,
                    "validation_policy": "trusted_ocr_title_absolute_likelihood",
                    "success_status": "aligned",
                }
            )
    if not probes:
        return empty_match, singing_ids, {}, [], []

    output_dir = job_dir / "song-alignment" / song.song_id / "ocr-title-direct"
    output_dir.mkdir(parents=True, exist_ok=True)
    _materialize_verification_stems(
        video,
        probes,
        output_dir,
        config.vocal_separation_device,
        job_dir / "vocal-candidates" / "manifest.json",
    )

    normalizer = JapaneseNormalizer()
    anchors: list[LyricAnchor] = []
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    next_line = 0
    started = False
    previous_end = -1.0
    for probe in probes:
        if started and float(probe["start"]) < previous_end - 1e-3:
            continue
        duration = float(probe["end"]) - float(probe["start"])
        line_ids = (
            _select_suffix_neighbor_lines(
                song,
                next_line,
                "",
                duration,
                config.lyric_neighbor_max_lines,
                normalizer,
            )
            if started
            else list(range(min(2, len(song.lines))))
        )
        if not line_ids:
            break
        probe["line_ids"] = line_ids
        probe["route"] = (
            "trusted_ocr_title_sequence" if started else "trusted_ocr_title_direct"
        )
        match = SongMatch(song, (), 1.0)
        recovered, recovered_alignments, audit = _verify_lyric_range(
            probe, cues, match, config, uv, worker, normalizer
        )
        audits.append(audit)
        if not recovered:
            continue

        cue_id = int(probe["cue_id"])
        cue_index = int(probe["cue_index"])
        replacements.setdefault(cue_id, []).extend(recovered)
        alignments.extend(recovered_alignments)
        anchors.append(LyricAnchor(cue_index, line_ids[0], line_ids[-1] + 1, 1.0, 0))
        started = True
        next_line = line_ids[-1] + 1
        previous_end = recovered[-1].end
        if next_line >= len(song.lines):
            break

    for values in replacements.values():
        values.sort(key=lambda value: (value.start, value.end))
    return (
        SongMatch(song, tuple(anchors), 1.0),
        singing_ids,
        replacements,
        alignments,
        audits,
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
            normalized_lyrics = "\n".join(
                unicodedata.normalize("NFKC", line).strip() for line in lines
            )
            source_hash = hashlib.sha256(normalized_lyrics.encode()).hexdigest()
            songs[source_hash] = LibrarySong(
                source_hash[:24],
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
    ocr_title: str | None,
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

    if ocr_title:
        normalized_title = _normalize_identity_text(ocr_title)
        if not any(name in normalized_title for name in confirmed_song_names):
            query = make_query(
                "ocr",
                ocr_title,
                "llm_selected_current_song_title",
                minimum_length=2,
            )
        else:
            query = None
        if query is not None:
            queries.append(query)
    return queries[:_MAX_WEB_SEARCH_QUERIES]


def select_ocr_song_titles(
    candidate_sets: list[list[OCRCandidate]],
    *,
    video_title: str,
    request: Callable[[dict[str, object]], dict[str, object]],
    model: str,
    json_mode: bool,
    thinking: str | None,
    context_size: int,
) -> list[str | None]:
    groups: list[dict[str, object]] = []
    for group_id, candidates in enumerate(candidate_sets):
        lines = [
            {
                "text": candidate.text,
                "confidence": round(candidate.score, 4),
                "bounds": [round(value, 1) for value in candidate.bounds],
            }
            for candidate in sorted(
                candidates,
                key=lambda item: (item.bounds[1], item.bounds[0], -item.score),
            )
        ]
        groups.append({"group_id": group_id, "ocr_lines": lines})

    def render(groups_batch: Sequence[dict[str, object]]) -> str:
        return "\n\n".join(
            (
                prompt_system(_OCR_TITLE_PROMPT),
                render_user_prompt(
                    _OCR_TITLE_PROMPT,
                    VIDEO_TITLE=video_title,
                    OCR_GROUPS=json.dumps(
                        groups_batch,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                ),
            )
        )

    def output_tokens(group_count: int) -> int:
        return max(512, min(4096, group_count * 48))

    # OCR JSON is dominated by coordinates and punctuation, which tokenize much
    # more densely than prose. One character per token is a conservative bound.
    batches = batch_prompt_items(
        groups,
        render_prompt=render,
        context_size=context_size,
        max_output_tokens=output_tokens,
        estimate_tokens=len,
    )
    logging.info(
        "split %d OCR search groups into %d prompt batch(es)",
        len(groups),
        len(batches),
    )
    titles: list[str | None] = [None for _ in candidate_sets]
    for batch in batches:
        prompt = render_user_prompt(
            _OCR_TITLE_PROMPT,
            VIDEO_TITLE=video_title,
            OCR_GROUPS=json.dumps(batch, ensure_ascii=False, separators=(",", ":")),
        )
        body = structured_request_body(
            model=model,
            prompt_name=_OCR_TITLE_PROMPT,
            prompt=prompt,
            max_tokens=output_tokens(len(batch)),
            temperature=0,
            json_mode=json_mode,
            thinking=thinking,
        )
        response = request(body)
        content = structured_response_content(response, finish_reason=finish_reason)
        parsed = parse_json_object(content)
        values = parsed.get("groups")
        if not isinstance(values, list):
            raise TypeError("OCR song-title response has no groups array")
        expected = {int(group["group_id"]) for group in batch}
        seen: set[int] = set()
        for item in values:
            if not isinstance(item, dict):
                raise TypeError("OCR song-title group must be an object")
            group_id = item.get("group_id")
            title = item.get("song_title")
            if not isinstance(group_id, int) or group_id not in expected:
                raise ValueError(f"invalid OCR song-title group_id: {group_id!r}")
            if group_id in seen:
                raise ValueError(f"duplicate OCR song-title group_id: {group_id}")
            if title is not None and not isinstance(title, str):
                raise TypeError("OCR song title must be text or null")
            seen.add(group_id)
            titles[group_id] = (
                title.strip() if isinstance(title, str) and title.strip() else None
            )
        if seen != expected:
            raise ValueError("OCR song-title response omitted search groups")
    return titles


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


def _normalize_identity_text(text: str) -> str:
    return re.sub(
        r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+",
        "",
        unicodedata.normalize("NFKC", text).casefold(),
    )


def _ensure_song_translations(
    library: LyricsLibrary,
    song: LibrarySong,
    translate_lyrics: Callable[..., object],
    translation_context: dict[str, object],
    model: str | None,
    *, cache: StageCache,
) -> LibrarySong:
    song_data = cache.remember("song:" + song.song_id, lambda: asdict(song))
    from .lyrics_library import LyricLine
    song = LibrarySong(**{**song_data, "lines": tuple(LyricLine(**line) for line in song_data["lines"])})
    cached = cache.get(song.song_id)
    if cached is not None:
        return replace(song, lines=tuple(replace(line, translation=cached[str(line.line_no)]) for line in song.lines))
    song = replace(song, lines=tuple(
        replace(line, translation=None, translation_source=None)
        if line.translation_source in {"llm", "local_mt"} else line for line in song.lines
    ))
    kept = {line.line_no: line.translation for line in song.lines if line.translation}
    source = "authored"
    translations = kept
    if len(kept) != len(song.lines):
        try:
            translated = translate_lyrics(
                song.title, song.artist, [line.text for line in song.lines],
                translation_context=translation_context,
            )
            if isinstance(translated, tuple) and len(translated) == 2 and translated[1] in {"llm", "local_mt"}:
                translations, source = translated
            elif isinstance(translated, dict):
                translations, source = translated, "llm"
            else:
                raise TypeError("lyrics translator returned an invalid result")
            translations = {**translations, **kept}
            if any(not isinstance(translations.get(line.line_no), str) or not translations[line.line_no].strip()
                   for line in song.lines):
                raise ValueError("lyrics translator did not return every canonical line")
        except BaseException as exc:
            cache.failed(song.song_id, exc)
            raise
    cache.put(song.song_id, translations, source=source,
              reason="local_machine_translation" if source == "local_mt" else None)
    return replace(song, lines=tuple(replace(line, translation=translations[line.line_no]) for line in song.lines))


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


def _validate_song_match(
    job_dir: Path,
    video: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
    source_maximum_units: float | None,
) -> tuple[
    SongMatch,
    list[int],
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    alignment_ids = singing_ids
    timing, audits = _align_match_with_pyshiro(
        job_dir,
        video,
        cues,
        alignment_ids,
        match,
        config,
    )
    replacements, alignments = _apply_local_match(
        cues,
        alignment_ids,
        match,
        timing,
        pyshiro_audit=audits,
        source_maximum_units=source_maximum_units,
    )
    recovered, recovered_alignments, recovery_audits = _recover_lyric_ranges(
        job_dir,
        cues,
        alignment_ids,
        match,
        config,
        video=video,
        verified_replacements=replacements,
    )
    for cue_id, values in recovered.items():
        replacements.setdefault(cue_id, []).extend(values)
        replacements[cue_id].sort(key=lambda value: (value.start, value.end))
    alignments.extend(recovered_alignments)
    audits.extend(recovery_audits)
    return match, alignment_ids, replacements, alignments, audits


def _verified_spans_for_match(
    match: SongMatch,
    replacements: dict[int, list[Cue]],
    alignments: list[dict[str, object]],
    audits: list[dict[str, object]],
) -> list[VerifiedLyricSpan]:
    likelihood_by_cue = {
        int(item["cue_id"]): float(item.get("likelihood_per_frame", -1e9))
        for item in audits
        if isinstance(item, dict)
        and item.get("status") in {"aligned", "range_verified"}
        and isinstance(item.get("cue_id"), int)
    }
    spans: list[VerifiedLyricSpan] = []
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
            spans.append(
                VerifiedLyricSpan(
                    value.start,
                    value.end,
                    cue_id,
                    match.song.song_id,
                    line_ids,
                    likelihood_by_cue.get(cue_id, -1e9),
                )
            )
    return spans


def _recover_lyric_ranges(
    job_dir: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
    *,
    video: Path | None = None,
    verified_replacements: dict[int, list[Cue]] | None = None,
) -> tuple[
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if not match.anchors:
        return {}, [], []
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if uv is None or not worker.is_file():
        return {}, [], []
    ranges: list[dict[str, object]] = []
    ordered = sorted(
        match.anchors, key=lambda value: (value.take_index, value.cue_index)
    )

    def add_range(
        start: float,
        end: float,
        line_ids: list[int],
        cue_id: int,
        take_index: int,
        route: str,
        anchor_sides: int,
    ) -> None:
        if end - start < 2 or not line_ids:
            return
        key = (round(start, 3), round(end, 3), tuple(line_ids))
        if any(value["key"] == key for value in ranges):
            return
        ranges.append(
            {
                "key": key,
                "start": start,
                "end": end,
                "line_ids": line_ids,
                "cue_id": cue_id,
                "take_index": take_index,
                "route": route,
                "anchor_sides": anchor_sides,
            }
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
        add_range(
            cues[left_cue_id].end,
            cues[right_cue_id].start,
            list(range(left.line_end, right.line_start)),
            left_cue_id,
            left.take_index,
            "between_anchors",
            2,
        )

    first = ordered[0]
    if first.cue_index > 0 and first.line_start > 0:
        cue_id = singing_ids[first.cue_index - 1]
        add_range(
            cues[cue_id].start,
            cues[cue_id].end,
            list(
                range(
                    max(0, first.line_start - config.lyric_neighbor_max_lines),
                    first.line_start,
                )
            ),
            cue_id,
            first.take_index,
            "take_prefix",
            1,
        )
    elif first.line_start > 0:
        cue_id = singing_ids[first.cue_index]
        verified = (verified_replacements or {}).get(cue_id, [])
        verified_start = min(
            (cue.start for cue in verified), default=cues[cue_id].start
        )
        add_range(
            cues[cue_id].start,
            verified_start,
            list(
                range(
                    max(0, first.line_start - config.lyric_neighbor_max_lines),
                    first.line_start,
                )
            ),
            cue_id,
            first.take_index,
            "take_prefix",
            1,
        )
    last = ordered[-1]
    if last.cue_index + 1 < len(singing_ids):
        cue_id = singing_ids[last.cue_index + 1]
        cue = cues[cue_id]
        normalizer = JapaneseNormalizer()
        add_range(
            cue.start,
            cue.end,
            _select_suffix_neighbor_lines(
                match.song,
                last.line_end,
                cue.text,
                cue.end - cue.start,
                config.lyric_neighbor_max_lines,
                normalizer,
            ),
            cue_id,
            last.take_index,
            "take_suffix",
            1,
        )

    analysis_path = job_dir / "audio-analysis.json"
    if analysis_path.is_file():
        try:
            analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
            phrases = [
                value
                for value in analysis["acoustic_phrases"]
                if isinstance(value, dict) and value.get("route_alt") is False
            ]
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            phrases = []
        first_cue_id = singing_ids[first.cue_index]
        before = [
            value
            for value in phrases
            if 0
            <= cues[first_cue_id].start - float(value.get("end", -1))
            <= config.song_search_group_gap_seconds
        ]
        if before and first.line_start > 0:
            phrase = max(before, key=lambda value: float(value["end"]))
            add_range(
                float(phrase["start"]),
                float(phrase["end"]),
                list(
                    range(
                        max(0, first.line_start - config.lyric_neighbor_max_lines),
                        first.line_start,
                    )
                ),
                first_cue_id,
                first.take_index,
                "acoustic_prefix",
                1,
            )
        last_cue_id = singing_ids[last.cue_index]
        after = [
            value
            for value in phrases
            if 0
            <= float(value.get("start", -1)) - cues[last_cue_id].end
            <= config.song_search_group_gap_seconds
        ]
        if after and last.line_end < len(match.song.lines):
            phrase = min(after, key=lambda value: float(value["start"]))
            duration = float(phrase["end"]) - float(phrase["start"])
            add_range(
                float(phrase["start"]),
                float(phrase["end"]),
                _select_suffix_neighbor_lines(
                    match.song,
                    last.line_end,
                    "",
                    duration,
                    config.lyric_neighbor_max_lines,
                    JapaneseNormalizer(),
                ),
                last_cue_id,
                last.take_index,
                "acoustic_suffix",
                1,
            )

    probes = [
        probe
        for value in ranges
        for probe in _split_lyric_range(value, config.pyshiro_max_window_seconds)
    ]
    if not probes or video is None:
        return {}, [], []
    output_dir = job_dir / "song-alignment" / match.song.song_id / "range-verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    _materialize_verification_stems(
        video,
        probes,
        output_dir,
        config.vocal_separation_device,
        job_dir / "vocal-candidates" / "manifest.json",
    )

    normalizer = JapaneseNormalizer()
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    for probe in probes:
        recovered, recovered_alignments, audit = _verify_lyric_range(
            probe, cues, match, config, uv, worker, normalizer
        )
        audits.append(audit)
        if recovered:
            cue_id = int(probe["cue_id"])
            replacements.setdefault(cue_id, []).extend(recovered)
            alignments.extend(recovered_alignments)
    return replacements, alignments, audits


def _split_lyric_range(
    value: dict[str, object], maximum_seconds: float
) -> list[dict[str, object]]:
    line_ids = list(value["line_ids"])
    start = float(value["start"])
    end = float(value["end"])
    count = min(len(line_ids), max(1, ceil((end - start) / maximum_seconds)))
    probes: list[dict[str, object]] = []
    for index in range(count):
        line_start = round(index * len(line_ids) / count)
        line_end = round((index + 1) * len(line_ids) / count)
        probe_start = start + (end - start) * index / count
        probe_end = start + (end - start) * (index + 1) / count
        probes.append(
            {
                **value,
                "line_ids": line_ids[line_start:line_end],
                "start": probe_start,
                "end": probe_end,
                "part": index,
                "part_count": count,
            }
        )
    return probes


def _materialize_verification_stems(
    video: Path,
    probes: list[dict[str, object]],
    output_dir: Path,
    device: str,
    manifest_path: Path | None = None,
) -> None:
    requests: list[tuple[float, float, Path]] = []
    manifests: list[dict[str, object]] = []
    if manifest_path is not None and manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifests = [value for value in raw if isinstance(value, dict)]
        except (OSError, TypeError, json.JSONDecodeError):
            manifests = []
    for index, probe in enumerate(probes):
        wav = output_dir / f"range-{index:04d}.vocals.wav"
        probe["wav"] = wav
        manifest = next(
            (
                value
                for value in manifests
                if float(value.get("start", -1)) <= float(probe["start"])
                and float(value.get("end", -1)) >= float(probe["end"])
            ),
            None,
        )
        if manifest is not None and manifest_path is not None:
            source = manifest_path.parent / str(manifest["path"])
            if _extract_vocal_window(
                source,
                wav,
                float(probe["start"]) - float(manifest["start"]),
                float(probe["end"]) - float(probe["start"]),
            ):
                continue
        requests.append((float(probe["start"]), float(probe["end"]), wav))
    if not requests:
        return
    from .audio_analysis import separate_vocal_ranges

    try:
        separate_vocal_ranges(video, requests, device)
    except Exception as exc:
        logging.warning("lyric range vocal separation failed: %s", exc)
        for probe in probes:
            probe["stem_error"] = str(exc)[:500]


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


def _verify_lyric_range(
    probe: dict[str, object],
    cues: list[Cue],
    match: SongMatch,
    config: SongIdentificationConfig,
    uv: str,
    worker: Path,
    normalizer: JapaneseNormalizer,
) -> tuple[list[Cue], list[dict[str, object]], dict[str, object]]:
    cue_id = int(probe["cue_id"])
    line_ids = list(probe["line_ids"])
    start = float(probe["start"])
    end = float(probe["end"])
    route = str(probe["route"])
    take_index = int(probe["take_index"])
    audit: dict[str, object] = {
        "status": "range_rejected",
        "route": route,
        "cue_id": cue_id,
        "take_index": take_index,
        "lyric_line_ids": line_ids,
        "start": start,
        "end": end,
        "part": probe.get("part"),
        "part_count": probe.get("part_count"),
    }
    if end <= start or end - start > config.pyshiro_max_window_seconds + 1e-3:
        audit["reason"] = "duration_outside_pyshiro_limit"
        return [], [], audit
    wav = probe.get("wav")
    if not isinstance(wav, Path) or not wav.is_file():
        audit["reason"] = "vocal_stem_unavailable"
        if probe.get("stem_error"):
            audit["stem_error"] = probe["stem_error"]
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
        audit["reason"] = "pyshiro_range_failed"
        audit["pyshiro_failure"] = candidate_failure
        return [], [], audit
    anchor_sides = int(probe["anchor_sides"])
    require_preference = bool(probe.get("require_preference", True))
    alternatives: list[dict[str, object]] = []
    alternative_failures: list[dict[str, object]] = []
    competing_sets = (
        _competing_line_sets(match.song, line_ids, normalizer=normalizer)
        if require_preference
        else []
    )
    for lines_value in competing_sets:
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
    required_margin = float(
        probe.get(
            "required_margin",
            config.pyshiro_gap_likelihood_margin
            if anchor_sides == 2
            else config.pyshiro_likelihood_margin,
        )
    )
    accepted, likelihood, best_alternative = _pyshiro_likelihood_wins(
        response,
        alternatives,
        config,
        margin=required_margin,
        require_preference=require_preference,
    )
    audit["likelihood_per_frame"] = likelihood
    audit["best_competing_likelihood_per_frame"] = best_alternative
    audit["validation_policy"] = str(
        probe.get(
            "validation_policy",
            "between_anchors_continuous_block"
            if anchor_sides == 2
            else "one_sided_neighbor_extension",
        )
    )
    audit["required_likelihood_margin"] = required_margin
    audit["requires_competitor_preference"] = require_preference
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
    previous_end = start
    for line_id, line, value, raw_units in zip(line_ids, lines, ranges, units_by_line):
        if (
            not isinstance(value, list)
            or len(value) != 2
            or not isinstance(raw_units, list)
        ):
            audit["reason"] = "pyshiro_timeline_rejected"
            return [], [], audit
        line_start = start + float(value[0])
        line_end = start + float(value[1])
        units = tuple(
            TimedTextUnit(
                str(unit["text"]),
                start + float(unit["start"]),
                start + float(unit["end"]),
            )
            for unit in raw_units
            if isinstance(unit, dict)
        )
        if line_start < previous_end - 1e-3 or line_end <= line_start or not units:
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
                line_start,
                line_end,
                line.text,
                cues[cue_id].speaker,
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
                "match": "lyrics_verified_range",
                "corrected_text": line.text,
                "start": line_start,
                "end": line_end,
                "score": round(likelihood, 6),
                "take_index": take_index,
            }
        )
        previous_end = line_end
    coverage = (recovered[-1].end - recovered[0].start) / (end - start)
    audit["timeline_coverage"] = round(coverage, 6)
    if coverage < config.lyric_neighbor_min_coverage:
        audit["reason"] = "insufficient_timeline_coverage"
        return [], [], audit
    audit["status"] = str(probe.get("success_status", "range_verified"))
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
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if uv is None or not worker.is_file():
        return {}, [{"status": "skipped", "reason": "pyshiro_worker_unavailable"}]
    output_dir = job_dir / "song-alignment" / match.song.song_id
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = JapaneseNormalizer()
    probes: list[dict[str, object]] = []
    for anchor in match.anchors:
        cue_id = singing_ids[anchor.cue_index]
        cue = cues[cue_id]
        strong_anchor = _anchor_has_continuous_support(anchor, match.anchors)
        absolute_likelihood_only = strong_anchor
        probes.append(
            {
                "start": cue.start,
                "end": cue.end,
                "line_ids": list(range(anchor.line_start, anchor.line_end)),
                "cue_id": cue_id,
                "take_index": anchor.take_index,
                "route": "initial_alt_anchor",
                "anchor_sides": 0,
                "require_preference": not absolute_likelihood_only,
                "required_margin": (
                    config.pyshiro_likelihood_margin
                    if not absolute_likelihood_only
                    else 0.0
                ),
                "validation_policy": (
                    "strong_continuous_alt_anchor"
                    if strong_anchor
                    else "isolated_single_line_alt_anchor"
                ),
                "success_status": "aligned",
            }
        )
    _materialize_verification_stems(
        video,
        probes,
        output_dir,
        config.device,
        job_dir / "vocal-candidates" / "manifest.json",
    )
    timings: dict[
        int, tuple[tuple[int, float, float, tuple[TimedTextUnit, ...]], ...]
    ] = {}
    audits: list[dict[str, object]] = []
    for probe in probes:
        recovered, _alignments, audit = _verify_lyric_range(
            probe, cues, match, config, uv, worker, normalizer
        )
        audits.append(audit)
        if not recovered:
            continue
        cue_id = int(probe["cue_id"])
        line_ids = list(probe["line_ids"])
        timings[cue_id] = tuple(
            (line_id, cue.start, cue.end, cue.source_units)
            for line_id, cue in zip(line_ids, recovered)
        )
    audit_path = output_dir / "alignment.json"
    audit_path.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return timings, audits


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



def collect_ocr_candidates(
    video: Path,
    group: SongSearchGroup,
    frame_dir: Path,
    config: SongIdentificationConfig,
    ocr: Any,
) -> list[OCRCandidate]:
    timestamp = max(0.0, group.start)
    frame = _extract_frame(video, frame_dir, timestamp)
    observations: list[
        tuple[str, float, int, float, tuple[float, float, float, float]]
    ] = []
    for text, score, bounds in ocr.read(frame):
        normalized = _normalize_ocr_text(text)
        if normalized and score >= config.minimum_ocr_score:
            observations.append((normalized, score, 0, timestamp, bounds))
    return aggregate_ocr_observations(observations, 1)


def aggregate_ocr_observations(
    observations: list[
        tuple[str, float, int, float, tuple[float, float, float, float]]
    ],
    minimum_frames: int,
) -> list[OCRCandidate]:
    clusters: list[
        list[tuple[str, float, int, float, tuple[float, float, float, float]]]
    ] = []
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
                best[4],
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

    def read(
        self, path: Path
    ) -> list[tuple[str, float, tuple[float, float, float, float]]]:
        response = self._exchange({"path": str(path.resolve())})
        values = response.get("values", [])
        if not isinstance(values, list):
            raise RuntimeError("song OCR worker returned malformed values")
        return [
            (
                str(item[0]),
                float(item[1]),
                tuple(float(value) for value in item[2]),
            )
            for item in values
            if isinstance(item, list)
            and len(item) == 3
            and isinstance(item[2], list)
            and len(item[2]) == 4
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

