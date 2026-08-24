from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import shutil
import socket
import subprocess
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

_CACHE_VERSION = 10
_PROMPT_VERSION = 7
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
_SUPPORTED_LYRIC_HOSTS = (
    "utaten.com",
    "oricon.co.jp",
    "awa.fm",
    "uta-net.com",
)


@dataclass(frozen=True)
class SongEpisode:
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
class SongIdentificationResult:
    corrected_cues: list[Cue]
    reports: list[dict[str, object]]


def identify_and_align_songs(
    video: Path,
    cues: list[Cue],
    metadata: dict[str, object],
    job_dir: Path,
    config: SongIdentificationConfig,
    *,
    source_maximum_units: float | None = None,
) -> SongIdentificationResult:
    episodes = group_singing_episodes(cues, config.song_gap_seconds)
    if not config.enabled or not episodes:
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
    ocr_signature = _ocr_signature(video, episodes, config)
    candidate_sets = _load_ocr_cache(ocr_cache_path, ocr_signature, len(episodes))
    if candidate_sets is None:
        candidate_sets = []
        ocr_succeeded = False
        try:
            ocr = _PaddleOCR(config)
        except Exception as exc:
            logging.warning(
                "song OCR worker failed to start; continuing without OCR: %s", exc
            )
            candidate_sets = [[] for _ in episodes]
        else:
            try:
                for index, episode in enumerate(episodes):
                    try:
                        candidates = collect_ocr_candidates(
                            video,
                            episode,
                            job_dir / "song-ocr-frames" / f"{index:03d}",
                            config,
                            ocr,
                        )
                    except Exception as exc:
                        logging.warning(
                            "song OCR failed for %.3f-%.3fs: %s",
                            episode.start,
                            episode.end,
                            exc,
                        )
                        candidates = []
                    candidate_sets.append(candidates)
                ocr_succeeded = True
            finally:
                ocr.close()
        if ocr_succeeded:
            _write_ocr_cache(ocr_cache_path, ocr_signature, candidate_sets)

    reports: list[dict[str, object]] = []
    lyric_replacements: dict[int, list[Cue]] = {}
    discarded_song_region_cues: set[int] = set()
    library = LyricsLibrary(Path(config.lyrics_library_path).resolve())
    try:
        for index, episode in enumerate(episodes):
            candidates = candidate_sets[index]
            singing_ids = [
                cue_id for cue_id in episode.cue_ids if cues[cue_id].kind == "singing"
            ]
            hypotheses = [cues[cue_id].text for cue_id in singing_ids]
            library_songs = library.songs()
            match = _match_candidates(hypotheses, library_songs, config)
            provenance = "local_library"
            web_search_audit: dict[str, object] | None = None
            if match is None:
                fetched, web_search_audit = _search_canonical_lyrics(
                    hypotheses,
                    candidates,
                    library_songs,
                    config,
                )
                match = _match_candidates(hypotheses, fetched, config)
                provenance = "web"
                if match is not None:
                    match = SongMatch(
                        library.store_canonical_song(
                            title=match.song.title,
                            artist=match.song.artist,
                            aliases=list(match.song.aliases),
                            source_url=match.song.source_url,
                            lines=[
                                (line.text, line.reading) for line in match.song.lines
                            ],
                        ),
                        match.anchors,
                        match.score,
                    )
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
                        "episode": asdict(episode),
                        "ocr_candidates": [asdict(item) for item in candidates],
                        "web_search": web_search_audit,
                    }
                )
                continue
            song = match.song
            match = SongMatch(song, match.anchors, match.score)
            timing, pyshiro_audit = _align_match_with_pyshiro(
                job_dir, cues, singing_ids, match, config
            )
            replacements, alignments = _apply_local_match(
                cues,
                singing_ids,
                match,
                timing,
                source_maximum_units=source_maximum_units,
            )
            recovered, recovered_alignments, gap_audit = _recover_lyric_gaps(
                job_dir,
                cues,
                singing_ids,
                match,
                config,
            )
            for cue_id, values in recovered.items():
                replacements.setdefault(cue_id, []).extend(values)
                replacements[cue_id].sort(key=lambda value: (value.start, value.end))
            alignments.extend(recovered_alignments)
            pyshiro_audit.extend(gap_audit)
            lyric_replacements.update(replacements)
            matched_ids = {item["asr_cue_ids"][0] for item in alignments}
            for cue_id in singing_ids:
                if cue_id in matched_ids:
                    continue
                discarded_song_region_cues.add(cue_id)
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
                    "episode": asdict(episode),
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
    # Song discovery may add canonical lyrics to the library.
    # Sign the cache against the resulting library state, not its state at startup.
    signature = _signature(video, cues, metadata, config)
    payload = {
        "version": _CACHE_VERSION,
        "signature": signature,
        "reports": reports,
        "corrected_cues": [asdict(cue) for cue in corrected],
    }
    if not any(report.get("error") or report.get("tool_errors") for report in reports):
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(cache_path)
    return SongIdentificationResult(corrected, reports)


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
            episode = report.get("episode")
            if not isinstance(episode, dict):
                continue
            start = float(episode.get("start", float("-inf")))
            end = float(episode.get("end", float("inf")))
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
    return SongIdentificationResult(corrected, result.reports)


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
    return SongIdentificationResult(corrected, result.reports)


def _match_candidates(
    hypotheses: list[str], songs: list[LibrarySong], config: SongIdentificationConfig
) -> SongMatch | None:
    return match_song(
        hypotheses,
        songs,
        anchor_threshold=config.match_anchor_threshold,
        minimum_anchors=config.match_minimum_anchors,
        minimum_score=config.match_minimum_score,
        minimum_margin=config.match_minimum_margin,
    )


def _search_canonical_lyrics(
    hypotheses: list[str],
    ocr: list[OCRCandidate],
    library_songs: list[LibrarySong],
    config: SongIdentificationConfig,
) -> tuple[list[LibrarySong], dict[str, object]]:
    tools = _WebTools(config)
    queries = _build_lyric_search_queries(hypotheses, ocr, library_songs)
    search_audit: list[dict[str, object]] = []
    songs: dict[str, LibrarySong] = {}
    for item in queries:
        query = str(item["query"])
        record = dict(item)
        try:
            raw_results = tools.search(query)
            parsed_results = json.loads(raw_results)
            if isinstance(parsed_results, list):
                record["results"] = parsed_results
            else:
                record["results"] = []
                if isinstance(parsed_results, dict) and parsed_results.get("error"):
                    record["error"] = str(parsed_results["error"])[:500]
        except Exception as exc:
            logging.warning("canonical lyric search failed for %r: %s", query, exc)
            record["error"] = str(exc)[:500]
        search_audit.append(record)
    fetch_audit: list[dict[str, object]] = []
    for url in sorted(tools.allowed_urls):
        record: dict[str, object] = {"url": url}
        try:
            value = tools.fetch_lyrics(url)
        except Exception as exc:
            logging.warning("canonical lyric fetch failed for %s: %s", url, exc)
            record.update(status="error", error=str(exc)[:500])
            fetch_audit.append(record)
            continue
        if value is None:
            record["status"] = "not_lyrics"
            fetch_audit.append(record)
            continue
        title, artist, lines = value
        record.update(
            status="fetched",
            title=title,
            artist=artist,
            lyric_line_count=len(lines),
        )
        fetch_audit.append(record)
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
    audit: dict[str, object] = {
        "queries": search_audit,
        "fetches": fetch_audit,
        "worker_errors": tools.errors,
    }
    return list(songs.values()), audit


def _build_lyric_search_queries(
    hypotheses: list[str],
    ocr: list[OCRCandidate],
    library_songs: list[LibrarySong],
) -> list[dict[str, object]]:
    queries: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(source: str, phrase: str, reason: str, *, minimum_length: int) -> None:
        compact = re.sub(r"\s+", " ", phrase).strip(' "')
        if len(compact) < minimum_length:
            return
        query = f"{compact} 歌詞"
        if query in seen:
            return
        seen.add(query)
        queries.append(
            {"source": source, "phrase": phrase, "reason": reason, "query": query}
        )

    for phrase in _asr_lyric_search_fragments(hypotheses):
        add("asr", phrase, "high_information_asr_fragment", minimum_length=8)
        if sum(item["source"] == "asr" for item in queries) >= 4:
            break

    ocr_count = 0
    for candidate in ocr:
        qualified = _qualified_ocr_song_evidence(candidate.text, library_songs)
        if qualified is None:
            continue
        phrase, reason = qualified
        before = len(queries)
        add("ocr", phrase, reason, minimum_length=3)
        if len(queries) > before:
            ocr_count += 1
        if ocr_count >= 4:
            break
    return queries


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
                excerpt = fragment[:20].strip()
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
        and translated[1] in {"llm", "machine"}
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
        model=model if source == "llm" else None,
        prompt_hash=(
            prompt_templates_digest("lyrics-translate.md") if source == "llm" else None
        ),
    )
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
    source_maximum_units: float | None = None,
) -> tuple[dict[int, list[Cue]], list[dict[str, object]]]:
    replacements: dict[int, list[Cue]] = {}
    alignments: list[dict[str, object]] = []
    for anchor in match.anchors:
        cue_id = singing_ids[anchor.cue_index]
        lines = match.song.lines[anchor.line_start : anchor.line_end]
        line_ids = list(range(anchor.line_start, anchor.line_end))
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

        text = "".join(line.text for line in lines)
        translation = "".join(line.translation or "" for line in lines) or None
        replacements[cue_id] = [
            Cue(
                **{
                    **asdict(cues[cue_id]),
                    "text": text,
                    "kind": "singing",
                    "language": language_for_text(text, "Japanese"),
                    "preferred_translation": translation,
                }
            )
        ]
        alignments.append(
            {
                "asr_cue_ids": [cue_id],
                "lyric_line_ids": line_ids,
                "match": "lyrics",
                "corrected_text": text,
                "score": round(anchor.score, 6),
                "take_index": anchor.take_index,
            }
        )
    return replacements, alignments


def _recover_lyric_gaps(
    job_dir: Path,
    cues: list[Cue],
    singing_ids: list[int],
    match: SongMatch,
    config: SongIdentificationConfig,
) -> tuple[
    dict[int, list[Cue]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    if len(match.anchors) < 2:
        return {}, [], []
    manifest_path = job_dir / "vocal-candidates" / "manifest.json"
    worker = Path(config.pyshiro_worker_project).resolve() / "worker.py"
    uv = shutil.which("uv")
    if not manifest_path.is_file() or uv is None or not worker.is_file():
        return {}, [], []
    raw_manifests = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        manifest = next(
            (
                value
                for value in manifests
                if float(value.get("start", -1))
                <= center
                <= float(value.get("end", -1))
            ),
            None,
        )
        if manifest is None:
            continue
        manifest_start = float(manifest["start"])
        manifest_end = float(manifest["end"])
        start = max(manifest_start, center - half)
        end = min(manifest_end, start + config.lyric_gap_recheck_seconds)
        start = max(manifest_start, end - config.lyric_gap_recheck_seconds)
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
            "take_index": left.take_index,
            "lyric_line_ids": missing_ids,
            "start": probe["start"],
            "end": probe["end"],
        }
        manifest = probe["manifest"]
        assert isinstance(manifest, dict)
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
        baseline = _run_pyshiro_lines(uv, worker, wav, adjacent, normalizer)
        aligned = _run_pyshiro_lines(uv, worker, wav, expanded, normalizer)
        if baseline is None or aligned is None:
            audit["reason"] = "pyshiro_recheck_failed"
            audits.append(audit)
            continue
        baseline_score = float(baseline.get("likelihood_per_frame", -1e9))
        aligned_score = float(aligned.get("likelihood_per_frame", -1e9))
        audit["baseline_likelihood_per_frame"] = baseline_score
        audit["expanded_likelihood_per_frame"] = aligned_score
        if (
            aligned_score < config.pyshiro_likelihood_floor
            or aligned_score < baseline_score - config.pyshiro_likelihood_margin
        ):
            audit["reason"] = "pyshiro_likelihood_rejected"
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
    return replacements, alignments, audits


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
    for left_take, right_take in pairwise(takes):
        left = left_take[-1]
        right = right_take[0]
        positions = list(range(left.cue_index + 1, right.cue_index))
        if not positions:
            continue
        candidates: list[tuple[int, list[int], str, int]] = []
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
    response = _run_pyshiro_lines(uv, worker, wav, lines, normalizer)
    if response is None:
        audit["reason"] = "pyshiro_neighbor_failed"
        return [], [], audit
    likelihood = float(response.get("likelihood_per_frame", -1e9))
    audit["likelihood_per_frame"] = likelihood
    if likelihood < config.pyshiro_likelihood_floor:
        audit["reason"] = "pyshiro_likelihood_rejected"
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
        if any(
            unit.end <= unit.start
            or unit.end - unit.start > config.lyric_neighbor_max_unit_seconds
            for unit in units
        ):
            audit["reason"] = "pyshiro_unit_duration_rejected"
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
) -> dict[str, object] | None:
    display_units = [
        normalizer.display_units(line.text, line.reading) for line in lines
    ]
    if any(not values for values in display_units):
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
    result = subprocess.run(
        [uv, "run", "--project", str(worker.parent), "python", str(worker)],
        input=json.dumps(request, ensure_ascii=False),
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    if result.returncode:
        return None
    response = _parse_worker_json_output(result.stdout)
    return response if response.get("ok") is True else None


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
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "skipped",
                    "reason": "no_covering_vocal_stem",
                }
            )
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
        lines = match.song.lines[anchor.line_start : anchor.line_end]
        display_units = [
            normalizer.display_units(line.text, line.reading) for line in lines
        ]
        if any(not units for units in display_units):
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": "lyric_display_units_empty",
                }
            )
            continue
        request = {
            "wav": str(wav.resolve()),
            "readings": [
                "".join(reading for _text, reading in units) for units in display_units
            ],
            "display_units": [
                [{"text": text, "reading": reading} for text, reading in units]
                for units in display_units
            ],
        }
        result = subprocess.run(
            [uv, "run", "--project", str(worker.parent), "python", str(worker)],
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        if result.returncode:
            audits.append(
                {
                    "cue_id": cue_id,
                    "status": "failed",
                    "reason": result.stderr[-300:] or result.stdout[-300:],
                }
            )
            continue
        try:
            response = _parse_worker_json_output(result.stdout)
        except ValueError as exc:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": str(exc),
                }
            )
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
        line_ids = list(range(anchor.line_start, anchor.line_end))
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
        for line_id, value, unit_values in zip(line_ids, ranges, owned_units):
            if not isinstance(unit_values, list) or not unit_values:
                line_ranges_values = []
                break
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
                line_ranges_values = []
                break
            line_ranges_values.append(
                (
                    line_id,
                    cue.start + float(value[0]),
                    cue.start + float(value[1]),
                    units,
                )
            )
        if not line_ranges_values:
            audits.append(
                {
                    "cue_id": cue_id,
                    "take_index": anchor.take_index,
                    "status": "failed",
                    "reason": "lyric_unit_ranges_invalid",
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
                "likelihood_per_frame": response.get("likelihood_per_frame"),
                "phonemes": response.get("phonemes"),
            }
        )
    audit_path = output_dir / "alignment.json"
    audit_path.write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return timings, audits


def group_singing_episodes(cues: list[Cue], maximum_gap: float) -> list[SongEpisode]:
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
        SongEpisode(
            group[0][1].start,
            group[-1][1].end,
            tuple(range(group[0][0], group[-1][0] + 1)),
        )
        for group in groups
    ]


def collect_ocr_candidates(
    video: Path,
    episode: SongEpisode,
    frame_dir: Path,
    config: SongIdentificationConfig,
    ocr: Any,
) -> list[OCRCandidate]:
    start = max(0.0, episode.start - config.seconds_before_start)
    end = episode.start + config.seconds_after_start
    frames = _extract_frames(
        video, frame_dir, start, end, config.sample_interval_seconds
    )
    observations: list[tuple[str, float, int, float]] = []
    for frame_index, path in enumerate(frames):
        timestamp = start + frame_index * config.sample_interval_seconds
        for text, score in ocr.read(path):
            normalized = _normalize_ocr_text(text)
            if normalized and score >= config.minimum_ocr_score:
                observations.append((normalized, score, frame_index, timestamp))
    return aggregate_ocr_observations(observations, config.minimum_persistent_frames)


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
        episode = report.get("episode", {})
        raw_allowed = episode.get("cue_ids", []) if isinstance(episode, dict) else []
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


class _PaddleOCR:
    def __init__(self, config: SongIdentificationConfig):
        self.process: subprocess.Popen[str] | None = None
        uv = shutil.which("uv")
        project = Path(config.ocr_worker_project).resolve()
        worker = project / "worker.py"
        if uv is None or not worker.is_file():
            raise RuntimeError(f"song OCR worker is unavailable at {worker}")
        self.process = subprocess.Popen(
            [uv, "run", "--project", str(project), "python", str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._exchange(
            {
                "device": config.device,
                "detection_model": config.detection_model,
                "recognition_model": config.recognition_model,
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


def _extract_frames(
    video: Path, directory: Path, start: float, end: float, interval: float
) -> list[Path]:
    cache_key = hashlib.sha256(
        f"{start:.3f}:{end:.3f}:{interval:.6f}".encode()
    ).hexdigest()[:12]
    directory = directory / cache_key
    directory.mkdir(parents=True, exist_ok=True)
    existing = sorted(directory.glob("frame-*.jpg"))
    if existing:
        return existing
    ffmpeg = require_command("ffmpeg")
    completed = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{max(0.001, end - start):.3f}",
            "-i",
            str(video),
            "-vf",
            f"fps=1/{interval:.6f}",
            "-q:v",
            "3",
            str(directory / "frame-%05d.jpg"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        raise RuntimeError(
            f"song OCR frame extraction failed: {completed.stderr[-500:]}"
        )
    return sorted(directory.glob("frame-*.jpg"))


def _episode_evidence(
    cues: list[Cue],
    episode: SongEpisode,
    metadata: dict[str, object],
    candidates: list[OCRCandidate],
) -> dict[str, object]:
    preceding = [
        {"id": i, "start": cue.start, "text": cue.text}
        for i, cue in enumerate(cues)
        if cue.kind != "singing" and episode.start - 45 <= cue.end <= episode.start + 5
    ]
    singing = [
        {
            "id": i,
            "start": cues[i].start,
            "end": cues[i].end,
            "kind": cues[i].kind,
            "text": cues[i].text,
        }
        for i in episode.cue_ids
    ]
    return {
        "video": {
            key: metadata.get(key)
            for key in ("title", "description", "channel", "uploader")
            if metadata.get(key)
        },
        "ocr": [asdict(item) for item in candidates[:30]],
        "announcement_asr": preceding[-20:],
        "singing_asr": singing,
    }


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
    hostname = (urlsplit(url).hostname or "").casefold()
    return any(
        hostname == allowed or hostname.endswith(f".{allowed}")
        for allowed in _SUPPORTED_LYRIC_HOSTS
    )


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
    episodes: list[SongEpisode],
    config: SongIdentificationConfig,
) -> str:
    stat = video.stat()
    payload = {
        "version": _CACHE_VERSION,
        "video": [stat.st_size, stat.st_mtime_ns],
        "episodes": [asdict(episode) for episode in episodes],
        "ocr": {
            key: value
            for key, value in asdict(config).items()
            if key
            in {
                "device",
                "detection_model",
                "recognition_model",
                "ocr_worker_project",
                "seconds_before_start",
                "seconds_after_start",
                "sample_interval_seconds",
                "minimum_ocr_score",
                "minimum_persistent_frames",
            }
        },
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()


def _load_ocr_cache(
    path: Path, signature: str, episode_count: int
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
        if not isinstance(groups, list) or len(groups) != episode_count:
            return None
        candidates = [
            [OCRCandidate(**item) for item in group]
            for group in groups
            if isinstance(group, list)
        ]
        if len(candidates) != episode_count:
            return None
        logging.info("using song OCR cache with %d episodes", len(candidates))
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
        logging.info("using song identification cache with %d reports", len(reports))
        return SongIdentificationResult(corrected, reports)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        logging.warning(
            "ignoring unreadable song identification cache %s: %s", path, exc
        )
        return None
