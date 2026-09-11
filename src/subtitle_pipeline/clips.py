from __future__ import annotations

import json
import logging
import math
import re
import statistics
import threading
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from .cache import CacheStore, config_snapshot, restore_config, job_lock
from .publication import publish_once
from .chat_context import (
    YouTubeChatMessage,
    read_youtube_live_chat,
    remove_youtube_chat_files,
)
from .commands import require_command, run
from .config import AppConfig
from .llm_response import (
    finish_reason,
    parse_json_object,
    structured_request_body,
    structured_response_content,
)
from .local_llm_server import LocalLLMServer
from .media import _download_youtube_chat_replay
from .pipeline import normalize_youtube_url, youtube_video_id
from .prompt_templates import render_user_prompt
from .subtitles import Cue, read_subtitles
from .upload import upload_videos_to_bilibili

from .prompt_budget import batch_requests, request_budget_validator

logger = logging.getLogger(__name__)

_ANALYSIS_VERSION = 1
_CHAT_WINDOW_SECONDS = 30.0
_CHAT_WINDOW_STEP_SECONDS = 5.0
_SPEECH_CONTEXT_SECONDS = 240.0
_SEMANTIC_WINDOW_SECONDS = 300.0
_MIN_SPEECH_SECONDS = 30.0
_REACTION_RE = re.compile(
    r"^(?:[wWｗＷ草笑8８]+|(?:拍手|パチ|ぱち)+|(?:888)+|[!！?？]+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ChatWindow:
    start: float
    end: float
    messages: int
    unique_authors: int
    reactions: int
    paid: int
    memberships: int
    acceleration: float
    zscore: float
    score: float
    snippets: tuple[str, ...] = ()


@dataclass(frozen=True)
class ClipPart:
    kind: str
    start: float
    end: float
    title: str
    reason: str
    confidence: str
    score: float
    song: str | None = None
    artist: str | None = None
    file: str | None = None


@dataclass(frozen=True)
class ClipsResult:
    job_dir: Path
    analysis: Path
    parts: tuple[Path, ...]
    uploaded: bool
    bilibili_aid: int | None = None
    bilibili_bvid: str | None = None


def run_clips(url: str, config: AppConfig, *, upload_override: bool | None = None,
              retry_degraded: str | None = None) -> ClipsResult:
    directory = config.work_dir.resolve() / youtube_video_id(normalize_youtube_url(url))
    with job_lock(directory):
        store = CacheStore(directory / "cache.sqlite3")
        store.restore_interrupted_retries()
        if retry_degraded is not None:
            store.retry_degraded(retry_degraded)
        return _run_clips_locked(url, config, upload_override=upload_override)


def _run_clips_locked(
    url: str,
    config: AppConfig,
    *,
    upload_override: bool | None = None,
) -> ClipsResult:
    url = normalize_youtube_url(url)
    job_dir = config.work_dir.resolve() / youtube_video_id(url)
    required = {
        "rendered video": job_dir / "translated.mp4",
        "translated subtitles": job_dir / "translated.zh-CN.srt",
        "translated metadata": job_dir / "translated.metadata.json",
        "source metadata": job_dir / "source.info.json",
    }
    missing = [
        f"{label}: {path}" for label, path in required.items() if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "clips requires a completed pipeline job; missing " + "; ".join(missing)
        )

    clips_dir = job_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = clips_dir / "analysis.json"
    store = CacheStore(job_dir / "cache.sqlite3")
    stage = store.stage("clip_analysis", lambda: {
        "clips": config_snapshot(config.clips), "llm": config_snapshot(config.llm),
        "translation": config_snapshot(config.translation),
    })
    config = replace(config, clips=restore_config(config.clips, stage.plan["clips"]),
                     llm=restore_config(config.llm, stage.plan["llm"]),
                     translation=restore_config(config.translation, stage.plan["translation"]))
    analysis = stage.get("__result__")
    analysis_cached = analysis is not None
    if analysis is None:
        analysis = _analyze(url, job_dir, clips_dir, required, config, stage)
        stage.finish(analysis)
    _write_json_atomic(analysis_path, analysis)

    part_records = _part_records(analysis)
    part_paths = _render_parts(
        required["rendered video"],
        clips_dir,
        part_records,
        config,
        force=not analysis_cached,
    )
    should_upload = config.clips.upload if upload_override is None else upload_override
    upload_path = clips_dir / "upload.json"
    aid: int | None = None
    bvid: str | None = None
    uploaded = False
    if should_upload and part_paths:
        metadata = analysis.get("upload_metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError("clip analysis has no upload metadata")
        tags = _upload_tags(required["translated metadata"], config.upload.tags)
        submission = publish_once(upload_path, {
            "title": metadata["title"], "description": metadata["description"],
            "parts": [str(path.relative_to(job_dir)) for path in part_paths],
        }, lambda: upload_videos_to_bilibili(
            part_paths, title=_required_text(metadata, "title"),
            description=_required_text(metadata, "description"), source_url=url,
            tags=tags, config=config.upload,
        ))
        aid, bvid, uploaded = submission.aid, submission.bvid, True

    return ClipsResult(job_dir, analysis_path, tuple(part_paths), uploaded, aid, bvid)


def analyze_chat_windows(
    messages: list[YouTubeChatMessage],
    *,
    duration: float,
) -> list[ChatWindow]:
    if not messages or duration <= 0:
        return []
    ordered = sorted(messages, key=lambda value: value.offset_seconds)
    starts = [
        index * _CHAT_WINDOW_STEP_SECONDS
        for index in range(max(1, math.ceil(duration / _CHAT_WINDOW_STEP_SECONDS)))
    ]
    bins: list[list[YouTubeChatMessage]] = []
    left = 0
    right = 0
    for start in starts:
        end = min(duration, start + _CHAT_WINDOW_SECONDS)
        while left < len(ordered) and ordered[left].offset_seconds < start:
            left += 1
        right = max(right, left)
        while right < len(ordered) and ordered[right].offset_seconds < end:
            right += 1
        bins.append(ordered[left:right])
    counts = [len(values) for values in bins]
    median = statistics.median(counts)
    deviations = [abs(value - median) for value in counts]
    mad = statistics.median(deviations)
    scale = max(1.0, mad * 1.4826, math.sqrt(max(1.0, median)))
    windows: list[ChatWindow] = []
    previous = median
    for index, values in enumerate(bins):
        current = len(values)
        zscore = max(0.0, (current - median) / scale)
        acceleration = max(0.0, (current - previous) / scale)
        authors = {message.author or message.message_id for message in values}
        authors.discard(None)
        reactions = sum(_is_reaction(message.text) for message in values)
        paid = sum(message.amount is not None for message in values)
        memberships = sum(message.membership for message in values)
        score = zscore + 0.25 * acceleration + 1.5 * paid + memberships
        snippets = tuple(
            dict.fromkeys(
                " ".join(message.text.split())[:180]
                for message in values
                if message.text.strip()
            )
        )[:20]
        windows.append(
            ChatWindow(
                start=starts[index],
                end=min(duration, starts[index] + _CHAT_WINDOW_SECONDS),
                messages=current,
                unique_authors=len(authors),
                reactions=reactions,
                paid=paid,
                memberships=memberships,
                acceleration=round(acceleration, 6),
                zscore=round(zscore, 6),
                score=round(score, 6),
                snippets=snippets,
            )
        )
        previous = current
    return windows


def select_chat_peaks(
    windows: list[ChatWindow],
    *,
    minimum_zscore: float,
    minimum_unique_authors: int,
) -> list[ChatWindow]:
    qualifying = [
        value
        for value in windows
        if (
            value.zscore >= minimum_zscore
            and value.unique_authors >= minimum_unique_authors
        )
        or value.paid > 0
        or value.memberships > 0
    ]
    groups: list[list[ChatWindow]] = []
    for value in qualifying:
        if groups and value.start <= groups[-1][-1].end:
            groups[-1].append(value)
        else:
            groups.append([value])
    peaks: list[ChatWindow] = []
    for group in groups:
        peak = max(group, key=lambda item: (item.score, item.messages))
        snippets = tuple(
            dict.fromkeys(text for item in group for text in item.snippets)
        )[:20]
        peaks.append(
            ChatWindow(
                start=group[0].start,
                end=group[-1].end,
                messages=sum(item.messages for item in group),
                unique_authors=max(item.unique_authors for item in group),
                reactions=sum(item.reactions for item in group),
                paid=sum(item.paid for item in group),
                memberships=sum(item.memberships for item in group),
                acceleration=peak.acceleration,
                zscore=peak.zscore,
                score=peak.score,
                snippets=snippets,
            )
        )
    return peaks


def _analyze(
    url: str,
    job_dir: Path,
    clips_dir: Path,
    required: dict[str, Path],
    config: AppConfig,
    stage,
) -> dict[str, object]:
    translated_cues = read_subtitles(required["translated subtitles"])
    source_metadata = _load_json_object(required["source metadata"], "source metadata")
    translated_metadata = _load_json_object(
        required["translated metadata"], "translated metadata"
    )
    duration = float(source_metadata.get("duration") or 0.0)
    if duration <= 0:
        duration = max((cue.end for cue in translated_cues), default=0.0)
    if duration <= 0 or not translated_cues:
        raise RuntimeError("completed job has no usable duration or translated cues")

    chat_messages = [YouTubeChatMessage(**value) for value in stage.remember(
        "chat", lambda: [asdict(value) for value in _download_chat(url, clips_dir, config)]
    )]
    windows = analyze_chat_windows(chat_messages, duration=duration)
    peaks = select_chat_peaks(
        windows,
        minimum_zscore=config.clips.chat_peak_zscore,
        minimum_unique_authors=config.clips.chat_min_unique_authors,
    )
    song_seeds = _song_seeds(translated_metadata, translated_cues, duration)
    speech_seeds = (
        [_chat_seed(peak, translated_cues, duration) for peak in peaks]
        if chat_messages
        else _semantic_seeds(translated_cues, duration)
    )

    seeds = stage.remember("candidates", lambda: {
        "songs": song_seeds, "speech": speech_seeds,
        "cues": [asdict(cue) for cue in translated_cues],
    })
    song_seeds, speech_seeds = seeds["songs"], seeds["speech"]
    translated_cues = [Cue(**value) for value in seeds["cues"]]
    server = LocalLLMServer(config.llm, clips_dir / "local-llm-server.log")
    from .config import llm_api_key
    from .translate import OpenAICompatibleTranslator

    translator = OpenAICompatibleTranslator(
        config.llm,
        config.translation,
        lambda: llm_api_key(config.llm),
        audit_path=clips_dir / "llm-audit.jsonl",
    )
    request_start_lock = threading.Lock()
    def before_request(llm_config):
        with request_start_lock:
            if server.config != llm_config:
                server.stop()
                server.config = llm_config
            server.start()
    translator.before_request = before_request
    try:
        songs = _review_seeds(
            song_seeds,
            translated_cues,
            translator,
            config,
            required=True, cache=stage, prefix="song",
        )
        speech = _review_seeds(
            speech_seeds,
            translated_cues,
            translator,
            config,
            required=False, cache=stage, prefix="speech",
        )
        selected_songs = [part for part in songs if part is not None]
        selected_speech = _merge_speech_parts(
            [part for part in speech if part is not None],
            selected_songs,
            config.clips.max_speech_seconds,
        )
        parts = sorted([*selected_songs, *selected_speech], key=lambda item: item.start)
        metadata = _upload_metadata(translated_metadata) if parts else {}
    finally:
        server.stop()

    records = []
    for index, part in enumerate(parts, 1):
        filename = f"{index:03d}_{_safe_filename(part.title)}.mp4"
        records.append({**asdict(part), "file": f"parts/{filename}"})
    return {
        "version": _ANALYSIS_VERSION,
        "source_url": url,
        "created_at": datetime.now(UTC).isoformat(),
        "chat_available": bool(chat_messages),
        "chat_windows": [asdict(value) for value in windows],
        "chat_peaks": [asdict(value) for value in peaks],
        "candidate_count": len(song_seeds) + len(speech_seeds),
        "parts": records,
        "upload_metadata": metadata,
    }


def _download_chat(
    url: str, clips_dir: Path, config: AppConfig
) -> list[YouTubeChatMessage]:
    directory = clips_dir / "chat-download"
    directory.mkdir(parents=True, exist_ok=True)
    try:
        try:
            path = _download_youtube_chat_replay(url, directory, config.download)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "live chat download failed; using semantic fallback: %s", exc
            )
            return []
        if path is None:
            return []
        try:
            return read_youtube_live_chat(path, include_low_information=True)
        except (OSError, UnicodeError) as exc:
            logger.warning(
                "live chat could not be read; using semantic fallback: %s", exc
            )
            return []
    finally:
        remove_youtube_chat_files(directory)


def _song_seeds(
    metadata: dict[str, object], cues: list[Cue], duration: float
) -> list[dict[str, object]]:
    reports = metadata.get("identified_songs")
    if not isinstance(reports, list):
        return []
    raw: list[dict[str, object]] = []
    for report in reports:
        if not isinstance(report, dict) or report.get("confidence") not in {
            "high",
            "medium",
        }:
            continue
        song = str(report.get("song") or "").strip()
        alignments = report.get("alignments")
        if not song or not isinstance(alignments, list):
            continue
        ranges = [
            (float(item["start"]), float(item["end"]))
            for item in alignments
            if isinstance(item, dict)
            and isinstance(item.get("start"), (int, float))
            and isinstance(item.get("end"), (int, float))
            and float(item["end"]) > float(item["start"])
        ]
        if not ranges:
            continue
        group = report.get("search_group")
        group_start = min(value[0] for value in ranges)
        group_end = max(value[1] for value in ranges)
        if isinstance(group, dict):
            candidate_start = group.get("start")
            candidate_end = group.get("end")
            if (
                isinstance(candidate_start, (int, float))
                and float(candidate_start) <= group_end
            ):
                group_start = max(0.0, float(candidate_start))
            if (
                isinstance(candidate_end, (int, float))
                and float(candidate_end) >= group_start
            ):
                group_end = min(duration, float(candidate_end))
        raw.append(
            {
                "kind": "song",
                "required_start": min(value[0] for value in ranges),
                "required_end": max(value[1] for value in ranges),
                "performance_start": group_start,
                "performance_end": group_end,
                "song": song,
                "artist": str(report.get("artist") or "").strip() or None,
                "score": float(report.get("score") or 1.0),
            }
        )
    raw.sort(key=lambda value: float(value["performance_start"]))
    for index, seed in enumerate(raw):
        previous_end = float(raw[index - 1]["performance_end"]) if index else 0.0
        next_start = (
            float(raw[index + 1]["performance_start"])
            if index + 1 < len(raw)
            else duration
        )
        seed["context_start"] = max(
            previous_end, float(seed["performance_start"]) - 120.0
        )
        seed["context_end"] = min(next_start, float(seed["performance_end"]) + 180.0)
    return raw


def _chat_seed(peak: ChatWindow, cues: list[Cue], duration: float) -> dict[str, object]:
    center = (peak.start + peak.end) / 2
    return {
        "kind": "speech",
        "context_start": max(0.0, center - _SPEECH_CONTEXT_SECONDS),
        "context_end": min(duration, center + _SPEECH_CONTEXT_SECONDS),
        "score": peak.score,
        "signals": asdict(peak),
    }


def _semantic_seeds(cues: list[Cue], duration: float) -> list[dict[str, object]]:
    seeds = []
    start = 0.0
    while start < duration:
        end = min(duration, start + _SEMANTIC_WINDOW_SECONDS)
        if any(cue.end > start and cue.start < end for cue in cues):
            seeds.append(
                {
                    "kind": "speech",
                    "context_start": start,
                    "context_end": end,
                    "score": 0.0,
                    "signals": {"fallback": "semantic_scan"},
                }
            )
        start = end
    return seeds


def _review_seed(seed, cues, translator, config, *, required: bool) -> ClipPart | None:
    context_start = float(seed["context_start"])
    context_end = float(seed["context_end"])
    selected = [
        (index, cue)
        for index, cue in enumerate(cues)
        if cue.end > context_start and cue.start < context_end
    ]
    def render(values):
        return _clip_request_body(config, "clip-review.md", _clip_prompt(seed, values, config))
    validate = (translator.validate_request if config.llm.local_server_enabled
                else request_budget_validator(config.llm.local_server_context_size))
    batches = batch_requests(selected, render_request=render, validate_request=validate)
    if not batches:
        validate(render([]))
        batches = [()]
    parts = [_review_seed_chunk(seed, list(batch), translator, config, required=required)
             for batch in batches]
    parts = [part for part in parts if part is not None]
    if not parts:
        return None
    if required:
        return replace(parts[0], start=min(part.start for part in parts), end=max(part.end for part in parts))
    return max(parts, key=lambda part: (part.score, part.end - part.start))


def _clip_prompt(seed, selected, config):
    payload = {
        **seed,
        "transcript_cues": [
            {
                "id": index,
                "start": round(cue.start, 3),
                "end": round(cue.end, 3),
                "text": cue.text,
            }
            for index, cue in selected
        ],
    }
    return render_user_prompt("clip-review.md", KIND=str(seed["kind"]),
        CANDIDATE_JSON=json.dumps(payload, ensure_ascii=False), MAX_SECONDS=config.clips.max_speech_seconds)


def _review_seed_chunk(seed, selected, translator, config, *, required):
    if not selected and not required:
        return None
    response = _request_json(translator, config, "clip-review.md", _clip_prompt(seed, selected, config))
    worthy = response.get("worthy") is True
    confidence = str(response.get("confidence") or "").strip().lower()
    if required and (not worthy or confidence != "high"):
        raise RuntimeError("song clip review was not high confidence")
    if not required and (not worthy or confidence != "high"):
        return None
    title = _required_text(response, "title")
    reason = _required_text(response, "reason")
    start, end = _validated_response_range(response, selected, seed, required)
    if not required:
        duration = end - start
        if duration < _MIN_SPEECH_SECONDS or duration > config.clips.max_speech_seconds:
            return None
    return ClipPart(
        kind=str(seed["kind"]),
        start=round(start, 3),
        end=round(end, 3),
        title=title[:70],
        reason=reason,
        confidence="high" if required else confidence,
        score=float(seed.get("score") or 0.0),
        song=str(seed.get("song") or "").strip() or None,
        artist=str(seed.get("artist") or "").strip() or None,
    )


def _review_seeds(seeds, cues, translator, config, *, required: bool, cache=None, prefix="seed"):
    results = []
    for index, seed in enumerate(seeds):
        key = f"{prefix}:{index}"
        saved = cache.get(key) if cache is not None else None
        if saved is not None:
            results.append(ClipPart(**saved["part"]) if saved["part"] is not None else None)
            continue
        error = None
        try:
            part = _review_seed(seed, cues, translator, config, required=required)
        except (RuntimeError, TypeError, ValueError, KeyError) as exc:
            logger.warning("discarding invalid %s clip review: %s", seed["kind"], exc)
            part = _required_song_fallback(seed) if required else None
            error = f"{type(exc).__name__}: {exc}"
        if cache is not None:
            cache.put(key, {"part": asdict(part) if part is not None else None},
                      source="clip_review", reason=error)
        results.append(part)
    return results


def _required_song_fallback(seed) -> ClipPart:
    song = str(seed["song"])
    artist = str(seed.get("artist") or "").strip() or None
    return ClipPart(
        kind="song",
        start=round(float(seed["performance_start"]), 3),
        end=round(float(seed["performance_end"]), 3),
        title=f"{song}{f' - {artist}' if artist else ''}"[:70],
        reason="LLM 边界复核失败，保留已验证的完整歌曲范围",
        confidence="high",
        score=float(seed.get("score") or 0.0),
        song=song,
        artist=artist,
    )


def _validated_response_range(response, selected, seed, required):
    by_id = {index: cue for index, cue in selected}
    start_id = response.get("start_id")
    end_id = response.get("end_id")
    if start_id is None and end_id is None and required:
        chosen_start = float(seed["performance_start"])
        chosen_end = float(seed["performance_end"])
    else:
        if not isinstance(start_id, int) or not isinstance(end_id, int):
            raise RuntimeError("clip LLM returned invalid start_id/end_id")
        ordered_ids = [index for index, _cue in selected]
        if start_id not in by_id or end_id not in by_id:
            raise RuntimeError("clip LLM selected a cue outside the candidate")
        if ordered_ids.index(start_id) > ordered_ids.index(end_id):
            raise RuntimeError("clip LLM returned reversed cue IDs")
        chosen_start = by_id[start_id].start
        chosen_end = by_id[end_id].end
    if required:
        chosen_start = min(chosen_start, float(seed["performance_start"]))
        chosen_end = max(chosen_end, float(seed["performance_end"]))
    if (
        chosen_start < float(seed["context_start"]) - 0.001
        or chosen_end > float(seed["context_end"]) + 0.001
    ):
        raise RuntimeError("clip LLM selected a range outside the supplied context")
    if chosen_end <= chosen_start:
        raise RuntimeError("clip LLM selected an empty range")
    return chosen_start, chosen_end


def _merge_speech_parts(
    speech: list[ClipPart], songs: list[ClipPart], maximum_seconds: float
) -> list[ClipPart]:
    without_songs = [
        part
        for part in speech
        if not any(
            _overlap(part.start, part.end, song.start, song.end) > 0 for song in songs
        )
    ]
    merged: list[ClipPart] = []
    for part in sorted(without_songs, key=lambda value: value.start):
        if merged and part.start <= merged[-1].end:
            previous = merged[-1]
            end = max(previous.end, part.end)
            if end - previous.start <= maximum_seconds:
                preferred = max((previous, part), key=lambda value: value.score)
                merged[-1] = ClipPart(
                    "speech",
                    previous.start,
                    end,
                    preferred.title,
                    preferred.reason,
                    "high",
                    max(previous.score, part.score),
                )
                continue
        merged.append(part)
    return merged


def _upload_metadata(metadata: dict[str, object]) -> dict[str, str]:
    return {
        "title": _required_text(metadata, "translated_title"),
        "description": _required_text(metadata, "translated_description"),
    }


def _clip_request_body(config, prompt_name, prompt):
    return structured_request_body(
        model=config.llm.model,
        prompt_name=prompt_name,
        prompt=prompt,
        max_tokens=min(4096, config.translation.max_tokens),
        temperature=0.1,

        thinking=config.llm.thinking,
    )


def _request_json(translator, config, prompt_name: str, prompt: str) -> dict[str, object]:
    body = _clip_request_body(config, prompt_name, prompt)
    last_error: Exception | None = None
    for attempt in range(1, config.llm.max_retries + 1):
        try:
            response = translator.request(body)
            content = structured_response_content(response, finish_reason=finish_reason)
            return parse_json_object(content)
        except Exception as exc:  # noqa: BLE001 - provider and parser failures retry alike
            last_error = exc
            logger.warning(
                "%s response attempt %d failed: %s", prompt_name, attempt, exc
            )
    raise RuntimeError(f"{prompt_name} exhausted LLM retries") from last_error


def _render_parts(
    source: Path,
    clips_dir: Path,
    records,
    config: AppConfig,
    *,
    force: bool = False,
) -> list[Path]:
    stage = CacheStore(clips_dir.parent / "cache.sqlite3").stage("clip_render", lambda: {
        "records": records, "render": config_snapshot(config.render),
    })
    records = stage.plan["records"]
    config = replace(config, render=restore_config(config.render, stage.plan["render"]))
    parts_dir = clips_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    output: list[Path] = []
    for record in records:
        relative = Path(_required_text(record, "file"))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parent != Path("parts")
        ):
            raise RuntimeError("clip analysis contains an unsafe output path")
        destination = clips_dir / relative
        if stage.get(str(relative)) is None or not destination.is_file() or destination.stat().st_size == 0:
            _render_clip(
                source,
                destination,
                float(record["start"]),
                float(record["end"]),
                config,
            )
            stage.put(str(relative), {"path": str(destination)})
        output.append(destination)
    stage.finish([str(path) for path in output])
    return output


def _render_clip(
    source: Path, destination: Path, start: float, end: float, config: AppConfig
) -> None:
    if end <= start:
        raise ValueError("clip end must be after start")
    ffmpeg = require_command("ffmpeg")
    temporary = destination.with_name(destination.stem + ".tmp.mp4")
    try:
        run(
            [
                ffmpeg,
                "-y",
                "-ss",
                f"{start:.3f}",
                "-i",
                str(source),
                "-t",
                f"{end - start:.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-map_metadata",
                "0",
                "-c:v",
                "libx264",
                "-preset",
                config.render.preset,
                "-crf",
                str(config.render.crf),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-avoid_negative_ts",
                "make_zero",
                "-movflags",
                "+faststart",
                str(temporary),
            ]
        )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _part_records(analysis: dict[str, object]) -> list[dict[str, object]]:
    values = analysis.get("parts")
    if not isinstance(values, list):
        raise TypeError("clip analysis parts must be a list")
    records = []
    for value in values:
        if not isinstance(value, dict):
            raise TypeError("clip analysis part must be an object")
        start = value.get("start")
        end = value.get("end")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or end <= start
        ):
            raise RuntimeError("clip analysis part has an invalid range")
        _required_text(value, "file")
        records.append(value)
    return sorted(
        records, key=lambda value: (float(value["start"]), float(value["end"]))
    )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _upload_tags(path: Path, fallback: list[str]) -> list[str]:
    metadata = _load_json_object(path, "translated metadata")
    tags = metadata.get("upload_tags")
    if isinstance(tags, list):
        values = [str(value).strip() for value in tags if str(value).strip()]
        if values:
            return values
    return fallback


def _required_text(value: dict[str, object], key: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"missing non-empty {key}")
    return text.strip()


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) else None


def _safe_filename(value: str) -> str:
    clean = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value).strip(" ._")
    return (clean or "highlight")[:70]


def _is_reaction(text: str) -> bool:
    clean = re.sub(r":[A-Za-z0-9_+-]+:", "", "".join(text.split()))
    return bool(clean and _REACTION_RE.fullmatch(clean)) or (
        not clean and bool(text.strip())
    )


def _overlap(
    left_start: float, left_end: float, right_start: float, right_end: float
) -> float:
    return max(0.0, min(left_end, right_end) - max(left_start, right_start))
