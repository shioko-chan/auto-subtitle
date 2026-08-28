from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .audio_buffer import AudioBuffer, AudioBufferPool
from .commands import require_command, run
from .config import AudioAnalysisConfig
from .telemetry import stage_metrics

logger = logging.getLogger(__name__)

# Transformers/Accelerate can temporarily install meta-device initialization
# hooks. Keep unrelated model construction out of that process-global context.
_MODEL_LOAD_LOCK = threading.Lock()

_CACHE_VERSION = 14
_SINGING_LABELS = {
    "chant",
    "child singing",
    "choir",
    "female singing",
    "male singing",
    "rapping",
    "singing",
    "synthetic singing",
    "yodeling",
}
_SPEECH_LABELS = {
    "child speech, kid speaking",
    "conversation",
    "female speech, woman speaking",
    "hubbub, speech noise, speech babble",
    "male speech, man speaking",
    "narration, monologue",
    "speech",
    "speech synthesizer",
}


def _is_music_activity_label(label: str) -> bool:
    return (
        label in {"music", "musical instrument", "orchestra"}
        or label.endswith(" music")
        or "(musical)" in label
    )


@dataclass(frozen=True)
class AudioRegion:
    start: float
    end: float
    kind: str
    speaker: str | None = None
    confidence: float | None = None
    overlap: bool = False
    source_path: str | None = None
    source_offset: float = 0.0
    anonymous_speaker: str | None = None
    overlap_seconds: float = 0.0
    overlap_speakers: tuple[str, ...] = ()
    asr_route: str = "qwen"
    speech_confidence: float | None = None
    music_confidence: float | None = None


@dataclass(frozen=True)
class AcousticPhrase:
    start: float
    end: float
    singing_level: str
    speech_level: str
    singing_score: float
    speech_score: float
    music_score: float
    vocal_score: float
    diarization_overlap_seconds: float
    route_alt: bool
    route_speech: bool
    source_path: str | None = None
    source_offset: float = 0.0


@dataclass(frozen=True)
class AudioAnalysis:
    speech: list[AudioRegion]
    singing: list[AudioRegion]
    diarization: list[AudioRegion] = field(default_factory=list)
    acoustic_phrases: list[AcousticPhrase] = field(default_factory=list)


def analyze_audio(
    video: Path,
    job_dir: Path,
    config: AudioAnalysisConfig,
    metadata: dict[str, object] | None = None,
    audio_pool: AudioBufferPool | None = None,
    force_runtime_sources: bool = False,
) -> AudioAnalysis:
    """Run reusable VAD/diarization and singing analysis before ASR."""
    job_dir.mkdir(parents=True, exist_ok=True)
    cache_path = job_dir / "audio-analysis.json"
    signature = _signature(video, config, metadata or {})
    cached = None if force_runtime_sources else _load_cache(cache_path, signature)
    if cached is not None:
        logger.info(
            "audio analysis cache: %d speech turns, %d ALT phrases",
            len(cached.speech),
            len(cached.singing),
        )
        return cached

    if audio_pool is not None:
        import torch

        buffer = audio_pool.main()
        waveform = torch.from_numpy(buffer.samples).unsqueeze(0)
        sample_rate = buffer.sample_rate
    else:
        wav_path = job_dir / "source.analysis.wav"
        _extract_audio(video, wav_path)
        waveform, sample_rate = _load_waveform(wav_path)
    ordinary_diarization, raw_scores = _run_initial_audio_analysis(
        video,
        job_dir,
        waveform,
        sample_rate,
        config,
        metadata or {},
        audio_buffer=audio_pool.main() if audio_pool is not None else None,
    )
    speech = _clean_speaker_timeline(ordinary_diarization)
    song_detection_audit: dict[str, object] = {}
    raw_candidates = _singing_regions_from_scores(
        raw_scores,
        threshold=config.singing_threshold,
        music_threshold=config.singing_music_threshold,
        smoothing_windows=config.singing_smoothing_windows,
        merge_gap_seconds=config.singing_merge_gap_seconds,
        audit=song_detection_audit,
    )
    vocal_candidates: list[tuple[AudioRegion, AudioBuffer]] = []
    vocal_scores: list[AudioRegion] = []
    fallback_vocals_path: Path | None = None
    if raw_candidates:
        phrase_candidates = _smart_acoustic_phrase_regions(
            raw_candidates,
            audio_pool.main() if audio_pool is not None else None,
            config,
        )
        with stage_metrics("audio.vocal_separation_and_detection", config.device):
            if audio_pool is not None:
                vocal_candidates = _separate_vocal_candidates(
                    video,
                    phrase_candidates,
                    config.device,
                    audio_pool,
                    # Persist candidate stems for later canonical-lyric alignment.
                    debug_dir=job_dir / "vocal-candidates",
                )
                vocal_scores = _score_singing_sources(
                    [
                        (
                            candidate.start,
                            _buffer_waveform(buffer),
                            buffer.sample_rate,
                        )
                        for candidate, buffer in vocal_candidates
                    ],
                    config,
                )
            else:
                vocals_path = job_dir / "source.vocals.wav"
                fallback_vocals_path = vocals_path.resolve()
                _separate_vocals(video, vocals_path, config.device)
                vocal_waveform, vocal_rate = _load_waveform(vocals_path)
                vocal_scores = _score_singing_windows(
                    vocal_waveform, vocal_rate, config
                )
    acoustic_phrases = _build_acoustic_phrases(
        raw_scores,
        phrase_candidates if raw_candidates else [],
        vocal_scores,
        ordinary_diarization,
        vocal_candidates,
        config,
        allow_unbound_vocal_source=fallback_vocals_path is not None,
    )
    alt_regions = [
        AudioRegion(phrase.start, phrase.end, "singing")
        for phrase in acoustic_phrases
        if phrase.route_alt
    ]
    from .speakers import identify_speakers, metadata_character

    known_character = metadata_character(metadata or {}, config.character_styles_file)
    with stage_metrics("audio.speaker_identity", config.device):
        speech = identify_speakers(
            waveform,
            sample_rate,
            speech,
            config,
            known_character=known_character,
            excluded_regions=alt_regions,
        )
        ordinary_diarization = _resolve_diarization_speakers(
            ordinary_diarization, speech
        )

    singing = [
        AudioRegion(
            phrase.start,
            phrase.end,
            "singing",
            known_character,
            phrase.singing_score,
            source_path=phrase.source_path
            or (
                str(fallback_vocals_path) if fallback_vocals_path is not None else None
            ),
            source_offset=phrase.source_offset,
            speech_confidence=phrase.speech_score,
            music_confidence=phrase.music_score,
        )
        for phrase in acoustic_phrases
        if phrase.route_alt
    ]
    result = AudioAnalysis(
        speech=speech,
        singing=singing,
        diarization=ordinary_diarization,
        acoustic_phrases=acoustic_phrases,
    )
    payload = {
        "version": _CACHE_VERSION,
        "signature": _signature(video, config, metadata or {}),
        "speech": [asdict(region) for region in speech],
        "singing": [asdict(region) for region in singing],
        "diarization": [asdict(region) for region in ordinary_diarization],
        "acoustic_phrases": [asdict(phrase) for phrase in acoustic_phrases],
        "song_detection": song_detection_audit,
    }
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    logger.info(
        "audio analysis wrote %d speech turns and %d ALT phrases",
        len(speech),
        len(singing),
    )
    return result


def _run_initial_audio_analysis(
    video: Path,
    job_dir: Path,
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
    metadata: dict[str, object],
    *,
    audio_buffer: AudioBuffer | None = None,
) -> tuple[list[AudioRegion], list[AudioRegion]]:
    def diarize() -> list[AudioRegion]:
        with stage_metrics("audio.diarization", config.device):
            if config.diarization_backend == "moss":
                from .moss_diarization import transcribe_and_diarize

                result = transcribe_and_diarize(
                    video,
                    job_dir,
                    config,
                    metadata,
                    audio_buffer=audio_buffer,
                    waveform=waveform,
                    sample_rate=sample_rate,
                )
                ordinary = _mark_overlaps(
                    [
                        AudioRegion(
                            segment.start,
                            segment.end,
                            "speech",
                            segment.speaker,
                            anonymous_speaker=segment.speaker,
                        )
                        for segment in result.segments
                    ]
                )
                return ordinary
            return _run_diarization(waveform, sample_rate, config)

    def detect_singing() -> list[AudioRegion]:
        with stage_metrics("audio.raw_singing_detection", config.device):
            return _score_singing_windows(waveform, sample_rate, config)

    if config.initial_analysis_concurrency == 1:
        return diarize(), detect_singing()
    logger.info(
        "running %s diarization and raw-audio AST concurrently",
        config.diarization_backend,
    )
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="audio-analysis") as pool:
        diarization = pool.submit(diarize)
        singing = pool.submit(detect_singing)
        return diarization.result(), singing.result()


def _run_diarization(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", module=r"pyannote\.audio\.core\.io")
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "speaker diarization is unavailable; run `uv sync --extra asr`"
        ) from exc

    logger.info("loading speaker diarization model %s", config.diarization_model)
    with _MODEL_LOAD_LOCK:
        pipeline = Pipeline.from_pretrained(config.diarization_model)
        if pipeline is None:
            raise RuntimeError(
                f"could not load gated diarization model {config.diarization_model}"
            )
        pipeline.to(torch.device(config.device))
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        ordinary_annotation = getattr(output, "speaker_diarization", output)
        ordinary = [
            AudioRegion(
                round(float(segment.start), 3),
                round(float(segment.end), 3),
                "speech",
                str(speaker),
            )
            for segment, _track, speaker in ordinary_annotation.itertracks(
                yield_label=True
            )
            if float(segment.end) > float(segment.start)
        ]
        return _mark_overlaps(ordinary)
    finally:
        del pipeline
        _release_cuda()


def _score_singing_windows(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    return _score_singing_sources([(0.0, waveform, sample_rate)], config)


def _score_singing_sources(
    sources: list[tuple[float, Any, int]],
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch
        from transformers import ASTForAudioClassification, AutoFeatureExtractor
    except ImportError as exc:
        raise RuntimeError(
            "singing detection is unavailable; run `uv sync --extra asr`"
        ) from exc

    target_rate = 16000
    prepared: list[tuple[float, Any]] = []
    for timeline_offset, waveform, sample_rate in sources:
        mono_tensor = waveform.mean(dim=0).detach().cpu()
        if sample_rate != target_rate:
            import torchaudio.functional as audio_functional

            mono_tensor = audio_functional.resample(
                mono_tensor, sample_rate, target_rate
            )
        prepared.append((timeline_offset, mono_tensor.numpy()))
    window_samples = max(1, round(config.singing_window_seconds * target_rate))
    stride_samples = max(1, round(config.singing_stride_seconds * target_rate))
    windows: list[tuple[Any, float, float]] = []
    for timeline_offset, mono in prepared:
        starts = list(range(0, max(1, len(mono) - window_samples + 1), stride_samples))
        if not starts or starts[-1] + window_samples < len(mono):
            starts.append(max(0, len(mono) - window_samples))
        windows.extend(
            (
                mono[offset : offset + window_samples],
                timeline_offset + offset / target_rate,
                timeline_offset + min(len(mono), offset + window_samples) / target_rate,
            )
            for offset in starts
        )

    logger.info("loading singing detector %s", config.singing_model)
    with _MODEL_LOAD_LOCK:
        extractor = AutoFeatureExtractor.from_pretrained(config.singing_model)
        model = ASTForAudioClassification.from_pretrained(config.singing_model).to(
            config.device
        )
    model.eval()
    labels = {
        int(index): str(label).casefold()
        for index, label in model.config.id2label.items()
    }
    singing_ids = [index for index, label in labels.items() if label in _SINGING_LABELS]
    speech_ids = [index for index, label in labels.items() if label in _SPEECH_LABELS]
    music_ids = [
        index for index, label in labels.items() if _is_music_activity_label(label)
    ]
    if not singing_ids:
        raise RuntimeError("AST singing detector exposes no recognized singing labels")

    scored_windows: list[AudioRegion] = []
    try:
        for batch_start in range(0, len(windows), 16):
            batch = windows[batch_start : batch_start + 16]
            inputs = extractor(
                [item[0] for item in batch],
                sampling_rate=target_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(config.device) for key, value in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
            for (_audio, start, end), row in zip(batch, probabilities):
                singing_score = sum(float(row[index]) for index in singing_ids)
                speech_score = sum(float(row[index]) for index in speech_ids)
                music_score = sum(float(row[index]) for index in music_ids)
                score = _singing_evidence_score(singing_score, speech_score)
                scored_windows.append(
                    AudioRegion(
                        round(start, 3),
                        round(end, 3),
                        "singing",
                        confidence=round(score, 4),
                        speech_confidence=round(speech_score, 4),
                        music_confidence=round(music_score, 4),
                    )
                )
    finally:
        del model
        _release_cuda()
    return scored_windows


def _singing_evidence_score(singing_score: float, speech_score: float) -> float:
    # AudioSet labels are evidence dimensions, not mutually exclusive routing
    # decisions. Calls and spoken lines over a song can legitimately score as
    # both speech and singing, so speech must not erase singing evidence.
    del speech_score
    return max(0.0, singing_score)


def _build_acoustic_phrases(
    raw_scores: list[AudioRegion],
    raw_candidates: list[AudioRegion],
    vocal_scores: list[AudioRegion],
    diarization: list[AudioRegion],
    vocal_candidates: list[tuple[AudioRegion, AudioBuffer]],
    config: AudioAnalysisConfig,
    *,
    allow_unbound_vocal_source: bool = False,
) -> list[AcousticPhrase]:
    """Split selected AST candidate spans into independent ASR routing units."""
    intervals = [(span.start, span.end) for span in raw_candidates]

    phrases: list[AcousticPhrase] = []
    for start, end in intervals:
        raw = [
            item
            for item in raw_scores
            if min(end, item.end) - max(start, item.start) > 0
        ]
        vocals = [
            item
            for item in vocal_scores
            if min(end, item.end) - max(start, item.start) > 0
        ]
        singing_score = max(
            (float(item.confidence or 0.0) for item in raw), default=0.0
        )
        speech_score = max(
            (float(item.speech_confidence or 0.0) for item in raw), default=0.0
        )
        music_score = max(
            (float(item.music_confidence or 0.0) for item in raw), default=0.0
        )
        vocal_score = max(
            (float(item.confidence or 0.0) for item in vocals), default=0.0
        )
        overlap_spans = [
            (max(start, turn.start), min(end, turn.end))
            for turn in diarization
            if min(end, turn.end) - max(start, turn.start) > 0
        ]
        diarization_overlap = sum(
            item.end - item.start
            for item in _merge_regions(
                [AudioRegion(left, right, "speech") for left, right in overlap_spans],
                0.0,
            )
        )
        singing_level, speech_level, route_alt, route_speech = _acoustic_phrase_route(
            singing_score,
            speech_score,
            vocal_score,
            diarization_overlap,
            config,
        )
        source = next(
            (
                item
                for item in vocal_candidates
                if item[0].start <= start + 1e-3 and item[0].end >= end - 1e-3
            ),
            None,
        )
        if route_alt and source is None and not allow_unbound_vocal_source:
            logger.warning(
                "dropping ALT route %.3f-%.3fs without complete vocal stem coverage",
                start,
                end,
            )
            route_alt = False
        phrases.append(
            AcousticPhrase(
                round(start, 3),
                round(end, 3),
                singing_level,
                speech_level,
                round(singing_score, 4),
                round(speech_score, 4),
                round(music_score, 4),
                round(vocal_score, 4),
                round(diarization_overlap, 3),
                route_alt,
                route_speech,
                source[1].uri if source is not None else None,
                source[0].start if source is not None else 0.0,
            )
        )
    return phrases


def _smart_acoustic_phrase_regions(
    candidates: list[AudioRegion],
    audio: AudioBuffer | None,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    """Cut AST ranges once, before Demucs and ALT, at stable acoustic valleys."""
    result: list[AudioRegion] = []
    for candidate in candidates:
        if candidate.end - candidate.start <= config.singing_asr_max_seconds:
            result.append(candidate)
            continue
        cut_points = _acoustic_cut_points(candidate, audio, config)
        cursor = candidate.start
        while candidate.end - cursor > config.singing_asr_max_seconds:
            minimum = cursor + config.singing_asr_min_seconds
            maximum = min(cursor + config.singing_asr_max_seconds, candidate.end)
            desired = min(cursor + config.singing_asr_target_seconds, maximum)
            search_start = max(minimum, desired - config.singing_asr_search_seconds)
            search_end = min(maximum, desired + config.singing_asr_search_seconds)
            options = [
                value for value in cut_points if search_start <= value[0] <= search_end
            ]
            cut = (
                min(options, key=lambda value: (value[1], abs(value[0] - desired)))[0]
                if options
                else desired
            )
            if candidate.end - cut < config.singing_asr_min_seconds:
                cut = max(minimum, candidate.end - config.singing_asr_min_seconds)
            result.append(AudioRegion(round(cursor, 3), round(cut, 3), "singing"))
            cursor = cut
        if candidate.end - cursor > 1e-3:
            result.append(
                AudioRegion(round(cursor, 3), round(candidate.end, 3), "singing")
            )
    return result


def _acoustic_cut_points(
    candidate: AudioRegion,
    audio: AudioBuffer | None,
    config: AudioAnalysisConfig,
) -> list[tuple[float, int]]:
    if audio is None:
        return []
    import numpy as np

    samples = np.asarray(audio.slice(candidate.start, candidate.end), dtype=np.float32)
    if not len(samples):
        return []
    sample_rate = audio.sample_rate
    frame_length = min(max(1, round(0.05 * sample_rate)), len(samples))
    hop_length = min(max(1, round(0.025 * sample_rate)), frame_length)
    starts = np.arange(0, len(samples), hop_length)
    ends = np.minimum(starts + frame_length, len(samples))
    prefix = np.concatenate(([0.0], np.cumsum(np.square(samples), dtype=np.float64)))
    rms = np.sqrt((prefix[ends] - prefix[starts]) / np.maximum(1, ends - starts))
    peak = max(float(rms.max()), 1e-9)
    low = rms <= peak * (10 ** (-35 / 20))
    points: list[tuple[float, int]] = []
    run_start: int | None = None
    for index in range(len(low) + 1):
        is_low = index < len(low) and bool(low[index])
        if is_low and run_start is None:
            run_start = index
        elif not is_low and run_start is not None:
            gap_start = starts[run_start] / sample_rate
            gap_end = min(len(samples), starts[index - 1] + frame_length) / sample_rate
            duration = gap_end - gap_start
            if duration >= config.singing_phrase_silence_seconds:
                priority = (
                    0
                    if duration >= max(0.8, config.singing_phrase_silence_seconds)
                    else 2
                )
                points.append((candidate.start + (gap_start + gap_end) / 2, priority))
            run_start = None
    if len(rms) >= 3:
        width = max(1, round(0.4 * sample_rate / hop_length))
        smoothed = np.convolve(rms, np.ones(width) / width, mode="same")
        for index in range(1, len(smoothed) - 1):
            if (
                smoothed[index] <= smoothed[index - 1]
                and smoothed[index] < smoothed[index + 1]
            ):
                points.append(
                    (
                        candidate.start
                        + (starts[index] + frame_length / 2) / sample_rate,
                        1,
                    )
                )
    return points


def _acoustic_phrase_route(
    singing_score: float,
    speech_score: float,
    vocal_score: float,
    diarization_overlap_seconds: float,
    config: AudioAnalysisConfig,
) -> tuple[str, str, bool, bool]:
    if vocal_score >= config.singing_vocal_threshold:
        singing_level = "high"
    elif singing_score >= config.singing_threshold:
        singing_level = "medium"
    else:
        singing_level = "low"
    if (
        diarization_overlap_seconds >= 0.08
        or speech_score >= config.singing_speech_takeover_threshold
    ):
        speech_level = "strong"
    elif speech_score > 0:
        speech_level = "weak"
    else:
        speech_level = "none"
    return (
        singing_level,
        speech_level,
        singing_level in {"high", "medium"},
        speech_level == "strong",
    )


def _overlap_duration(left: AudioRegion, right: AudioRegion) -> float:
    return max(0.0, min(left.end, right.end) - max(left.start, right.start))


def _singing_regions_from_scores(
    windows: list[AudioRegion],
    *,
    threshold: float,
    smoothing_windows: int,
    merge_gap_seconds: float = 0.0,
    music_threshold: float | None = None,
    audit: dict[str, object] | None = None,
) -> list[AudioRegion]:
    """Select independent ALT candidates from overlapping classifier windows."""
    if not windows:
        if audit is not None:
            audit.update({"windows": [], "candidates": []})
        return []

    radius = smoothing_windows // 2
    smoothed: list[AudioRegion] = []
    for index, window in enumerate(windows):
        nearby = windows[max(0, index - radius) : index + radius + 1]
        score = _padded_median(
            [float(item.confidence or 0.0) for item in nearby], smoothing_windows
        )
        speech_score = _padded_median(
            [float(item.speech_confidence or 0.0) for item in nearby],
            smoothing_windows,
        )
        music_score = _padded_median(
            [float(item.music_confidence or 0.0) for item in nearby],
            smoothing_windows,
        )
        smoothed.append(
            AudioRegion(
                window.start,
                window.end,
                "singing",
                confidence=round(score, 4),
                speech_confidence=round(speech_score, 4),
                music_confidence=round(music_score, 4),
            )
        )

    window_audit: list[dict[str, object]] = []
    candidates: list[AudioRegion] = []
    for raw, smooth in zip(windows, smoothed):
        music_support = (
            music_threshold is not None
            and float(smooth.music_confidence or 0.0) >= music_threshold
        )
        selected = float(smooth.confidence or 0.0) >= threshold or (
            float(raw.confidence or 0.0) >= threshold and music_support
        )
        if selected:
            candidates.append(
                AudioRegion(
                    smooth.start,
                    smooth.end,
                    "singing",
                    confidence=max(
                        float(raw.confidence or 0.0),
                        float(smooth.confidence or 0.0),
                    ),
                )
            )
        window_audit.append(
            {
                "start": round(smooth.start, 3),
                "end": round(smooth.end, 3),
                "raw_singing": round(float(raw.confidence or 0.0), 4),
                "singing": round(float(smooth.confidence or 0.0), 4),
                "speech": round(float(smooth.speech_confidence or 0.0), 4),
                "music": round(float(smooth.music_confidence or 0.0), 4),
                "music_support": music_support,
                "selected": selected,
            }
        )

    result = _merge_regions(candidates, merge_gap_seconds)
    if audit is not None:
        audit.update(
            {
                "policy": {
                    "singing_threshold": threshold,
                    "music_threshold": music_threshold,
                    "smoothing_windows": smoothing_windows,
                },
                "windows": window_audit,
                "candidates": [asdict(region) for region in result],
            }
        )
    return result


def _padded_median(values: list[float], size: int) -> float:
    padded = [*values, *([0.0] * (size - len(values)))]
    padded.sort()
    return padded[len(padded) // 2]


def _separate_vocals(source: Path, destination: Path, device: str) -> None:
    if destination.is_file():
        return
    try:
        import soundfile as sf
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "vocal separation requires the ASR optional dependencies"
        ) from exc

    logger.info("separating vocals with Demucs htdemucs")
    separator = Separator(model="htdemucs", device=device, progress=True)
    try:
        _origin, stems = separator.separate_audio_file(source)
        vocals = stems.get("vocals")
        if vocals is None:
            raise RuntimeError("Demucs did not return a vocals stem")
        data = vocals.detach().cpu().numpy().T
        sf.write(destination, data, separator.samplerate)
    finally:
        del separator
        _release_cuda()


def _separate_vocal_candidates(
    video: Path,
    candidates: list[AudioRegion],
    device: str,
    audio_pool: AudioBufferPool,
    *,
    debug_dir: Path | None = None,
) -> list[tuple[AudioRegion, AudioBuffer]]:
    try:
        import soundfile as sf
        import torchaudio.functional as audio_functional
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "vocal separation requires the ASR optional dependencies"
        ) from exc

    logger.info(
        "separating vocals for %d source-quality song candidate ranges",
        len(candidates),
    )
    separator = Separator(model="htdemucs", device=device, progress=True)
    outputs: list[tuple[AudioRegion, AudioBuffer]] = []
    try:
        for index, candidate in enumerate(candidates):
            waveform, sample_rate = _decode_stereo_range(
                video,
                candidate.start,
                candidate.end,
                separator.samplerate,
            )
            _origin, stems = separator.separate_tensor(waveform, sr=sample_rate)
            vocals = stems.get("vocals")
            if vocals is None:
                raise RuntimeError("Demucs did not return a vocals stem")
            mono = vocals.mean(dim=0).detach().cpu()
            if separator.samplerate != 16000:
                mono = audio_functional.resample(mono, separator.samplerate, 16000)
            buffer = audio_pool.add(mono.numpy(), 16000)
            outputs.append((candidate, buffer))
            if debug_dir is not None:
                debug_dir.mkdir(parents=True, exist_ok=True)
                sf.write(
                    debug_dir / f"candidate-{index:04d}.vocals.wav",
                    buffer.samples,
                    buffer.sample_rate,
                )
    finally:
        del separator
        _release_cuda()
    if debug_dir is not None:
        (debug_dir / "manifest.json").write_text(
            json.dumps(
                [
                    {
                        "start": candidate.start,
                        "end": candidate.end,
                        "path": f"candidate-{index:04d}.vocals.wav",
                        "source": "ast_candidate",
                    }
                    for index, candidate in enumerate(candidates)
                ],
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return outputs


def separate_vocal_ranges(
    video: Path,
    ranges: list[tuple[float, float, Path]],
    device: str,
) -> None:
    """Persist Demucs vocal stems for explicit timeline ranges in one model load."""
    if not ranges:
        return
    try:
        import soundfile as sf
        import torchaudio.functional as audio_functional
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError(
            "vocal separation requires the ASR optional dependencies"
        ) from exc

    logger.info("separating vocals for %d verified lyric gap ranges", len(ranges))
    separator = Separator(model="htdemucs", device=device, progress=True)
    try:
        for start, end, destination in ranges:
            if destination.is_file():
                continue
            waveform, sample_rate = _decode_stereo_range(
                video, start, end, separator.samplerate
            )
            _origin, stems = separator.separate_tensor(waveform, sr=sample_rate)
            vocals = stems.get("vocals")
            if vocals is None:
                raise RuntimeError("Demucs did not return a vocals stem")
            mono = vocals.mean(dim=0).detach().cpu()
            if separator.samplerate != 16000:
                mono = audio_functional.resample(mono, separator.samplerate, 16000)
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, mono.numpy(), 16000)
    finally:
        del separator
        _release_cuda()


def _decode_stereo_range(
    video: Path,
    start: float,
    end: float,
    sample_rate: int,
) -> tuple[Any, int]:
    import numpy as np
    import torch

    ffmpeg = require_command("ffmpeg")
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-ss",
            f"{start:.3f}",
            "-t",
            f"{end - start:.3f}",
            "-i",
            str(video.resolve()),
            "-vn",
            "-ac",
            "2",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "could not decode source-quality song candidate: "
            + result.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    values = np.frombuffer(result.stdout, dtype=np.float32)
    if len(values) < 2:
        raise RuntimeError("song candidate contains no decoded audio")
    values = values[: len(values) - len(values) % 2].reshape(-1, 2)
    return torch.from_numpy(values.T.copy()), sample_rate


def _buffer_waveform(buffer: AudioBuffer) -> Any:
    import torch

    return torch.from_numpy(buffer.samples).unsqueeze(0)


def _mark_overlaps(regions: list[AudioRegion]) -> list[AudioRegion]:
    marked: list[AudioRegion] = []
    for index, region in enumerate(regions):
        overlap = any(
            other.speaker != region.speaker
            and min(region.end, other.end) > max(region.start, other.start)
            for other in regions[index + 1 :]
            if other.start < region.end
        ) or any(
            other.speaker != region.speaker
            and min(region.end, other.end) > max(region.start, other.start)
            for other in regions[:index]
            if other.end > region.start
        )
        marked.append(AudioRegion(**{**asdict(region), "overlap": overlap}))
    return marked


def _overlap_intersections(regions: list[AudioRegion]) -> list[AudioRegion]:
    ordered = sorted(regions, key=lambda item: (item.start, item.end))
    intersections: list[AudioRegion] = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if right.start >= left.end:
                break
            left_label = left.speaker or left.anonymous_speaker
            right_label = right.speaker or right.anonymous_speaker
            if not left_label or not right_label or left_label == right_label:
                continue
            start = max(left.start, right.start)
            end = min(left.end, right.end)
            if end <= start:
                continue
            speakers = tuple(sorted({left_label, right_label}))
            intersections.append(
                AudioRegion(
                    start,
                    end,
                    "overlap",
                    overlap=True,
                    overlap_seconds=end - start,
                    overlap_speakers=speakers,
                )
            )
    return intersections


def _clean_speaker_timeline(ordinary: list[AudioRegion]) -> list[AudioRegion]:
    intersections = _overlap_intersections(ordinary)
    overlap_spans = [(item.start, item.end) for item in intersections]
    clean: list[AudioRegion] = []
    for region in ordinary:
        for fragment in _subtract_regions(region, overlap_spans):
            clean.append(
                AudioRegion(
                    **{
                        **asdict(fragment),
                        "anonymous_speaker": (
                            region.anonymous_speaker or region.speaker
                        ),
                        "overlap": False,
                        "overlap_seconds": 0.0,
                        "overlap_speakers": (),
                        "asr_route": "qwen",
                    }
                )
            )
    return sorted(clean, key=lambda item: (item.start, item.end, item.speaker or ""))


def _resolve_diarization_speakers(
    ordinary: list[AudioRegion], resolved_exclusive: list[AudioRegion]
) -> list[AudioRegion]:
    mapping = _speaker_resolution_mapping(resolved_exclusive)
    return [
        AudioRegion(
            **{
                **asdict(region),
                "speaker": mapping.get(region.speaker or "", region.speaker),
                "anonymous_speaker": region.anonymous_speaker or region.speaker,
                "overlap_speakers": tuple(
                    mapping.get(speaker, speaker) for speaker in region.overlap_speakers
                ),
            }
        )
        for region in ordinary
    ]


def _resolve_overlap_speakers(
    regions: list[AudioRegion], resolved_exclusive: list[AudioRegion]
) -> list[AudioRegion]:
    mapping = _speaker_resolution_mapping(resolved_exclusive)
    return [
        AudioRegion(
            **{
                **asdict(region),
                "overlap_speakers": tuple(
                    mapping.get(speaker, speaker) for speaker in region.overlap_speakers
                ),
            }
        )
        for region in regions
    ]


def _speaker_resolution_mapping(
    resolved_exclusive: list[AudioRegion],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for region in resolved_exclusive:
        anonymous = region.anonymous_speaker
        if anonymous and region.speaker and anonymous != region.speaker:
            mapping[anonymous] = region.speaker
    return mapping


def _subtract_regions(
    region: AudioRegion, spans: list[tuple[float, float]]
) -> list[AudioRegion]:
    fragments = [(region.start, region.end)]
    for cut_start, cut_end in spans:
        updated: list[tuple[float, float]] = []
        for start, end in fragments:
            if cut_end <= start or cut_start >= end:
                updated.append((start, end))
                continue
            if start < cut_start:
                updated.append((start, cut_start))
            if cut_end < end:
                updated.append((cut_end, end))
        fragments = updated
    return [
        AudioRegion(
            start,
            end,
            region.kind,
            region.speaker,
            region.confidence,
            False,
            region.source_path,
            region.source_offset,
        )
        for start, end in fragments
        if end - start >= 0.08
    ]


def _exclude_timeline_regions(
    regions: list[AudioRegion], excluded: list[AudioRegion]
) -> list[AudioRegion]:
    result: list[AudioRegion] = []
    for region in regions:
        fragments = [(region.start, region.end)]
        for item in excluded:
            updated: list[tuple[float, float]] = []
            for start, end in fragments:
                if item.end <= start or item.start >= end:
                    updated.append((start, end))
                    continue
                if start < item.start:
                    updated.append((start, item.start))
                if item.end < end:
                    updated.append((item.end, end))
            fragments = updated
        result.extend(
            AudioRegion(**{**asdict(region), "start": start, "end": end})
            for start, end in fragments
            if end - start >= 0.08
        )
    return result


def _merge_regions(regions: list[AudioRegion], maximum_gap: float) -> list[AudioRegion]:
    if not regions:
        return []
    ordered = sorted(regions, key=lambda item: (item.start, item.end))
    merged = [ordered[0]]
    for region in ordered[1:]:
        previous = merged[-1]
        if region.start <= previous.end + maximum_gap:
            scores = [
                score
                for score in (previous.confidence, region.confidence)
                if score is not None
            ]
            merged[-1] = AudioRegion(
                previous.start,
                max(previous.end, region.end),
                previous.kind,
                confidence=max(scores) if scores else None,
            )
        else:
            merged.append(region)
    return merged


def _load_waveform(path: Path) -> tuple[Any, int]:
    import soundfile as sf
    import torch

    data, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    return torch.from_numpy(data.T.copy()), int(sample_rate)


def _extract_audio(video: Path, destination: Path) -> None:
    ffmpeg = require_command("ffmpeg")
    run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video.resolve()),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(destination.resolve()),
        ]
    )


def _release_cuda() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _signature(
    video: Path, config: AudioAnalysisConfig, metadata: dict[str, object]
) -> str:
    stat = video.stat()
    profile_dir = Path(config.speaker_profiles_dir).expanduser()
    profile_state = (
        [
            (path.name, path.stat().st_size, path.stat().st_mtime_ns)
            for path in sorted(profile_dir.glob("*.json"))
        ]
        if profile_dir.is_dir()
        else []
    )
    payload = {
        "version": _CACHE_VERSION,
        "video_size": stat.st_size,
        "video_mtime_ns": stat.st_mtime_ns,
        "config": asdict(config),
        "channel_identity": {
            key: metadata.get(key)
            for key in ("channel", "channel_id", "uploader", "uploader_id")
        },
        "speaker_profiles": profile_state,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_cache(path: Path, signature: str) -> AudioAnalysis | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("version") != _CACHE_VERSION
            or value.get("signature") != signature
        ):
            return None
        result = AudioAnalysis(
            speech=[_decode_audio_region(item) for item in value["speech"]],
            singing=[_decode_audio_region(item) for item in value["singing"]],
            diarization=[
                _decode_audio_region(item) for item in value.get("diarization", [])
            ],
            acoustic_phrases=[
                AcousticPhrase(**item)
                for item in value.get("acoustic_phrases", [])
                if isinstance(item, dict)
            ],
        )
        return result
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logger.warning("ignoring unreadable audio analysis cache %s: %s", path, exc)
        return None


def _decode_audio_region(value: dict[str, object]) -> AudioRegion:
    normalized = dict(value)
    normalized["overlap_speakers"] = tuple(value.get("overlap_speakers", ()))
    return AudioRegion(**normalized)
