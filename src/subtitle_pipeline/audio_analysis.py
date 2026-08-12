from __future__ import annotations

import hashlib
import json
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .commands import require_command, run
from .config import AudioAnalysisConfig


_CACHE_VERSION = 2
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


@dataclass(frozen=True)
class AudioAnalysis:
    speech: list[AudioRegion]
    singing: list[AudioRegion]


def analyze_audio(
    video: Path,
    job_dir: Path,
    config: AudioAnalysisConfig,
    metadata: dict[str, object] | None = None,
) -> AudioAnalysis:
    """Run reusable VAD/diarization and singing analysis before ASR."""
    job_dir.mkdir(parents=True, exist_ok=True)
    cache_path = job_dir / "audio-analysis.json"
    signature = _signature(video, config, metadata or {})
    cached = _load_cache(cache_path, signature)
    if cached is not None:
        logging.info(
            "audio analysis cache: %d speech turns, %d singing regions",
            len(cached.speech),
            len(cached.singing),
        )
        return cached

    wav_path = job_dir / "source.analysis.wav"
    _extract_audio(video, wav_path)
    waveform, sample_rate = _load_waveform(wav_path)
    speech = _run_diarization(waveform, sample_rate, config)
    if any(region.overlap for region in speech):
        speech = _run_overlap_separation(
            waveform,
            sample_rate,
            speech,
            config,
            job_dir / "separated-speakers",
        )
    singing_windows = _run_singing_detection(waveform, sample_rate, config)
    from .speakers import identify_speakers, metadata_character

    known_character = metadata_character(
        metadata or {}, config.character_styles_file
    )
    speech = identify_speakers(
        waveform,
        sample_rate,
        speech,
        config,
        known_character=known_character,
        excluded_regions=singing_windows,
    )

    vocals_path = job_dir / "source.vocals.wav"
    singing = singing_windows
    if singing_windows:
        _separate_vocals(wav_path, vocals_path, config.device)
        vocal_waveform, vocal_rate = _load_waveform(vocals_path)
        singing = _singing_phrases(
            vocal_waveform,
            vocal_rate,
            singing_windows,
            silence_seconds=config.singing_phrase_silence_seconds,
        )
        singing = [
            region
            for region in singing
            if region.end - region.start >= config.singing_min_phrase_seconds
        ]
        singing = [
            AudioRegion(
                **{
                    **asdict(region),
                    "speaker": known_character,
                    "source_path": str(vocals_path.resolve()),
                }
            )
            for region in singing
        ]

    result = AudioAnalysis(speech=speech, singing=singing)
    payload = {
        "version": _CACHE_VERSION,
        "signature": _signature(video, config, metadata or {}),
        "speech": [asdict(region) for region in speech],
        "singing": [asdict(region) for region in singing],
    }
    temporary = cache_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(cache_path)
    logging.info(
        "audio analysis wrote %d speech turns and %d singing phrases",
        len(speech),
        len(singing),
    )
    return result


def _run_diarization(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", module=r"pyannote\.audio\.core\.io"
            )
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "speaker diarization is unavailable; run `uv sync --extra asr`"
        ) from exc

    logging.info("loading speaker diarization model %s", config.diarization_model)
    pipeline = Pipeline.from_pretrained(config.diarization_model)
    if pipeline is None:
        raise RuntimeError(
            f"could not load gated diarization model {config.diarization_model}"
        )
    pipeline.to(torch.device(config.device))
    try:
        output = pipeline({"waveform": waveform, "sample_rate": sample_rate})
        annotation = getattr(output, "speaker_diarization", output)
        raw = [
            AudioRegion(
                round(float(segment.start), 3),
                round(float(segment.end), 3),
                "speech",
                str(speaker),
            )
            for segment, _track, speaker in annotation.itertracks(yield_label=True)
            if float(segment.end) > float(segment.start)
        ]
        return _mark_overlaps(raw)
    finally:
        del pipeline
        _release_cuda()


def _run_singing_detection(
    waveform: Any,
    sample_rate: int,
    config: AudioAnalysisConfig,
) -> list[AudioRegion]:
    try:
        import torch
        from transformers import AutoFeatureExtractor, ASTForAudioClassification
    except ImportError as exc:
        raise RuntimeError(
            "singing detection is unavailable; run `uv sync --extra asr`"
        ) from exc

    mono = waveform.mean(dim=0).detach().cpu().numpy()
    window_samples = max(1, round(config.singing_window_seconds * sample_rate))
    stride_samples = max(1, round(config.singing_stride_seconds * sample_rate))
    starts = list(range(0, max(1, len(mono) - window_samples + 1), stride_samples))
    if not starts or starts[-1] + window_samples < len(mono):
        starts.append(max(0, len(mono) - window_samples))

    logging.info("loading singing detector %s", config.singing_model)
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
    if not singing_ids:
        raise RuntimeError("AST singing detector exposes no recognized singing labels")

    windows: list[AudioRegion] = []
    try:
        for batch_start in range(0, len(starts), 16):
            batch_offsets = starts[batch_start : batch_start + 16]
            audio = [mono[offset : offset + window_samples] for offset in batch_offsets]
            inputs = extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True,
            )
            inputs = {key: value.to(config.device) for key, value in inputs.items()}
            with torch.inference_mode():
                probabilities = torch.softmax(model(**inputs).logits, dim=-1).cpu()
            for offset, row in zip(batch_offsets, probabilities):
                score = sum(float(row[index]) for index in singing_ids)
                if score >= config.singing_threshold:
                    windows.append(
                        AudioRegion(
                            round(offset / sample_rate, 3),
                            round(min(len(mono), offset + window_samples) / sample_rate, 3),
                            "singing",
                            confidence=round(score, 4),
                        )
                    )
    finally:
        del model
        _release_cuda()
    return _merge_regions(windows, config.singing_merge_gap_seconds)


def _run_overlap_separation(
    waveform: Any,
    sample_rate: int,
    diarized_regions: list[AudioRegion],
    config: AudioAnalysisConfig,
    output_dir: Path,
) -> list[AudioRegion]:
    """Separate overlapping speakers while retaining full-recording timestamps."""
    try:
        import soundfile as sf
        import torch
        from pyannote.audio.pipelines import SpeechSeparation
    except ImportError as exc:
        raise RuntimeError(
            "overlap separation is unavailable; install the ASR optional dependencies"
        ) from exc

    overlap_spans = _overlap_spans(diarized_regions, float(waveform.shape[-1]) / sample_rate)
    if not overlap_spans:
        return diarized_regions
    logging.info(
        "loading overlap separator %s for %d local regions",
        config.overlap_separation_model,
        len(overlap_spans),
    )
    pipeline = SpeechSeparation(
        segmentation=config.overlap_separation_model,
        segmentation_step=0.1,
        embedding=config.speaker_embedding_model,
        embedding_exclude_overlap=False,
        clustering="AgglomerativeClustering",
        embedding_batch_size=8,
        segmentation_batch_size=8,
    )
    pipeline.instantiate(
        {
            "segmentation": {"min_duration_off": 0.0, "threshold": 0.82},
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 15,
                "threshold": 0.68,
            },
            "separation": {"leakage_removal": True, "asr_collar": 0.32},
        }
    )
    pipeline.to(torch.device(config.device))
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        separated: list[AudioRegion] = []
        for span_index, (span_start, span_end) in enumerate(overlap_spans):
            start_sample = round(span_start * sample_rate)
            end_sample = round(span_end * sample_rate)
            output = pipeline(
                {
                    "waveform": waveform[:, start_sample:end_sample],
                    "sample_rate": sample_rate,
                }
            )
            if isinstance(output, tuple) and len(output) == 2:
                annotation, sources = output
            else:
                annotation = getattr(output, "speaker_diarization", None)
                sources = getattr(output, "separation", None)
            if annotation is None or sources is None:
                raise RuntimeError("pyannote overlap separator returned no sources")
            labels = list(annotation.labels())
            data = sources.data
            if data.ndim != 2 or data.shape[1] < len(labels):
                raise RuntimeError("pyannote overlap separator returned malformed sources")
            source_paths: dict[str, str] = {}
            for source_index, label in enumerate(labels):
                path = output_dir / f"overlap-{span_index:05d}-{label}.wav"
                sf.write(path, data[:, source_index], sample_rate)
                source_paths[str(label)] = str(path.resolve())
            separated.extend(
                AudioRegion(
                    round(span_start + float(segment.start), 3),
                    round(span_start + float(segment.end), 3),
                    "speech",
                    f"OVERLAP_{span_index:05d}_{speaker}",
                    overlap=True,
                    source_path=source_paths.get(str(speaker)),
                    source_offset=span_start,
                )
                for segment, _track, speaker in annotation.itertracks(yield_label=True)
                if float(segment.end) > float(segment.start)
            )
        untouched = [
            fragment
            for region in diarized_regions
            for fragment in _subtract_regions(region, overlap_spans)
        ]
        return sorted([*untouched, *separated], key=lambda item: (item.start, item.end))
    finally:
        del pipeline
        _release_cuda()


def _separate_vocals(source: Path, destination: Path, device: str) -> None:
    if destination.is_file():
        return
    try:
        import soundfile as sf
        from demucs.api import Separator
    except ImportError as exc:
        raise RuntimeError("vocal separation requires the ASR optional dependencies") from exc

    logging.info("separating vocals with Demucs htdemucs")
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


def _singing_phrases(
    waveform: Any,
    sample_rate: int,
    regions: list[AudioRegion],
    *,
    silence_seconds: float,
) -> list[AudioRegion]:
    """Use silence in the separated vocal stem as approximate lyric-line edges."""
    import librosa

    mono = waveform.mean(dim=0).detach().cpu().numpy()
    phrases: list[AudioRegion] = []
    for region in regions:
        offset = max(0, round(region.start * sample_rate))
        end = min(len(mono), round(region.end * sample_rate))
        intervals = librosa.effects.split(
            mono[offset:end], top_db=35, frame_length=2048, hop_length=256
        )
        current: list[int] | None = None
        for local_start, local_end in intervals:
            if current is None:
                current = [int(local_start), int(local_end)]
                continue
            gap = (int(local_start) - current[1]) / sample_rate
            if gap <= silence_seconds:
                current[1] = int(local_end)
            else:
                phrases.append(_vocal_interval(offset, current, sample_rate, region))
                current = [int(local_start), int(local_end)]
        if current is not None:
            phrases.append(_vocal_interval(offset, current, sample_rate, region))
    return phrases


def _vocal_interval(
    offset: int, interval: list[int], sample_rate: int, parent: AudioRegion
) -> AudioRegion:
    return AudioRegion(
        round((offset + interval[0]) / sample_rate, 3),
        round((offset + interval[1]) / sample_rate, 3),
        "singing",
        confidence=parent.confidence,
    )


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


def _overlap_spans(
    regions: list[AudioRegion], duration: float
) -> list[tuple[float, float]]:
    regions = sorted(regions, key=lambda item: (item.start, item.end))
    intersections: list[AudioRegion] = []
    for index, left in enumerate(regions):
        for right in regions[index + 1 :]:
            if right.start >= left.end:
                break
            if left.speaker == right.speaker:
                continue
            start = max(left.start, right.start)
            end = min(left.end, right.end)
            if end > start:
                intersections.append(
                    AudioRegion(max(0.0, start - 0.75), min(duration, end + 0.75), "speech")
                )
    merged = _merge_regions(intersections, 0.25)
    spans: list[tuple[float, float]] = []
    for region in merged:
        cursor = region.start
        while cursor < region.end:
            end = min(region.end, cursor + 30.0)
            spans.append((cursor, end))
            if end >= region.end:
                break
            cursor = end
    return spans


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
    if destination.is_file():
        return
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
    profile_state = [
        (path.name, path.stat().st_size, path.stat().st_mtime_ns)
        for path in sorted(profile_dir.glob("*.json"))
    ] if profile_dir.is_dir() else []
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
        if value.get("version") != _CACHE_VERSION or value.get("signature") != signature:
            return None
        return AudioAnalysis(
            speech=[AudioRegion(**item) for item in value["speech"]],
            singing=[AudioRegion(**item) for item in value["singing"]],
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        logging.warning("ignoring unreadable audio analysis cache %s: %s", path, exc)
        return None
