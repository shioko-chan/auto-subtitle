from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass, replace
from importlib import resources
from multiprocessing import shared_memory
from pathlib import Path
from typing import Any

from .audio_analysis import AudioRegion
from .audio_buffer import AudioBufferPool, is_shared_audio_uri
from .config import AudioAnalysisConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CharacterStyle:
    id: str
    source_name: str
    primary_color: str
    outline_color: str


def load_character_styles(path_value: str | None = None) -> dict[str, CharacterStyle]:
    if path_value:
        value = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    else:
        value = json.loads(
            resources.files("subtitle_pipeline")
            .joinpath("character_styles.json")
            .read_text(encoding="utf-8")
        )
    styles: dict[str, CharacterStyle] = {}
    for character in value.get("characters", []):
        style = character["subtitle_style"]
        item = CharacterStyle(
            str(character["id"]),
            str(character["source_name"]),
            _hex_color(str(style["primary_color"])),
            _hex_color(str(style.get("outline_color", "#000000"))),
        )
        styles[item.id] = item
    return styles


def metadata_character(
    metadata: dict[str, object], path_value: str | None = None
) -> str | None:
    if path_value:
        value = json.loads(Path(path_value).expanduser().read_text(encoding="utf-8"))
    else:
        value = json.loads(
            resources.files("subtitle_pipeline")
            .joinpath("character_styles.json")
            .read_text(encoding="utf-8")
        )
    evidence = " ".join(
        str(metadata.get(key) or "")
        for key in ("channel", "channel_id", "uploader", "uploader_id")
    ).casefold()
    matches = [
        str(character["id"])
        for character in value.get("characters", [])
        if any(
            str(alias).casefold() in evidence
            for alias in character.get("identity_aliases", [])
        )
    ]
    return matches[0] if len(matches) == 1 else None


def identify_speakers(
    waveform: Any,
    sample_rate: int,
    regions: list[AudioRegion],
    config: AudioAnalysisConfig,
    *,
    known_character: str | None = None,
    excluded_regions: list[AudioRegion] | None = None,
    audio_pool: AudioBufferPool | None = None,
) -> list[AudioRegion]:
    """Enroll clean solo audio and map anonymous clusters to member prototypes."""
    import numpy as np
    import torch

    candidates = [
        region
        for region in _identity_candidates(regions)
        if not any(
            min(region.end, excluded.end) > max(region.start, excluded.start)
            for excluded in (excluded_regions or [])
        )
    ]
    if not candidates:
        return regions
    snippets = _candidate_snippets(
        waveform, sample_rate, candidates, audio_pool=audio_pool
    )
    embeddings = _extract_embeddings(snippets, config)
    by_label: dict[str, list[Any]] = {}
    for region, embedding in zip(candidates, embeddings):
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.size and np.isfinite(embedding).all():
            by_label.setdefault(region.speaker or "unknown", []).append(embedding)
    try:
        centroids = {
            label: _normalize(np.mean(values, axis=0))
            for label, values in by_label.items()
            if values
        }
        profile_dir = Path(config.speaker_profiles_dir).expanduser()
        model_signature = _model_signature(config)
        profiles = _load_profiles(
            profile_dir,
            model_signature,
            max_centers=config.speaker_profile_max_centers,
            min_samples_per_center=config.speaker_profile_min_samples_per_center,
        )
        enrollment_labels = {
            region.speaker or "unknown"
            for region in candidates
            if not region.overlap
        }
        enrollment = {
            label: values
            for label, values in by_label.items()
            if label in enrollment_labels
        }
        if known_character and enrollment:
            dominant = max(enrollment, key=lambda label: len(enrollment[label]))
            selected = _evenly_spaced(
                enrollment[dominant],
                config.speaker_enrollment_samples_per_video,
            )
            _update_profile(
                profile_dir,
                known_character,
                selected,
                model_signature,
            )
            profiles = _load_profiles(
                profile_dir,
                model_signature,
                max_centers=config.speaker_profile_max_centers,
                min_samples_per_center=config.speaker_profile_min_samples_per_center,
            )
        mapping: dict[str, tuple[str, float]] = {}
        for label, centroid in centroids.items():
            distances = sorted(
                (
                    _profile_distance(centroid, profile_centers),
                    character,
                )
                for character, profile_centers in profiles.items()
            )
            threshold = (
                config.speaker_overlap_match_threshold
                if label.startswith("OVERLAP_")
                else config.speaker_match_threshold
            )
            if _profile_match_is_confident(
                distances,
                threshold=threshold,
                margin=config.speaker_match_margin,
            ):
                mapping[label] = (distances[0][1], distances[0][0])
            elif distances:
                runner_up = distances[1][0] if len(distances) > 1 else float("inf")
                logger.info(
                    "speaker identity %s unresolved; closest=%s "
                    "cosine_distance=%.3f runner_up=%.3f threshold=%.3f "
                    "margin=%.3f",
                    label,
                    distances[0][1],
                    distances[0][0],
                    runner_up,
                    threshold,
                    config.speaker_match_margin,
                )
        for label, (character, distance) in mapping.items():
            logger.info(
                "speaker identity %s -> %s cosine_distance=%.3f",
                label,
                character,
                distance,
            )
        return [
            replace(
                region,
                speaker=mapping.get(region.speaker or "", (region.speaker, 0.0))[0],
                confidence=(
                    round(1.0 - mapping[region.speaker or ""][1], 4)
                    if region.speaker in mapping
                    else region.confidence
                ),
            )
            for region in regions
        ]
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _candidate_snippets(
    waveform: Any,
    sample_rate: int,
    candidates: list[AudioRegion],
    *,
    audio_pool: AudioBufferPool | None = None,
) -> list[tuple[Any, int]]:
    import soundfile as sf
    import torch

    snippets: list[tuple[Any, int]] = []
    separated_waveforms: dict[str, tuple[Any, int]] = {}
    for region in candidates:
        source_waveform, source_rate = waveform, sample_rate
        if region.source_path:
            if region.source_path not in separated_waveforms:
                if is_shared_audio_uri(region.source_path):
                    if audio_pool is None:
                        raise RuntimeError("shared speaker track has no audio pool")
                    buffer = audio_pool.resolve(region.source_path)
                    separated_waveforms[region.source_path] = (
                        torch.from_numpy(buffer.samples).unsqueeze(0),
                        buffer.sample_rate,
                    )
                else:
                    data, loaded_rate = sf.read(
                        region.source_path, always_2d=True, dtype="float32"
                    )
                    separated_waveforms[region.source_path] = (
                        torch.from_numpy(data.T.copy()),
                        int(loaded_rate),
                    )
            source_waveform, source_rate = separated_waveforms[region.source_path]
        start = round((region.start - region.source_offset) * source_rate)
        end = round((region.end - region.source_offset) * source_rate)
        snippets.append((source_waveform[:, start:end], source_rate))
    return snippets


def _extract_embeddings(
    snippets: list[tuple[Any, int]], config: AudioAnalysisConfig
) -> list[Any]:
    if config.speaker_embedding_backend == "eres2netv2":
        return _extract_eres2netv2_embeddings(snippets, config)
    return _extract_wespeaker_embeddings(snippets, config)


def _extract_wespeaker_embeddings(
    snippets: list[tuple[Any, int]], config: AudioAnalysisConfig
) -> list[Any]:
    import numpy as np
    import torch
    from pyannote.audio import Inference, Model

    model = Model.from_pretrained(config.speaker_embedding_model)
    if model is None:
        raise RuntimeError(
            f"could not load speaker model {config.speaker_embedding_model}"
        )
    inference = Inference(model, window="whole")
    inference.to(torch.device(config.device))
    try:
        return [
            np.asarray(
                inference({"waveform": audio, "sample_rate": rate})
            ).reshape(-1)
            for audio, rate in snippets
        ]
    finally:
        del inference, model


def _extract_eres2netv2_embeddings(
    snippets: list[tuple[Any, int]], config: AudioAnalysisConfig
) -> list[Any]:
    import numpy as np
    import torchaudio.functional as audio_functional

    uv = shutil.which("uv")
    project = Path(config.speaker_embedding_worker_project).resolve()
    worker = project / "worker.py"
    if uv is None or not worker.is_file():
        raise RuntimeError(f"speaker embedding worker is unavailable at {worker}")
    values: list[np.ndarray] = []
    items: list[dict[str, int]] = []
    cursor = 0
    for index, (audio, rate) in enumerate(snippets):
        mono = audio.mean(dim=0).detach().cpu()
        if rate != 16000:
            mono = audio_functional.resample(mono, rate, 16000)
        value = np.asarray(mono.numpy(), dtype=np.float32)
        values.append(value)
        items.append(
            {"id": index, "start_sample": cursor, "end_sample": cursor + len(value)}
        )
        cursor += len(value)
    combined = np.concatenate(values) if values else np.empty(0, dtype=np.float32)
    memory = shared_memory.SharedMemory(create=True, size=max(1, combined.nbytes))
    shared = np.ndarray(combined.shape, dtype=np.float32, buffer=memory.buf)
    shared[:] = combined
    try:
        payload = {
            "model": config.speaker_embedding_model,
            "device": config.device,
            "audio": {
                "shared_memory": memory.name,
                "dtype": "float32",
                "shape": list(shared.shape),
                "sample_rate": 16000,
            },
            "items": items,
        }
        result = subprocess.run(
            [uv, "run", "--project", str(project), "python", str(worker)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
    finally:
        del shared
        memory.close()
        memory.unlink()
    if result.returncode != 0:
        raise RuntimeError(
            "ERes2NetV2 speaker worker failed: "
            + (result.stderr or result.stdout)[-2000:]
        )
    try:
        response = json.loads(result.stdout)
        values = response["embeddings"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError("ERes2NetV2 speaker worker returned malformed JSON") from exc
    if not isinstance(values, list) or len(values) != len(snippets):
        raise RuntimeError(
            f"ERes2NetV2 returned {len(values) if isinstance(values, list) else 0} "
            f"embeddings for {len(snippets)} snippets"
        )
    return values


def _can_embed_region(region: AudioRegion) -> bool:
    duration = region.end - region.start
    maximum = 30.0 if region.overlap and region.source_path else 15.0
    return 2.0 <= duration <= maximum and (
        not region.overlap or bool(region.source_path)
    )


def _identity_candidates(regions: list[AudioRegion]) -> list[AudioRegion]:
    candidates = [region for region in regions if _can_embed_region(region)]
    represented = {region.speaker for region in candidates}
    separated: dict[tuple[str | None, str, float], list[AudioRegion]] = {}
    for region in regions:
        if region.overlap and region.source_path:
            key = (region.speaker, region.source_path, region.source_offset)
            separated.setdefault(key, []).append(region)

    for (speaker, _source_path, _source_offset), fragments in separated.items():
        if speaker in represented:
            continue
        start = min(fragment.start for fragment in fragments)
        end = max(fragment.end for fragment in fragments)
        voiced = sum(fragment.end - fragment.start for fragment in fragments)
        if voiced >= 1.0 and 1.0 <= end - start <= 30.0:
            candidates.append(replace(fragments[0], start=start, end=end))
    return candidates


def _evenly_spaced(values: list[Any], limit: int) -> list[Any]:
    if len(values) <= limit:
        return values.copy()
    if limit == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))]
        for index in range(limit)
    ]


def _model_signature(config: AudioAnalysisConfig) -> str:
    return f"{config.speaker_embedding_backend}:{config.speaker_embedding_model}"


def _profile_centers(
    embeddings: Any,
    *,
    max_centers: int,
    min_samples_per_center: int,
) -> Any:
    """Build deterministic cosine k-means centers for one speaker profile."""
    import numpy as np

    values = np.asarray(embeddings, dtype=np.float32)
    values = np.asarray([_normalize(value) for value in values], dtype=np.float32)
    maximum = min(max_centers, max(1, len(values) // min_samples_per_center))
    random = np.random.default_rng(0)

    for count in range(maximum, 0, -1):
        best: tuple[float, Any] | None = None
        attempts = 1 if count == 1 else 32
        for _ in range(attempts):
            seeds = random.choice(len(values), size=count, replace=False)
            centers_array = values[seeds].copy()
            assignments = np.full(len(values), -1, dtype=np.int32)
            valid = True
            for _ in range(50):
                updated = np.argmax(values @ centers_array.T, axis=1)
                sizes = np.bincount(updated, minlength=count)
                if not np.all(sizes):
                    valid = False
                    break
                if np.array_equal(updated, assignments):
                    break
                assignments = updated
                centers_array = np.asarray(
                    [
                        _normalize(np.mean(values[assignments == index], axis=0))
                        for index in range(count)
                    ],
                    dtype=np.float32,
                )

            if not valid:
                continue
            sizes = np.bincount(assignments, minlength=count)
            if count > 1 and int(np.min(sizes)) < min_samples_per_center:
                continue
            score = float(np.sum(np.max(values @ centers_array.T, axis=1)))
            if best is None or score > best[0]:
                best = (score, centers_array)

        if best is not None:
            centers_array = best[1]
            order = np.lexsort(centers_array.T[::-1])
            return centers_array[order]

    raise AssertionError("at least one speaker profile center must be produced")


def _profile_distance(embedding: Any, centers: Any) -> float:
    import numpy as np

    normalized = _normalize(np.asarray(embedding, dtype=np.float32))
    return min(
        float(1.0 - np.dot(normalized, center))
        for center in np.asarray(centers, dtype=np.float32)
    )


def _profile_match_is_confident(
    distances: list[tuple[float, str]],
    *,
    threshold: float,
    margin: float,
) -> bool:
    if not distances or distances[0][0] > threshold:
        return False
    return len(distances) == 1 or distances[1][0] - distances[0][0] >= margin


def _load_profiles(
    directory: Path,
    model_signature: str,
    *,
    max_centers: int = 5,
    min_samples_per_center: int = 20,
) -> dict[str, Any]:
    import numpy as np

    profiles: dict[str, Any] = {}
    if not directory.is_dir():
        return profiles
    for path in directory.glob("*.json"):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("model") != model_signature:
                logger.warning(
                    "ignoring speaker profile from another embedding model: %s",
                    path,
                )
                continue
            embeddings = np.asarray(value["embeddings"], dtype=np.float32)
            if embeddings.ndim == 2 and len(embeddings):
                profiles[path.stem] = _profile_centers(
                    embeddings,
                    max_centers=max_centers,
                    min_samples_per_center=min_samples_per_center,
                )
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            logger.warning("ignoring malformed speaker profile %s", path)
    return profiles


def _update_profile(
    directory: Path,
    character: str,
    embeddings: list[Any],
    model_signature: str,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{character}.json"
    existing: list[list[float]] = []
    if path.is_file():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if value.get("model") == model_signature:
                existing = value.get("embeddings", [])
        except (OSError, json.JSONDecodeError):
            pass
    additions = [embedding.astype(float).tolist() for embedding in embeddings]
    payload = {
        "version": 2,
        "character_id": character,
        "model": model_signature,
        "embeddings": (existing + additions)[-400:],
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
    logger.info(
        "speaker profile %s now contains %d samples",
        character,
        len(payload["embeddings"]),
    )


def _normalize(vector: Any) -> Any:
    import numpy as np

    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _hex_color(value: str) -> str:
    normalized = value.strip().upper()
    if len(normalized) != 7 or not normalized.startswith("#"):
        raise ValueError(f"invalid character subtitle color: {value}")
    int(normalized[1:], 16)
    return normalized
