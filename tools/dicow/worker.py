from __future__ import annotations

import json
import re
import shutil
import sys
from contextlib import redirect_stdout
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any

_TIMESTAMP_SEGMENT_RE = re.compile(
    r"<\|(?P<start>\d+(?:\.\d+)?)\|>"
    r"(?P<text>.*?)"
    r"<\|(?P<end>\d+(?:\.\d+)?)\|>",
    re.DOTALL,
)


def _prepare_remote_code(model: str, revision: str) -> None:
    """Work around Transformers 4.55 omitting second-level relative imports."""
    from huggingface_hub import snapshot_download
    from transformers.dynamic_module_utils import HF_MODULES_CACHE, init_hf_modules

    snapshot = Path(
        snapshot_download(
            model,
            revision=revision,
            allow_patterns=["*.py", "*.json", "*.txt"],
        )
    )
    init_hf_modules()
    target = (
        Path(HF_MODULES_CACHE)
        / "transformers_modules"
        / Path(*model.split("/"))
        / revision
    )
    target.mkdir(parents=True, exist_ok=True)
    for parent in [target, *target.parents]:
        if parent == Path(HF_MODULES_CACHE).parent:
            break
        (parent / "__init__.py").touch(exist_ok=True)
    for source in snapshot.glob("*.py"):
        shutil.copy2(source, target / source.name)


def _case_mapping(tokenizer: Any) -> None:
    tokenizer.upper_cased_tokens = {}
    vocabulary = tokenizer.get_vocab()
    for token, index in vocabulary.items():
        if not token:
            continue
        if token[0] == "Ġ" and len(token) > 1:
            lowered = token[0] + token[1].lower() + token[2:]
        else:
            lowered = token[0].lower() + token[1:]
        lower_index = vocabulary.get(lowered)
        if lower_index is not None and lowered != token:
            tokenizer.upper_cased_tokens[lower_index] = index


def _diarization_masks(
    speakers: list[str], turns: list[dict[str, object]], start: float, frames: int
) -> Any:
    import torch

    activity = torch.zeros((len(speakers), frames), dtype=torch.float32)
    speaker_index = {speaker: index for index, speaker in enumerate(speakers)}
    for turn in turns:
        speaker = str(turn["speaker"])
        if speaker not in speaker_index:
            continue
        left = max(0, min(frames, round((float(turn["start"]) - start) * 50)))
        right = max(0, min(frames, round((float(turn["end"]) - start) * 50)))
        if right > left:
            activity[speaker_index[speaker], left:right] = 1
    masks = []
    for target in range(len(speakers)):
        others = torch.ones(len(speakers), dtype=torch.bool)
        others[target] = False
        silence = (1 - activity).prod(dim=0)
        anyone_else = (
            (1 - activity[others]).prod(dim=0)
            if bool(others.any())
            else torch.ones(frames)
        )
        target_only = activity[target] * anyone_else
        non_target = (1 - activity[target]) * (1 - anyone_else)
        overlap = activity[target] - target_only
        masks.append(torch.stack((silence, target_only, non_target, overlap)))
    return torch.stack(masks)


def _decode_segments(
    tokenizer: Any,
    sequences: Any,
    speakers: list[str],
    offset: float,
    duration: float,
) -> list[dict[str, object]]:
    decoded = tokenizer.batch_decode(
        sequences, decode_with_timestamps=True, skip_special_tokens=True
    )
    cues: list[dict[str, object]] = []
    for speaker, text in zip(speakers, decoded):
        matches = list(_TIMESTAMP_SEGMENT_RE.finditer(text))
        if text.strip() and not matches:
            raise RuntimeError(
                f"DiCoW returned text without timestamp pairs for {speaker}: "
                f"{text[-500:]!r}"
            )
        for match in matches:
            start = max(0.0, min(duration, float(match.group("start"))))
            end = max(0.0, min(duration, float(match.group("end"))))
            value = match.group("text").strip()
            if end > start and value:
                cues.append(
                    {
                        "start": round(offset + start, 3),
                        "end": round(offset + end, 3),
                        "speaker": speaker,
                        "text": value,
                    }
                )
    return cues


def _transcribe_windows(
    model: Any,
    feature_extractor: Any,
    tokenizer: Any,
    audio: list[Any],
    windows: list[dict[str, object]],
    language: str,
) -> list[dict[str, object]]:
    import torch

    samples = feature_extractor(
        audio,
        sampling_rate=16000,
        return_attention_mask=True,
        return_tensors="pt",
    )
    input_features = samples.input_features
    frames = input_features.shape[-1] // 2
    feature_rows: list[int] = []
    masks = []
    speaker_groups: list[list[str]] = []
    for index, window in enumerate(windows):
        start = float(window["start"])
        speakers = [str(item) for item in window["speakers"]]
        if not speakers:
            raise RuntimeError("conditioned ASR window has no speakers")
        feature_rows.extend([index] * len(speakers))
        speaker_groups.append(speakers)
        masks.append(
            _diarization_masks(
                speakers,
                list(window["turns"]),
                start,
                frames,
            )
        )
    row_index = torch.tensor(feature_rows, dtype=torch.long)
    input_features = input_features.index_select(0, row_index).to(
        model.device, dtype=model.dtype
    )
    stno_mask = torch.cat(masks).to(model.device, dtype=model.dtype)
    attention_mask = torch.ones(
        (len(feature_rows), input_features.shape[-1]),
        dtype=torch.bool,
        device=model.device,
    )
    with torch.inference_mode():
        generated = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            stno_mask=stno_mask,
            language=language,
            task="transcribe",
            return_timestamps=True,
            do_sample=False,
        )
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    cues: list[dict[str, object]] = []
    cursor = 0
    for window, speakers in zip(windows, speaker_groups):
        end_cursor = cursor + len(speakers)
        start = float(window["start"])
        end = float(window["end"])
        cues.extend(
            _decode_segments(
                tokenizer,
                sequences[cursor:end_cursor],
                speakers,
                start,
                end - start,
            )
        )
        cursor = end_cursor
    return cues


def _handle(request: dict[str, object]) -> dict[str, object]:
    import numpy as np
    import torch
    from transformers import (
        AutoFeatureExtractor,
        AutoModelForSpeechSeq2Seq,
        AutoTokenizer,
    )
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.utils import logging as transformers_logging

    if not torch.cuda.is_available():
        raise RuntimeError("DiCoW requires a CUDA device")
    descriptor = request["audio"]
    if not isinstance(descriptor, dict) or int(descriptor["sample_rate"]) != 16000:
        raise ValueError("DiCoW input must be 16 kHz shared audio")
    memory = shared_memory.SharedMemory(name=str(descriptor["shared_memory"]))
    source = np.ndarray(
        tuple(int(value) for value in descriptor["shape"]),
        dtype=np.dtype(str(descriptor["dtype"])),
        buffer=memory.buf,
    )
    try:
        model_name = str(request["model"])
        revision = str(request["revision"])
        _prepare_remote_code(model_name, revision)
        get_class_from_dynamic_module(
            "modeling_dicow.DiCoWForConditionalGeneration",
            model_name,
            revision=revision,
        )
        transformers_logging.set_verbosity_error()
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        ).to(str(request["device"]))
        model.eval()
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, revision=revision, trust_remote_code=True
        )
        _case_mapping(tokenizer)
        if hasattr(model, "set_tokenizer"):
            model.set_tokenizer(tokenizer)
        cues: list[dict[str, object]] = []
        windows = list(request["windows"])
        batch_size = max(1, int(request.get("batch_size") or 1))
        for batch_start in range(0, len(windows), batch_size):
            batch = windows[batch_start : batch_start + batch_size]
            audio = []
            for window in batch:
                start = float(window["start"])
                end = float(window["end"])
                left = max(0, round(start * 16000))
                right = min(len(source), round(end * 16000))
                audio.append(np.asarray(source[left:right], dtype=np.float32).copy())
            cues.extend(
                _transcribe_windows(
                    model,
                    feature_extractor,
                    tokenizer,
                    audio,
                    batch,
                    str(request.get("language") or "ja"),
                )
            )
        return {"cues": cues}
    finally:
        name = memory._name
        del source
        memory.close()
        resource_tracker.unregister(name, "shared_memory")


def main() -> None:
    try:
        request = json.loads(sys.stdin.read())
        with redirect_stdout(sys.stderr):
            response = _handle(request)
    except Exception as exc:  # noqa: BLE001 - worker boundary reports JSON failures
        response = {"error": str(exc)}
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
