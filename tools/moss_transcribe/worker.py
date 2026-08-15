from __future__ import annotations

import json
import re
import sys
from contextlib import redirect_stdout
from multiprocessing import resource_tracker, shared_memory
from typing import Any

_SEGMENT_RE = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]"
    r"\[(?P<speaker>S\d+)\]"
    r"(?P<text>.*?)"
    r"\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)


def _parse_transcript(text: str, offset: float, window: int) -> list[dict[str, object]]:
    segments: list[dict[str, object]] = []
    matches = list(_SEGMENT_RE.finditer(text))
    for match in matches:
        start = float(match.group("start")) + offset
        end = float(match.group("end")) + offset
        value = match.group("text").strip()
        if end <= start or not value:
            continue
        segments.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": f"MOSS_W{window:03d}_{match.group('speaker')}",
                "text": value,
            }
        )
    if text.strip() and not matches:
        raise RuntimeError("MOSS returned a non-empty but unparseable transcript")
    return segments


def _generate(
    model: Any, processor: Any, audio: Any, prompt: str, request: dict[str, object]
) -> str:

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": "shared-memory"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    device = next(model.parameters()).device
    inputs = processor(
        text=rendered,
        audio=[audio],
        max_length=131072,
        audio_kwargs={"device": str(device)},
        return_tensors="pt",
    ).to(device)
    prompt_length = int(inputs["attention_mask"][0].sum().item())
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        input_features=inputs["input_features"],
        audio_feature_lengths=inputs["audio_feature_lengths"],
        audio_chunk_mapping=inputs["audio_chunk_mapping"],
        max_new_tokens=int(request["max_new_tokens"]),
        do_sample=False,
    )
    return processor.tokenizer.decode(
        outputs[0][prompt_length:], skip_special_tokens=True
    ).strip()


def _handle(request: dict[str, object]) -> dict[str, object]:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("MOSS transcription requires a CUDA device")
    descriptor = request["audio"]
    if not isinstance(descriptor, dict):
        raise TypeError("audio descriptor must be an object")
    if int(descriptor["sample_rate"]) != 16000:
        raise ValueError("MOSS shared input must be 16 kHz")
    memory = shared_memory.SharedMemory(name=str(descriptor["shared_memory"]))
    shape = tuple(int(value) for value in descriptor["shape"])
    source = np.ndarray(
        shape, dtype=np.dtype(str(descriptor["dtype"])), buffer=memory.buf
    )
    dtype = getattr(torch, str(request.get("dtype") or "float16"))
    device = torch.device(str(request.get("device") or "cuda:0"))
    try:
        model = (
            AutoModelForCausalLM.from_pretrained(
                str(request["model"]),
                trust_remote_code=True,
                dtype=dtype,
                attn_implementation="sdpa",
            )
            .to(device)
            .eval()
        )
        processor = AutoProcessor.from_pretrained(
            str(request["model"]), trust_remote_code=True
        )
        transcripts: list[dict[str, object]] = []
        segments: list[dict[str, object]] = []
        for index, window in enumerate(request["windows"]):
            if not isinstance(window, dict):
                raise TypeError("window must be an object")
            start = float(window["start"])
            end = float(window["end"])
            start_sample = max(0, round(start * 16000))
            end_sample = min(len(source), round(end * 16000))
            audio = np.asarray(source[start_sample:end_sample], dtype=np.float32).copy()
            raw = _generate(model, processor, audio, str(request["prompt"]), request)
            transcripts.append(
                {"window": index, "start": start, "end": end, "raw": raw}
            )
            try:
                parsed = _parse_transcript(raw, start, index)
            except RuntimeError as exc:
                return {
                    "error": str(exc),
                    "failed_window": index,
                    "raw": raw,
                    "transcripts": transcripts,
                    "segments": segments,
                }
            segments.extend(parsed)
        return {"transcripts": transcripts, "segments": segments}
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
    except Exception as exc:  # noqa: BLE001 - process boundary returns failures as JSON
        response = {"error": str(exc)}
    print(json.dumps(response, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
