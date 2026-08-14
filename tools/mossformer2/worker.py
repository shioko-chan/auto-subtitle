from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path
from typing import Any


def _attach(descriptor: dict[str, object]) -> tuple[Any, Any]:
    import numpy as np

    memory = shared_memory.SharedMemory(name=str(descriptor["shared_memory"]))
    shape = tuple(int(value) for value in descriptor["shape"])
    values = np.ndarray(
        shape, dtype=np.dtype(str(descriptor["dtype"])), buffer=memory.buf
    )
    return memory, values


def _close_attachment(memory: shared_memory.SharedMemory) -> None:
    name = memory._name  # Child must not unlink memory owned by the main process.
    memory.close()
    resource_tracker.unregister(name, "shared_memory")


def _handle(request: dict[str, object], models: dict[str, Any]) -> dict[str, object]:
    import numpy as np
    import soundfile as sf
    import torch
    from clearvoice import ClearVoice

    items = request.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    model_name = str(request.get("model") or "MossFormer2_SS_16K")
    if model_name not in models:
        models[model_name] = ClearVoice(
            task="speech_separation", model_names=[model_name]
        )
    model = models[model_name]

    source_memory = None
    source = None
    audio_descriptor = request.get("audio")
    if isinstance(audio_descriptor, dict):
        source_memory, source = _attach(audio_descriptor)
        if int(audio_descriptor.get("sample_rate") or 0) != 16000:
            del source
            _close_attachment(source_memory)
            raise ValueError("MossFormer2 shared input must be 16 kHz")

    response: list[dict[str, object]] = []
    try:
        for item in items:
            item_id = item.get("id") if isinstance(item, dict) else None
            attachments: list[shared_memory.SharedMemory] = []
            try:
                if not isinstance(item, dict):
                    raise ValueError("item must be an object")
                if source is not None:
                    start = int(item["start_sample"])
                    end = int(item["end_sample"])
                    mono = np.asarray(source[start:end], dtype=np.float32).copy()
                    output_descriptors = item.get("outputs")
                    if not isinstance(output_descriptors, list) or len(output_descriptors) != 2:
                        raise ValueError("outputs must contain two shared buffers")
                else:
                    audio, sample_rate = sf.read(
                        str(item["input_path"]), always_2d=True, dtype="float32"
                    )
                    if int(sample_rate) != 16000:
                        raise ValueError(
                            f"MossFormer2 input must be 16 kHz, got {sample_rate}"
                        )
                    mono = audio.mean(axis=1, dtype=np.float32)
                    output_descriptors = None

                separated = np.asarray(
                    model(mono.reshape(1, -1), False), dtype=np.float32
                )
                if separated.ndim != 3 or separated.shape[:2] != (2, 1):
                    raise RuntimeError(
                        f"unexpected MossFormer2 output shape {separated.shape}"
                    )

                if output_descriptors is not None:
                    names: list[str] = []
                    for source_index, descriptor in enumerate(output_descriptors):
                        if not isinstance(descriptor, dict):
                            raise ValueError("shared output descriptor must be an object")
                        memory, target = _attach(descriptor)
                        attachments.append(memory)
                        values = separated[source_index, 0]
                        count = min(len(values), len(target))
                        target[:count] = values[:count]
                        if count < len(target):
                            target[count:] = 0
                        names.append(str(descriptor["shared_memory"]))
                        del target
                    response.append({"id": item_id, "output_shared_memory": names})
                else:
                    output_paths = item.get("output_paths")
                    if not isinstance(output_paths, list) or len(output_paths) != 2:
                        raise ValueError("output_paths must contain exactly two paths")
                    written: list[str] = []
                    for source_index, path_value in enumerate(output_paths):
                        path = Path(str(path_value))
                        path.parent.mkdir(parents=True, exist_ok=True)
                        values = separated[source_index, 0]
                        if len(values) > len(mono):
                            values = values[: len(mono)]
                        elif len(values) < len(mono):
                            values = np.pad(values, (0, len(mono) - len(values)))
                        sf.write(path, values, 16000)
                        written.append(str(path.resolve()))
                    response.append({"id": item_id, "output_paths": written})
            except Exception as exc:  # Keep independent spans recoverable.
                response.append({"id": item_id, "error": str(exc)})
            finally:
                for memory in attachments:
                    _close_attachment(memory)
    finally:
        if source_memory is not None:
            del source
            _close_attachment(source_memory)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {"device": device, "items": response}


def main() -> None:
    models: dict[str, Any] = {}
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            with redirect_stdout(sys.stderr):
                response = _handle(request, models)
        except Exception as exc:
            response = {"error": str(exc)}
        print(json.dumps(response, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
