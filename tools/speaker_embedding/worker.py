from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from multiprocessing import resource_tracker, shared_memory
from typing import Any


def _normalize_embeddings(results: list[object], expected: int) -> list[list[float]]:
    embeddings: list[list[float]] = []
    for result in results:
        value = result.get("spk_embedding") if isinstance(result, dict) else None
        if value is None:
            raise RuntimeError("ERes2NetV2 returned no spk_embedding")
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        if not hasattr(value, "reshape"):
            raise RuntimeError("ERes2NetV2 returned a malformed spk_embedding")
        if getattr(value, "ndim", 1) <= 1:
            rows = [value.reshape(-1)]
        else:
            rows = value.reshape(value.shape[0], -1)
        embeddings.extend([float(item) for item in row] for row in rows)
    if len(embeddings) != expected:
        raise RuntimeError(
            f"ERes2NetV2 returned {len(embeddings)} embeddings for {expected} inputs"
        )
    return embeddings


def _handle(request: dict[str, object], models: dict[tuple[str, str], Any]) -> dict[str, object]:
    import numpy as np
    import torch
    from funasr import AutoModel

    requested_device = str(request.get("device") or "cpu")
    device = requested_device if torch.cuda.is_available() else "cpu"
    model_name = str(request["model"])
    key = (model_name, device)
    if key not in models:
        models[key] = AutoModel(
            model=model_name,
            device=device,
            disable_update=True,
        )
    model = models[key]

    paths = request.get("paths")
    memory = None
    if isinstance(paths, list) and all(isinstance(path, str) for path in paths):
        inputs: list[object] = paths
    else:
        descriptor = request.get("audio")
        items = request.get("items")
        if not isinstance(descriptor, dict) or not isinstance(items, list):
            raise ValueError("request must contain paths or shared audio items")
        if int(descriptor.get("sample_rate") or 0) != 16000:
            raise ValueError("ERes2NetV2 shared input must be 16 kHz")
        memory = shared_memory.SharedMemory(name=str(descriptor["shared_memory"]))
        shape = tuple(int(value) for value in descriptor["shape"])
        source = np.ndarray(
            shape, dtype=np.dtype(str(descriptor["dtype"])), buffer=memory.buf
        )
        inputs = [
            np.asarray(
                source[int(item["start_sample"]) : int(item["end_sample"])],
                dtype=np.float32,
            ).copy()
            for item in items
            if isinstance(item, dict)
        ]
    try:
        results = model.generate(input=inputs, batch_size=16)
        return {"embeddings": _normalize_embeddings(results, len(inputs))}
    finally:
        if memory is not None:
            name = memory._name
            del source
            memory.close()
            resource_tracker.unregister(name, "shared_memory")


def main() -> None:
    models: dict[tuple[str, str], Any] = {}
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
