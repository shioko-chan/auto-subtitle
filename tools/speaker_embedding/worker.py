from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout


def main() -> None:
    request = json.load(sys.stdin)
    paths = request.get("paths")
    if not isinstance(paths, list) or not all(isinstance(path, str) for path in paths):
        raise ValueError("paths must be a list of audio paths")

    with redirect_stdout(sys.stderr):
        import torch
        from funasr import AutoModel

        requested_device = str(request.get("device") or "cpu")
        device = requested_device if torch.cuda.is_available() else "cpu"
        model = AutoModel(
            model=str(request["model"]),
            device=device,
            disable_update=True,
        )
        results = model.generate(input=paths, batch_size=16)

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
    if len(embeddings) != len(paths):
        raise RuntimeError(
            f"ERes2NetV2 returned {len(embeddings)} embeddings for {len(paths)} paths"
        )
    print(json.dumps({"embeddings": embeddings}, separators=(",", ":")))


if __name__ == "__main__":
    main()
