from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path


def main() -> None:
    request = json.load(sys.stdin)
    items = request.get("items")
    if not isinstance(items, list):
        raise ValueError("items must be a list")

    with redirect_stdout(sys.stderr):
        import numpy as np
        import soundfile as sf
        import torch
        from clearvoice import ClearVoice

        model = ClearVoice(
            task="speech_separation",
            model_names=[str(request.get("model") or "MossFormer2_SS_16K")],
        )

        response: list[dict[str, object]] = []
        for item in items:
            item_id = item.get("id") if isinstance(item, dict) else None
            try:
                if not isinstance(item, dict):
                    raise ValueError("item must be an object")
                output_paths = item.get("output_paths")
                if not isinstance(output_paths, list) or len(output_paths) != 2:
                    raise ValueError("output_paths must contain exactly two paths")
                audio, sample_rate = sf.read(
                    str(item["input_path"]), always_2d=True, dtype="float32"
                )
                if int(sample_rate) != 16000:
                    raise ValueError(
                        f"MossFormer2 input must be 16 kHz, got {sample_rate}"
                    )
                mono = audio.mean(axis=1, dtype=np.float32)
                separated = np.asarray(
                    model(mono.reshape(1, -1), False), dtype=np.float32
                )
                if separated.ndim != 3 or separated.shape[:2] != (2, 1):
                    raise RuntimeError(
                        f"unexpected MossFormer2 output shape {separated.shape}"
                    )
                written: list[str] = []
                for source_index, path_value in enumerate(output_paths):
                    path = Path(str(path_value))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    source = separated[source_index, 0]
                    if len(source) > len(mono):
                        source = source[: len(mono)]
                    elif len(source) < len(mono):
                        source = np.pad(source, (0, len(mono) - len(source)))
                    sf.write(path, source, 16000)
                    written.append(str(path.resolve()))
                response.append({"id": item_id, "output_paths": written})
            except Exception as exc:  # Keep independent spans recoverable.
                response.append({"id": item_id, "error": str(exc)})

        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(json.dumps({"device": device, "items": response}, separators=(",", ":")))


if __name__ == "__main__":
    main()
