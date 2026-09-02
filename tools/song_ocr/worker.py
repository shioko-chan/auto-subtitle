from __future__ import annotations

import json
import sys
import traceback
from contextlib import redirect_stdout


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def main() -> None:
    settings = json.loads(sys.stdin.readline())
    with redirect_stdout(sys.stderr):
        import easyocr

        configured_device = str(settings.get("device") or "cpu")
        gpu: bool | str = (
            configured_device if configured_device.startswith("cuda") else False
        )
        reader = easyocr.Reader(
            ["ja", "en"],
            gpu=gpu,
            verbose=False,
        )
    emit({"ok": True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            values: list[list[object]] = []
            with redirect_stdout(sys.stderr):
                results = reader.readtext(
                    request["path"],
                    detail=1,
                    paragraph=False,
                )
            for result in results:
                if not isinstance(result, (list, tuple)) or len(result) < 3:
                    continue
                values.append([str(result[1]), float(result[2])])
            emit({"ok": True, "values": values})
        except Exception as exc:
            emit(
                {
                    "ok": False,
                    "error": str(exc),
                    "trace": traceback.format_exc()[-2000:],
                }
            )


if __name__ == "__main__":
    main()
