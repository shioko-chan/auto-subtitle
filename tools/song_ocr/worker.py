from __future__ import annotations

import json
import sys
import traceback
from contextlib import redirect_stdout


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def result_payload(result: object) -> dict[str, object]:
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        value = value["res"]
    return value if isinstance(value, dict) else {}


def main() -> None:
    settings = json.loads(sys.stdin.readline())
    with redirect_stdout(sys.stderr):
        from paddleocr import PaddleOCR

        pipeline = PaddleOCR(
            device=settings["device"],
            text_detection_model_name=settings["detection_model"],
            text_recognition_model_name=settings["recognition_model"],
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    emit({"ok": True})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            values: list[list[object]] = []
            with redirect_stdout(sys.stderr):
                results = list(pipeline.predict(request["path"]))
            for result in results:
                payload = result_payload(result)
                for text, score in zip(
                    payload.get("rec_texts", []), payload.get("rec_scores", [])
                ):
                    values.append([str(text), float(score)])
            emit({"ok": True, "values": values})
        except Exception as exc:
            emit({"ok": False, "error": str(exc), "trace": traceback.format_exc()[-2000:]})


if __name__ == "__main__":
    main()
