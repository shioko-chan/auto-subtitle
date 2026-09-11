"""Durable publication records, deliberately independent of computation caches."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from .upload import BilibiliSubmission, BiliupCommandError, UploadNotStartedError


def read_record(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"publication record is unreadable; inspect {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"publication record is invalid: {path}")
    return value


def write_record(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def completed(record: dict) -> bool:
    return record.get("uploaded") is True or record.get("status") == "success"


def publish_once(path: Path, intent: dict, submit: Callable[[], BilibiliSubmission]) -> BilibiliSubmission:
    previous = read_record(path)
    if completed(previous):
        return BilibiliSubmission(previous.get("bilibili_aid", previous.get("aid")),
                                 previous.get("bilibili_bvid", previous.get("bvid")),
                                 str(previous.get("response", "")))
    if previous.get("status") in {"submitting", "unknown"}:
        raise RuntimeError(f"publication outcome is unknown; use publication resolve after checking Bilibili: {path}")
    record = {**previous, **intent, "status": "submitting", "uploaded": False,
              "submitted_at": datetime.now(UTC).isoformat()}
    write_record(path, record)
    try:
        submission = submit()
    except Exception as exc:
        not_uploaded = isinstance(exc, UploadNotStartedError) or (
            isinstance(exc, BiliupCommandError) and exc.submission_rejected
        )
        status = "not_uploaded" if not_uploaded else "unknown"
        write_record(path, {**record, "status": status, "error": f"{type(exc).__name__}: {exc}"})
        raise
    write_record(path, {**record, "status": "success", "uploaded": True,
                        "aid": submission.aid, "bvid": submission.bvid,
                        "bilibili_aid": submission.aid, "bilibili_bvid": submission.bvid,
                        "response": submission.response, "uploaded_at": datetime.now(UTC).isoformat()})
    return submission


def resolve(path: Path, *, uploaded: bool, aid: int | None, bvid: str | None) -> None:
    if uploaded and (aid is None or aid <= 0 or not bvid or not bvid.startswith("BV")):
        raise ValueError("--uploaded requires positive --aid and --bvid BV...")
    record = read_record(path)
    record.update(status="success" if uploaded else "not_uploaded", uploaded=uploaded,
                  aid=aid if uploaded else None, bvid=bvid if uploaded else None,
                  bilibili_aid=aid if uploaded else None, bilibili_bvid=bvid if uploaded else None,
                  resolved_at=datetime.now(UTC).isoformat())
    write_record(path, record)
