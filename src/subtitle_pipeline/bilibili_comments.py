from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import UploadConfig

logger = logging.getLogger(__name__)

_TASK_NAME = "bilibili-setlist-comment.json"
_REPLIES_URL = "https://api.bilibili.com/x/v2/reply"
_ADD_REPLY_URL = "https://api.bilibili.com/x/v2/reply/add"
_PERMANENT_CODES = {12016, 12025, 12035}
_RETRY_DELAYS_SECONDS = (900, 1800, 3600, 7200, 14400, 21600)
_INITIAL_DELAY_SECONDS = 3600


@dataclass(frozen=True)
class CommentProcessSummary:
    posted: int = 0
    already_exists: int = 0
    pending: int = 0
    permanent_errors: int = 0

    def add(self, status: str) -> CommentProcessSummary:
        return CommentProcessSummary(
            posted=self.posted + (status == "posted"),
            already_exists=self.already_exists + (status == "already_exists"),
            pending=self.pending + (status in {"pending_review", "retryable_error"}),
            permanent_errors=self.permanent_errors + (status == "permanent_error"),
        )


def build_song_setlist_comment(
    source_title: str,
    reports: list[dict[str, object]],
) -> str | None:
    if "歌枠" not in source_title:
        return None
    entries: list[tuple[float, float, str, str]] = []
    for report in reports:
        song = report.get("song")
        if not isinstance(song, str) or not song.strip():
            continue
        start = _report_song_start(report)
        if start is not None:
            group = report.get("search_group")
            end = (
                float(group["end"])
                if isinstance(group, dict)
                and isinstance(group.get("end"), (int, float))
                else start
            )
            identity = str(report.get("song_id") or song).strip()
            entries.append((start, end, identity, song.strip()))
    entries.sort(key=lambda item: item[0])
    if not entries:
        return None
    performances: list[tuple[float, float, str, str]] = []
    for entry in entries:
        if (
            performances
            and entry[2] == performances[-1][2]
            and entry[0] - performances[-1][1] <= 120.0
        ):
            previous = performances[-1]
            performances[-1] = (
                previous[0],
                max(previous[1], entry[1]),
                previous[2],
                previous[3],
            )
            continue
        performances.append(entry)
    lines = [
        f"{_format_timestamp(start)} {song}"
        for start, _end, _identity, song in performances
    ]
    comment = "\n".join(lines)
    if len(comment) > 1000:
        logger.warning("setlist comment exceeds Bilibili's 1000-character limit")
        return None
    return comment


def create_comment_task(
    job_dir: Path,
    *,
    aid: int,
    bvid: str,
    message: str,
    source_url: str,
    upload_response: str,
) -> Path:
    path = job_dir / _TASK_NAME
    now = time.time()
    payload = {
        "version": 1,
        "status": "pending_review",
        "aid": aid,
        "bvid": bvid,
        "message": message,
        "source_url": source_url,
        "rpid": None,
        "created_at": now,
        "updated_at": now,
        "next_attempt_at": now + _INITIAL_DELAY_SECONDS,
        "attempts": [],
        "upload_response_tail": upload_response[-4000:],
    }
    _write_json(path, payload)
    return path


def process_pending_bilibili_comments(
    work_dir: Path,
    config: UploadConfig,
) -> CommentProcessSummary:
    summary = CommentProcessSummary()
    for path in sorted(work_dir.glob(f"*/{_TASK_NAME}")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("cannot read Bilibili comment task %s: %s", path, exc)
            continue
        if payload.get("status") in {"posted", "already_exists", "permanent_error"}:
            continue
        if float(payload.get("next_attempt_at") or 0) > time.time():
            summary = summary.add("pending_review")
            continue
        try:
            status = process_bilibili_comment_task(path, config)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("cannot process Bilibili comment task %s: %s", path, exc)
            continue
        summary = summary.add(status)
    return summary


def process_bilibili_comment_task(path: Path, config: UploadConfig) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cookies = _load_cookies(Path(config.cookie_file))
    mid = cookies.get("DedeUserID", "")
    csrf = cookies.get("bili_jct", "")
    if not mid or not csrf or not cookies.get("SESSDATA"):
        raise RuntimeError("Bilibili cookie file lacks DedeUserID, bili_jct, or SESSDATA")

    aid = int(payload["aid"])
    message = str(payload["message"])
    try:
        existing_rpid, query_response = _find_existing_comment(
            aid, message, mid=mid, cookies=cookies
        )
        if existing_rpid is not None:
            return _record_attempt(
                path,
                payload,
                status="already_exists",
                response=query_response,
                rpid=existing_rpid,
            )
        if int(query_response.get("code", -1)) != 0:
            return _record_attempt(
                path,
                payload,
                status="pending_review",
                response=query_response,
            )
        response = _post_comment(aid, message, csrf=csrf, cookies=cookies)
    except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
        return _record_attempt(
            path,
            payload,
            status="retryable_error",
            response={"exception": f"{type(exc).__name__}: {exc}"},
        )

    code = int(response.get("code", -1))
    if code == 0:
        data = response.get("data")
        rpid = _as_int(data.get("rpid")) if isinstance(data, dict) else None
        return _record_attempt(
            path, payload, status="posted", response=response, rpid=rpid
        )
    if code == 12051:
        return _record_attempt(
            path, payload, status="already_exists", response=response
        )
    status = "permanent_error" if code in _PERMANENT_CODES else "retryable_error"
    return _record_attempt(path, payload, status=status, response=response)


def _report_song_start(report: dict[str, object]) -> float | None:
    alignments = report.get("alignments")
    starts = (
        [
            float(item["start"])
            for item in alignments
            if isinstance(item, dict)
            and isinstance(item.get("start"), (int, float))
        ]
        if isinstance(alignments, list)
        else []
    )
    if starts:
        return min(starts)
    group = report.get("search_group")
    if isinstance(group, dict) and isinstance(group.get("start"), (int, float)):
        return float(group["start"])
    return None


def _format_timestamp(seconds: float) -> str:
    total = max(0, round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _load_cookies(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    raw = value.get("cookie_info", {}).get("cookies", [])
    if not isinstance(raw, list):
        raise TypeError("invalid biliup cookie file")
    return {
        str(item["name"]): str(item["value"])
        for item in raw
        if isinstance(item, dict) and item.get("name") and item.get("value")
    }


def _find_existing_comment(
    aid: int,
    message: str,
    *,
    mid: str,
    cookies: dict[str, str],
) -> tuple[int | None, dict[str, Any]]:
    last_response: dict[str, Any] = {"code": 0, "data": {}}
    for page in range(1, 11):
        query = urllib.parse.urlencode(
            {"type": 1, "oid": aid, "sort": 2, "pn": page, "ps": 49}
        )
        response = _request_json(f"{_REPLIES_URL}?{query}", cookies=cookies)
        last_response = response
        if int(response.get("code", -1)) != 0:
            return None, response
        data = response.get("data")
        replies = data.get("replies") if isinstance(data, dict) else None
        if not isinstance(replies, list) or not replies:
            break
        for reply in replies:
            if not isinstance(reply, dict):
                continue
            member = reply.get("member")
            content = reply.get("content")
            sender = str(member.get("mid", "")) if isinstance(member, dict) else ""
            text = str(content.get("message", "")) if isinstance(content, dict) else ""
            if sender == mid and _normalize_message(text) == _normalize_message(message):
                return _as_int(reply.get("rpid")), response
        page_info = data.get("page") if isinstance(data, dict) else None
        count = _as_int(page_info.get("count")) if isinstance(page_info, dict) else None
        if count is not None and page * 49 >= count:
            break
    return None, last_response


def _post_comment(
    aid: int,
    message: str,
    *,
    csrf: str,
    cookies: dict[str, str],
) -> dict[str, Any]:
    body = urllib.parse.urlencode(
        {"type": 1, "oid": aid, "message": message, "plat": 1, "csrf": csrf}
    ).encode()
    return _request_json(_ADD_REPLY_URL, cookies=cookies, body=body)


def _request_json(
    url: str,
    *,
    cookies: dict[str, str],
    body: bytes | None = None,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Cookie": "; ".join(f"{key}={value}" for key, value in cookies.items()),
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.bilibili.com/",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST" if body is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Bilibili returned a non-object JSON response")
    return value


def _record_attempt(
    path: Path,
    payload: dict[str, Any],
    *,
    status: str,
    response: dict[str, Any],
    rpid: int | None = None,
) -> str:
    now = time.time()
    attempts = payload.setdefault("attempts", [])
    attempts.append({"at": now, "status": status, "response": response})
    payload["status"] = status
    payload["updated_at"] = now
    payload["last_response"] = response
    if status in {"pending_review", "retryable_error"}:
        delay_index = min(len(attempts) - 1, len(_RETRY_DELAYS_SECONDS) - 1)
        payload["next_attempt_at"] = now + _RETRY_DELAYS_SECONDS[delay_index]
    else:
        payload["next_attempt_at"] = None
    if rpid is not None:
        payload["rpid"] = rpid
    _write_json(path, payload)
    logger.info(
        "Bilibili setlist comment %s for %s (rpid=%s)",
        status,
        payload.get("bvid"),
        payload.get("rpid"),
    )
    return status


def _normalize_message(value: str) -> str:
    return "\n".join(line.rstrip() for line in value.replace("\r\n", "\n").strip().splitlines())


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)
