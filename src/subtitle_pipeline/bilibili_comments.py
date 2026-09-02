from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from .config import UploadConfig
from .speakers import metadata_character, title_characters

logger = logging.getLogger(__name__)

_TASK_NAME = "bilibili-setlist-comment.json"
_REPLIES_URL = "https://api.bilibili.com/x/v2/reply"
_REPLY_DETAIL_URL = "https://api.bilibili.com/x/v2/reply/reply"
_PERMANENT_CODES = {12016, 12025, 12035}
_REPLIES_PAGE_SIZE = 20
_POST_VERIFY_DELAY_SECONDS = 5
_PUBLICATION_DELAY_SECONDS = 5 * 60
_BROWSER_TIMEOUT_MS = 60_000
_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
)
_SETLIST_MARKER_RE = re.compile(r"(?:SETLIST|SET\s*LIST|セトリ|歌单)", re.IGNORECASE)
_TIMESTAMP_LINE_RE = re.compile(r"(?m)^\s*\d{1,2}:(?:\d{2}:)?\d{2}\s+\S")
_CHARACTER_EMOJI = {
    "nakamachi_arale": "[梦限大_阿拉蕾耶]",
    "minetsuki_ritsu": "[梦限大_律敬礼]",
    "miyanaga_nonoka": "[梦限大_野乃花来啦]",
    "fuji_miyako": "[梦限大_都子期待]",
    "sengoku_yuno": "[梦限大_由乃坏笑]",
}


class BrowserCommentError(RuntimeError):
    pass


def build_song_setlist_comment(
    source_metadata: dict[str, object],
    reports: list[dict[str, object]],
    youtube_comments: list[object] | None = None,
    *,
    minimum_comment_likes: int = 10,
) -> str | None:
    source_title = str(source_metadata.get("title") or "")
    if "歌枠" not in source_title:
        return None
    existing_setlist = _highest_liked_setlist_comment(
        youtube_comments or [], minimum_likes=minimum_comment_likes
    )
    if existing_setlist is not None:
        return _with_character_emoji(existing_setlist, source_metadata)
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
    comment = _with_character_emoji("歌单：\n" + "\n".join(lines), source_metadata)
    if len(comment) > 1000:
        logger.warning("setlist comment exceeds Bilibili's 1000-character limit")
        return None
    return comment


def _with_character_emoji(
    comment: str, source_metadata: dict[str, object]
) -> str:
    character_id = metadata_character(source_metadata)
    if character_id is None:
        title_matches = title_characters(source_metadata)
        character_id = title_matches[0] if len(title_matches) == 1 else None
    emoji = _CHARACTER_EMOJI.get(character_id or "")
    if emoji is None:
        return comment
    decorated = f"{emoji}\n{comment}"
    return decorated if len(decorated) <= 1000 else comment


def _highest_liked_setlist_comment(
    comments: list[object], *, minimum_likes: int
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for value in comments:
        if not isinstance(value, dict) or value.get("parent") not in (None, "root"):
            continue
        text = str(value.get("text") or "").strip()
        try:
            likes = int(value.get("like_count") or 0)
        except (TypeError, ValueError):
            continue
        if (
            likes < minimum_likes
            or len(text) > 1000
            or _SETLIST_MARKER_RE.search(text) is None
            or len(_TIMESTAMP_LINE_RE.findall(text)) < 3
        ):
            continue
        candidates.append((likes, text))
    return max(candidates, default=None, key=lambda item: item[0])[1] if candidates else None


def create_comment_task(
    job_dir: Path,
    *,
    aid: int,
    bvid: str,
    message: str,
    source_url: str,
) -> Path:
    path = job_dir / _TASK_NAME
    now = time.time()
    payload = {
        "status": "scheduled",
        "aid": aid,
        "bvid": bvid,
        "message": message,
        "source_url": source_url,
        "rpid": None,
        "created_at": now,
        "publish_at": now + _PUBLICATION_DELAY_SECONDS,
        "updated_at": now,
    }
    _write_json(path, payload)
    logger.info(
        "scheduled Bilibili comment for publication in %ds: %s",
        _PUBLICATION_DELAY_SECONDS,
        path,
    )
    return path


def publish_comment_task(path: Path, config: UploadConfig) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "scheduled":
        raise ValueError(f"comment is not scheduled: {path}")
    publish_at = float(payload["publish_at"])
    delay = max(0.0, publish_at - time.time())
    if delay:
        logger.info("waiting %.1fs before publishing Bilibili comment", delay)
        time.sleep(delay)
    try:
        cookies = _load_cookies(Path(config.cookie_file))
        mid = cookies.get("DedeUserID", "")
        if not mid or not cookies.get("bili_jct") or not cookies.get("SESSDATA"):
            raise RuntimeError(
                "Bilibili cookie file lacks DedeUserID, bili_jct, or SESSDATA"
            )
        aid = int(payload["aid"])
        message = str(payload["message"])
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
                status="failed",
                response=query_response,
            )
        response = _post_comment_with_browser(
            str(payload["bvid"]),
            message,
            cookies=cookies,
            failure_screenshot=path.with_name("bilibili-comment-failure.png"),
        )
    except (
        OSError,
        RuntimeError,
        TimeoutError,
        urllib.error.URLError,
        ValueError,
    ) as exc:
        return _record_attempt(
            path,
            payload,
            status="failed",
            response={"exception": f"{type(exc).__name__}: {exc}"},
        )

    code = int(response.get("code", -1))
    if code == 0:
        data = response.get("data")
        rpid = _as_int(data.get("rpid")) if isinstance(data, dict) else None
        time.sleep(_POST_VERIFY_DELAY_SECONDS)
        try:
            review_state, verification_response = _query_comment_by_rpid(
                aid,
                rpid,
                message=message,
                mid=mid,
                cookies=cookies,
            )
        except (OSError, TimeoutError, urllib.error.URLError, ValueError) as exc:
            review_state = "verification_failed"
            verification_response = {
                "exception": f"{type(exc).__name__}: {exc}"
            }
        if review_state != "visible":
            return _record_attempt(
                path,
                payload,
                status=review_state,
                response={
                    "post_response": response,
                    "verification_response": verification_response,
                },
                rpid=rpid,
            )
        return _record_attempt(
            path,
            payload,
            status="posted",
            response=response,
            rpid=rpid,
        )
    if code == 12051:
        return _record_attempt(
            path, payload, status="already_exists", response=response
        )
    status = "rejected" if code in _PERMANENT_CODES else "failed"
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
            {
                "type": 1,
                "oid": aid,
                "sort": 2,
                "pn": page,
                "ps": _REPLIES_PAGE_SIZE,
            }
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
        if count is not None and page * _REPLIES_PAGE_SIZE >= count:
            break
    return None, last_response


def _query_comment_by_rpid(
    aid: int,
    rpid: int | None,
    *,
    message: str,
    mid: str,
    cookies: dict[str, str],
) -> tuple[str, dict[str, Any]]:
    if rpid is None:
        return "rejected", {"code": -1, "message": "missing rpid"}
    query = urllib.parse.urlencode(
        {"type": 1, "oid": aid, "root": rpid, "pn": 1, "ps": 1}
    )
    response = _request_json(f"{_REPLY_DETAIL_URL}?{query}", cookies=cookies)
    data = response.get("data")
    root = data.get("root") if isinstance(data, dict) else None
    if int(response.get("code", -1)) != 0 or not isinstance(root, dict):
        return "rejected", response
    member = root.get("member")
    content = root.get("content")
    sender = str(member.get("mid", "")) if isinstance(member, dict) else ""
    text = str(content.get("message", "")) if isinstance(content, dict) else ""
    if sender != mid or _normalize_message(text) != _normalize_message(message):
        return "rejected", response
    state = _as_int(root.get("state"))
    if state == 17:
        return "hidden", response
    if state == 0:
        return "visible", response
    return "rejected", response


def _post_comment_with_browser(
    bvid: str,
    message: str,
    *,
    cookies: dict[str, str],
    failure_screenshot: Path,
) -> dict[str, Any]:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise BrowserCommentError("Playwright is not installed") from exc

    executable = next(
        (
            value
            for name in ("chromium", "chromium-browser", "google-chrome")
            if (value := shutil.which(name)) is not None
        ),
        None,
    )
    if executable is None:
        raise BrowserCommentError("no Chromium executable found")

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=executable,
            headless=not bool(os.environ.get("DISPLAY")),
            args=["--disable-dev-shm-usage"],
        )
        page = None
        try:
            context = browser.new_context(
                locale="zh-CN",
                user_agent=_BROWSER_USER_AGENT,
                viewport={"width": 1440, "height": 1000},
            )
            context.add_cookies(
                [
                    {
                        "name": name,
                        "value": value,
                        "domain": ".bilibili.com",
                        "path": "/",
                        "secure": True,
                    }
                    for name, value in cookies.items()
                ]
            )
            page = context.new_page()
            return _submit_comment_on_page(page, bvid, message)
        except BrowserCommentError:
            _save_failure_screenshot(page, failure_screenshot)
            raise
        except PlaywrightError as exc:
            _save_failure_screenshot(page, failure_screenshot)
            raise BrowserCommentError(
                f"Playwright comment publication failed: {exc}"
            ) from exc
        finally:
            browser.close()


def _submit_comment_on_page(page: Any, bvid: str, message: str) -> dict[str, Any]:
    page.goto(
        f"https://www.bilibili.com/video/{bvid}",
        wait_until="domcontentloaded",
        timeout=_BROWSER_TIMEOUT_MS,
    )
    comment_area = _first_visible_locator(
        page,
        ("#commentapp", "#comment", "[class*='comment-container']"),
        timeout_ms=15_000,
    )
    if comment_area is not None:
        comment_area.scroll_into_view_if_needed(timeout=10_000)
    editor = _first_visible_locator(
        page,
        (
            (
                "bili-comments bili-comments-header-renderer "
                "bili-comment-box bili-comment-rich-textarea .brt-editor"
            ),
            "textarea.reply-box-textarea",
            "textarea[placeholder*='评论']",
            "[contenteditable='true'][data-placeholder*='评论']",
            "[contenteditable='true'][class*='reply']",
        ),
        timeout_ms=30_000,
    )
    if editor is None:
        raise BrowserCommentError(
            "Bilibili top-level comment editor was not found; login may have expired"
        )
    _fill_comment_editor(editor, message)
    send = _first_visible_locator(
        page,
        (
            (
                "bili-comments bili-comments-header-renderer "
                "bili-comment-box #pub button"
            ),
            'button:has-text("发布")',
            '[role="button"]:has-text("发布")',
            ".reply-box-send",
            '[class*="send"]:has-text("发布")',
        ),
        timeout_ms=10_000,
    )
    if send is None:
        raise BrowserCommentError("Bilibili comment publish button was not found")
    with page.expect_response(
        lambda response: (
            "/x/v2/reply/add" in response.url and response.request.method == "POST"
        ),
        timeout=30_000,
    ) as response_info:
        send.click(timeout=10_000)
    value = response_info.value.json()
    if not isinstance(value, dict):
        raise BrowserCommentError(
            "Bilibili comment page returned a non-object response"
        )
    return value


def _fill_comment_editor(editor: Any, message: str) -> None:
    editor.fill("")
    for index, line in enumerate(message.splitlines()):
        if index:
            editor.press("Shift+Enter")
        if line:
            editor.press_sequentially(line)


def _first_visible_locator(
    root: Any,
    selectors: tuple[str, ...],
    *,
    timeout_ms: int,
) -> Any | None:
    deadline = time.monotonic() + timeout_ms / 1000
    while time.monotonic() < deadline:
        for selector in selectors:
            matches = root.locator(selector)
            for index in range(matches.count()):
                candidate = matches.nth(index)
                if candidate.is_visible():
                    return candidate
        page = root.page if hasattr(root, "page") else root
        page.wait_for_timeout(250)
    return None


def _save_failure_screenshot(page: Any | None, path: Path) -> None:
    if page is None:
        return
    try:
        page.screenshot(path=str(path), full_page=True)
    except Exception:
        logger.exception("could not save Bilibili comment failure screenshot")


def _request_json(
    url: str,
    *,
    cookies: dict[str, str],
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Cookie": "; ".join(f"{key}={value}" for key, value in cookies.items()),
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36"
            ),
            "sec-ch-ua": '"Chromium";v="151", "Not=A?Brand";v="99"',
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "Referer": "https://www.bilibili.com/",
        },
        method="GET",
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
    payload["status"] = status
    payload["updated_at"] = now
    payload["last_response"] = response
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
