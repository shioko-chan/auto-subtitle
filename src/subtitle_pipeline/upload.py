from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .commands import require_command
from .config import UploadConfig

logger = logging.getLogger(__name__)


class UploadNotStartedError(RuntimeError):
    """Upload preparation or process creation failed before biliup started."""


class BiliupCommandError(RuntimeError):
    def __init__(self, returncode: int, output: str):
        super().__init__(f"biliup failed with exit code {returncode}")
        self.returncode = returncode
        self.output = output

    @property
    def submission_rejected(self) -> bool:
        responses = re.findall(r"ResponseData\s*\{\s*code:\s*(-?\d+),\s*data:\s*(None|Some)", self.output)
        # An explicit rejection is different from a transport failure. If any
        # response reports success, do not authorize a duplicate submission.
        if responses:
            return all(int(code) != 0 and data == "None" for code, data in responses)
        return _bilibili_failure_code(self.output) in {406, 429, 21566}


@dataclass(frozen=True)
class BilibiliSubmission:
    aid: int | None
    bvid: str | None
    response: str


def upload_to_bilibili(
    video: Path,
    *,
    title: str,
    source_url: str,
    tags: list[str],
    config: UploadConfig,
) -> BilibiliSubmission:
    return upload_videos_to_bilibili(
        [video],
        title=title,
        source_url=source_url,
        tags=tags,
        config=config,
    )


def upload_videos_to_bilibili(
    videos: Sequence[Path],
    *,
    title: str,
    source_url: str,
    tags: list[str],
    config: UploadConfig,
    append_aid: int | None = None,
) -> BilibiliSubmission:
    if not videos:
        raise UploadNotStartedError("at least one video is required for Bilibili upload")
    with _upload_lock(Path(config.throttle_state_file)):
        return _upload_videos_locked(
            videos, title=title, source_url=source_url, tags=tags, config=config, append_aid=append_aid,
        )


@contextmanager
def _upload_lock(state_path: Path) -> Iterator[None]:
    import fcntl

    try:
        state_path = state_path.resolve()
        path = state_path.with_name(state_path.name + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("a+")
    except OSError as exc:
        raise UploadNotStartedError(f"cannot lock Bilibili upload state: {state_path}") from exc
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX)
        except OSError as exc:
            raise UploadNotStartedError(f"cannot lock Bilibili upload state: {path}") from exc
        yield


def _upload_videos_locked(
    videos: Sequence[Path], *, title: str, source_url: str,
    tags: list[str], config: UploadConfig, append_aid: int | None = None,
) -> BilibiliSubmission:
    try:
        command, pause_marker = _prepare_upload_command(
            videos, title=title, source_url=source_url,
            tags=tags, config=config, append_aid=append_aid,
        )
    except Exception as exc:
        raise UploadNotStartedError(str(exc)) from exc
    delays = config.rate_limit_retry_delays_seconds
    for attempt in range(len(delays) + 1):
        # A pause can be requested while the cooldown or retry delay is sleeping.
        _check_upload_pause(pause_marker)
        try:
            output = _run_biliup(command)
            break
        except BiliupCommandError as exc:
            code = _bilibili_failure_code(exc.output)
            if code == 412:
                _write_pause_marker(pause_marker, code, exc.output)
                raise
            if code not in {406, 429, 21566} or not exc.submission_rejected:
                raise
            if attempt == len(delays):
                _write_pause_marker(pause_marker, code, exc.output)
                logger.error("Bilibili rate limit persisted after %d attempts; batch paused via %s",
                             attempt + 1, pause_marker)
                raise
            retry_after = _retry_after_from_output(exc.output)
            delay = retry_after if retry_after is not None else delays[attempt]
            logger.warning("Bilibili rate limit code=%d; retry %d/%d in %.1fs (%s)",
                           code, attempt + 1, len(delays), delay,
                           "Retry-After" if retry_after is not None else "configured backoff")
            time.sleep(delay)
    try:
        aid, bvid = _submission_ids(output)
    except RuntimeError:
        logger.warning("upload succeeded without parseable aid/bvid")
        aid, bvid = append_aid, None
    try:
        _record_upload_cooldown(config)
    except OSError as exc:
        logger.warning("could not record upload cooldown after successful submission: %s", exc)
    return BilibiliSubmission(aid=aid, bvid=bvid, response=output)


def _prepare_upload_command(
    videos: Sequence[Path], *, title: str, source_url: str,
    tags: list[str], config: UploadConfig, append_aid: int | None = None,
) -> tuple[list[str], Path]:
    if not videos:
        raise ValueError("at least one video is required for Bilibili upload")
    biliup = require_command("biliup")
    cookie_file = Path(config.cookie_file)
    if not cookie_file.is_file():
        raise RuntimeError(
            f"Bilibili cookie file not found: {cookie_file}; run 'biliup login' first"
        )
    pause_marker = Path(config.pause_marker_file)
    _check_upload_pause(pause_marker)
    _wait_for_upload_cooldown(Path(config.throttle_state_file))
    if append_aid is not None:
        if isinstance(append_aid, bool) or not isinstance(append_aid, int) or append_aid <= 0:
            raise ValueError("append requires a positive AID")
        command = [biliup, "--user-cookie", str(cookie_file), "append",
                   "--vid", f"av{append_aid}", "--limit", str(config.limit)]
        if config.line:
            command.extend(["--line", config.line])
        command.extend(str(video) for video in videos)
        logger.info("appending %d clips to Bilibili aid=%d", len(videos), append_aid)
        return command, pause_marker
    description_prefix = config.description_prefix.replace("{youtube_url}", source_url)
    upload_description = _prepare_description(
        description_prefix, max_chars=config.description_max_chars,
    )
    logger.info(
        "Bilibili description: %d -> %d characters, %d UTF-16 units",
        len(description_prefix),
        len(upload_description),
        _utf16_units(upload_description),
    )
    logger.info("Bilibili submission category: tid_v2=%d", config.tid_v2)
    command = [
        biliup,
        "--user-cookie",
        str(cookie_file),
        "upload",
        "--copyright",
        str(config.copyright),
        "--extra-fields",
        json.dumps({"tid_v2": config.tid_v2}),
        "--title",
        (config.title_prefix + title)[:80],
        "--desc",
        upload_description,
        "--tag",
        ",".join(tags),
        "--limit",
        str(config.limit),
    ]
    if config.copyright == 2:
        command.extend(["--source", source_url])
    if config.line:
        command.extend(["--line", config.line])
    command.extend(str(video) for video in videos)
    return command, pause_marker


def _check_upload_pause(path: Path) -> None:
    if path.is_file():
        raise UploadNotStartedError(
            f"Bilibili uploads are paused by {path}; inspect and remove it before resuming"
        )


def _submission_ids(output: str) -> tuple[int, str]:
    clean = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", output)
    match = re.search(
        r'["\']?aid["\']?\s*:\s*(?:Number\()?([0-9]+)\)?'
        r'.{0,500}?["\']?bvid["\']?\s*:\s*(?:String\()?'
        r'["\'](BV[0-9A-Za-z]+)["\']\)?',
        clean,
        flags=re.DOTALL,
    )
    if match is None:
        raise RuntimeError(
            "biliup reported success but its output did not contain aid and bvid"
        )
    return int(match.group(1)), match.group(2)


def _run_biliup(command: Sequence[str]) -> str:
    logger.info("running biliup upload")
    try:
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
    except OSError as exc:
        raise UploadNotStartedError(str(exc)) from exc
    output: deque[str] = deque(maxlen=2000)
    assert process.stdout is not None
    for line in process.stdout:
        output.append(line)
        logger.info("biliup: %s", line.rstrip())
    returncode = process.wait()
    combined = "".join(output)
    if returncode != 0:
        raise BiliupCommandError(returncode, combined)
    return combined


def _bilibili_failure_code(output: str) -> int | None:
    if re.search(r'''["']?code["']?\s*[:=]\s*0\b''', output):
        return None
    for code in (412, 429, 406, 21566):
        patterns = (
            rf'["\']?code["\']?\s*[:=]\s*{code}\b',
            rf"\bHTTP(?: status)?\s*{code}\b",
        )
        if any(re.search(pattern, output, flags=re.IGNORECASE) for pattern in patterns):
            return code
    return None


def _retry_after_from_output(output: str) -> float | None:
    match = re.search(
        r"retry[-_ ]after[\"']?\s*[:=]\s*[\"']?(\d+(?:\.\d+)?)",
        output,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return max(0.0, float(match.group(1)))


def _wait_for_upload_cooldown(path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        next_allowed_at = float(payload["next_allowed_at"])
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        TypeError,
        KeyError,
        json.JSONDecodeError,
    ):
        return
    delay = next_allowed_at - time.time()
    if delay > 0:
        logger.info("waiting %.1fs for Bilibili upload cooldown", delay)
        time.sleep(delay)


def _record_upload_cooldown(config: UploadConfig) -> None:
    delay = random.uniform(
        config.cooldown_min_seconds,
        config.cooldown_max_seconds,
    )
    path = Path(config.throttle_state_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"next_allowed_at": time.time() + delay, "cooldown_seconds": delay},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_pause_marker(path: Path, code: int, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {"code": code, "output_tail": output[-2000:], "paused_at": time.time()},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _truncate_utf16(value: str, max_units: int) -> str:
    encoded = value.encode("utf-16-le")
    if len(encoded) <= max_units * 2:
        return value
    return encoded[: max_units * 2].decode("utf-16-le", errors="ignore")


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _bounded_prefix(value: str, *, max_chars: int, max_utf16_units: int) -> str:
    if max_chars <= 0 or max_utf16_units <= 0:
        return ""
    return _truncate_utf16(value[:max_chars], max_utf16_units)


def _prepare_description(prefix: str, *, max_chars: int) -> str:
    clean_prefix = prefix.replace("\r\n", "\n").replace("\r", "\n").strip()
    return _bounded_prefix(
        clean_prefix, max_chars=max_chars, max_utf16_units=max_chars,
    ).rstrip()
