from __future__ import annotations

import json
import logging
import random
import re
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import Sequence

from .commands import require_command
from .config import UploadConfig

logger = logging.getLogger(__name__)


class BiliupCommandError(RuntimeError):
    def __init__(self, returncode: int, output: str):
        super().__init__(f"biliup failed with exit code {returncode}")
        self.returncode = returncode
        self.output = output


def upload_to_bilibili(
    video: Path,
    *,
    title: str,
    description: str,
    source_url: str,
    tags: list[str],
    config: UploadConfig,
) -> None:
    biliup = require_command("biliup")
    cookie_file = Path(config.cookie_file)
    if not cookie_file.is_file():
        raise RuntimeError(
            f"Bilibili cookie file not found: {cookie_file}; run 'biliup login' first"
        )
    pause_marker = Path(config.pause_marker_file)
    if pause_marker.is_file():
        raise RuntimeError(
            f"Bilibili uploads are paused by {pause_marker}; inspect and remove it "
            "before resuming"
        )
    _wait_for_upload_cooldown(Path(config.throttle_state_file))
    source = config.source or source_url
    upload_description = _prepare_description(
        description,
        suffix=config.description_suffix,
        max_chars=config.description_max_chars,
    )
    logger.info(
        "Bilibili description: %d -> %d characters, %d UTF-16 units",
        len(description + config.description_suffix),
        len(upload_description),
        _utf16_units(upload_description),
    )
    command = [
        biliup,
        "--user-cookie",
        str(cookie_file),
        "upload",
        "--copyright",
        str(config.copyright),
        "--tid",
        str(config.tid),
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
        command.extend(["--source", source])
    if config.line:
        command.extend(["--line", config.line])
    command.append(str(video))
    for attempt in range(len(config.rate_limit_retry_delays_seconds) + 1):
        try:
            _run_biliup(command)
            _record_upload_cooldown(config)
            return
        except BiliupCommandError as exc:
            code = _bilibili_failure_code(exc.output)
            if code == 412:
                _write_pause_marker(pause_marker, code, exc.output)
                raise RuntimeError(
                    f"Bilibili risk control returned 412; upload queue paused via "
                    f"{pause_marker}"
                ) from exc
            if code not in {406, 429}:
                raise
            if attempt >= len(config.rate_limit_retry_delays_seconds):
                raise RuntimeError(
                    f"Bilibili upload remained rate-limited with code {code} after "
                    f"{attempt + 1} attempts"
                ) from exc
            retry_after = _retry_after_from_output(exc.output)
            delay = (
                retry_after
                if retry_after is not None
                else config.rate_limit_retry_delays_seconds[attempt]
            )
            logger.warning(
                "Bilibili upload returned %d; retrying the complete upload in %.1fs",
                code,
                delay,
            )
            time.sleep(delay)


def _run_biliup(command: Sequence[str]) -> str:
    logger.info("running biliup upload")
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
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
    for code in (412, 429, 406):
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
    except (FileNotFoundError, OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
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
    path = Path(config.throttle_state_file)
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


def _truncate_description_body(
    value: str,
    *,
    max_chars: int,
    max_utf16_units: int,
) -> str:
    candidate = _bounded_prefix(
        value,
        max_chars=max_chars,
        max_utf16_units=max_utf16_units,
    )
    if candidate == value:
        return candidate.rstrip()

    minimum_boundary = int(len(candidate) * 0.65)
    paragraph_end = candidate.rfind("\n\n")
    if paragraph_end > 0:
        return candidate[:paragraph_end].rstrip()

    line_end = candidate.rfind("\n")
    if line_end > 0:
        return candidate[:line_end].rstrip()

    sentence_end = max(candidate.rfind(mark) for mark in "。！？；.!?;")
    if sentence_end >= minimum_boundary:
        return candidate[: sentence_end + 1].rstrip()
    return candidate.rstrip()


def _prepare_description(description: str, *, suffix: str, max_chars: int) -> str:
    body = description.replace("\r\n", "\n").replace("\r", "\n").strip()
    clean_suffix = suffix.replace("\r\n", "\n").replace("\r", "\n").strip()
    clean_suffix = _bounded_prefix(
        clean_suffix,
        max_chars=max_chars,
        max_utf16_units=max_chars,
    ).rstrip()

    separator = "\n\n" if body and clean_suffix else ""
    reserved_chars = len(separator) + len(clean_suffix)
    reserved_units = _utf16_units(separator + clean_suffix)
    body = _truncate_description_body(
        body,
        max_chars=max(0, max_chars - reserved_chars),
        max_utf16_units=max(0, max_chars - reserved_units),
    )
    separator = "\n\n" if body and clean_suffix else ""
    return body + separator + clean_suffix
