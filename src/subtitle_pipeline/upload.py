from __future__ import annotations

import logging
from pathlib import Path

from .commands import require_command, run
from .config import UploadConfig

logger = logging.getLogger(__name__)


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
    run(command)


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
