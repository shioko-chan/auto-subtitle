from __future__ import annotations

from pathlib import Path

from .commands import require_command, run
from .config import UploadConfig


def upload_to_bilibili(
    video: Path,
    *,
    title: str,
    description: str,
    source_url: str,
    config: UploadConfig,
) -> None:
    biliup = require_command("biliup")
    cookie_file = Path(config.cookie_file)
    if not cookie_file.is_file():
        raise RuntimeError(
            f"Bilibili cookie file not found: {cookie_file}; run 'biliup login' first"
        )
    source = config.source or source_url
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
        (description + config.description_suffix)[:2000],
        "--tag",
        ",".join(config.tags),
        "--limit",
        str(config.limit),
    ]
    if config.copyright == 2:
        command.extend(["--source", source])
    if config.line:
        command.extend(["--line", config.line])
    command.append(str(video))
    run(command)
