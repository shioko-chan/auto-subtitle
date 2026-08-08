from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Sequence


class CommandError(RuntimeError):
    pass


def require_command(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise CommandError(f"required command not found on PATH: {name}")
    return path


def run(command: Sequence[str], *, cwd: Path | None = None) -> None:
    logging.info("running: %s", " ".join(_display_arg(arg) for arg in command))
    try:
        subprocess.run(list(command), cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise CommandError(
            f"command failed with exit code {exc.returncode}: {command[0]}"
        ) from exc


def _display_arg(value: str) -> str:
    if any(char.isspace() for char in value):
        return repr(value)
    return value
