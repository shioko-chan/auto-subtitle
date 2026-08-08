from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadConfig:
    subtitle_languages: list[str] = field(default_factory=lambda: ["en.*", "ja.*"])
    cookies_from_browser: str | None = None
    cookies_file: str | None = None


@dataclass(frozen=True)
class WhisperConfig:
    model: str = "small"
    device: str = "auto"
    compute_type: str = "auto"
    language: str | None = None
    initial_prompt: str | None = None


@dataclass(frozen=True)
class SegmentationConfig:
    enabled: bool = True
    max_gap_seconds: float = 0.8
    max_duration_seconds: float = 10.0
    max_source_chars: int = 180


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    api_key_pass_entry: str | None = "api/deepseek"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    target_language: str = "简体中文"
    batch_size: int = 30
    timeout_seconds: int = 120
    max_retries: int = 3
    json_mode: bool = True


@dataclass(frozen=True)
class RenderConfig:
    font_name: str = "Noto Sans CJK SC"
    font_size: int = 18
    margin_vertical: int = 32
    crf: int = 20
    preset: str = "medium"


@dataclass(frozen=True)
class UploadConfig:
    enabled: bool = False
    cookie_file: str = "cookies.json"
    copyright: int = 2
    tid: int = 171
    tags: list[str] = field(default_factory=lambda: ["中文字幕"])
    source: str = ""
    title_prefix: str = ""
    description_suffix: str = ""
    line: str | None = None
    limit: int = 3


@dataclass(frozen=True)
class AppConfig:
    work_dir: Path = Path("work")
    download: DownloadConfig = field(default_factory=DownloadConfig)
    whisper: WhisperConfig = field(default_factory=WhisperConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    upload: UploadConfig = field(default_factory=UploadConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be a TOML table")
    return value


def load_config(path: Path) -> AppConfig:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc

    try:
        config = AppConfig(
            work_dir=Path(data.get("work_dir", "work")),
            download=DownloadConfig(**_section(data, "download")),
            whisper=WhisperConfig(**_section(data, "whisper")),
            segmentation=SegmentationConfig(**_section(data, "segmentation")),
            llm=LLMConfig(**_section(data, "llm")),
            render=RenderConfig(**_section(data, "render")),
            upload=UploadConfig(**_section(data, "upload")),
        )
    except TypeError as exc:
        raise ConfigError(f"unknown or missing configuration field: {exc}") from exc

    if config.llm.batch_size < 1:
        raise ConfigError("llm.batch_size must be at least 1")
    if config.llm.max_retries < 1:
        raise ConfigError("llm.max_retries must be at least 1")
    if config.segmentation.max_gap_seconds < 0:
        raise ConfigError("segmentation.max_gap_seconds cannot be negative")
    if config.segmentation.max_duration_seconds <= 0:
        raise ConfigError("segmentation.max_duration_seconds must be positive")
    if config.segmentation.max_source_chars < 1:
        raise ConfigError("segmentation.max_source_chars must be at least 1")
    if config.upload.copyright not in (1, 2):
        raise ConfigError("upload.copyright must be 1 (original) or 2 (repost)")
    return config


def llm_api_key(config: LLMConfig) -> str:
    if config.api_key_pass_entry:
        pass_command = shutil.which("pass")
        if pass_command is None:
            raise ConfigError(
                "password-store command 'pass' was not found; install pass or set "
                "llm.api_key_pass_entry to an empty value to use api_key_env"
            )
        try:
            result = subprocess.run(
                [pass_command, "show", config.api_key_pass_entry],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise ConfigError(
                f"failed to read LLM API key from pass entry: {config.api_key_pass_entry}"
            ) from exc
        first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        if not first_line:
            raise ConfigError(f"pass entry is empty: {config.api_key_pass_entry}")
        return first_line

    value = os.environ.get(config.api_key_env)
    if not value:
        raise ConfigError(
            f"environment variable {config.api_key_env} is required when no pass entry is configured"
        )
    return value
