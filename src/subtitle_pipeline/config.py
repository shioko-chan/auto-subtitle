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
    cookies_from_browser: str | None = None
    cookies_file: str | None = None
    js_runtime: str | None = "auto"


@dataclass(frozen=True)
class ASRConfig:
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    device: str = "cuda:0"
    dtype: str = "float16"
    language: str = "Japanese"
    context: str = ""
    chunk_seconds: float = 170.0
    chunk_context_seconds: float = 2.0
    max_inference_batch_size: int = 1
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class SegmentationConfig:
    model_window_cues: int = 600


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    api_key_pass_entry: str | None = "api/deepseek"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    target_language: str = "简体中文"
    batch_size: int = 30
    timeout_seconds: int = 120
    max_retries: int = 5
    max_tokens: int = 16384
    context_cues: int = 3
    json_mode: bool = True
    thinking: str | None = None
    translate_metadata: bool = True
    metadata_description_max_chars: int = 6000
    metadata_tag_count: int = 5
    metadata_subtitle_max_chars: int = 4000
    ip_aliases_file: str | None = None
    glossary_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RenderConfig:
    font_name: str = "Noto Sans CJK SC"
    font_size_ratio: float = 0.066
    portrait_font_size_ratio: float = 0.077
    min_font_size: int = 28
    max_font_size: int = 144
    margin_horizontal_ratio: float = 0.075
    portrait_margin_horizontal_ratio: float = 0.025
    margin_vertical_ratio: float = 0.05
    outline_ratio: float = 0.003
    crf: int = 20
    preset: str = "medium"


@dataclass(frozen=True)
class UploadConfig:
    enabled: bool = False
    cookie_file: str = "cookies.json"
    copyright: int = 2
    tid: int = 171
    tags: list[str] = field(default_factory=lambda: ["中文字幕"])
    max_tags: int = 10
    tag_catalog_file: str | None = None
    source: str = ""
    title_prefix: str = ""
    description_suffix: str = ""
    line: str | None = None
    limit: int = 3


@dataclass(frozen=True)
class AppConfig:
    work_dir: Path = Path("work")
    download: DownloadConfig = field(default_factory=DownloadConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
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
            asr=ASRConfig(**_section(data, "asr")),
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
    if config.llm.max_tokens < 1:
        raise ConfigError("llm.max_tokens must be at least 1")
    if config.llm.context_cues < 0:
        raise ConfigError("llm.context_cues cannot be negative")
    if config.llm.thinking not in (None, "enabled", "disabled"):
        raise ConfigError("llm.thinking must be 'enabled' or 'disabled'")
    if config.llm.metadata_description_max_chars < 0:
        raise ConfigError("llm.metadata_description_max_chars cannot be negative")
    if not 1 <= config.llm.metadata_tag_count <= 10:
        raise ConfigError("llm.metadata_tag_count must be between 1 and 10")
    if config.llm.metadata_subtitle_max_chars < 0:
        raise ConfigError("llm.metadata_subtitle_max_chars cannot be negative")
    if not isinstance(config.llm.glossary_files, list) or not all(
        isinstance(path, str) and path.strip() for path in config.llm.glossary_files
    ):
        raise ConfigError("llm.glossary_files must be a list of non-empty paths")
    if not config.asr.model.strip():
        raise ConfigError("asr.model cannot be empty")
    if not config.asr.aligner_model.strip():
        raise ConfigError("asr.aligner_model cannot be empty")
    if config.asr.dtype not in {"float16", "bfloat16", "float32"}:
        raise ConfigError("asr.dtype must be 'float16', 'bfloat16', or 'float32'")
    if config.asr.chunk_seconds <= 0 or config.asr.chunk_seconds > 175:
        raise ConfigError("asr.chunk_seconds must be between 0 and 175")
    if config.asr.chunk_context_seconds < 0:
        raise ConfigError("asr.chunk_context_seconds cannot be negative")
    if config.asr.chunk_seconds + 2 * config.asr.chunk_context_seconds > 180:
        raise ConfigError(
            "asr chunk plus both context windows cannot exceed 180 seconds"
        )
    if config.asr.max_inference_batch_size < 1:
        raise ConfigError("asr.max_inference_batch_size must be at least 1")
    if config.asr.max_new_tokens < 1:
        raise ConfigError("asr.max_new_tokens must be at least 1")
    if config.segmentation.model_window_cues < 1:
        raise ConfigError("segmentation.model_window_cues must be at least 1")
    if config.render.font_size_ratio <= 0:
        raise ConfigError("render.font_size_ratio must be positive")
    if config.render.portrait_font_size_ratio <= 0:
        raise ConfigError("render.portrait_font_size_ratio must be positive")
    if config.render.min_font_size < 1:
        raise ConfigError("render.min_font_size must be at least 1")
    if config.render.max_font_size < config.render.min_font_size:
        raise ConfigError(
            "render.max_font_size must be at least render.min_font_size"
        )
    if not 0 <= config.render.margin_horizontal_ratio < 0.5:
        raise ConfigError(
            "render.margin_horizontal_ratio must be between 0 and 0.5"
        )
    if not 0 <= config.render.portrait_margin_horizontal_ratio < 0.5:
        raise ConfigError(
            "render.portrait_margin_horizontal_ratio must be between 0 and 0.5"
        )
    if not 0 <= config.render.margin_vertical_ratio < 0.5:
        raise ConfigError("render.margin_vertical_ratio must be between 0 and 0.5")
    if config.render.outline_ratio < 0:
        raise ConfigError("render.outline_ratio cannot be negative")
    if config.upload.copyright not in (1, 2):
        raise ConfigError("upload.copyright must be 1 (original) or 2 (repost)")
    if not 1 <= config.upload.max_tags <= 10:
        raise ConfigError("upload.max_tags must be between 1 and 10")
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
