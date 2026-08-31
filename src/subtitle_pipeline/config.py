from __future__ import annotations

import os
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadConfig:
    cookies_from_browser: str | None = None
    cookies_file: str | None = None
    js_runtime: str | None = "auto"
    concurrent_fragments: int = 8
    download_chat_replay: bool = True
    download_top_comments: bool = True
    top_comment_fetch_limit: int = 100
    top_comment_min_likes: int = 10
    top_comment_background_limit: int = 30
    video_format: str = (
        "bv*[vcodec^=vp9]+ba/bv*[vcodec^=vp09]+ba/bv*[vcodec^=avc1]+ba/b"
    )


@dataclass(frozen=True)
class ASRConfig:
    model: str = "Qwen/Qwen3-ASR-1.7B"
    aligner_model: str = "Qwen/Qwen3-ForcedAligner-0.6B"
    singing_model: str = "HeartMuLa/HeartTranscriptor-oss"
    singing_max_new_tokens: int = 256
    singing_num_beams: int = 2
    device: str = "cuda:0"
    dtype: str = "float16"
    # None lets Qwen detect the language independently for each ASR window.
    language: str | None = None
    context: str = ""
    chunk_seconds: float = 170.0
    chunk_context_seconds: float = 0.5
    max_inference_batch_size: int = 4
    max_new_tokens: int = 2048
    speech_window_target_seconds: float = 60.0
    speech_window_max_seconds: float = 90.0
    speech_window_max_gap_seconds: float = 2.0
    speech_window_max_silence_seconds: float = 15.0
    speech_window_min_coverage: float = 0.6


@dataclass(frozen=True)
class AudioAnalysisConfig:
    enabled: bool = True
    debug_audio_artifacts: bool = False
    initial_analysis_concurrency: int = 1
    diarization_backend: str = "pyannote"
    diarization_model: str = "pyannote/speaker-diarization-community-1"
    overlap_conditioned_asr_seconds: float = 0.5
    overlap_context_seconds: float = 2.0
    conditioned_asr_backend: str = "dicow"
    conditioned_asr_model: str = "BUT-FIT/DiCoW_v3_3"
    conditioned_asr_revision: str = "c34b64d9a9c5148c65fd355bb188d60343a6b44f"
    conditioned_asr_worker_project: str = "tools/dicow"
    conditioned_asr_batch_size: int = 4
    moss_transcribe_model: str = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    moss_transcribe_worker_project: str = "tools/moss_transcribe"
    moss_window_seconds: float = 480.0
    moss_max_window_seconds: float = 540.0
    moss_window_search_seconds: float = 60.0
    moss_max_new_tokens: int = 8192
    speaker_embedding_backend: str = "eres2netv2"
    speaker_embedding_model: str = "iic/speech_eres2netv2_sv_zh-cn_16k-common"
    speaker_embedding_worker_project: str = "tools/speaker_embedding"
    singing_model: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
    device: str = "cuda:0"
    singing_window_seconds: float = 5.0
    singing_stride_seconds: float = 2.5
    singing_threshold: float = 0.015
    singing_music_threshold: float = 0.05
    singing_speech_takeover_threshold: float = 0.5
    singing_vocal_threshold: float = 0.15
    singing_merge_gap_seconds: float = 1.5
    singing_smoothing_windows: int = 3
    singing_phrase_silence_seconds: float = 0.45
    singing_asr_target_seconds: float = 10.0
    singing_asr_min_seconds: float = 6.0
    singing_asr_max_seconds: float = 15.0
    singing_asr_search_seconds: float = 4.0
    singing_asr_overlap_seconds: float = 1.0
    speaker_profiles_dir: str = "work/speaker-profiles-eres2netv2"
    speaker_match_threshold: float = 0.42
    speaker_match_margin: float = 0.025
    speaker_identity_trim_ratio: float = 0.15
    speaker_identity_max_weight_seconds: float = 10.0
    speaker_identity_edge_trim_seconds: float = 0.15
    speaker_identity_min_segment_seconds: float = 1.5
    speaker_enrollment_samples_per_video: int = 40
    speaker_profile_max_centers: int = 5
    speaker_profile_min_samples_per_center: int = 20
    character_styles_file: str | None = None


@dataclass(frozen=True)
class SongIdentificationConfig:
    enabled: bool = False
    device: str = "gpu:0"
    detection_model: str = "PP-OCRv5_mobile_det"
    recognition_model: str = "PP-OCRv5_server_rec"
    ocr_worker_project: str = "tools/song_ocr"
    search_worker_project: str = "tools/song_search"
    seconds_before_start: float = 30.0
    seconds_after_start: float = 15.0
    sample_interval_seconds: float = 1.0
    song_search_group_gap_seconds: float = 35.0
    minimum_ocr_score: float = 0.45
    minimum_persistent_frames: int = 2
    max_search_results: int = 5
    lyrics_library_path: str = "databases/lyrics.sqlite3"
    match_anchor_threshold: float = 0.48
    match_minimum_anchors: int = 3
    match_minimum_score: float = 0.48
    match_minimum_margin: float = 0.05
    pyshiro_worker_project: str = "tools/pyshiro"
    vocal_separation_device: str = "cuda:0"
    pyshiro_max_window_seconds: float = 20.0
    lyric_gap_recheck_seconds: float = 20.0
    lyric_gap_vocal_active_ratio: float = 0.08
    pyshiro_likelihood_floor: float = -30.0
    pyshiro_gap_likelihood_margin: float = 0.1
    pyshiro_likelihood_margin: float = 0.2
    lyric_neighbor_max_lines: int = 12
    lyric_neighbor_min_coverage: float = 0.45


@dataclass(frozen=True)
class ASRCorrectionConfig:
    batch_windows: int = 6
    batch_chars: int = 3000
    max_tokens: int = 8192


@dataclass(frozen=True)
class SegmentationConfig:
    boundary_score_threshold: int = 3
    local_unit_min_fallback_seconds: float = 2.0
    local_unit_max_seconds: float = 6.0
    speaker_episode_gap_seconds: float = 2.0
    model_window_units: int = 240
    model_window_chars: int = 3000
    max_tokens: int = 8192
    dialogue_context_before_seconds: float = 20.0
    dialogue_context_after_seconds: float = 10.0
    dialogue_context_max_chars: int = 4000


@dataclass(frozen=True)
class TranslationConfig:
    target_language: str = "简体中文"
    batch_cues: int = 32
    batch_chars: int = 3000
    max_tokens: int = 8192
    context_before_seconds: float = 20.0
    context_after_seconds: float = 10.0
    context_max_chars: int = 4000
    local_model: str = "facebook/m2m100_418M"
    local_device: str = "cpu"
    translate_metadata: bool = True
    metadata_description_max_chars: int = 6000
    metadata_tag_count: int = 5
    metadata_subtitle_max_chars: int = 4000
    ip_aliases_file: str | None = None
    glossary_files: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FanKnowledgeConfig:
    enabled: bool = True
    database_path: str = "databases/fan-knowledge.sqlite3"
    collection_cache_dir: str = "work/knowledge/youtube"
    chunk_target_seconds: float = 45.0
    chunk_max_chars: int = 1400
    youtube_subtitle_languages: list[str] = field(
        default_factory=lambda: ["ja.*", "ja", "en.*", "en"]
    )
    include_regular_chat: bool = False
    current_video_chat_lookback_seconds: float = 30.0
    current_video_chat_lookahead_seconds: float = 15.0
    top_k_asr: int = 6
    top_k_translation: int = 12
    translation_query_chars: int = 12000
    auto_update_enabled: bool = True
    update_interval_hours: float = 24.0
    youtube_sources: list[str] = field(default_factory=list)
    youtube_playlist_limit: int = 10000
    official_sources: list[str] = field(default_factory=list)
    sns_sources: list[str] = field(default_factory=list)
    embedding_model: str | None = "intfloat/multilingual-e5-base"
    vector_index_path: str = "databases/fan-knowledge.faiss"
    vector_minimum_score: float = 0.62
    reranker_model: str | None = "BAAI/bge-reranker-v2-m3"
    reranker_minimum_score: float = 0.05
    term_extraction_context_size: int = 262144
    term_extraction_target_input_tokens: int = 220000
    term_extraction_max_output_tokens: int = 16384


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    api_style: str = "chat_completions"
    api_key_pass_entry: str | None = "api/deepseek"
    api_key_env: str = "DEEPSEEK_API_KEY"
    model: str = "deepseek-chat"
    timeout_seconds: int = 120
    max_retries: int = 5
    max_concurrency: int = 16
    local_server_enabled: bool = False
    local_server_command: str = "llama-server"
    local_server_model_path: str | None = None
    local_server_hf_repo: str | None = None
    local_server_host: str = "127.0.0.1"
    local_server_port: int = 8080
    local_server_context_size: int = 8192
    local_server_gpu_layers: int = 99
    local_server_parallel: int = 2
    local_server_reasoning: str = "off"
    local_server_startup_timeout_seconds: int = 1800
    json_mode: bool = True
    thinking: str | None = None
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class RenderConfig:
    backend: str = "auto"
    font_name: str = "Noto Sans CJK SC"
    font_size_ratio: float = 0.066
    portrait_font_size_ratio: float = 0.077
    min_font_size: int = 28
    max_font_size: int = 144
    margin_horizontal_ratio: float = 0.075
    portrait_margin_horizontal_ratio: float = 0.025
    margin_vertical_ratio: float = 0.05
    outline_ratio: float = 0.0045
    crf: int = 20
    preset: str = "medium"
    nvenc_preset: str = "p4"
    nvenc_cq: int = 20


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
    description_prefix: str = ""
    description_max_chars: int = 1800
    line: str | None = None
    limit: int = 3
    cooldown_min_seconds: float = 60.0
    cooldown_max_seconds: float = 120.0
    rate_limit_retry_delays_seconds: list[float] = field(
        default_factory=lambda: [120.0, 300.0, 600.0, 1200.0]
    )
    throttle_state_file: str = "work/bilibili-upload-throttle.json"
    pause_marker_file: str = "work/bilibili-upload-paused.json"
    song_setlist_comment: bool = True


@dataclass(frozen=True)
class AppConfig:
    work_dir: Path = Path("work")
    download: DownloadConfig = field(default_factory=DownloadConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    audio_analysis: AudioAnalysisConfig = field(default_factory=AudioAnalysisConfig)
    song_identification: SongIdentificationConfig = field(
        default_factory=SongIdentificationConfig
    )
    asr_correction: ASRCorrectionConfig = field(default_factory=ASRCorrectionConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    translation: TranslationConfig = field(default_factory=TranslationConfig)
    fan_knowledge: FanKnowledgeConfig = field(default_factory=FanKnowledgeConfig)
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
            audio_analysis=AudioAnalysisConfig(**_section(data, "audio_analysis")),
            song_identification=SongIdentificationConfig(
                **_section(data, "song_identification")
            ),
            asr_correction=ASRCorrectionConfig(**_section(data, "asr_correction")),
            segmentation=SegmentationConfig(**_section(data, "segmentation")),
            translation=TranslationConfig(**_section(data, "translation")),
            fan_knowledge=FanKnowledgeConfig(**_section(data, "fan_knowledge")),
            llm=LLMConfig(**_section(data, "llm")),
            render=RenderConfig(**_section(data, "render")),
            upload=UploadConfig(**_section(data, "upload")),
        )
    except TypeError as exc:
        raise ConfigError(f"unknown or missing configuration field: {exc}") from exc

    if not 1 <= config.download.concurrent_fragments <= 32:
        raise ConfigError("download.concurrent_fragments must be between 1 and 32")
    if config.download.top_comment_fetch_limit < 1:
        raise ConfigError("download.top_comment_fetch_limit must be at least 1")
    if config.download.top_comment_min_likes < 0:
        raise ConfigError("download.top_comment_min_likes cannot be negative")
    if config.download.top_comment_background_limit < 1:
        raise ConfigError("download.top_comment_background_limit must be at least 1")
    if config.llm.max_retries < 1:
        raise ConfigError("llm.max_retries must be at least 1")
    if config.llm.max_concurrency < 1:
        raise ConfigError("llm.max_concurrency must be at least 1")
    for section_name, max_tokens in (
        ("asr_correction", config.asr_correction.max_tokens),
        ("segmentation", config.segmentation.max_tokens),
        ("translation", config.translation.max_tokens),
    ):
        if max_tokens < 1:
            raise ConfigError(f"{section_name}.max_tokens must be at least 1")
    if config.llm.local_server_enabled:
        if config.llm.api_style != "chat_completions":
            raise ConfigError(
                "llm.local_server_enabled requires api_style='chat_completions'"
            )
        if not config.llm.local_server_command.strip():
            raise ConfigError("llm.local_server_command cannot be empty")
        model_sources = (
            bool(
                config.llm.local_server_model_path
                and config.llm.local_server_model_path.strip()
            ),
            bool(
                config.llm.local_server_hf_repo
                and config.llm.local_server_hf_repo.strip()
            ),
        )
        if sum(model_sources) != 1:
            raise ConfigError(
                "local LLM server requires exactly one of local_server_model_path "
                "or local_server_hf_repo"
            )
        if config.llm.local_server_host not in {"127.0.0.1", "localhost", "::1"}:
            raise ConfigError("llm.local_server_host must be a loopback address")
        if not 1 <= config.llm.local_server_port <= 65535:
            raise ConfigError("llm.local_server_port must be between 1 and 65535")
        if config.llm.local_server_context_size < 1:
            raise ConfigError("llm.local_server_context_size must be at least 1")
        if config.llm.local_server_gpu_layers < 0:
            raise ConfigError("llm.local_server_gpu_layers cannot be negative")
        if config.llm.local_server_parallel < 1:
            raise ConfigError("llm.local_server_parallel must be at least 1")
        if config.llm.local_server_reasoning not in {"on", "off", "auto"}:
            raise ConfigError(
                "llm.local_server_reasoning must be 'on', 'off', or 'auto'"
            )
        if config.llm.local_server_startup_timeout_seconds <= 0:
            raise ConfigError(
                "llm.local_server_startup_timeout_seconds must be positive"
            )
        endpoint = urlsplit(config.llm.base_url)
        endpoint_port = endpoint.port or (443 if endpoint.scheme == "https" else 80)
        if (
            endpoint.scheme != "http"
            or endpoint.hostname not in {"127.0.0.1", "localhost", "::1"}
            or endpoint_port != config.llm.local_server_port
        ):
            raise ConfigError(
                "llm.base_url must point to the configured local HTTP server port"
            )
    if config.llm.api_style not in {"chat_completions", "responses"}:
        raise ConfigError("llm.api_style must be 'chat_completions' or 'responses'")
    knowledge = config.fan_knowledge
    if knowledge.update_interval_hours <= 0:
        raise ConfigError("fan_knowledge.update_interval_hours must be positive")
    if knowledge.youtube_playlist_limit < 1:
        raise ConfigError("fan_knowledge.youtube_playlist_limit must be positive")
    if not 0 <= knowledge.vector_minimum_score <= 1:
        raise ConfigError("fan_knowledge.vector_minimum_score must be between 0 and 1")
    if not 0 <= knowledge.reranker_minimum_score <= 1:
        raise ConfigError(
            "fan_knowledge.reranker_minimum_score must be between 0 and 1"
        )
    if config.llm.thinking not in (None, "enabled", "disabled"):
        raise ConfigError("llm.thinking must be 'enabled' or 'disabled'")
    if config.llm.reasoning_effort not in (
        None,
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
    ):
        raise ConfigError(
            "llm.reasoning_effort must be minimal, low, medium, high, or xhigh"
        )
    if config.llm.api_style == "responses" and config.llm.thinking is not None:
        raise ConfigError(
            "llm.thinking is only supported by chat_completions; use "
            "llm.reasoning_effort with responses"
        )
    if (
        config.llm.api_style == "chat_completions"
        and config.llm.reasoning_effort is not None
    ):
        raise ConfigError(
            "llm.reasoning_effort is only supported when api_style='responses'"
        )
    if not config.asr.model.strip():
        raise ConfigError("asr.model cannot be empty")
    if not config.asr.aligner_model.strip():
        raise ConfigError("asr.aligner_model cannot be empty")
    if not config.asr.singing_model.strip():
        raise ConfigError("asr.singing_model cannot be empty")
    if config.asr.singing_max_new_tokens < 1:
        raise ConfigError("asr.singing_max_new_tokens must be at least 1")
    if config.asr.singing_num_beams < 1:
        raise ConfigError("asr.singing_num_beams must be at least 1")
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
    if config.asr.speech_window_target_seconds <= 0:
        raise ConfigError("asr.speech_window_target_seconds must be positive")
    if not (
        config.asr.speech_window_target_seconds
        <= config.asr.speech_window_max_seconds
        <= 180
    ):
        raise ConfigError(
            "asr.speech_window_max_seconds must be between the target and 180"
        )
    if config.asr.speech_window_max_seconds <= 2 * config.asr.chunk_context_seconds:
        raise ConfigError("asr speech window must be longer than both context margins")
    if config.asr.speech_window_max_gap_seconds < 0:
        raise ConfigError("asr.speech_window_max_gap_seconds cannot be negative")
    if config.asr.speech_window_max_silence_seconds < 0:
        raise ConfigError("asr.speech_window_max_silence_seconds cannot be negative")
    if not 0 < config.asr.speech_window_min_coverage <= 1:
        raise ConfigError("asr.speech_window_min_coverage must be between 0 and 1")
    analysis = config.audio_analysis
    if analysis.initial_analysis_concurrency not in {1, 2}:
        raise ConfigError("audio_analysis.initial_analysis_concurrency must be 1 or 2")
    if analysis.diarization_backend not in {"moss", "pyannote"}:
        raise ConfigError(
            "audio_analysis.diarization_backend must be 'moss' or 'pyannote'"
        )
    if analysis.overlap_conditioned_asr_seconds < 0:
        raise ConfigError(
            "audio_analysis.overlap_conditioned_asr_seconds cannot be negative"
        )
    if analysis.overlap_context_seconds < 0:
        raise ConfigError("audio_analysis.overlap_context_seconds cannot be negative")
    if analysis.conditioned_asr_backend not in {"dicow", "disabled"}:
        raise ConfigError(
            "audio_analysis.conditioned_asr_backend must be 'dicow' or 'disabled'"
        )
    if analysis.conditioned_asr_backend == "dicow" and not (
        analysis.conditioned_asr_model.strip()
        and analysis.conditioned_asr_revision.strip()
    ):
        raise ConfigError(
            "audio_analysis conditioned ASR model and revision cannot be empty"
        )
    if analysis.conditioned_asr_batch_size < 1:
        raise ConfigError(
            "audio_analysis.conditioned_asr_batch_size must be at least 1"
        )
    if not analysis.moss_transcribe_model.strip():
        raise ConfigError("audio_analysis.moss_transcribe_model cannot be empty")
    if analysis.moss_window_seconds <= 0:
        raise ConfigError("audio_analysis.moss_window_seconds must be positive")
    if not (analysis.moss_window_seconds <= analysis.moss_max_window_seconds <= 900):
        raise ConfigError(
            "audio_analysis.moss_max_window_seconds must be between "
            "moss_window_seconds and 900"
        )
    if not 0 <= analysis.moss_window_search_seconds < analysis.moss_window_seconds:
        raise ConfigError(
            "audio_analysis.moss_window_search_seconds must be non-negative and "
            "smaller than moss_window_seconds"
        )
    if analysis.moss_max_new_tokens < 1:
        raise ConfigError("audio_analysis.moss_max_new_tokens must be at least 1")
    if analysis.speaker_embedding_backend not in {"eres2netv2", "wespeaker"}:
        raise ConfigError(
            "audio_analysis.speaker_embedding_backend must be 'eres2netv2' or "
            "'wespeaker'"
        )
    if analysis.singing_window_seconds <= 0:
        raise ConfigError("audio_analysis.singing_window_seconds must be positive")
    if not 0 < analysis.singing_stride_seconds <= analysis.singing_window_seconds:
        raise ConfigError(
            "audio_analysis.singing_stride_seconds must be positive and no larger "
            "than singing_window_seconds"
        )
    if not 0 <= analysis.singing_threshold <= 1:
        raise ConfigError("audio_analysis.singing_threshold must be between 0 and 1")
    if not 0 <= analysis.singing_music_threshold <= 1:
        raise ConfigError(
            "audio_analysis.singing_music_threshold must be between 0 and 1"
        )
    if not 0 <= analysis.singing_speech_takeover_threshold <= 1:
        raise ConfigError(
            "audio_analysis.singing_speech_takeover_threshold must be between 0 and 1"
        )
    if not 0 <= analysis.singing_vocal_threshold <= 1:
        raise ConfigError(
            "audio_analysis.singing_vocal_threshold must be between 0 and 1"
        )
    if analysis.singing_merge_gap_seconds < 0:
        raise ConfigError("audio_analysis.singing_merge_gap_seconds cannot be negative")
    if (
        analysis.singing_smoothing_windows < 1
        or analysis.singing_smoothing_windows % 2 == 0
    ):
        raise ConfigError(
            "audio_analysis.singing_smoothing_windows must be a positive odd integer"
        )
    if analysis.singing_phrase_silence_seconds <= 0:
        raise ConfigError(
            "audio_analysis.singing_phrase_silence_seconds must be positive"
        )
    if analysis.singing_asr_min_seconds <= 0:
        raise ConfigError("audio_analysis.singing_asr_min_seconds must be positive")
    if not (
        analysis.singing_asr_min_seconds
        <= analysis.singing_asr_target_seconds
        <= analysis.singing_asr_max_seconds
    ):
        raise ConfigError(
            "audio_analysis singing ASR target must be between min and max"
        )
    if analysis.singing_asr_search_seconds < 0:
        raise ConfigError(
            "audio_analysis.singing_asr_search_seconds cannot be negative"
        )
    if not (
        0 <= analysis.singing_asr_overlap_seconds < analysis.singing_asr_min_seconds
    ):
        raise ConfigError(
            "audio_analysis.singing_asr_overlap_seconds must be non-negative and "
            "smaller than singing_asr_min_seconds"
        )
    if analysis.singing_asr_max_seconds < (
        2 * analysis.singing_asr_min_seconds - analysis.singing_asr_overlap_seconds
    ):
        raise ConfigError(
            "audio_analysis.singing_asr_max_seconds must meet the minimum "
            "two-window span"
        )
    if not 0 <= analysis.speaker_match_threshold <= 2:
        raise ConfigError(
            "audio_analysis.speaker_match_threshold must be between 0 and 2"
        )
    if not 0 <= analysis.speaker_match_margin <= 2:
        raise ConfigError("audio_analysis.speaker_match_margin must be between 0 and 2")
    if not 0 <= analysis.speaker_identity_trim_ratio < 0.5:
        raise ConfigError(
            "audio_analysis.speaker_identity_trim_ratio must be between 0 and 0.5"
        )
    if analysis.speaker_identity_max_weight_seconds <= 0:
        raise ConfigError(
            "audio_analysis.speaker_identity_max_weight_seconds must be positive"
        )
    if analysis.speaker_identity_edge_trim_seconds < 0:
        raise ConfigError(
            "audio_analysis.speaker_identity_edge_trim_seconds cannot be negative"
        )
    if analysis.speaker_identity_min_segment_seconds <= 0:
        raise ConfigError(
            "audio_analysis.speaker_identity_min_segment_seconds must be positive"
        )
    if analysis.speaker_enrollment_samples_per_video < 1:
        raise ConfigError(
            "audio_analysis.speaker_enrollment_samples_per_video must be at least 1"
        )
    if analysis.speaker_profile_max_centers < 1:
        raise ConfigError(
            "audio_analysis.speaker_profile_max_centers must be at least 1"
        )
    if analysis.speaker_profile_min_samples_per_center < 1:
        raise ConfigError(
            "audio_analysis.speaker_profile_min_samples_per_center must be at least 1"
        )
    songs = config.song_identification
    if songs.seconds_before_start < 0 or songs.seconds_after_start < 0:
        raise ConfigError("song identification OCR windows cannot be negative")
    if songs.sample_interval_seconds <= 0:
        raise ConfigError(
            "song_identification.sample_interval_seconds must be positive"
        )
    if songs.song_search_group_gap_seconds < 0:
        raise ConfigError(
            "song_identification.song_search_group_gap_seconds cannot be negative"
        )
    if not 0 <= songs.minimum_ocr_score <= 1:
        raise ConfigError(
            "song_identification.minimum_ocr_score must be between 0 and 1"
        )
    if songs.minimum_persistent_frames < 1:
        raise ConfigError(
            "song_identification.minimum_persistent_frames must be at least 1"
        )
    if songs.max_search_results < 1:
        raise ConfigError("song identification search limits must be at least 1")
    if not 0 <= songs.match_anchor_threshold <= 1:
        raise ConfigError(
            "song_identification.match_anchor_threshold must be between 0 and 1"
        )
    if songs.match_minimum_anchors < 2:
        raise ConfigError(
            "song_identification.match_minimum_anchors must be at least 2"
        )
    if not 0 <= songs.match_minimum_score <= 1 or songs.match_minimum_margin < 0:
        raise ConfigError("song identification match score settings are invalid")
    if songs.pyshiro_max_window_seconds <= 0 or songs.pyshiro_max_window_seconds > 20:
        raise ConfigError(
            "song_identification.pyshiro_max_window_seconds must be in (0, 20]"
        )
    if not 0 < songs.lyric_gap_recheck_seconds <= 20:
        raise ConfigError(
            "song_identification.lyric_gap_recheck_seconds must be in (0, 20]"
        )
    if not 0 <= songs.lyric_gap_vocal_active_ratio <= 1:
        raise ConfigError(
            "song_identification.lyric_gap_vocal_active_ratio must be between 0 and 1"
        )
    if songs.pyshiro_gap_likelihood_margin < 0 or songs.pyshiro_likelihood_margin < 0:
        raise ConfigError(
            "song identification pySHIRO likelihood margins cannot be negative"
        )
    if songs.lyric_neighbor_max_lines < 1:
        raise ConfigError(
            "song_identification.lyric_neighbor_max_lines must be at least 1"
        )
    if not 0 <= songs.lyric_neighbor_min_coverage <= 1:
        raise ConfigError(
            "song_identification.lyric_neighbor_min_coverage must be between 0 and 1"
        )
    correction = config.asr_correction
    if correction.batch_windows < 1:
        raise ConfigError("asr_correction.batch_windows must be at least 1")
    if correction.batch_chars < 1:
        raise ConfigError("asr_correction.batch_chars must be at least 1")
    segmentation = config.segmentation
    if segmentation.boundary_score_threshold < 0:
        raise ConfigError("segmentation.boundary_score_threshold cannot be negative")
    if segmentation.local_unit_min_fallback_seconds <= 0:
        raise ConfigError(
            "segmentation.local_unit_min_fallback_seconds must be positive"
        )
    if (
        segmentation.local_unit_max_seconds
        < segmentation.local_unit_min_fallback_seconds
    ):
        raise ConfigError(
            "segmentation.local_unit_max_seconds must be at least "
            "local_unit_min_fallback_seconds"
        )
    if segmentation.speaker_episode_gap_seconds <= 0:
        raise ConfigError("segmentation.speaker_episode_gap_seconds must be positive")
    if segmentation.model_window_units < 1:
        raise ConfigError("segmentation.model_window_units must be at least 1")
    if segmentation.model_window_chars < 1:
        raise ConfigError("segmentation.model_window_chars must be at least 1")
    if segmentation.dialogue_context_before_seconds < 0:
        raise ConfigError(
            "segmentation.dialogue_context_before_seconds cannot be negative"
        )
    if segmentation.dialogue_context_after_seconds < 0:
        raise ConfigError(
            "segmentation.dialogue_context_after_seconds cannot be negative"
        )
    if segmentation.dialogue_context_max_chars < 1:
        raise ConfigError("segmentation.dialogue_context_max_chars must be at least 1")
    translation = config.translation
    if not translation.target_language.strip():
        raise ConfigError("translation.target_language cannot be empty")
    if translation.batch_cues < 1:
        raise ConfigError("translation.batch_cues must be at least 1")
    if translation.batch_chars < 1:
        raise ConfigError("translation.batch_chars must be at least 1")
    if translation.context_before_seconds < 0:
        raise ConfigError("translation.context_before_seconds cannot be negative")
    if translation.context_after_seconds < 0:
        raise ConfigError("translation.context_after_seconds cannot be negative")
    if translation.context_max_chars < 1:
        raise ConfigError("translation.context_max_chars must be at least 1")
    if not translation.local_model.strip():
        raise ConfigError("translation.local_model cannot be empty")
    if not translation.local_device.strip():
        raise ConfigError("translation.local_device cannot be empty")
    if translation.metadata_description_max_chars < 0:
        raise ConfigError(
            "translation.metadata_description_max_chars cannot be negative"
        )
    if not 1 <= translation.metadata_tag_count <= 10:
        raise ConfigError("translation.metadata_tag_count must be between 1 and 10")
    if translation.metadata_subtitle_max_chars < 0:
        raise ConfigError("translation.metadata_subtitle_max_chars cannot be negative")
    if not isinstance(translation.glossary_files, list) or not all(
        isinstance(path, str) and path.strip() for path in translation.glossary_files
    ):
        raise ConfigError(
            "translation.glossary_files must be a list of non-empty paths"
        )
    knowledge = config.fan_knowledge
    if not knowledge.database_path.strip():
        raise ConfigError("fan_knowledge.database_path cannot be empty")
    if not knowledge.collection_cache_dir.strip():
        raise ConfigError("fan_knowledge.collection_cache_dir cannot be empty")
    if knowledge.chunk_target_seconds <= 0:
        raise ConfigError("fan_knowledge.chunk_target_seconds must be positive")
    if knowledge.chunk_max_chars < 100:
        raise ConfigError("fan_knowledge.chunk_max_chars must be at least 100")
    if knowledge.current_video_chat_lookback_seconds < 0:
        raise ConfigError(
            "fan_knowledge.current_video_chat_lookback_seconds cannot be negative"
        )
    if knowledge.current_video_chat_lookahead_seconds < 0:
        raise ConfigError(
            "fan_knowledge.current_video_chat_lookahead_seconds cannot be negative"
        )
    if not knowledge.youtube_subtitle_languages or not all(
        isinstance(value, str) and value.strip()
        for value in knowledge.youtube_subtitle_languages
    ):
        raise ConfigError(
            "fan_knowledge.youtube_subtitle_languages must contain strings"
        )
    if knowledge.top_k_asr < 1 or knowledge.top_k_translation < 1:
        raise ConfigError("fan_knowledge top-k limits must be at least 1")
    if knowledge.translation_query_chars < 1:
        raise ConfigError("fan_knowledge.translation_query_chars must be at least 1")
    if knowledge.term_extraction_context_size < 4096:
        raise ConfigError(
            "fan_knowledge.term_extraction_context_size must be at least 4096"
        )
    if knowledge.term_extraction_target_input_tokens < 1024:
        raise ConfigError(
            "fan_knowledge.term_extraction_target_input_tokens must be at least 1024"
        )
    if knowledge.term_extraction_max_output_tokens < 256:
        raise ConfigError(
            "fan_knowledge.term_extraction_max_output_tokens must be at least 256"
        )
    if (
        knowledge.term_extraction_target_input_tokens
        + knowledge.term_extraction_max_output_tokens
        > knowledge.term_extraction_context_size
    ):
        raise ConfigError(
            "fan_knowledge term extraction input and output budgets exceed context"
        )
    if config.render.font_size_ratio <= 0:
        raise ConfigError("render.font_size_ratio must be positive")
    if config.render.portrait_font_size_ratio <= 0:
        raise ConfigError("render.portrait_font_size_ratio must be positive")
    if config.render.min_font_size < 1:
        raise ConfigError("render.min_font_size must be at least 1")
    if config.render.max_font_size < config.render.min_font_size:
        raise ConfigError("render.max_font_size must be at least render.min_font_size")
    if not 0 <= config.render.margin_horizontal_ratio < 0.5:
        raise ConfigError("render.margin_horizontal_ratio must be between 0 and 0.5")
    if not 0 <= config.render.portrait_margin_horizontal_ratio < 0.5:
        raise ConfigError(
            "render.portrait_margin_horizontal_ratio must be between 0 and 0.5"
        )
    if not 0 <= config.render.margin_vertical_ratio < 0.5:
        raise ConfigError("render.margin_vertical_ratio must be between 0 and 0.5")
    if config.render.outline_ratio < 0:
        raise ConfigError("render.outline_ratio cannot be negative")
    if config.render.backend not in {"auto", "cpu", "cuda"}:
        raise ConfigError("render.backend must be 'auto', 'cpu', or 'cuda'")
    if not 0 <= config.render.nvenc_cq <= 51:
        raise ConfigError("render.nvenc_cq must be between 0 and 51")
    if config.upload.copyright not in (1, 2):
        raise ConfigError("upload.copyright must be 1 (original) or 2 (repost)")
    if not 1 <= config.upload.max_tags <= 10:
        raise ConfigError("upload.max_tags must be between 1 and 10")
    if config.upload.description_max_chars < 1:
        raise ConfigError("upload.description_max_chars must be at least 1")
    if config.upload.cooldown_min_seconds < 0:
        raise ConfigError("upload.cooldown_min_seconds cannot be negative")
    if config.upload.cooldown_max_seconds < config.upload.cooldown_min_seconds:
        raise ConfigError(
            "upload.cooldown_max_seconds must be at least cooldown_min_seconds"
        )
    if not config.upload.rate_limit_retry_delays_seconds or any(
        delay <= 0 for delay in config.upload.rate_limit_retry_delays_seconds
    ):
        raise ConfigError(
            "upload.rate_limit_retry_delays_seconds must contain positive delays"
        )
    return config


def llm_api_key(config: LLMConfig) -> str:
    if config.local_server_enabled:
        return "local-llama-cpp"
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
                "failed to read LLM API key from pass entry: "
                f"{config.api_key_pass_entry}"
            ) from exc
        first_line = result.stdout.splitlines()[0].strip() if result.stdout else ""
        if not first_line:
            raise ConfigError(f"pass entry is empty: {config.api_key_pass_entry}")
        return first_line

    value = os.environ.get(config.api_key_env)
    if not value:
        raise ConfigError(
            f"environment variable {config.api_key_env} is required when no pass "
            "entry is configured"
        )
    return value
