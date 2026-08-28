from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from dataclasses import replace
from pathlib import Path

from .chat_context import remove_youtube_chat_files
from .config import AppConfig, ConfigError, load_config
from .fan_knowledge import FanKnowledgeRetriever
from .knowledge_collection import collect_official_documents, collect_sns_documents
from .knowledge_ingestion import (
    IngestionSummary,
    download_youtube_subtitles,
    ingest_document_mapping,
    ingest_jsonl,
    ingest_work_directory,
    ingest_youtube_cache,
)
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-pipeline",
        description=(
            "Download YouTube, translate/generate subtitles, upload to Bilibili."
        ),
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--verbose", action="store_true")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run the complete pipeline")
    run_parser.add_argument("url", help="a single YouTube video URL")
    upload_group = run_parser.add_mutually_exclusive_group()
    upload_group.add_argument(
        "--upload", action="store_true", help="upload even if upload.enabled is false"
    )
    upload_group.add_argument(
        "--no-upload", action="store_true", help="render locally but never upload"
    )

    subparsers.add_parser("check", help="check local executables and configuration")

    knowledge_parser = subparsers.add_parser(
        "knowledge", help="collect and inspect the local fan knowledge base"
    )
    knowledge_commands = knowledge_parser.add_subparsers(
        dest="knowledge_command", required=True
    )
    work_parser = knowledge_commands.add_parser(
        "ingest-work", help="import existing pipeline ASR and metadata"
    )
    work_parser.add_argument("--path", type=Path)

    youtube_parser = knowledge_commands.add_parser(
        "ingest-youtube", help="download and import YouTube subtitles"
    )
    youtube_parser.add_argument("urls", nargs="+")
    youtube_parser.add_argument("--cache-dir", type=Path)
    youtube_parser.add_argument("--browser")
    youtube_parser.add_argument("--playlist-end", type=int, default=100)
    youtube_parser.add_argument("--languages", nargs="+")

    jsonl_parser = knowledge_commands.add_parser(
        "ingest-jsonl", help="import normalized SNS or website documents"
    )
    jsonl_parser.add_argument("paths", nargs="+", type=Path)
    official_parser = knowledge_commands.add_parser(
        "ingest-official", help="collect official webpages or RSS/Atom feeds"
    )
    official_parser.add_argument("urls", nargs="+")
    official_parser.add_argument("--source-type", default="official_news")
    official_parser.add_argument("--follow-links", action="store_true")
    official_parser.add_argument("--link-pattern")
    official_parser.add_argument("--maximum-documents", type=int, default=1000)
    official_parser.add_argument("--required-text", nargs="+")
    official_parser.add_argument("--maximum-depth", type=int)
    sns_parser = knowledge_commands.add_parser(
        "ingest-sns", help="collect X or Instagram metadata through gallery-dl"
    )
    sns_parser.add_argument("urls", nargs="+")
    sns_parser.add_argument("--browser")
    knowledge_commands.add_parser("stats", help="show knowledge database counts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = load_config(args.config)
        if args.command == "check":
            return _check(config)
        if args.command == "knowledge":
            return _knowledge(config, args)
        override = True if args.upload else False if args.no_upload else None
        result = run_pipeline(args.url, config, upload_override=override)
        logging.info("complete: %s", result.rendered_video)
        logging.info("uploaded to Bilibili: %s", "yes" if result.uploaded else "no")
        return 0
    except (ConfigError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1


def _knowledge(config: AppConfig, args: argparse.Namespace) -> int:
    knowledge = config.fan_knowledge
    with FanKnowledgeRetriever(
        Path(knowledge.database_path).expanduser(),
        embedding_model=knowledge.embedding_model,
        vector_index_path=Path(knowledge.vector_index_path).expanduser(),
        vector_minimum_score=knowledge.vector_minimum_score,
    ) as retriever:
        command = args.knowledge_command
        if command == "stats":
            logging.info(
                "knowledge database: documents=%d chunks=%d",
                retriever.document_count(),
                retriever.chunk_count(),
            )
            return 0
        if command == "ingest-work":
            summary = ingest_work_directory(
                retriever,
                args.path or config.work_dir,
                target_seconds=knowledge.chunk_target_seconds,
                maximum_chars=knowledge.chunk_max_chars,
                include_regular_chat=knowledge.include_regular_chat,
            )
        elif command == "ingest-jsonl":
            summary = IngestionSummary()
            for path in args.paths:
                summary = summary.merge(
                    ingest_jsonl(
                        retriever,
                        path,
                        maximum_chars=knowledge.chunk_max_chars,
                    )
                )
        elif command == "ingest-youtube":
            if args.playlist_end < 1:
                raise ValueError("--playlist-end must be at least 1")
            cache_dir = args.cache_dir or Path(knowledge.collection_cache_dir)
            download = (
                replace(config.download, cookies_from_browser=args.browser)
                if args.browser
                else config.download
            )
            languages = tuple(args.languages or knowledge.youtube_subtitle_languages)
            summary = IngestionSummary()

            def ingest_downloaded_batch(video_ids: tuple[str, ...]) -> None:
                nonlocal summary
                summary = summary.merge(
                    ingest_youtube_cache(
                        retriever,
                        cache_dir,
                        target_seconds=knowledge.chunk_target_seconds,
                        maximum_chars=knowledge.chunk_max_chars,
                        language_priority=tuple(
                            value.removesuffix(".*") for value in languages
                        ),
                        include_regular_chat=knowledge.include_regular_chat,
                        video_ids=set(video_ids),
                    )
                )
                for video_id in video_ids:
                    remove_youtube_chat_files(cache_dir / video_id)

            download_youtube_subtitles(
                args.urls,
                cache_dir,
                download,
                languages=languages,
                playlist_end=args.playlist_end,
                after_batch=ingest_downloaded_batch,
            )
        elif command in {"ingest-official", "ingest-sns"}:
            values = (
                collect_official_documents(
                    args.urls,
                    source_type=args.source_type,
                    follow_links=args.follow_links,
                    link_pattern=args.link_pattern,
                    maximum_documents=args.maximum_documents,
                    required_terms=tuple(args.required_text or ()),
                    maximum_depth=args.maximum_depth,
                )
                if command == "ingest-official"
                else collect_sns_documents(
                    args.urls,
                    cookies_from_browser=args.browser
                    or config.download.cookies_from_browser,
                )
            )
            summary = IngestionSummary()
            for index, value in enumerate(values):
                summary = summary.add(
                    ingest_document_mapping(
                        retriever,
                        value,
                        fallback_external_id=f"{command}:{index}",
                        maximum_chars=knowledge.chunk_max_chars,
                    )
                )
        else:
            raise ValueError(f"unknown knowledge command: {command}")
        _log_ingestion_summary(summary)
        added_vectors, removed_vectors = retriever.sync_vector_index()
        logging.info(
            "knowledge vector index: added=%d removed=%d",
            added_vectors,
            removed_vectors,
        )
    return 0


def _log_ingestion_summary(summary: IngestionSummary) -> None:
    logging.info(
        "knowledge ingestion: scanned=%d changed=%d unchanged=%d chunks=%d skipped=%d",
        summary.scanned,
        summary.inserted_or_updated,
        summary.unchanged,
        summary.chunks,
        summary.skipped,
    )


def _check(config: AppConfig) -> int:
    missing = []
    for executable in ("yt-dlp", "ffmpeg"):
        path = shutil.which(executable)
        if path:
            logging.info("found %s: %s", executable, path)
        else:
            missing.append(executable)
            logging.error("missing executable: %s", executable)
    if config.upload.enabled:
        path = shutil.which("biliup")
        if path:
            logging.info("found biliup: %s", path)
        else:
            missing.append("biliup")
            logging.error("missing executable: biliup")
    try:
        import qwen_asr  # noqa: F401
        import torch

        logging.info("found Qwen3-ASR runtime")
        if config.asr.device.startswith("cuda"):
            if torch.cuda.is_available():
                logging.info("found CUDA device: %s", torch.cuda.get_device_name(0))
            else:
                missing.append("CUDA")
                logging.error(
                    "Qwen3-ASR is configured for CUDA, but CUDA is unavailable"
                )
        if config.audio_analysis.enabled:
            import demucs  # noqa: F401
            import transformers  # noqa: F401

            analysis = config.audio_analysis
            workers: list[tuple[str, Path]] = []
            if config.song_identification.enabled:
                workers.extend(
                    [
                        (
                            "song search worker",
                            Path(config.song_identification.search_worker_project)
                            .resolve()
                            .joinpath("worker.py"),
                        ),
                        (
                            "pySHIRO worker",
                            Path(config.song_identification.pyshiro_worker_project)
                            .resolve()
                            .joinpath("worker.py"),
                        ),
                    ]
                )
            if analysis.diarization_backend == "moss":
                workers.append(
                    (
                        "MOSS transcription worker",
                        Path(analysis.moss_transcribe_worker_project)
                        .resolve()
                        .joinpath("worker.py"),
                    )
                )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", module=r"pyannote\.audio\.core\.io"
                    )
                    import pyannote.audio  # noqa: F401
                if analysis.conditioned_asr_backend == "dicow":
                    workers.append(
                        (
                            "DiCoW worker",
                            Path(analysis.conditioned_asr_worker_project)
                            .resolve()
                            .joinpath("worker.py"),
                        )
                    )
            for worker_name, worker in workers:
                if shutil.which("uv") is None or not worker.is_file():
                    missing.append(worker_name)
                    logging.error("missing %s: %s", worker_name, worker)
                else:
                    logging.info("found %s: %s", worker_name, worker)
            logging.info("found diarization and singing runtimes")
    except ImportError:
        missing.append("qwen-asr")
        logging.error("missing Qwen3-ASR runtime; run `uv sync --extra asr`")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
