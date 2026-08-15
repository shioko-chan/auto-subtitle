from __future__ import annotations

import argparse
import logging
import shutil
import sys
import warnings
from pathlib import Path

from .config import AppConfig, ConfigError, load_config
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
        override = True if args.upload else False if args.no_upload else None
        result = run_pipeline(args.url, config, upload_override=override)
        logging.info("complete: %s", result.rendered_video)
        logging.info("uploaded to Bilibili: %s", "yes" if result.uploaded else "no")
        return 0
    except (ConfigError, RuntimeError, ValueError) as exc:
        logging.error("%s", exc)
        return 1


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
            if analysis.diarization_backend == "moss":
                worker_name = "MOSS transcription worker"
                worker = (
                    Path(analysis.moss_transcribe_worker_project)
                    .resolve()
                    .joinpath("worker.py")
                )
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", module=r"pyannote\.audio\.core\.io"
                    )
                    import pyannote.audio  # noqa: F401
                worker_name = "MossFormer2 worker"
                worker = (
                    Path(analysis.overlap_separation_worker_project)
                    .resolve()
                    .joinpath("worker.py")
                )
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
