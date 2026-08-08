from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .config import AppConfig, ConfigError, load_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="subtitle-pipeline",
        description="Download YouTube, translate/generate subtitles, upload to Bilibili.",
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
    if config.whisper.enabled:
        try:
            import faster_whisper  # noqa: F401

            logging.info("found optional faster-whisper fallback")
        except ImportError:
            logging.warning(
                "Whisper fallback is enabled but faster-whisper is not installed"
            )
    else:
        logging.info("Whisper fallback is disabled by configuration")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
