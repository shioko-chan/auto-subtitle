#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/config.toml}"

VIDEO_IDS=(
    "R8E2TQa0hQY"
    "VJV4VdE395s"
    "0qxn9UjCyYU"
)

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Config file not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi

cd "$ROOT_DIR"

for video_id in "${VIDEO_IDS[@]}"; do
    printf '\nRunning %s without upload\n' "$video_id"
    uv run --extra asr subtitle-pipeline \
        --config "$CONFIG_PATH" \
        run --no-upload "https://www.youtube.com/watch?v=$video_id"
done
