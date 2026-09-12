#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${IN_NIX_SHELL:-}" ]]; then
    if ! command -v nix >/dev/null 2>&1; then
        printf 'Required command not found: nix\n' >&2
        exit 1
    fi
    printf 'Entering the subtitle-pipeline Nix development shell...\n'
    exec nix develop "$ROOT_DIR#default" --command bash "${BASH_SOURCE[0]}" "$@"
fi

CONFIG_PATH="${1:-$ROOT_DIR/config.toml}"
WORK_DIR="$ROOT_DIR/work"
STATUS_PATH="$WORK_DIR/yumemita-retry-10-14-status.log"
STOP_PATH="$WORK_DIR/yumemita.stop"
BILIBILI_PAUSE_PATH="$WORK_DIR/bilibili-upload-paused.json"
UPLOADED_PATH="$WORK_DIR/uploaded.txt"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/auto-subtitle-uv-cache}"

log_status() {
    printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" >>"$STATUS_PATH"
}

trap 'log_status "INTERRUPTED"; printf "\nBatch interrupted; rerun this script to resume.\n" >&2; exit 130' INT TERM

require_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$1" >&2
        exit 1
    fi
}

require_command uv
require_command biliup
require_command jq

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Config file not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi

if [[ ! -f "$ROOT_DIR/cookies.json" ]]; then
    printf 'Bilibili cookie file not found: %s\n' "$ROOT_DIR/cookies.json" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"
touch "$STATUS_PATH"
touch "$UPLOADED_PATH"

cd "$ROOT_DIR"

printf 'Checking pipeline dependencies...\n'
uv run --extra asr subtitle-pipeline --config "$CONFIG_PATH" check

printf 'Checking Bilibili login...\n'
if ! biliup --user-cookie cookies.json list >/dev/null; then
    printf 'Bilibili login check failed; run: biliup login\n' >&2
    exit 1
fi

# Successful history plus public archived streams published from 2026-08-08 through 2026-08-22.
# Members-only streams are deliberately excluded.
RECORDS=(
    "2026-09-03|arale|9yh-GFAnsS8"
    "2026-09-03|ritsu|0o96ZlU8lKY"
    "2026-09-02|yuno|m-IWA2Sf5kE"
    "2026-09-02|ritsu|G5KZVkFNJmE"
    "2026-09-01|miyako|rLfpBK5PUcU"
)

total="${#RECORDS[@]}"
succeeded=0
skipped=0
failed=0

for index in "${!RECORDS[@]}"; do
    IFS='|' read -r published channel video_id <<<"${RECORDS[$index]}"
    url="https://www.youtube.com/watch?v=$video_id"
    manifest="$WORK_DIR/$video_id/manifest.json"
    position=$((index + 1))

    if [[ -f "$STOP_PATH" ]]; then
        log_status "STOP before=$position/$total"
        printf 'Stop marker found; batch paused before [%d/%d].\n' "$position" "$total"
        printf 'Remove %s before resuming.\n' "$STOP_PATH"
        exit 0
    fi
    if [[ -f "$BILIBILI_PAUSE_PATH" ]]; then
        log_status "STOP position=$position/$total reason=bilibili-risk-control"
        printf 'Bilibili upload pause marker found: %s\n' "$BILIBILI_PAUSE_PATH" >&2
        exit 1
    fi

    if grep -Fxq "$video_id" "$UPLOADED_PATH" || \
        { [[ -f "$manifest" ]] && jq -e '.uploaded == true' "$manifest" >/dev/null; }; then
        printf '[%d/%d] SKIP %s %s %s (already uploaded)\n' \
            "$position" "$total" "$published" "$channel" "$video_id"
        log_status "SKIP position=$position/$total published=$published channel=$channel video=$video_id reason=already-uploaded"
        skipped=$((skipped + 1))
        continue
    fi

    printf '\n[%d/%d] RUN  %s %s %s\n' \
        "$position" "$total" "$published" "$channel" "$video_id"
    log_status "RUN position=$position/$total published=$published channel=$channel video=$video_id"

    if uv run --extra asr subtitle-pipeline --config "$CONFIG_PATH" run --upload "$url"; then
        if ! grep -Fxq "$video_id" "$UPLOADED_PATH"; then
            printf '%s\n' "$video_id" >>"$UPLOADED_PATH"
        fi
        log_status "OK position=$position/$total video=$video_id"
        succeeded=$((succeeded + 1))
    else
        status=$?
        log_status "FAIL position=$position/$total video=$video_id exit=$status"
        printf '[%d/%d] FAILED %s (exit %d); continuing.\n' \
            "$position" "$total" "$video_id" "$status" >&2
        failed=$((failed + 1))
        if [[ -f "$BILIBILI_PAUSE_PATH" ]]; then
            log_status "STOP after=$position/$total reason=bilibili-risk-control"
            printf 'Bilibili risk control paused the batch; inspect %s\n' \
                "$BILIBILI_PAUSE_PATH" >&2
            exit 1
        fi
    fi
done

printf '\nBatch complete: %d succeeded, %d skipped, %d failed.\n' \
    "$succeeded" "$skipped" "$failed"
log_status "COMPLETE succeeded=$succeeded skipped=$skipped failed=$failed"
printf 'Status log: %s\n' "$STATUS_PATH"

if ((failed > 0)); then
    exit 1
fi
