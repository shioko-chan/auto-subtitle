#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/config.toml}"
WORK_DIR="$ROOT_DIR/work"
STATUS_PATH="$WORK_DIR/yumemita-2026-08-10-status.log"
STOP_PATH="$WORK_DIR/yumemita-2026-08-10.stop"
UPLOADED_PATH="$WORK_DIR/yumemita-2026-08-10-uploaded.txt"

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
require_command pass
require_command biliup
require_command jq
require_command sha256sum

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

printf 'Unlocking the DeepSeek API key...\n'
if ! pass show api/deepseek >/dev/null; then
    printf 'Could not read pass entry: api/deepseek\n' >&2
    exit 1
fi

cd "$ROOT_DIR"

printf 'Checking pipeline dependencies...\n'
uv run --extra asr subtitle-pipeline --config "$CONFIG_PATH" check

printf 'Checking Bilibili login...\n'
if ! biliup --user-cookie cookies.json list >/dev/null; then
    printf 'Bilibili login check failed; run: biliup login\n' >&2
    exit 1
fi

# Public archived streams published from 2026-07-27 through 2026-08-10.
# Members-only streams are deliberately excluded.
RECORDS=(
    "2026-07-27|nonoka|iDX9seQqdbs"
    "2026-07-27|yuno|8r2aqmmBZEY"
    "2026-07-28|arale|Xz52b5YZazw"
    "2026-07-28|nonoka|vJPs8vO2b5c"
    "2026-07-28|ritsu|EBlXcBc_fc0"
    "2026-07-29|arale|Nc7hT2vYsLQ"
    "2026-07-29|nonoka|r4ugYsVssAE"
    "2026-07-29|ritsu|cu1srt3zd0Y"
    "2026-07-29|yuno|vy4qBWKlGfM"
    "2026-07-30|group|Al7aJdw-oic"
    "2026-07-30|miyako|TxbV_9g7YZY"
    "2026-07-30|yuno|tiyrmBJWVNY"
    "2026-07-31|ritsu|y-xzyetM-FI"
    "2026-08-01|ritsu|rpogVTapfWY"
    "2026-08-01|miyako|MfUA2snR1nA"
    "2026-08-02|ritsu|_aQBzRWfsU0"
    "2026-08-02|miyako|OHpuxUkdNUQ"
    "2026-08-02|miyako|tN9mEXyjKHY"
    "2026-08-03|ritsu|Sfq4BMb5ync"
    "2026-08-04|arale|QkDtDVGbY9s"
    "2026-08-04|ritsu|RoTo0TVCbKA"
    "2026-08-04|yuno|7dUt5p0ueP4"
    "2026-08-04|yuno|7M_ePReieMs"
    "2026-08-05|arale|89V47TNVrO4"
    "2026-08-05|arale|UCV-0SPr53w"
    "2026-08-05|arale|kXUrjnbmc4Q"
    "2026-08-05|ritsu|3oBi7dvEAyU"
    "2026-08-06|yuno|Ez9KFh8x9QI"
    "2026-08-07|group|YuT4MawXQxk"
    "2026-08-07|nonoka|brSArzAwakE"
    "2026-08-08|arale|8Qtx3gFWmtI"
    "2026-08-08|nonoka|3EYGzu9v7Fc"
    "2026-08-08|ritsu|tzzU0bmVLXA"
    "2026-08-08|miyako|htt3MrWSjNQ"
    "2026-08-09|arale|v_3fGOhByw4"
    "2026-08-09|nonoka|2nf1sFpZCYs"
    "2026-08-09|ritsu|Ba62K5k56zA"
    "2026-08-09|miyako|AwTrSRxs9jo"
    "2026-08-09|yuno|jPCGHnY_jOQ"
)

total="${#RECORDS[@]}"
succeeded=0
skipped=0
failed=0

for index in "${!RECORDS[@]}"; do
    IFS='|' read -r published channel video_id <<<"${RECORDS[$index]}"
    url="https://www.youtube.com/watch?v=$video_id"
    hash="$(printf '%s' "$url" | sha256sum)"
    job_id="${hash%% *}"
    job_id="${job_id:0:12}"
    manifest="$WORK_DIR/$job_id/manifest.json"
    position=$((index + 1))

    if [[ -f "$STOP_PATH" ]]; then
        log_status "STOP before=$position/$total"
        printf 'Stop marker found; batch paused before [%d/%d].\n' "$position" "$total"
        printf 'Remove %s before resuming.\n' "$STOP_PATH"
        exit 0
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
    fi
done

printf '\nBatch complete: %d succeeded, %d skipped, %d failed.\n' \
    "$succeeded" "$skipped" "$failed"
log_status "COMPLETE succeeded=$succeeded skipped=$skipped failed=$failed"
printf 'Status log: %s\n' "$STATUS_PATH"

if ((failed > 0)); then
    exit 1
fi
