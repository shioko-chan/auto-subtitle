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
STATUS_PATH="$WORK_DIR/yumemita-2026-08-10-status.log"
STOP_PATH="$WORK_DIR/yumemita-2026-08-10.stop"
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
    "2026-08-31|yuno|d-ICiEWvlSQ"
    "2026-08-31|miyako|R3tmiB-31ak"
    "2026-08-31|arale|R8E2TQa0hQY"
    "2026-08-30|miyako|VJV4VdE395s"
    "2026-08-30|ritsu|zjSgepNhZQY"
    "2026-08-30|ritsu|Igx--BDE0n8"
    "2026-08-30|nonoka|blxgN0vjLBI"
    "2026-08-30|miyako|--7cN8iGGB4"
    "2026-08-30|arale|MGVRS_MYXSw"
    "2026-08-29|yuno|0qxn9UjCyYU"
    "2026-08-29|nonoka|LmI79HkAsNc"
    "2026-08-29|ritsu|PYjJ4_lCNg8"
    "2026-08-29|arale|-j4eXICUPhc"
    "2026-08-29|nonoka|5APd1wj9QlA"
    "2026-08-28|group|gsNby5aGDKc"
    "2026-08-27|miyako|NHACpW6SpMc"
    "2026-08-27|arale|W-KQV-pCUlo"
    "2026-08-27|ritsu|LetHoS1RecQ"
    "2026-08-27|yuno|PpiAS9wzRcE"
    "2026-08-27|nonoka|c9zFMwjV7k0"
    "2026-08-26|arale|ws0Ha2YJauc"
    "2026-08-26|nonoka|UFx2RyP1-bw"
    "2026-08-26|yuno|i1-CRdYcsGA"
    "2026-08-25|ritsu|Nn2Ho9tu11Y"
    "2026-08-25|yuno|oOCFu4bVw0w"
    "2026-08-25|nonoka|-NkdhlHMIcc"
    "2026-08-25|ritsu|-fTIi3AbZSk"
    "2026-08-24|miyako|GODHmmJtNJg"
    "2026-08-24|nonoka|EM0dUwFmyXc"
    "2026-08-24|miyako|eUXnHg_yRGQ"
    "2026-08-23|ritsu|6kx8QLFtN_s"
    "2026-08-23|miyako|WeLp4J9BEM4"
    "2026-08-23|yuno|knUjBxky7pw"
    "2026-08-23|arale|3_LmU0XIwXs"
    "2026-08-22|ritsu|z7thQ5a4bII"
    "2026-08-22|group|lo8ogUEvtaA"
    "2026-08-21|group|8cch4m8t8ME"
    "2026-08-21|group|VRYwuzsIoV8"
    "2026-08-20|nonoka|p-R2Mluzdtc"
    "2026-08-20|miyako|7e7uClmC-Ps"
    "2026-08-20|arale|HhZIXb2YEfk"
    "2026-08-20|ritsu|YO2noyg1Hps"
    "2026-08-20|arale|EmGZ08TBFOg"
    "2026-08-19|yuno|d0uqb1ntAng"
    "2026-08-19|ritsu|fuiTW78dby0"
    "2026-08-19|yuno|BK8p8J28G6w"
    "2026-08-18|yuno|pfwAKQQXwu4"
    "2026-08-18|yuno|4yH9F6coc_o"
    "2026-08-12|ritsu|0GFgZ1DU5n0"
    "2026-08-12|ritsu|_bh0IurOZTg"
    "2026-08-12|yuno|FdGQCelgYkQ"
    "2026-08-11|yuno|9LdQLQgD_Lg"
    "2026-08-10|ritsu|Y7e056lLCoc"
    "2026-08-10|miyako|_oZxIYFOcZE"
    "2026-08-09|miyako|AwTrSRxs9jo"
    "2026-08-09|yuno|jPCGHnY_jOQ"
    "2026-08-08|ritsu|tzzU0bmVLXA"
    "2026-08-08|miyako|htt3MrWSjNQ"
    "2026-08-02|ritsu|_aQBzRWfsU0"
    "2026-08-01|ritsu|rpogVTapfWY"
    "2026-07-31|ritsu|y-xzyetM-FI"
    "2026-07-30|miyako|TxbV_9g7YZY"
    "2026-07-30|yuno|tiyrmBJWVNY"
    "2026-07-30|group|Al7aJdw-oic"
    "2026-07-29|arale|Nc7hT2vYsLQ"
    "2026-07-29|nonoka|r4ugYsVssAE"
    "2026-07-29|ritsu|cu1srt3zd0Y"
    "2026-07-29|yuno|vy4qBWKlGfM"
    "2026-07-28|arale|Xz52b5YZazw"
    "2026-07-28|nonoka|vJPs8vO2b5c"
    "2026-07-28|ritsu|EBlXcBc_fc0"
    "2026-07-27|nonoka|iDX9seQqdbs"
    "2026-07-27|yuno|8r2aqmmBZEY"
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
