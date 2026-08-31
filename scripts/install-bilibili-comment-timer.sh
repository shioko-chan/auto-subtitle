#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-$ROOT_DIR/config.toml}"
NIX_BIN="$(command -v nix)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
SERVICE_PATH="$UNIT_DIR/subtitle-comments.service"
TIMER_PATH="$UNIT_DIR/subtitle-comments.timer"

for value in "$ROOT_DIR" "$CONFIG_PATH" "$NIX_BIN"; do
    if [[ "$value" == *$'\n'* || "$value" == *' '* ]]; then
        printf 'Paths containing spaces or newlines are not supported: %s\n' "$value" >&2
        exit 1
    fi
done

if [[ ! -f "$CONFIG_PATH" ]]; then
    printf 'Config file not found: %s\n' "$CONFIG_PATH" >&2
    exit 1
fi

mkdir -p "$UNIT_DIR"
printf '%s\n' \
    '[Unit]' \
    'Description=Post pending Bilibili setlist comments' \
    'After=network-online.target' \
    '' \
    '[Service]' \
    'Type=oneshot' \
    "WorkingDirectory=$ROOT_DIR" \
    "ExecStart=$NIX_BIN develop $ROOT_DIR#default --command uv run --extra asr subtitle-pipeline --config $CONFIG_PATH retry-comments" \
    >"$SERVICE_PATH"

printf '%s\n' \
    '[Unit]' \
    'Description=Check pending Bilibili setlist comments' \
    '' \
    '[Timer]' \
    'OnCalendar=*:0/15' \
    'Persistent=true' \
    'RandomizedDelaySec=2min' \
    'Unit=subtitle-comments.service' \
    '' \
    '[Install]' \
    'WantedBy=timers.target' \
    >"$TIMER_PATH"

systemctl --user daemon-reload
systemctl --user enable --now subtitle-comments.timer
printf 'Installed and enabled %s\n' "$TIMER_PATH"
systemctl --user list-timers subtitle-comments.timer --no-pager
