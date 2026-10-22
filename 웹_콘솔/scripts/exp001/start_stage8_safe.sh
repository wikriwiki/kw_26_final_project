#!/usr/bin/env bash
# Start Stage 8 outside the SSH session so a client disconnect does not stop it.
# This script only launches /data/stage8.sh; it does not alter experiment inputs.
set -euo pipefail
C="${1:-v5}"
PIDFILE=/data/stage8/stage8.pid
mkdir -p /data/stage8 /data/logs
if [ -s "$PIDFILE" ]; then
  OLD_PID=$(cat "$PIDFILE")
  if kill -0 "$OLD_PID" 2>/dev/null \
      && tr '\0' ' ' <"/proc/$OLD_PID/cmdline" | grep -q '/data/stage8.sh'; then
    echo "stage8 already running: pid=$OLD_PID"
    exit 1
  fi
fi
STAMP=$(date +%Y%m%dT%H%M%S)
LOG="/data/logs/stage8_${C}_${STAMP}.log"
nohup setsid /data/stage8.sh "$C" >"$LOG" 2>&1 </dev/null &
PID=$!
printf '%s\n' "$PID" > "$PIDFILE"
printf 'started pid=%s log=%s\n' "$PID" "$LOG"
