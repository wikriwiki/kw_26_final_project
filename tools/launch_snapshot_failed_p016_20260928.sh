#!/usr/bin/env bash
set -Eeuo pipefail
root=/data/multipolicy_v53_20260928/p016
session=mv53_p016_failed_snapshot
test "$(cat "$root/on/launch.exitcode")" = 1
test ! -e "$root/failed_prepatch_snapshot.exitcode"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing session $session" >&2
  exit 1
fi
tmux new-session -d -s "$session" \
  "bash /data/pilot_repo_20260927/tools/snapshot_failed_p016_20260928.sh > '$root/failed_prepatch_snapshot.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$root/failed_prepatch_snapshot.exitcode'; sleep 43200"
echo "Started $session"
