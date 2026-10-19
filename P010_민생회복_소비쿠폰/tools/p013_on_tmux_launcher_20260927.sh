#!/usr/bin/env bash
# Keep the Neo4j child process's tmux session alive after the paired pilot exits.
set -u
OUT=/data/p013_v53_pilot_20260927
bash "$OUT/on_from_restore.sh" > "$OUT/on.runner.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/on.runner.exitcode"
sleep 7200
