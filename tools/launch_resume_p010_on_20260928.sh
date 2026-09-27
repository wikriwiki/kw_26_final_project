#!/usr/bin/env bash
set -Eeuo pipefail
out=/data/multipolicy_v53_20260928/p010/on
session=mv53_p010_on_resume
test "$(cat "$out/launch.exitcode")" != 0
test ! -e "$out/resume.exitcode"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing session $session" >&2; exit 1
fi
tmux new-session -d -s "$session" \
  "bash /data/pilot_repo_20260927/tools/resume_p010_on_20260928.sh > '$out/resume.launch.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$out/resume.exitcode'; sleep 43200"
echo "Started $session"
