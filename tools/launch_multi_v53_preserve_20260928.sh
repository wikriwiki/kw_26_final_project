#!/usr/bin/env bash
set -Eeuo pipefail
case_id=${1:?case id}
arm=${2:?arm}
[[ $case_id =~ ^(p010|distancing|p016|p014|p012)$ && $arm =~ ^(on|off)$ ]] || exit 2
out="/data/multipolicy_v53_20260928/$case_id/$arm"
session="mv53_preserve_${case_id}_${arm}"
if [[ $(cat "$out/launch.exitcode") != 0 ]]; then
  test "$(cat "$out/resume.exitcode")" = 0
fi
test ! -e "$out/preserve.exitcode"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing preserve session $session" >&2; exit 1
fi
tmux new-session -d -s "$session" \
  "bash /data/pilot_repo_20260927/tools/preserve_multi_v53_short_arm_20260928.sh '$case_id' '$arm' > '$out/preserve.launch.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$out/preserve.exitcode'; sleep 43200"
echo "Started preserve session $session"
