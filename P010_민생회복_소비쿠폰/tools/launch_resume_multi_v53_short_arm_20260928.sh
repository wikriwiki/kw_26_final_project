#!/usr/bin/env bash
set -Eeuo pipefail
case_id=${1:?case id}
arm=${2:?arm}
attempt_id=${3:-1}
[[ $case_id =~ ^(p010|p012|distancing|p016|p014)$ && $arm =~ ^(on|off)$ && $attempt_id =~ ^[1-9][0-9]*$ ]]
session="mv53_resume_${case_id}_${arm}_${attempt_id}"
out="/data/multipolicy_v53_20260928/$case_id/$arm"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing session $session; refusing duplicate" >&2; exit 1
fi
test -s "$out/launch.exitcode"
test ! -e "$out/resume_${attempt_id}.exitcode"
tmux new-session -d -s "$session" \
  "bash /data/pilot_repo_20260927/tools/resume_multi_v53_short_arm_20260928.sh '$case_id' '$arm' '$attempt_id' > '$out/resume_${attempt_id}.launch.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$out/resume_${attempt_id}.exitcode'; sleep 43200"
echo "Started continuation $session; graph retained"
