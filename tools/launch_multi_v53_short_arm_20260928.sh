#!/usr/bin/env bash
set -Eeuo pipefail
case_id=${1:?case id}
arm=${2:?arm}
[[ $case_id =~ ^(p010|distancing|p016|p014|p012)$ && $arm =~ ^(on|off)$ ]] || exit 2
session="mv53_${case_id}_${arm}"
out="/data/multipolicy_v53_20260928/$case_id/$arm"
if tmux has-session -t "$session" 2>/dev/null; then
  echo "Existing session $session; refusing duplicate" >&2; exit 1
fi
mkdir -p "$out"
test ! -e "$out/launch.exitcode" || { echo 'Existing exit code; refusing rerun' >&2; exit 1; }
tmux new-session -d -s "$session" \
  "bash /data/pilot_repo_20260927/tools/run_multi_v53_short_arm_20260928.sh '$case_id' '$arm' > '$out/launch.log' 2>&1; rc=\$?; printf '%s\\n' \"\$rc\" > '$out/launch.exitcode'; sleep 43200"
echo "Started session $session; output $out"
