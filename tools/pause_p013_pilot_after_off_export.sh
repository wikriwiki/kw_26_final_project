#!/usr/bin/env bash
# Operational guard for the active 2026-09-27 pilot. Stop only the bash runner
# while its OFF ledger exporter runs, so OFF is complete before graph restore.
set -euo pipefail
runner_pid="$1"
out=/data/p013_v53_pilot_20260927
while kill -0 "$runner_pid" 2>/dev/null; do
  if ps -ww -o args= --ppid "$runner_pid" | grep -q '^python scripts/report/export_policy_daily_ledger.py .*--arm off '; then
    kill -STOP "$runner_pid"
    printf '%s paused runner %s during OFF ledger export\n' "$(date '+%F %T %Z')" "$runner_pid" >> "$out/operational_pause.log"
    exit 0
  fi
  sleep 0.2
done
printf '%s runner ended before OFF export\n' "$(date '+%F %T %Z')" >> "$out/operational_pause.log"
exit 1
