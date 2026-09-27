#!/usr/bin/env bash
# Run ONLY after the original pilot runner is stopped, OFF ledger is complete,
# and the OFF output has been copied outside the A100 with verified SHA256.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
OUT=/data/p013_v53_pilot_20260927
NEO=/data/neo4j-community-5.26.0
BACKUP=/data/backup_20260927_pre_pilot
POLICY=data/experiments/P013_v53_policy_20260926.json
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONIOENCODING=utf-8 PYTHONPATH="$REPO" LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 SIM_ENVIRONMENT=covid_2021
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
unset SIM_ALLOW_STAGE2_FALLBACK
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$OUT/restored_on.log"; }
trap 'log "FAILED line $LINENO; graph and outputs remain for audit"' ERR

test -s "$OUT/off.ledger.jsonl"
test -s "$OUT/off.ledger.jsonl.manifest.json"
(cd "$OUT" && sha256sum -c off.ledger.sha256)
python - "$OUT/off/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('completed_at') and len(d['summary'])==5
assert all(x['ok']==80 and x['err']==0 for x in d['summary'])
PY
test -f "$OUT/off_external_copy_verified.txt"
test -f "$OUT/off_graph_external_copy_verified.txt"
(cd "$BACKUP" && sha256sum -c SHA256SUMS)
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py' | grep -q .; then
  log 'A simulation or ledger exporter is still active; refusing restore'
  exit 1
fi

if [ ! -f "$OUT/on_graph_restored.marker" ]; then
  log 'Restore pre-pilot neo4j database to remove all OFF-arm graph mutations'
  stopped=0
  restart_if_needed() {
    rc=$?
    trap - EXIT
    if [ "$stopped" -eq 1 ]; then "$NEO/bin/neo4j" start || rc=1; fi
    exit "$rc"
  }
  trap restart_if_needed EXIT
  "$NEO/bin/neo4j" stop
  stopped=1
  if pgrep -af '[o]rg.neo4j.server.CommunityEntryPoint' | grep -q .; then
    log 'Neo4j still mounted; refusing load'
    exit 1
  fi
  "$NEO/bin/neo4j-admin" database load neo4j --from-path="$BACKUP" --overwrite-destination=true
  "$NEO/bin/neo4j" start
  stopped=0
  "$NEO/bin/neo4j" status
  log 'Database restored; reset run artifacts and seed identical Day0'
  python scripts/neo4j_load/97_reset_run_artifacts.py
  DAY_ZERO=2020-05-08 python scripts/neo4j_load/08_initial_state.py
  python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
  python scripts/sim/policy_preflight.py --require-db "$POLICY"
  printf 'pre-pilot dump restored; policy wired; %s\n' "$(date -Is)" > "$OUT/on_graph_restored.marker"
else
  log 'Reusing the already restored ON graph after an incomplete day'
  python scripts/sim/policy_preflight.py --require-db "$POLICY"
fi

export SIM_OUTPUT_DIR="$OUT/on" SIM_RUN_ID=p013-v53-pilot-20260927-on
log 'Start ON: same 80 citizens x 5 days'
for day in 2020-05-09 2020-05-10 2020-05-11 2020-05-12 2020-05-13; do
  if [ -s "$OUT/on.day_$day.json" ]; then
    log "$day already completed and recorded; skip"
    continue
  fi
  success=0
  for attempt in 1 2 3; do
    log "Run ON $day attempt $attempt (committed citizen-days reused on retry)"
    if python -u scripts/sim/run_simulation.py --start "$day" --days 1 --limit 80 \
         --workers 32 --environment covid_2021 > "$OUT/on.${day}_${attempt}.run.log" 2>&1; then
      python - "$OUT/on/summary.json" "$OUT/on.day_$day.json" "$day" <<'PY'
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1])); day=sys.argv[3]
assert d.get('completed_at') and len(d['summary'])==1
row=d['summary'][0]
assert row['day']==day and row['ok']==80 and row['err']==0
Path(sys.argv[2]).write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
PY
      success=1
      break
    fi
  done
  if [ "$success" -ne 1 ]; then log "Three failed ON attempts for $day"; exit 1; fi
done
python - "$OUT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
root=Path(sys.argv[1]); days=['2020-05-09','2020-05-10','2020-05-11','2020-05-12','2020-05-13']
rows=[json.loads((root/f'on.day_{day}.json').read_text()) for day in days]
assert [row['day'] for row in rows]==days
assert all(row['ok']==80 and row['err']==0 for row in rows)
path=root/'on/summary.json'; d=json.loads(path.read_text())
summary={'summary':rows,'args':{**d['args'],'start':days[0],'days':len(days)},
         'completed_at':datetime.now().isoformat(),
         'resume_audit':{'one_day_invocations':True,'committed_citizen_days_reused_on_retry':True}}
tmp=path.with_suffix('.json.tmp'); tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)
PY
python scripts/report/audit_stage2_generation.py --metrics-dir "$OUT/on/metrics" \
  --expected-per-day 80 --json-out "$OUT/on.stage2.json" --strict
python scripts/report/export_policy_daily_ledger.py --roster "$OUT/roster.json" \
  --start 2020-05-09 --end 2020-05-13 --arm on --policy-id P013 \
  --policy-file "$POLICY" --metrics-dir "$OUT/on/metrics" \
  --out "$OUT/on.ledger.jsonl"
sha256sum "$OUT/on.ledger.jsonl" "$OUT/on.ledger.jsonl.manifest.json" > "$OUT/on.ledger.sha256"
PILOT_REPO_ROOT="$REPO" python "$OUT/sector_export.py" export --arm on \
  --roster "$OUT/roster.json" --days 2020-05-11 2020-05-12 2020-05-13 \
  --out "$OUT/on.sector.json"
sha256sum "$OUT/on.sector.json" > "$OUT/on.sector.sha256"
python scripts/report/paired_grant_effect.py --on "$OUT/on.ledger.jsonl" \
  --off "$OUT/off.ledger.jsonl" --roster "$OUT/roster.json" \
  --start 2020-05-09 --end 2020-05-13 --effect-start 2020-05-11 \
  --effect-end 2020-05-13 --policy-id P013 --expected-recipients 80 \
  --expected-issued-won 22400000 --json-out "$OUT/paired_effect.json"
PILOT_REPO_ROOT="$REPO" python "$OUT/sector_export.py" pair \
  --on "$OUT/on.sector.json" --off "$OUT/off.sector.json" \
  --out "$OUT/paired_sector.json"
sha256sum "$OUT/paired_effect.json" >> "$OUT/frozen_inputs.sha256"
sha256sum "$OUT/paired_sector.json" >> "$OUT/frozen_inputs.sha256"
log 'PILOT_COMPLETE'
