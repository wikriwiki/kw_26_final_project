#!/usr/bin/env bash
# Continue a failed arm on its *existing* graph. Never restore/reset the DB.
set -Eeuo pipefail
umask 077

case_id=${1:?case id}
arm=${2:?on|off}
attempt_id=${3:-1}
[[ $case_id =~ ^(p010|p012|distancing|p016|p014)$ ]]
[[ $arm =~ ^(on|off)$ && $attempt_id =~ ^[1-9][0-9]*$ ]]

REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928
OUT=$BASE/$case_id
ARM=$OUT/$arm
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8 LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
export SIM_OUTPUT_DIR="$ARM" SIM_RUN_ID="multipolicy-v53-20260928-$case_id-$arm"
unset SIM_ALLOW_STAGE2_FALLBACK

readarray -t cfg < <(python - "$ARM/run_manifest.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
for key in ('case','arm','start','days','citizens','environment','policy_file','request_mode'):
    print(d[key])
PY
)
[[ ${cfg[0]} == "$case_id" && ${cfg[1]} == "$arm" ]]
START=${cfg[2]}; DAYS=${cfg[3]}; N=${cfg[4]}; ENV_ID=${cfg[5]}; POLICY=${cfg[6]}
export SIM_ENVIRONMENT="$ENV_ID"
if [[ ${cfg[7]} == qwen8b && $case_id == p010 ]]; then
  unset LLM_MODE
else
  [[ ${cfg[7]} == exaone_4_5 ]]
  export LLM_MODE=exaone_4_5
fi

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ARM/resume_${attempt_id}.log"; }
trap 'log "FAILED at line $LINENO; graph/output remain intact"' ERR
test -s "$ARM/graph_restored.marker"
test -s "$ARM/served_model_evidence.json"
test "$(cat "$ARM/launch.exitcode")" != 0
test ! -e "$ARM/resume_${attempt_id}.exitcode"
test ! -e "$ARM/graph_backup/SHA256SUMS"
sha256sum -c "$ARM/frozen_inputs.sha256"
test "$(curl -fsS -m 8 localhost:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py|[e]xport_multi_policy_sector_ledger.py' | grep -q .; then
  log 'Active simulator/exporter; refusing simultaneous resume'; exit 1
fi
if [[ $arm == on && -n $POLICY ]]; then
  python scripts/sim/policy_preflight.py --require-db "$POLICY"
else
  python scripts/sim/policy_preflight.py --expect-no-policy
fi

for offset in $(seq 0 $((DAYS - 1))); do
  day=$(date -d "$START + $offset days" +%F)
  daily="$ARM/day_${day}.json"
  if [[ -s $daily ]]; then
    python - "$daily" "$day" "$N" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]))
assert r['day']==sys.argv[2] and r['ok']==int(sys.argv[3]) and r['err']==0
PY
    log "Reuse complete day $day"
    continue
  fi
  success=0
  for trial in 1 2 3 4 5; do
    args=(--start "$day" --days 1 --limit "$N" --workers 32)
    if [[ -n $ENV_ID ]]; then args+=(--environment "$ENV_ID"); fi
    log "Resume day $day trial $trial"
    if python -u scripts/sim/run_simulation.py "${args[@]}" \
         > "$ARM/resume_${attempt_id}_${day}_trial${trial}.run.log" 2>&1 && \
       python - "$ARM/summary.json" "$daily" "$day" "$N" <<'PY'
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1]))
assert d.get('completed_at') and len(d['summary'])==1
r=d['summary'][0]
assert r['day']==sys.argv[3] and r['ok']==int(sys.argv[4]) and r['err']==0
Path(sys.argv[2]).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
PY
    then
      success=1; break
    fi
  done
  [[ $success == 1 ]] || { log "Five failed continuation trials for $day"; exit 1; }
done

python - "$ARM" "$START" "$DAYS" "$N" "$attempt_id" <<'PY'
import json,sys
from datetime import date,datetime,timedelta
from pathlib import Path
p=Path(sys.argv[1]); start=date.fromisoformat(sys.argv[2]); count=int(sys.argv[3]); n=int(sys.argv[4])
days=[(start+timedelta(days=i)).isoformat() for i in range(count)]
rows=[json.loads((p/f'day_{day}.json').read_text()) for day in days]
assert [r['day'] for r in rows]==days
assert all(r['ok']==n and r['err']==0 for r in rows)
d=json.loads((p/'summary.json').read_text())
summary={'summary':rows,'args':{**d['args'],'start':days[0],'days':count},
         'completed_at':datetime.now().isoformat(),
         'resume_audit':{'original_launch_exitcode':(p/'launch.exitcode').read_text().strip(),
                         'resume_attempt':int(sys.argv[5]),
                         'committed_citizen_days_reused_on_retry':True,
                         'raw_attempts_dir':'metrics/attempts'}}
tmp=p/'summary.json.tmp'; tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
tmp.replace(p/'summary.json')
PY
python scripts/report/audit_stage2_generation.py --metrics-dir "$ARM/metrics" \
  --expected-per-day "$N" --json-out "$ARM/stage2.json" --strict
log 'Resumed to all-ok days and strict Stage2 audit; graph retained for export/dump'
