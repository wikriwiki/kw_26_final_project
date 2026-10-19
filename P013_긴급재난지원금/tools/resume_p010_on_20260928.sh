#!/usr/bin/env bash
# Continue the existing P010 ON graph after a failed citizen-day. Never reset it.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
OUT=/data/multipolicy_v53_20260928/p010
ARM=$OUT/on
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONIOENCODING=utf-8 PYTHONPATH="$REPO" LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 SIM_ENVIRONMENT=''
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
export SIM_OUTPUT_DIR="$ARM" SIM_RUN_ID=multipolicy-v53-20260928-p010-on
unset SIM_ALLOW_STAGE2_FALLBACK LLM_MODE
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ARM/resume.log"; }
trap 'log "FAILED at line $LINENO; graph/output remain intact"' ERR

test -s "$ARM/graph_restored.marker"
test -s "$ARM/served_model_evidence.json"
test "$(cat "$ARM/launch.exitcode")" != 0
test ! -e "$ARM/resume.exitcode"
(cd "$REPO" && sha256sum -c "$ARM/frozen_inputs.sha256")
python scripts/sim/policy_preflight.py --require-db data/experiments/P010_v53_policy_20260927.json
test "$(curl -fsS -m 8 localhost:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py|[e]xport_multi_policy_sector_ledger.py' | grep -q .; then
  log 'Simulator or exporter still active; refusing simultaneous resume'; exit 1
fi

# The failed five-day invocation saved completed day 0/1 in summary.json.
test ! -e "$ARM/summary.before_resume.json" || { log 'Resume already attempted; inspect before rerun'; exit 1; }
cp "$ARM/summary.json" "$ARM/summary.before_resume.json"
python - "$ARM" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads((p/'summary.before_resume.json').read_text())
assert not d.get('completed_at') and len(d['summary'])==2
assert [x['day'] for x in d['summary']]==['2025-07-19','2025-07-20']
assert all(x['ok']==80 and x['err']==0 for x in d['summary'])
for x in d['summary']:
    (p/f"day_{x['day']}.json").write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
PY
log 'Resuming failed 2025-07-21 from committed per-citizen DB outbox'
for day in 2025-07-21 2025-07-22 2025-07-23; do
  success=0
  for attempt in 1 2 3; do
    log "Day $day attempt $attempt"
    if python -u scripts/sim/run_simulation.py --start "$day" --days 1 --limit 80 --workers 32 \
         > "$ARM/resume_${day}_attempt${attempt}.run.log" 2>&1 && \
       python - "$ARM/summary.json" "$ARM/day_${day}.json" "$day" <<'PY'
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1])); day=sys.argv[3]
assert d.get('completed_at') and len(d['summary'])==1
r=d['summary'][0]
assert r['day']==day and r['ok']==80 and r['err']==0
Path(sys.argv[2]).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
PY
    then
      success=1; break
    fi
  done
  [[ $success == 1 ]] || { log "Three failed attempts on $day"; exit 1; }
done
python - "$ARM" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
p=Path(sys.argv[1]); days=['2025-07-19','2025-07-20','2025-07-21','2025-07-22','2025-07-23']
rows=[json.loads((p/f'day_{day}.json').read_text()) for day in days]
assert [x['day'] for x in rows]==days
assert all(x['ok']==80 and x['err']==0 for x in rows)
d=json.loads((p/'summary.before_resume.json').read_text())
summary={'summary':rows,'args':{**d['args'],'start':days[0],'days':len(days)},
         'completed_at':datetime.now().isoformat(),
         'resume_audit':{'original_five_day_run_failed_on':'2025-07-21',
                         'original_summary':'summary.before_resume.json',
                         'raw_attempts_dir':'metrics/attempts',
                         'committed_citizen_days_reused_on_retry':True}}
tmp=p/'summary.json.tmp'; tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
tmp.replace(p/'summary.json')
PY
python scripts/report/audit_stage2_generation.py --metrics-dir "$ARM/metrics" \
  --expected-per-day 80 --json-out "$ARM/stage2.json" --strict
log 'P010 ON resumed to five all-ok days; graph remains intact for export and dump'
