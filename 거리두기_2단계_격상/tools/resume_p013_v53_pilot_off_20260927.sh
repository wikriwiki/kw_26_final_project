#!/usr/bin/env bash
# Resume the OFF arm after one citizen's Stage1 JSON parse failed on 2020-05-10.
# Never reset the graph; committed citizen-days are read from the Neo4j outbox.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
OUT=/data/p013_v53_pilot_20260927
POLICY=data/experiments/P013_v53_policy_20260926.json
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONIOENCODING=utf-8 PYTHONPATH="$REPO" LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 SIM_ENVIRONMENT=covid_2021
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
export SIM_OUTPUT_DIR="$OUT/off" SIM_RUN_ID=p013-v53-pilot-20260927-off
unset SIM_ALLOW_STAGE2_FALLBACK
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$OUT/off.resume.log"; }
trap 'log "FAILED line $LINENO; completed days and graph remain intact"' ERR
python scripts/sim/policy_preflight.py --expect-no-policy
if [ ! -f "$OUT/off.summary_before_resume.json" ]; then
  cp "$OUT/off/summary.json" "$OUT/off.summary_before_resume.json"
fi
python - "$OUT/off.summary_before_resume.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert len(d['summary'])==1 and d['summary'][0]['day']=='2020-05-09'
assert d['summary'][0]['ok']==80 and d['summary'][0]['err']==0
PY
if [ ! -f "$OUT/off.initial_failed_2020-05-10.json" ]; then
  cp "$OUT/off/checkpoints/failed_2020-05-10.json" "$OUT/off.initial_failed_2020-05-10.json"
fi

for day in 2020-05-10 2020-05-11 2020-05-12 2020-05-13; do
  if [ -s "$OUT/off.resume_day_$day.json" ]; then
    log "$day already completed and recorded; skip"
    continue
  fi
  success=0
  for attempt in 1 2 3; do
    log "Resume $day attempt $attempt (committed citizen-days reused)"
    if python -u scripts/sim/run_simulation.py --start "$day" --days 1 --limit 80 \
         --workers 32 --environment covid_2021 > "$OUT/off.resume_${day}_${attempt}.run.log" 2>&1; then
      python - "$OUT/off/summary.json" "$OUT/off.resume_day_$day.json" "$day" <<'PY'
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
  if [ "$success" -ne 1 ]; then log "Three failed resume attempts for $day"; exit 1; fi
done
python - "$OUT" <<'PY'
import json,sys
from datetime import datetime
from pathlib import Path
root=Path(sys.argv[1]); initial=json.loads((root/'off.summary_before_resume.json').read_text())
days=['2020-05-10','2020-05-11','2020-05-12','2020-05-13']
rows=initial['summary']+[json.loads((root/f'off.resume_day_{day}.json').read_text()) for day in days]
assert [row['day'] for row in rows]==['2020-05-09',*days]
assert all(row['ok']==80 and row['err']==0 for row in rows)
summary={'summary':rows,'args':initial['args'],'completed_at':datetime.now().isoformat(),
         'resume_audit':{'original_failed_day':'2020-05-10',
                         'initial_failure':'one Stage1 malformed JSON after three attempts',
                         'raw_failed_metrics_preserved':True,
                         'committed_citizen_days_reused':True}}
path=root/'off/summary.json'; tmp=path.with_suffix('.json.tmp')
tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n'); tmp.replace(path)
PY
python scripts/report/audit_stage2_generation.py --metrics-dir "$OUT/off/metrics" \
  --expected-per-day 80 --json-out "$OUT/off.stage2.json" --strict
python scripts/report/export_policy_daily_ledger.py --roster "$OUT/roster.json" \
  --start 2020-05-09 --end 2020-05-13 --arm off --policy-id P013 \
  --policy-file "$POLICY" --metrics-dir "$OUT/off/metrics" \
  --out "$OUT/off.ledger.jsonl"
sha256sum "$OUT/off.ledger.jsonl" "$OUT/off.ledger.jsonl.manifest.json" > "$OUT/off.ledger.sha256"
PILOT_REPO_ROOT="$REPO" python "$OUT/sector_export.py" export --arm off \
  --roster "$OUT/roster.json" --days 2020-05-11 2020-05-12 2020-05-13 \
  --out "$OUT/off.sector.json"
sha256sum "$OUT/off.sector.json" > "$OUT/off.sector.sha256"
log 'OFF_COMPLETE_READY_FOR_EXTERNAL_COPY'
