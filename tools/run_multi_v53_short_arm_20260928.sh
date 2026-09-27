#!/usr/bin/env bash
# A separately frozen arm of the multi-policy v53 technical screen.
# Invoke in a persistent tmux session. Never advances to another arm itself.
set -Eeuo pipefail
umask 077

case_id=${1:?case id required: p010|distancing|p016|p014|p012}
arm=${2:?arm required: on|off}
[[ $arm == on || $arm == off ]] || { echo 'arm must be on or off' >&2; exit 2; }
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928
PRE=/data/backup_20260927_pre_pilot
NEO=/data/neo4j-community-5.26.0
N=40
case "$case_id" in
  p010)
    START=2025-07-19; DAYS=5; DAY0=2025-07-18; N=80
    ENV_ID=''; POLICY=data/experiments/P010_v53_policy_20260927.json; PID=P010;;
  distancing)
    START=2020-11-24; DAYS=3; DAY0=2020-11-23
    POLICY=''; PID=''
    if [[ $arm == on ]]; then ENV_ID=covid_2021; else ENV_ID=covid_no_distancing; fi;;
  p016)
    START=2020-07-28; DAYS=5; DAY0=2020-07-27
    ENV_ID=covid_2021; POLICY=data/neo4j_load/policies/P016.json; PID=P016;;
  p014)
    START=2020-09-19; DAYS=5; DAY0=2020-09-18
    ENV_ID=covid_2021; POLICY=data/neo4j_load/policies/P014.json; PID=P014;;
  p012)
    START=2021-10-01; DAYS=31; DAY0=2021-09-30; N=12
    ENV_ID=covid_2021; POLICY=data/experiments/P012_v53_october_policy_20260928.json; PID=P012;;
  *) echo "unknown case $case_id" >&2; exit 2;;
esac
OUT=$BASE/$case_id
ARM=$OUT/$arm
mkdir -p "$ARM"
# A previous arm cannot be overwritten merely because its graph has a local dump.
# The operator records verified copies of both the graph and run artifacts off A100.
previous=''
case "$case_id/$arm" in
  p010/off) previous=p010/on;;
  p012/on) previous=p010/off;;
  p012/off) previous=p012/on;;
  distancing/on) previous=p012/off;;
  distancing/off) previous=distancing/on;;
  p016/on) previous=distancing/off;;
  p016/off) previous=p016/on;;
  p014/on) previous=p016/off;;
  p014/off) previous=p014/on;;
esac
if [[ -n $previous ]]; then
  prevdir=$BASE/$previous
  test -s "$prevdir/external_copy_verified.txt"
  (cd "$prevdir/graph_backup" && sha256sum -c SHA256SUMS)
  (cd "$(dirname "$prevdir")" && sha256sum -c "$(basename "$prevdir")_artifacts.sha256")
fi
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONIOENCODING=utf-8 PYTHONPATH="$REPO" LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 SIM_ENVIRONMENT="$ENV_ID"
# P010 ON was frozen with the default request alias. Keep its paired control
# identical; subsequent policy pairs explicitly request the served model ID.
if [[ $case_id == p010 ]]; then unset LLM_MODE; else export LLM_MODE=exaone_4_5; fi
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
unset SIM_ALLOW_STAGE2_FALLBACK
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ARM/arm.log"; }
trap 'log "FAILED at line $LINENO; graph/output left intact"' ERR

# A previous completed graph and the baseline restore source must be recoverable.
test -s /data/p013_v53_pilot_20260927/all_external_copy_verified.txt
test -s /data/p013_v53_pilot_20260927/on_graph_backup/SHA256SUMS
(cd /data/p013_v53_pilot_20260927/on_graph_backup && sha256sum -c SHA256SUMS)
(cd "$PRE" && sha256sum -c SHA256SUMS)
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py|[e]xport_multi_policy_sector_ledger.py' | grep -q .; then
  log 'Another simulation/exporter is active; refusing graph restore'; exit 1
fi
test ! -e "$ARM/summary.json" || { log 'Arm output already exists; refusing overwrite'; exit 1; }
test ! -e "$ARM/graph_restored.marker" || { log 'Arm graph marker already exists; refusing reinitialize'; exit 1; }

# P010 retains its original 80-person preregistered cohort. Other cases use
# smaller stratified cohorts, frozen from the same pre-pilot graph after restore.
if [[ $case_id == p010 ]]; then
  cp -n /data/p013_v53_pilot_20260927/roster.json "$OUT/roster.json"
  cp -n /data/p013_v53_pilot_20260927/frozen_income.json "$OUT/frozen_income.json"
  cmp /data/p013_v53_pilot_20260927/roster.json "$OUT/roster.json"
  cmp /data/p013_v53_pilot_20260927/frozen_income.json "$OUT/frozen_income.json"
fi
if [[ $case_id == p010 ]]; then python - "$OUT/roster.json" "$OUT/frozen_income.json" <<'PY'
import json,sys
roster=json.load(open(sys.argv[1])); income=json.load(open(sys.argv[2]))
assert len(roster)==80 and len(set(roster))==80
assert set(roster)==set(income['daily_income_by_aid'])
PY
fi
test "$(curl -fsS -m 8 localhost:8000/v1/models | python -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')" = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
if [[ -n $POLICY ]]; then
  test -s "$REPO/$POLICY"
  SIM_OUTPUT_DIR="$ARM/preflight" NEO4J_URI='' python scripts/sim/policy_preflight.py "$POLICY"
fi
sha256sum scripts/sim/prompts/v53.py scripts/sim/stage1_intent.py scripts/sim/stage2_poi.py \
  scripts/sim/run_simulation.py scripts/sim/environments/registry.py \
  ${POLICY:+"$POLICY"} > "$ARM/frozen_inputs.sha256"
if [[ $case_id == p010 ]]; then
  sha256sum "$OUT/roster.json" "$OUT/frozen_income.json" >> "$ARM/frozen_inputs.sha256"
fi
cat > "$ARM/run_manifest.json" <<EOF
{"case":"$case_id","arm":"$arm","start":"$START","days":$DAYS,"day0":"$DAY0","environment":"$ENV_ID","policy_file":"$POLICY","prompt":"v53","served_model":"LGAI-EXAONE/EXAONE-4.5-33B-AWQ","request_mode":"${LLM_MODE:-qwen8b}","citizens":$N,"source_commit":"$(git rev-parse HEAD)","prepared_at":"$(date -Is)"}
EOF
sha256sum "$ARM/run_manifest.json" >> "$ARM/frozen_inputs.sha256"
python tools/capture_served_model_evidence_20260928.py \
  --out "$ARM/served_model_evidence.json"
sha256sum "$ARM/served_model_evidence.json" >> "$ARM/frozen_inputs.sha256"
log 'Frozen manifest and SHA256 saved before restore'

# Every arm starts from the same verified pre-pilot graph, never the preceding arm.
stopped=0
restart_on_failure() {
  rc=$?
  trap - EXIT
  if [[ $stopped == 1 ]]; then "$NEO/bin/neo4j" start || rc=1; fi
  exit "$rc"
}
trap restart_on_failure EXIT
"$NEO/bin/neo4j" stop
stopped=1
if pgrep -af '[o]rg.neo4j.server.CommunityEntryPoint' | grep -q .; then
  log 'Neo4j is still mounted; refusing restore'; exit 1
fi
"$NEO/bin/neo4j-admin" database load neo4j --from-path="$PRE" --overwrite-destination=true
"$NEO/bin/neo4j" start
stopped=0
"$NEO/bin/neo4j" status
ready=0
for attempt in $(seq 1 45); do
  if python - <<'PY' >/dev/null 2>&1
from scripts.neo4j_load._common import driver_session
with driver_session() as s: assert s.run('RETURN 1 AS ready').single()['ready']==1
PY
  then ready=1; break; fi
  sleep 2
done
[[ $ready == 1 ]] || { log 'Neo4j not query-ready'; exit 1; }
trap - EXIT
log 'Pre-pilot graph restored; seeding Day0'
if [[ $case_id != p010 ]]; then
  SIM_OUTPUT_DIR="$ARM/preflight" python tools/freeze_multi_small_cohort_20260928.py \
    --out "$OUT" --citizens "$N" --backup "$PRE"
  sha256sum "$OUT/roster.json" "$OUT/frozen_income.json" >> "$ARM/frozen_inputs.sha256"
fi
python scripts/neo4j_load/97_reset_run_artifacts.py
DAY_ZERO="$DAY0" python scripts/neo4j_load/08_initial_state.py
if [[ -n $POLICY && $arm == on ]]; then
  python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
  python scripts/sim/policy_preflight.py --require-db "$POLICY"
else
  python scripts/sim/policy_preflight.py --expect-no-policy
fi
printf 'restored_from=%s\nrestored_at=%s\n' "$PRE" "$(date -Is)" > "$ARM/graph_restored.marker"

export SIM_OUTPUT_DIR="$ARM" SIM_RUN_ID="multipolicy-v53-20260928-$case_id-$arm"
log "RUNNING $case_id/$arm: $N citizens x $DAYS days"
for offset in $(seq 0 $((DAYS - 1))); do
  day=$(date -d "$START + $offset days" +%F)
  success=0
  for attempt in 1 2 3; do
    args=(--start "$day" --days 1 --limit "$N" --workers 32)
    if [[ -n $ENV_ID ]]; then args+=(--environment "$ENV_ID"); fi
    log "Day $day attempt $attempt"
    if python -u scripts/sim/run_simulation.py "${args[@]}" \
         > "$ARM/day_${day}_attempt${attempt}.run.log" 2>&1 && \
       python - "$ARM/summary.json" "$ARM/day_${day}.json" "$day" "$N" <<'PY'
import json,sys
from pathlib import Path
d=json.load(open(sys.argv[1])); day=sys.argv[3]
assert d.get('completed_at') and len(d['summary'])==1
r=d['summary'][0]
assert r['day']==day and r['ok']==int(sys.argv[4]) and r['err']==0
Path(sys.argv[2]).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
PY
    then
      success=1; break
    fi
  done
  [[ $success == 1 ]] || { log "Three failed attempts for $day"; exit 1; }
done
python - "$ARM" "$START" "$DAYS" "$N" <<'PY'
import json,sys
from datetime import date,datetime,timedelta
from pathlib import Path
root=Path(sys.argv[1]); start=date.fromisoformat(sys.argv[2]); n=int(sys.argv[3])
days=[(start+timedelta(days=i)).isoformat() for i in range(n)]
rows=[json.loads((root/f'day_{day}.json').read_text()) for day in days]
assert [r['day'] for r in rows]==days
assert all(r['ok']==int(sys.argv[4]) and r['err']==0 for r in rows)
d=json.loads((root/'summary.json').read_text())
summary={'summary':rows,'args':{**d['args'],'start':days[0],'days':n},
         'completed_at':datetime.now().isoformat(),
         'resume_audit':{'one_day_invocations':True,'committed_citizen_days_reused_on_retry':True}}
tmp=root/'summary.json.tmp'; tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
tmp.replace(root/'summary.json')
PY
python - "$ARM/summary.json" "$DAYS" "$N" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2])
assert d.get('completed_at') and len(d['summary'])==n
assert all(x['ok']==int(sys.argv[3]) and x['err']==0 for x in d['summary'])
PY
python scripts/report/audit_stage2_generation.py --metrics-dir "$ARM/metrics" \
  --expected-per-day "$N" --json-out "$ARM/stage2.json" --strict
log 'Simulation and Stage2 quality audit completed; graph remains intact for export and dump'
