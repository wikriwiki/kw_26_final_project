#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
OUT=/data/p013_v53_pilot_20260927
POLICY=data/experiments/P013_v53_policy_20260926.json
cd "$REPO"
source /data/venv/bin/activate
# Read the existing credential assignment without executing its simulation script.
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONIOENCODING=utf-8 PYTHONPATH="$REPO" LLM_BASE_URL=http://localhost:8000/v1
export SIM_PROMPT_VARIANT=v53 SIM_ENVIRONMENT=covid_2021
export EXP_SANGSAENG_BASE_RATIO=0.268 EXP_SEED_SANGSAENG=1 EXP_BALANCE_DAYS=39
export EXP_DURABLES=1 EXP_CATLINE=fold EXP_POLICY_ANONYMOUS=1 POLICY_POI_SORT_BOOST=0
export EXP_DAILY_INCOME=baseline EXP_DAILY_INCOME_MAP="$OUT/frozen_income.json"
unset SIM_ALLOW_STAGE2_FALLBACK
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$OUT/pilot.log"; }
trap 'log "FAILED line $LINENO; graph and all output left intact for audit"' ERR
test -s /data/backup_20260927_pre_pilot/neo4j.dump
test -s /data/backup_20260927_pre_pilot/SHA256SUMS
test -s "$OUT/prereg.md"
sha256sum "$OUT/prereg.md" "$REPO/$POLICY" > "$OUT/frozen_inputs.sha256"

# Freeze the same deterministic stratified cohort and policy-independent budget.
SIM_OUTPUT_DIR="$OUT/preflight" python - <<'PY'
import hashlib, json
from pathlib import Path
from scripts.sim.run_simulation import fetch_agents
from scripts.neo4j_load._common import driver_session

root=Path('/data/p013_v53_pilot_20260927')
ids=fetch_agents(limit=80)
assert len(ids)==80 and len(set(ids))==80
with driver_session() as s:
    q='MATCH (a:Agent) WHERE a.id IN $ids RETURN a.id AS id, a.s_daily_wd AS wd, a.s_daily_we AS we ORDER BY a.id'
    first=[dict(r) for r in s.run(q,ids=ids)]
    second=[dict(r) for r in s.run(q,ids=ids)]
assert first==second and len(first)==80
def anchor(row):
    wd,we=row['wd'],row['we']
    wd=float(wd) if wd is not None and float(wd)>0 else None
    we=float(we) if we is not None and float(we)>0 else None
    # Mirror Day0's one-sided anchor fill; a citizen with no anchor receives
    # a fixed 1,500,000/39 daily budget, declared below as a synthetic fallback.
    if wd is None and we is None:
        return round(1500000/39), 'both_missing'
    if wd is None: wd=we
    if we is None: we=wd
    return round((5*wd+2*we)/7), 'observed_or_one_sided'
computed={r['id']:anchor(r) for r in first}
budgets={aid:v for aid,(v,kind) in computed.items()}
assert all(v>0 for v in budgets.values())
def sha(b): return hashlib.sha256(b).hexdigest()
proj=json.dumps(first,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
roster=json.dumps(ids,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
backup=Path('/data/backup_20260927_pre_pilot')
data={'schema':'fixed_persona_budget_v1','source_kind':'stable_persona_spending_anchors',
      'source_field_stability_verified':True,'policy_outcome_used':False,
      'source_fields':['s_daily_wd','s_daily_we'],
      'formula':'round((5*s_daily_wd + 2*s_daily_we)/7)',
      'source_archive_sha256':sha((backup/'neo4j.dump').read_bytes()),
      'confirmation_archive_sha256':sha((backup/'SHA256SUMS').read_bytes()),
      'source_roster_sha256':sha(roster),
      'source_agent_projection_sha256':sha(proj),
      'citizen_count':len(ids),'total_daily_budget_won':sum(budgets.values()),
      'daily_income_by_aid':budgets,
      'normalization_note':'One missing anchor uses the other; both missing use policy-independent 1,500,000/39 won per day.',
      'both_missing_anchor_count':sum(kind=='both_missing' for v,kind in computed.values()),
      'provenance_note':'Projection read twice identically from restored graph; SHA256SUMS is backup manifest, not independent archive.'}
(root/'roster.json').write_text(json.dumps(ids,ensure_ascii=False,indent=2)+'\n')
(root/'frozen_income.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
print(f'Frozen 80 citizens; missing both anchors={data["both_missing_anchor_count"]}; graph projection verified')
PY
sha256sum "$OUT/roster.json" "$OUT/frozen_income.json" >> "$OUT/frozen_inputs.sha256"

run_arm() {
  local arm="$1"
  log "Reset and Day0 seed for $arm"
  python scripts/neo4j_load/97_reset_run_artifacts.py
  DAY_ZERO=2020-05-08 python scripts/neo4j_load/08_initial_state.py
  if [ "$arm" = on ]; then
    python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
    python scripts/sim/policy_preflight.py --require-db "$POLICY"
  else
    python scripts/sim/policy_preflight.py --expect-no-policy
  fi
  export SIM_OUTPUT_DIR="$OUT/$arm" SIM_RUN_ID="p013-v53-pilot-20260927-$arm"
  log "Start $arm: 80 citizens x 5 days"
  python -u scripts/sim/run_simulation.py --start 2020-05-09 --days 5 --limit 80 \
    --workers 32 --environment covid_2021 > "$OUT/$arm.run.log" 2>&1
  python - "$OUT/$arm/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d.get('completed_at') and len(d['summary'])==5
assert all(x['ok']==80 and x['err']==0 for x in d['summary'])
PY
  python scripts/report/audit_stage2_generation.py --metrics-dir "$OUT/$arm/metrics" \
    --expected-per-day 80 --json-out "$OUT/$arm.stage2.json" --strict
  python scripts/report/export_policy_daily_ledger.py --roster "$OUT/roster.json" \
    --start 2020-05-09 --end 2020-05-13 --arm "$arm" --policy-id P013 \
    --policy-file "$POLICY" --metrics-dir "$OUT/$arm/metrics" \
    --out "$OUT/$arm.ledger.jsonl"
  sha256sum "$OUT/$arm.ledger.jsonl" "$OUT/$arm.ledger.jsonl.manifest.json" > "$OUT/$arm.ledger.sha256"
  log "Completed and exported $arm"
}
run_arm off
run_arm on
python scripts/report/paired_grant_effect.py --on "$OUT/on.ledger.jsonl" \
  --off "$OUT/off.ledger.jsonl" --roster "$OUT/roster.json" \
  --start 2020-05-09 --end 2020-05-13 --effect-start 2020-05-11 \
  --effect-end 2020-05-13 --policy-id P013 --expected-recipients 80 \
  --expected-issued-won 22400000 --json-out "$OUT/paired_effect.json"
sha256sum "$OUT/paired_effect.json" >> "$OUT/frozen_inputs.sha256"
log 'PILOT_COMPLETE'
