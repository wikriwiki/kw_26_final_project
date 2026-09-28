#!/usr/bin/env bash
# Export and preserve one completed arm. Never resets for a subsequent arm.
set -Eeuo pipefail
umask 077
case_id=${1:?case id}
arm=${2:?arm}
[[ $arm == on || $arm == off ]] || exit 2
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928
OUT=$BASE/$case_id
ARM=$OUT/$arm
NEO=/data/neo4j-community-5.26.0
N=40
case "$case_id" in
  p010) START=2025-07-19; END=2025-07-23; DAYS=5; N=80; POLICY=data/experiments/P010_v53_policy_20260927.json; PID=P010; FROM=2025-07-21; UNTIL=2025-11-30;;
  distancing) START=2020-11-24; END=2020-11-26; DAYS=3; POLICY=''; PID=''; FROM=''; UNTIL='';;
  p016) START=2020-07-28; END=2020-08-01; DAYS=5; POLICY=data/neo4j_load/policies/P016.json; PID=P016; FROM=2020-07-30; UNTIL=2020-11-30;;
  p014) START=2020-09-19; END=2020-09-23; DAYS=5; POLICY=data/neo4j_load/policies/P014.json; PID=P014; FROM=2020-09-21; UNTIL=2020-10-11;;
  p012) START=2021-10-01; END=2021-10-31; DAYS=31; N=12; POLICY=data/experiments/P012_v53_october_policy_20260928.json; PID=P012; FROM=2021-10-01; UNTIL=2021-11-30;;
  # 출력 배관 검수용 압축월(7일). 실측 대조에 쓰지 않는다.
  p012t) START=2021-10-01; END=2021-10-07; DAYS=7; N=40; POLICY=data/experiments/P012_v53_compressed7_TESTONLY.json; PID=P012; FROM=2021-10-01; UNTIL=2021-10-07;;
  *) echo "unknown case $case_id" >&2; exit 2;;
esac
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8 SIM_PROMPT_VARIANT=v53
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$ARM/preserve.log"; }
trap 'log "FAILED line $LINENO; graph/output left intact"' ERR
if [[ $(cat "$ARM/launch.exitcode") != 0 ]]; then
  test "$(cat "$ARM/resume.exitcode")" = 0
fi
test -s "$ARM/summary.json"
test -s "$ARM/stage2.json"
test -s "$ARM/served_model_evidence.json"
if [[ $case_id == distancing ]]; then
  test -s "$ARM/receipt_coordinates.jsonl"
  test -s "$ARM/receipt_coordinates.jsonl.manifest.json"
  test -s "$ARM/poi_coordinates.json"
fi
test ! -e "$ARM/graph_backup/SHA256SUMS" || { log 'Graph already dumped; refusing duplicate'; exit 1; }
python - "$ARM/summary.json" "$DAYS" "$N" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2])
assert d.get('completed_at') and len(d['summary'])==n
assert all(r['ok']==int(sys.argv[3]) and r['err']==0 for r in d['summary'])
PY
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py|[e]xport_multi_policy_sector_ledger.py' | grep -q .; then
  log 'Active simulator/exporter; refusing graph stop'; exit 1
fi

extra=()
if [[ $arm == on && -n $PID ]]; then
  extra=(--policy-id "$PID" --policy-file "$POLICY" --effective-from "$FROM" --effective-until "$UNTIL")
fi
python scripts/report/export_multi_policy_sector_ledger.py \
  --roster "$OUT/roster.json" --start "$START" --end "$END" --arm "$arm" \
  "${extra[@]}" --metrics-dir "$ARM/metrics" --out "$ARM/sector.ledger.jsonl"
if [[ $case_id == p010 ]]; then
  python scripts/report/export_policy_daily_ledger.py \
    --roster "$OUT/roster.json" --start "$START" --end "$END" --arm "$arm" \
    --policy-id "$PID" --policy-file "$POLICY" \
    --metrics-dir "$ARM/metrics" --out "$ARM/policy.ledger.jsonl"
fi
if [[ $case_id == p012 || $case_id == p012t ]]; then
  python scripts/report/export_cashback_month.py \
    --month 2021-10 --arm "$arm" --policy-id P012 --policy-file "$POLICY" \
    --base-ratio 0.268 --roster "$OUT/roster.json" \
    --metrics-dir "$ARM/metrics" --out "$ARM/cashback.ledger.jsonl"
fi
log 'Completed read-only graph ledger exports'

# 기억·스케줄·지출 내역은 그래프 안에만 있고, 다음 팔이 그래프를 덮는다. 덮이기
# 전에 사람 단위로 빼 둔다 — 1대1 인터뷰가 덤프 복원 없이 읽을 수 있어야 한다.
# 날이 하나라도 빠지면 이 스크립트가 여기서 멈춘다(반쪽 보존을 남기지 않는다).
python scripts/report/export_agent_dossier.py \
  --roster "$OUT/roster.json" --start "$START" --end "$END" --arm "$arm" \
  --out "$ARM/dossier.jsonl"
test -s "$ARM/dossier.jsonl"
test -s "$ARM/dossier.jsonl.manifest.json"
python - "$ARM/dossier.jsonl.manifest.json" "$DAYS" "$N" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
assert m["expected_days"] == int(sys.argv[2]), m["expected_days"]
assert m["agents"] == int(sys.argv[3]), m["agents"]
assert not m["state_day_gaps"], m["state_day_gaps"][:1]
assert m["totals"]["states"] == m["agents"] * m["expected_days"], m["totals"]
assert m["totals"]["memories"] > 0 and m["totals"]["plan_items"] > 0, m["totals"]
PY
log "Agent dossier preserved: memories, schedules, spending, wallet states"

DEST=$ARM/graph_backup
mkdir -p "$DEST"
chmod 700 "$DEST"
exec 9>"$DEST/.lock"
flock -n 9
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
  log 'Neo4j still mounted; refusing dump'; exit 1
fi
"$NEO/bin/neo4j-admin" database dump neo4j --to-path="$DEST"
"$NEO/bin/neo4j-admin" database dump system --to-path="$DEST"
tar -C "$NEO" -czf "$DEST/config_plugins.tar.gz" conf plugins
"$NEO/bin/neo4j" version > "$DEST/neo4j_version.txt"
(cd "$DEST" && sha256sum neo4j.dump system.dump config_plugins.tar.gz neo4j_version.txt > SHA256SUMS)
"$NEO/bin/neo4j" start
stopped=0
"$NEO/bin/neo4j" status
trap - EXIT
(cd "$DEST" && sha256sum -c SHA256SUMS)
log 'Full restore graph backup completed and SHA256 verified'

tar --exclude='graph_backup' -C "$OUT" -czf "$OUT/${arm}_artifacts.tar.gz" \
  "$arm" roster.json frozen_income.json
(cd "$OUT" && sha256sum "${arm}_artifacts.tar.gz" > "${arm}_artifacts.sha256")
sha256sum "$ARM/summary.json" "$ARM/sector.ledger.jsonl" \
  "$ARM/sector.ledger.jsonl.manifest.json" "$ARM/stage2.json" \
  "$ARM/run_manifest.json" > "$ARM/outputs.sha256"
sha256sum "$ARM/served_model_evidence.json" >> "$ARM/outputs.sha256"
sha256sum "$ARM/dossier.jsonl" "$ARM/dossier.jsonl.manifest.json" >> "$ARM/outputs.sha256"
if [[ $case_id == p010 ]]; then
  sha256sum "$ARM/policy.ledger.jsonl" "$ARM/policy.ledger.jsonl.manifest.json" >> "$ARM/outputs.sha256"
fi
if [[ $case_id == p012 || $case_id == p012t ]]; then
  sha256sum "$ARM/cashback.ledger.jsonl" "$ARM/cashback.ledger.jsonl.manifest.json" >> "$ARM/outputs.sha256"
fi
if [[ $case_id == distancing ]]; then
  # Geography is a descriptive proxy. Require its graph-backed input archive
  # before this graph may be reset for the next arm.
  sha256sum "$ARM/receipt_coordinates.jsonl" \
    "$ARM/receipt_coordinates.jsonl.manifest.json" \
    "$ARM/poi_coordinates.json" >> "$ARM/outputs.sha256"
fi
log 'PRESERVED_ON_SERVER: copy both archives/manifests outside A100 and verify before any next restore'
