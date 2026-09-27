#!/usr/bin/env bash
# Check real Neo4j policy rendering before the first model call of a new arm.
# The preceding arm must already have verified off-server restore archives.
set -Eeuo pipefail
umask 077

case_id=${1:?case id required}
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928
PRE=/data/backup_20260927_pre_pilot
NEO=/data/neo4j-community-5.26.0
case "$case_id" in
  p016)
    PREVIOUS=distancing/off
    POLICY=data/neo4j_load/policies/P016.json
    TODAY=2020-07-30;;
  p014)
    PREVIOUS=p016/off
    POLICY=data/neo4j_load/policies/P014.json
    TODAY=2020-09-21;;
  *) echo 'preflight supports p016 or p014' >&2; exit 2;;
esac
OUT=$BASE/$case_id
mkdir -p "$OUT"
test -s "$BASE/$PREVIOUS/external_copy_verified.txt"
(cd "$BASE/$PREVIOUS/graph_backup" && sha256sum -c SHA256SUMS)
(cd "$BASE/$(dirname "$PREVIOUS")" && sha256sum -c "$(basename "$PREVIOUS")_artifacts.sha256")
(cd "$PRE" && sha256sum -c SHA256SUMS)
test ! -e "$OUT/on/graph_restored.marker"
if pgrep -af '[r]un_simulation.py|[e]xport_.*ledger.py' | grep -q .; then
  echo 'simulator or exporter is active' >&2; exit 1
fi

cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' tools/run_p013_ruler.sh | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
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
  echo 'Neo4j remained active; refusing restore' >&2; exit 1
fi
"$NEO/bin/neo4j-admin" database load neo4j --from-path="$PRE" --overwrite-destination=true
"$NEO/bin/neo4j" start
stopped=0
for attempt in $(seq 1 45); do
  if python - <<'PY' >/dev/null 2>&1
from scripts.neo4j_load._common import driver_session
with driver_session() as s:
    assert s.run('RETURN 1 AS ready').single()['ready'] == 1
PY
  then break; fi
  if [[ $attempt == 45 ]]; then echo 'Neo4j not query-ready' >&2; exit 1; fi
  sleep 2
done
trap - EXIT
python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
python scripts/sim/policy_preflight.py --require-db "$POLICY"
python - "$OUT/preflight_roster.json" <<'PY'
import json,sys
from pathlib import Path
from scripts.sim.run_simulation import fetch_agents
ids=fetch_agents(limit=40)
assert len(ids)==40 and len(set(ids))==40
Path(sys.argv[1]).write_text(json.dumps(ids,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
PY
python tools/audit_policy_graph_render_20260928.py \
  --policy-file "$POLICY" --roster "$OUT/preflight_roster.json" \
  --today "$TODAY" --out "$OUT/live_policy_render_audit_pre_run.json"
sha256sum "$OUT/preflight_roster.json" "$OUT/live_policy_render_audit_pre_run.json" \
  > "$OUT/live_policy_render_audit_pre_run.sha256"
echo "PREFLIGHT_PASS $case_id: graph was read for facts/status; 0 model calls"
