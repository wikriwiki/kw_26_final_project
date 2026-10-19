#!/usr/bin/env bash
# Preserve the failed, incomplete P016 arm before any amended experiment.
set -Eeuo pipefail
umask 077
ROOT=/data/multipolicy_v53_20260928/p016
ARM=$ROOT/on
DEST=$ROOT/failed_prepatch_graph_backup
NEO=/data/neo4j-community-5.26.0
test "$(cat "$ARM/launch.exitcode")" = 1
test -s "$ARM/metrics/day_2020-07-30.jsonl"
test ! -e "$DEST/SHA256SUMS"
if pgrep -af '[r]un_simulation.py|[e]xport_.*ledger.py' | grep -q .; then
  echo 'simulation or ledger export is active; refusing snapshot' >&2
  exit 1
fi
python - "$ARM" <<'PY'
import collections,json,pathlib,sys
arm=pathlib.Path(sys.argv[1])
summary=json.loads((arm/'summary.json').read_text())
assert len(summary.get('summary',[]))<5, 'this is not a completed five-day arm'
rows=[json.loads(line) for line in (arm/'metrics/day_2020-07-30.jsonl').read_text().splitlines()]
counts=collections.Counter((row.get('status'),row.get('error')) for row in rows)
assert counts.get(('error',"'P016'"),0)>0, counts
print('P016 no-eligible-purchase KeyError attempts:',counts[('error',"'P016'")])
PY
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
  echo 'Neo4j still mounted; refusing dump' >&2
  exit 1
fi
"$NEO/bin/neo4j-admin" database dump neo4j --to-path="$DEST"
"$NEO/bin/neo4j-admin" database dump system --to-path="$DEST"
tar -C "$NEO" -czf "$DEST/config_plugins.tar.gz" conf plugins
"$NEO/bin/neo4j" version > "$DEST/neo4j_version.txt"
(cd "$DEST" && sha256sum neo4j.dump system.dump config_plugins.tar.gz neo4j_version.txt > SHA256SUMS)
"$NEO/bin/neo4j" start
stopped=0
trap - EXIT
(cd "$DEST" && sha256sum -c SHA256SUMS)
tar -C "$ROOT" -czf "$ROOT/failed_prepatch_artifacts.tar.gz" \
  on roster.json frozen_income.json preflight_first_failure.log \
  preflight_first_failure.exitcode preflight_first_failure.manifest.json \
  preflight_retry.log live_policy_render_audit_pre_run.json
(cd "$ROOT" && sha256sum failed_prepatch_artifacts.tar.gz > failed_prepatch_artifacts.sha256)
python - "$ROOT" <<'PY'
import collections,datetime,hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); arm=r/'on'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(line) for line in (arm/'metrics/day_2020-07-30.jsonl').read_text().splitlines()]
out={'schema':'failed_p016_prepatch_snapshot_v1','created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
     'run_status':'incomplete_failed','model_calls_occurred':True,'date_of_failure':'2020-07-30',
     'raw_attempt_rows':len(rows),'raw_keyerror_rows':sum(x.get('status')=='error' and x.get('error')=="'P016'" for x in rows),
     'source_frozen_inputs_sha256':sha(arm/'frozen_inputs.sha256'),
     'source_run_manifest_sha256':sha(arm/'run_manifest.json'),
     'graph_manifest_sha256':sha(r/'failed_prepatch_graph_backup/SHA256SUMS'),
     'artifacts_sha256':sha(r/'failed_prepatch_artifacts.tar.gz'),
     'finding':'Inactive closing-balance key for an active instant-discount policy causes KeyError when a citizen makes no eligible purchase. Not an effect estimate.'}
(r/'failed_prepatch_snapshot.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
PY
(cd "$ROOT" && sha256sum failed_prepatch_snapshot.json > failed_prepatch_snapshot.sha256)
echo 'FAILED_ARM_SNAPSHOT_ON_SERVER: copy graph and artifacts outside A100 before reset'
