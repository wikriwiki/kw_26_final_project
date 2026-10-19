#!/usr/bin/env bash
# Preserve the completed OFF graph before restoring the shared Neo4j database.
set -Eeuo pipefail
umask 077
OUT=/data/p013_v53_pilot_20260927
NEO=/data/neo4j-community-5.26.0
DEST="$OUT/off_graph_backup"
test -s "$OUT/off.ledger.jsonl"
test -s "$OUT/off.ledger.jsonl.manifest.json"
test -s "$OUT/off.sector.json"
(cd "$OUT" && sha256sum -c off.ledger.sha256 && sha256sum -c off.sector.sha256)
python - "$OUT/off/summary.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d.get('completed_at') and len(d['summary'])==5
assert all(x['ok']==80 and x['err']==0 for x in d['summary'])
PY
if pgrep -af '[r]un_simulation.py|[e]xport_policy_daily_ledger.py|[s]ector_export.py' | grep -q .; then
  echo 'simulation or export still active; refusing stop' >&2
  exit 1
fi
mkdir -p "$DEST"
chmod 700 "$DEST"
exec 9>"$DEST/.lock"
flock -n 9
stopped=0
restart_on_exit() {
  rc=$?
  trap - EXIT
  if [ "$stopped" -eq 1 ]; then "$NEO/bin/neo4j" start || rc=1; fi
  exit "$rc"
}
trap restart_on_exit EXIT
printf 'started_at=%s\n' "$(date -Is)" > "$DEST/dump_state.txt"
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
printf 'dumps_complete_at=%s\n' "$(date -Is)" >> "$DEST/dump_state.txt"
"$NEO/bin/neo4j" start
stopped=0
"$NEO/bin/neo4j" status
printf 'restarted_at=%s\n' "$(date -Is)" >> "$DEST/dump_state.txt"
