#!/usr/bin/env bash
# Offline, restorable snapshot of the active A100 Neo4j Community databases.
# Run as the Neo4j owner. A trap restarts the service even if dumping fails.
set -Eeuo pipefail

NEO=/data/neo4j-community-5.26.0
DEST=/data/backup_20260927_pre_pilot
mkdir -p "$DEST"
chmod 700 "$DEST"
exec 9>"$DEST/.lock"
flock -n 9 || { echo 'another backup is running' >&2; exit 1; }

if pgrep -af '[r]un_simulation.py' | grep -q .; then
  echo 'simulation is still running; refusing Neo4j stop' >&2
  exit 1
fi

stopped=0
restart_on_exit() {
  rc=$?
  trap - EXIT
  if [ "$stopped" -eq 1 ]; then
    "$NEO/bin/neo4j" start || rc=1
  fi
  exit "$rc"
}
trap restart_on_exit EXIT

printf 'started_at=%s\n' "$(date -Is)" > "$DEST/backup_state.txt"
"$NEO/bin/neo4j" stop
stopped=1
if pgrep -af '[o]rg.neo4j.server.CommunityEntryPoint' | grep -q .; then
  echo 'Neo4j process remains mounted; refusing dump' >&2
  exit 1
fi

"$NEO/bin/neo4j-admin" database dump neo4j --to-path="$DEST"
"$NEO/bin/neo4j-admin" database dump system --to-path="$DEST"
tar -C "$NEO" -czf "$DEST/config_plugins.tar.gz" conf plugins
"$NEO/bin/neo4j" version > "$DEST/neo4j_version.txt"
(cd "$DEST" && sha256sum neo4j.dump system.dump config_plugins.tar.gz neo4j_version.txt > SHA256SUMS)
printf 'dumps_complete_at=%s\n' "$(date -Is)" >> "$DEST/backup_state.txt"

"$NEO/bin/neo4j" start
stopped=0
"$NEO/bin/neo4j" status
printf 'restarted_at=%s\n' "$(date -Is)" >> "$DEST/backup_state.txt"
