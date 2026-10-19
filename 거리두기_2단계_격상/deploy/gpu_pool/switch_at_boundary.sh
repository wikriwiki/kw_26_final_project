#!/usr/bin/env bash
# One controlled hand-over of the compute processes to the GPU-pool controller.
#
# Waits until DAY_DONE has a verified complete-day backup and the simulator is
# in the agent phase of DAY_NEXT, then: stop the old recovery controller, stop
# the supervisor with SIGTERM (its own clean interrupt path), start the GPU-pool
# controller, and verify that the resumed simulator uses the pool and that no
# completed day file changed. The model server, Neo4j and the instance are
# never restarted. Any failed check rolls back to the night-progress controller.
set -uo pipefail
DAY_DONE=${1:?usage: switch_at_boundary.sh DAY_DONE DAY_NEXT}
DAY_NEXT=${2:?usage: switch_at_boundary.sh DAY_DONE DAY_NEXT}
PREFIX=integration-main-v22-1154
R=/workspace/no-smoking-results
RUN=$R/$PREFIX-shared-pre
POOL=/workspace/no-smoking-gpu-pool
LAYER=/workspace/no-smoking-runtime-hotfix-v22-gpu-pool
OLD=/workspace/no-smoking-runtime-hotfix-v22-night-progress
PY=/workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python
LOG=$R/$PREFIX-gpu-pool-switch.log
STAMP=$(date -u +%Y%m%dT%H%M%SZ)

say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
pid_of() { pgrep -f "$1" | head -1; }
controller_old() { pid_of "^$PY $OLD/resume_runtime[.]py"; }
controller_new() { pid_of "^$PY $LAYER/resume_runtime[.]py"; }
supervisor() { pid_of "deploy/vast/run_shared[.]py --config"; }
simulator() { pid_of "scripts/sim/run_simulation[.]py --start"; }
backup() { pid_of "deploy/vast/backup_checkpoint[.]py"; }
day_hashes() { (cd "$RUN/metrics" && for f in day_*.jsonl; do [ "$f" = "day_$DAY_NEXT.jsonl" ] || sha256sum "$f"; done); }
rows_next() { [ -f "$RUN/metrics/day_$DAY_NEXT.jsonl" ] && wc -l < "$RUN/metrics/day_$DAY_NEXT.jsonl" || echo 0; }
wait_gone() { local what=$1 seconds=$2; for _ in $(seq "$seconds"); do [ -z "$($what)" ] && return 0; sleep 1; done; return 1; }

exec 8>"$POOL/switch.lock"
flock -n 8 || { say "another switch is already armed; exiting"; exit 1; }
say "armed: waiting for verified backup of $DAY_DONE and the agent phase of $DAY_NEXT"

while true; do
  [ -e "$POOL/ABORT_SWITCH" ] && { say "ABORT_SWITCH present; nothing changed"; exit 0; }
  if [ -f "$RUN/backup_completed_$DAY_DONE.json" ] && [ -f "$RUN/cohort_$DAY_NEXT.json" ] \
     && [ -z "$(backup)" ] && [ -n "$(simulator)" ] \
     && python3 - "$RUN/recoverable_backup.json" "$DAY_DONE" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
sys.exit(0 if record.get('day') == sys.argv[2] and record.get('kind') == 'complete_day'
         and record.get('verified_at_utc') and record.get('graph_sha256') else 1)
PY
  then break; fi
  sleep 10
done
say "boundary reached: $(python3 -c "import json;d=json.load(open('$RUN/recoverable_backup.json'));print(d['day'],d['kind'],d['verified_at_utc'],d['checkpoint_sha256'])")"

# ---- pre-checks (nothing changed yet) ----
$PY "$LAYER/resume_runtime.py" --dry-run >>"$LOG" 2>&1 || { say "PRECHECK FAILED: new controller dry-run; nothing changed"; exit 1; }
python3 - <<'PY' >>"$LOG" 2>&1 || { say "PRECHECK FAILED: pool proxy not ready; nothing changed"; exit 1; }
import json, urllib.request
status = json.load(urllib.request.urlopen('http://127.0.0.1:30000/pool/status', timeout=5))
assert [b for b in status['backends'] if b['local'] and b['healthy']], status
print('pool backends', [(b['name'], b['healthy']) for b in status['backends']])
PY
[ -n "$(controller_old)" ] || { say "PRECHECK FAILED: night-progress controller not running; nothing changed"; exit 1; }
[ -z "$(controller_new)" ] || { say "PRECHECK FAILED: pool controller already running; nothing changed"; exit 1; }
BEFORE_HASHES=$(day_hashes); BEFORE_ROWS=$(rows_next)
say "pre-state: completed day files $(echo "$BEFORE_HASHES" | wc -l), $DAY_NEXT rows $BEFORE_ROWS, supervisor $(supervisor), simulator $(simulator)"
echo "$BEFORE_HASHES" > "$POOL/switch-pre-hashes-$STAMP.txt"

rollback() {
  say "ROLLBACK: $1"
  local p; p=$(controller_new); [ -n "$p" ] && kill "$p"; wait_gone controller_new 30
  p=$(supervisor); [ -n "$p" ] && { kill "$p"; wait_gone supervisor 120; wait_gone simulator 60; }
  cp -p "/workspace/onstart.sh.pre-gpu-pool-$STAMP" /workspace/onstart.sh
  touch "$POOL/DISABLED"
  bash /workspace/onstart.sh
  say "ROLLBACK done: night-progress controller restarted (Vast only); pool DISABLED flag set"
  exit 2
}

# ---- hand-over ----
[ -z "$(backup)" ] || { say "backup started during pre-checks; nothing changed, re-run later"; exit 1; }
cp -p /workspace/onstart.sh "/workspace/onstart.sh.pre-gpu-pool-$STAMP"
OLDC=$(controller_old); kill "$OLDC"; wait_gone controller_old 30 || { say "old controller did not exit; nothing else changed"; bash /workspace/onstart.sh; exit 1; }
say "old controller $OLDC stopped"
SUP=$(supervisor); SIM=$(simulator)
kill "$SUP"                                   # SIGTERM: run_shared stops its children and records the interrupt
wait_gone supervisor 120 || say "supervisor still present after 120s"
if ! wait_gone simulator 60; then p=$(simulator); say "simulator $p still present; SIGTERM"; kill "$p"; wait_gone simulator 60 || { kill -9 "$(simulator)"; wait_gone simulator 30; }; fi
[ -z "$(supervisor)$(simulator)" ] || rollback "old compute processes did not exit"
say "supervisor $SUP / simulator $SIM stopped; pipeline: $(python3 -c "import json;d=json.load(open('$R/$PREFIX-pipeline.json'));print(d['status'],d.get('error'))")"
cp "$LAYER/onstart.sh" /workspace/onstart.sh && chmod +x /workspace/onstart.sh
bash /workspace/onstart.sh
say "pool controller started via /workspace/onstart.sh ($(sha256sum /workspace/onstart.sh | cut -c1-16))"

# ---- verification ----
for _ in $(seq 60); do [ -n "$(simulator)" ] && break; sleep 5; done
NEWSIM=$(simulator); [ -n "$NEWSIM" ] || rollback "no simulator within 5 minutes"
URL=$(tr '\0' '\n' < "/proc/$NEWSIM/environ" | grep '^SGLANG_BASE_URL=' | cut -d= -f2-)
PP=$(tr '\0' '\n' < "/proc/$NEWSIM/environ" | grep '^PYTHONPATH=' | cut -d= -f2- | cut -d: -f1)
say "new simulator $NEWSIM SGLANG_BASE_URL=$URL first PYTHONPATH=$PP controller $(controller_new) supervisor $(supervisor)"
[ "$URL" = "http://127.0.0.1:30000/v1" ] && [ "$PP" = "$LAYER" ] || rollback "resumed simulator is not using the pool layer"
[ "$(day_hashes)" = "$BEFORE_HASHES" ] || { say "CRITICAL: a completed day file changed; stopping the pool controller for inspection"; rollback "completed day file changed"; }
for _ in $(seq 90); do [ "$(rows_next)" -gt "$BEFORE_ROWS" ] && break; sleep 10; done
AFTER_ROWS=$(rows_next)
[ "$AFTER_ROWS" -gt "$BEFORE_ROWS" ] || rollback "no new $DAY_NEXT result within 15 minutes"
[ "$(day_hashes)" = "$BEFORE_HASHES" ] || rollback "completed day file changed"
say "SWITCH OK: $DAY_NEXT rows $BEFORE_ROWS -> $AFTER_ROWS; completed days unchanged; pool: $(python3 -c "
import json,urllib.request
s=json.load(urllib.request.urlopen('http://127.0.0.1:30000/pool/status',timeout=5))
print([(b['name'],b['healthy'],b['served'],b['inflight']) for b in s['backends']])")"
