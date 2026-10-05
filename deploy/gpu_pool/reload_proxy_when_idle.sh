#!/usr/bin/env bash
# Swap the proxy config only while the experiment is quiescent.
#
# The proxy has no hot reload; restarting it drops in-flight requests, which
# would cost those agents a retry attempt. During a post-day backup the
# simulator holds no LLM requests, so wait for: a backup_checkpoint process is
# running AND every backend shows inflight 0 on three checks 5 s apart. Then
# install the new config, let keeper.sh restart the proxy, and verify.
set -uo pipefail
NEW=${1:?usage: reload_proxy_when_idle.sh NEW_CONFIG_JSON}
POOL=/workspace/no-smoking-gpu-pool
LOG=/workspace/no-smoking-results/integration-main-v22-1154-gpu-pool-reload.log
say() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
python3 -c "import json,sys; c=json.load(open(sys.argv[1])); assert c['listen_port']==30000 and c['local_url']=='http://127.0.0.1:8000'" "$NEW" \
  || { say "new config invalid; nothing changed"; exit 1; }
exec 8>"$POOL/reload.lock"; flock -n 8 || { say "another reload is armed"; exit 1; }
idle() {
  python3 - <<'PY'
import json, sys, urllib.request
s = json.load(urllib.request.urlopen('http://127.0.0.1:30000/pool/status', timeout=5))
sys.exit(0 if all(b['inflight'] == 0 for b in s['backends']) else 1)
PY
}
say "armed: waiting for a backup window with no in-flight requests"
while true; do
  [ -e "$POOL/ABORT_RELOAD" ] && { say "ABORT_RELOAD present; nothing changed"; exit 0; }
  if pgrep -f "deploy/vast/backup_checkpoint[.]py" >/dev/null && idle && sleep 5 && idle && sleep 5 && idle; then break; fi
  sleep 10
done
cp -p "$POOL/config.json" "$POOL/config.json.pre-reload-$(date -u +%Y%m%dT%H%M%SZ)"
cp "$NEW" "$POOL/config.json"
OLD=$(pgrep -f "gpu_pool_proxy[.]py --config")
say "quiescent backup window; restarting proxy $OLD with new config"
kill "$OLD"
for _ in $(seq 30); do
  NEWPID=$(pgrep -f "gpu_pool_proxy[.]py --config"); [ -n "$NEWPID" ] && [ "$NEWPID" != "$OLD" ] && break; sleep 1
done
sleep 20
python3 - <<'PY' | tee -a "$LOG"
import json, urllib.request
s = json.load(urllib.request.urlopen('http://127.0.0.1:30000/pool/status', timeout=5))
print('pool after reload', [(b['name'], b['healthy'], b['capacity']) for b in s['backends']])
PY
say "reload done: proxy pid $(pgrep -f 'gpu_pool_proxy[.]py --config')"
