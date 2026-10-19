#!/usr/bin/env bash
# 두 팔의 그래프 덤프를 **실제로 복원**하고, 복원된 그래프가 dossier 와 사람 단위로
# 같은지 대조한다. 체크섬은 "파일이 그대로다" 까지만 말한다 — 그 안에 기억이 정말
# 있는지는 복원해서 읽어야 안다.
#
# 런이 **모두 끝난 뒤에만** 돌린다. Community 는 DB 가 하나라서 복원하면 지금 그래프를
# 덮는다. 시뮬이 돌고 있으면 거부한다.
#
# 복원 명령은 러너(run_multi_v53_short_arm_20260928.sh)가 팔마다 쓰는 것과 같다.
# Neo4j 는 setsid 로 세션 밖에서 띄운다 — 이 스크립트가 끝나도 DB 가 살아 있게.
#
#   bash tools/verify_graph_restore_20260929.sh /data/multipolicy_v53_20260928/p012m
set -Eeuo pipefail
umask 077
BASE=${1:?base dir}
REPO=/data/pilot_repo_20260927
NEO=/data/neo4j-community-5.26.0
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
OUT=$BASE/restore_check
mkdir -p "$OUT"
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/restore.log"; }
trap 'log "FAILED at line $LINENO"; setsid nohup "$NEO/bin/neo4j" start >/dev/null 2>&1 < /dev/null || true' ERR

if pgrep -af '[r]un_simulation.py|[e]xport_agent_dossier.py' | grep -q .; then
  log '시뮬이나 추출이 돌고 있다 — 복원하지 않는다'; exit 1
fi
for arm in on off; do
  test -s "$BASE/$arm/graph_backup/neo4j.dump"
  test -s "$BASE/$arm/dossier.jsonl"
  (cd "$BASE/$arm/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
done

wait_ready() {
  for _ in $(seq 1 90); do
    if python - <<'PY' >/dev/null 2>&1
from scripts.neo4j_load._common import driver_session
with driver_session() as s: assert s.run('RETURN 1 AS x').single()['x'] == 1
PY
    then return 0; fi
    sleep 2
  done
  return 1
}

rc=0
for arm in on off; do
  log "=== $arm 팔 덤프 복원"
  "$NEO/bin/neo4j" stop >/dev/null 2>&1 || true
  if pgrep -af '[o]rg.neo4j.server.CommunityEntryPoint' | grep -q .; then
    log 'Neo4j 가 아직 떠 있다 — 복원하지 않는다'; exit 1
  fi
  "$NEO/bin/neo4j-admin" database load neo4j \
    --from-path="$BASE/$arm/graph_backup" --overwrite-destination=true >> "$OUT/restore.log" 2>&1
  setsid nohup "$NEO/bin/neo4j" start >> "$OUT/restore.log" 2>&1 < /dev/null
  wait_ready || { log 'Neo4j 가 질의 가능 상태가 되지 않았다'; exit 1; }
  log "  복원 완료 — dossier 와 대조"
  if python scripts/report/verify_graph_against_dossier.py \
       --dossier "$BASE/$arm/dossier.jsonl" --sample 60 \
       --json-out "$OUT/${arm}.json" | tee -a "$OUT/restore.log"; then
    log "  $arm 팔: 복원한 그래프가 dossier 와 일치"
  else
    log "  $arm 팔: **불일치** — $OUT/${arm}.json"
    rc=1
  fi
done
if [[ $rc == 0 ]]; then
  log "결론: 두 팔의 덤프가 복원 가능하고, 복원한 그래프가 dossier 와 사람 단위로 같다"
else
  log "결론: **복원 대조 실패**"
fi
exit $rc
