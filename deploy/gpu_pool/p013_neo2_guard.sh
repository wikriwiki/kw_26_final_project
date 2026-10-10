#!/usr/bin/env bash
# P013 지킴이 — 두 번째 Neo4j(/data/neo4j2-community-5.26.0)가 P013 의 다음 단계를 막지 않게 한다.
# P013 의 보존·되돌리기 단계는 "서버에 Neo4j 가 하나라도 떠 있으면 중단"한다. 사용자가 그 확인 범위를 좁히는 것을
# 승인하지 않은 동안(/data/gpu_pool/ALLOW_PARALLEL_WITH_P013 없음)은, 지원금 없는 쪽 17일째(2020-05-23)가 끝나면
# 두 번째 Neo4j 를 멈춘다. 그 전에 다른 시뮬레이션을 두 번째 Neo4j 에서 돌리고 있으면 크게 실패만 남긴다.
M=/data/multipolicy_v53_20260928/p013m_main2000/off
N2=/data/neo4j2-community-5.26.0
LOG=/data/gpu_pool/p013_neo2_guard.log
log() { printf '[%s] %s\n' "$(date -Is)" "$*" >> "$LOG"; }
log "지킴이 시작 — 기준 파일 $M/day_2020-05-23.json"
while true; do
  if [[ -e /data/gpu_pool/ALLOW_PARALLEL_WITH_P013 ]]; then log "동시 실행 승인 표시가 있다 — 지킴이 끝"; exit 0; fi
  if [[ -e $M/day_2020-05-23.json ]]; then
    if pgrep -af '[r]un_simulation.py' | grep -v 'p013m_main2000' | grep -q .; then
      log "FAIL: P013 밖의 시뮬레이션이 돌고 있다 — 두 번째 Neo4j 를 멈추지 못한다. 사람이 봐야 한다"
      echo "$(date -Is) 두 번째 Neo4j 정지 실패 — 다른 시뮬레이션 실행 중" > /data/gpu_pool/P013_GUARD.FAILED; exit 1
    fi
    if pgrep -f "CommunityEntryPoint --home-dir=$N2" >/dev/null; then
      NEO4J_HOME=$N2 NEO4J_CONF=$N2/conf "$N2/bin/neo4j" stop >> "$LOG" 2>&1
      sleep 5
      if pgrep -f "CommunityEntryPoint --home-dir=$N2" >/dev/null; then
        log "FAIL: 두 번째 Neo4j 가 멈추지 않았다"; echo "$(date -Is) 정지 실패" > /data/gpu_pool/P013_GUARD.FAILED; exit 1
      fi
      log "두 번째 Neo4j 를 멈췄다 — P013 이 지원금 있는 쪽으로 넘어갈 수 있다"
    else
      log "두 번째 Neo4j 는 이미 멈춰 있다"
    fi
    echo "$(date -Is)" > /data/gpu_pool/P013_GUARD.DONE; exit 0
  fi
  sleep 60
done
