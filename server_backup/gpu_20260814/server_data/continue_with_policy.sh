#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# 대조군(nopolicy) 이어달리기 → P010 정책충격 주입 + 우리 개선코드로 14일 런
#
# 전제: 정책 OFF 대조군(exp001_nopolicy)이 14일 완주해 Neo4j에 State·Memory가
#       누적된 상태. 그 상태를 리셋하지 않고(!!) 이어받아, 다음날 P010을 주입하고
#       14일 더 돌려 same-seed 반사실(정책 ON) 비교를 만든다.
#
# 안전장치(메가존 안내 대응):
#   - 실행 전 Neo4j를 NAS(dumps_continue)에 dump 백업 (워크로드 죽어도 복구)
#   - 매일 하루 끝에 State/metrics를 NAS(BACKUP_DIR)에 저장 (_daily_backup)
#   - 병목 계측: Stage2 5단계 + Dawn 메모리(DAWN_MEM_COUNT=1) 세부 로그
#
# 사용: bash continue_with_policy.sh            # 전체
#       bash continue_with_policy.sh check       # 대조군 종료 여부만 확인
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

NAS=/home/ubuntu/data
REPO=/data/exp001_repo                 # 현재(대조군) 실행 repo — 종료 후 우리 코드로 갱신
OURS=$NAS/ours_staging                 # 우리 개선코드 스테이징 (NAS)
NEO=/data/neo4j-community-5.26.0
NEO4J_PW=exp001pass
VENV=/data/venv                        # 시뮬 클라이언트 venv (sglang venv 아님)

# 실험 파라미터
SIM_START=2025-07-28                   # 대조군 마지막날(07-27) 다음날 = 정책 주입일
SIM_DAYS=14                            # 2주
LIMIT=7500                             # 대조군과 동일 표본
WORKERS=${WORKERS:-64}
OUT=$NAS/exp001_policy_cont/sim_output
BK=$NAS/exp001_policy_cont/backup
POLICY=$REPO/data/neo4j_load/policies/P010_continue.json

env_common() {
  export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=$NEO4J_PW NEO4J_DATABASE=neo4j
  export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
  export SIM_OUTPUT_DIR=$OUT PYTHONUNBUFFERED=1
  export BACKUP_DIR=$BK                 # ★ 매일 NAS 백업 켜기
  export STAGE2_SLOW_SEC=45             # 느린 Stage2 상세 로그 임계
  export DAWN_MEM_COUNT=1               # ★ 에이전트 메모리 누적 병목 계측(총 Memory 수)
  source $VENV/bin/activate
}

check_control_done() {
  # 대조군 프로세스가 살아있으면 아직 진행중
  if ps -eo cmd | grep -q "[r]un_simulation.py"; then
    echo "⏳ 대조군(nopolicy) 아직 실행 중 — 종료 후 재실행하세요."
    ls -t $NAS/exp001_nopolicy/sim_output/metrics/day_*.jsonl 2>/dev/null | head -1 | \
      xargs -I{} sh -c 'echo "   최신: $(basename {}) $(wc -l < {})행"'
    return 1
  fi
  # 대조군 14일(07-14~07-27) 완료 확인
  local last=$NAS/exp001_nopolicy/sim_output/metrics/day_2025-07-27.jsonl
  if [ ! -f "$last" ]; then
    echo "⚠️ 대조군 마지막날(07-27) metrics 없음 — 대조군이 14일 완주하지 않았을 수 있음. 확인 필요."
    return 1
  fi
  echo "✅ 대조군 종료 확인 (07-27: $(wc -l < $last)행)"
  return 0
}

deploy_our_code() {
  echo "══ 우리 개선코드 반영 (staging → repo) ══"
  cp $OURS/scripts/sim/*.py $REPO/scripts/sim/
  cp $OURS/scripts/neo4j_load/10_load_grant_policy.py $REPO/scripts/neo4j_load/
  cp $OURS/data/neo4j_load/policies/P010_continue.json $REPO/data/neo4j_load/policies/
  echo "  복사 완료: sim/*.py(계측·백업·10분위) + 10_load + P010_continue.json"
}

backup_before() {
  echo "══ 이어달리기 직전 Neo4j → NAS dump 백업 ══"
  mkdir -p $NAS/dumps_continue
  $NEO/bin/neo4j stop || true
  $NEO/bin/neo4j-admin database dump neo4j --to-path=$NAS/dumps_continue --overwrite-destination=true
  $NEO/bin/neo4j start
  for i in $(seq 1 40); do
    $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p $NEO4J_PW 'RETURN 1' >/dev/null 2>&1 && { echo "  Neo4j READY"; break; }
    sleep 3
  done
}

inject_policy() {
  echo "══ P010 정책 주입 (10분위 균등 15만, effective 2025-07-28) — 리셋 없음(State·Memory 보존) ══"
  env_common; cd $REPO
  python scripts/neo4j_load/10_load_grant_policy.py "$POLICY"
  # coupon_eligible은 대조군에서 이미 백필됨(537,489 POI) → 재백필 불필요. 확인만.
  $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p $NEO4J_PW \
    "MATCH (p:POI) WHERE p.coupon_eligible IS NOT NULL RETURN count(p) AS coupon_labeled" 2>/dev/null | tail -1
  echo "-- preflight --"
  python scripts/sim/policy_preflight.py "$POLICY"
}

run_sim() {
  echo "══ 정책 ON 14일 런 (start=$SIM_START days=$SIM_DAYS workers=$WORKERS, 매일백업+병목계측) ══"
  env_common; cd $REPO
  mkdir -p $OUT $BK
  nohup python scripts/sim/run_simulation.py \
      --start $SIM_START --days $SIM_DAYS --limit $LIMIT --workers $WORKERS \
      > $NAS/exp001_policy_cont/run.log 2>&1 &
  echo "  PID $! — 진행: tail -f $NAS/exp001_policy_cont/run.log"
  echo "  일별 병목요약: grep -E 'Stage2 병목|Dawn 병목|done in' $NAS/exp001_policy_cont/run.log"
}

case "${1:-all}" in
  check)   check_control_done ;;
  all)     check_control_done && deploy_our_code && backup_before && inject_policy && run_sim ;;
  deploy)  deploy_our_code ;;
  backup)  backup_before ;;
  policy)  inject_policy ;;
  run)     run_sim ;;
  *) echo "usage: $0 [all|check|deploy|backup|policy|run]"; exit 1 ;;
esac
