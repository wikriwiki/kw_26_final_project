#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# EXP-001 GPU LIVE 원샷 세팅 — EXAONE-4.5-33B-AWQ(TP2) + 신규 Neo4j + 14일 런
#
# 대상: GPU LIVE 워크로드 (A100 80GB×2, ubuntu, sudo NOPASSWD, /data=스토리지 마운트 가정)
# 사용: bash setup_gpulive_exp001.sh all          # 전체
#       bash setup_gpulive_exp001.sh s3_neo4j     # 특정 단계만 (s1~s9)
#
# 사전 준비(로컬에서 업로드해야 하는 것):
#   /data/kw_26_final_project/          ← 코드 (scp/rsync)
#   /data/dumps/neo4j.dump              ← neo4j_3day_p009_20260601_1515.dump 를 이 이름으로
# ═══════════════════════════════════════════════════════════════════════════
set -euo pipefail

BASE=/data
REPO=$BASE/kw_26_final_project
VENV=$BASE/venv
NEO4J_VER=5.26.0
NEO4J_HOME=$BASE/neo4j-community-$NEO4J_VER
NEO4J_PW=${NEO4J_PW:-exp001pass}
export HF_HOME=$BASE/hf_cache
LOG=$BASE/exp001
mkdir -p "$LOG" "$BASE/dumps" "$HF_HOME"

# EXP-001 실험 파라미터 (docs/EXPERIMENT_LOG.md 장표와 일치시킬 것)
# 실제 정책일 기준: 민생회복 소비쿠폰 1차 신청·지급 개시 = 2025-07-21(월)
# 14일 = 시행 전 7일(07-14 월~07-20 일) + 시행 후 7일(07-21 월~07-27 일) — 월~일 2주 정렬
SIM_START=2025-07-14
SIM_DAYS=14
DAY_ZERO=2025-07-13           # = SIM_START - 1
WORKERS=${WORKERS:-32}
AGENTS=${AGENTS:-7500}        # EXP-001 표본 (미지정 시 LIVES_AT 전체 ~14.7k 실행되므로 명시 필수)
LLM_PORT=8000

env_common() {
  export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=$NEO4J_PW NEO4J_DATABASE=neo4j
  export LLM_BASE_URL=http://localhost:$LLM_PORT/v1 LLM_MODE=exaone_4_5
  export SIM_OUTPUT_DIR=$LOG/sim_output PYTHONUNBUFFERED=1
  source $VENV/bin/activate
}

s1_deps() {  # 파이썬 환경 + vLLM(최신 — AWQ schema 지원) + 런타임
  echo "══ s1: deps ══"
  python3 -m venv $VENV || true
  source $VENV/bin/activate
  pip install -q -U pip
  pip install -q -U vllm huggingface_hub \
      "neo4j>=5.20,<7" "openai>=1.54,<2" "pydantic>=2.7,<3" \
      "pyyaml>=6.0" "openpyxl>=3.1" "requests>=2.31"
  python -c "import vllm; print('vllm', vllm.__version__)" | tee $LOG/versions.txt
  nvidia-smi --query-gpu=name,memory.total --format=csv | tee -a $LOG/versions.txt
  sudo apt-get update -qq && sudo apt-get install -y -qq openjdk-17-jre-headless > /dev/null
  java -version 2>&1 | head -1 | tee -a $LOG/versions.txt
}

s2_model() {  # 모델 프리페치 (스토리지 캐시 — 워크로드 재생성에도 보존)
  echo "══ s2: EXAONE-4.5-33B-AWQ 다운로드 (HF_HOME=$HF_HOME) ══"
  source $VENV/bin/activate
  huggingface-cli download LGAI-EXAONE/EXAONE-4.5-33B-AWQ --quiet | tail -1
}

s3_neo4j() {  # Neo4j 5.26 신규 설치 + 덤프 로드 + 기동
  echo "══ s3: Neo4j 신규 세팅 ══"
  [ -d $NEO4J_HOME ] || { cd $BASE && wget -q https://dist.neo4j.org/neo4j-community-$NEO4J_VER-unix.tar.gz && tar xzf neo4j-community-$NEO4J_VER-unix.tar.gz; }
  grep -q "^server.directories.data=$BASE/neo4j_data" $NEO4J_HOME/conf/neo4j.conf || \
    echo "server.directories.data=$BASE/neo4j_data" >> $NEO4J_HOME/conf/neo4j.conf
  # Community는 파일명이 <db명>.dump 여야 함
  [ -f $BASE/dumps/neo4j.dump ] || { echo "❌ $BASE/dumps/neo4j.dump 없음 — 로컬에서 업로드 필요"; exit 1; }
  $NEO4J_HOME/bin/neo4j stop 2>/dev/null || true
  $NEO4J_HOME/bin/neo4j-admin database load neo4j --from-path=$BASE/dumps --overwrite-destination=true
  $NEO4J_HOME/bin/neo4j-admin dbms set-initial-password $NEO4J_PW 2>/dev/null || true
  $NEO4J_HOME/bin/neo4j start
  for i in $(seq 1 40); do
    $NEO4J_HOME/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p $NEO4J_PW 'RETURN 1' >/dev/null 2>&1 && { echo "Neo4j READY"; return; }
    sleep 3
  done
  echo "❌ Neo4j 기동 실패 — $NEO4J_HOME/logs/neo4j.log 확인"; exit 1
}

s4_reset() {  # 포스트런 덤프 → clean Day0 (런 산출물 제거)
  echo "══ s4: 런 산출물 리셋 ══"
  env_common; cd $REPO
  python scripts/neo4j_load/97_reset_run_artifacts.py --dry-run
  python scripts/neo4j_load/97_reset_run_artifacts.py
}

s5_seed() {  # Day0 State 시드 (시작일-1)
  echo "══ s5: Day0 State 시드 (DAY_ZERO=$DAY_ZERO) ══"
  env_common; cd $REPO
  DAY_ZERO=$DAY_ZERO python scripts/neo4j_load/08_initial_state.py
}

s6_policy() {  # 정책(P010) 적재 + 쿠폰 사용처 백필(실측 가맹점 CSV + 룰)
  echo "══ s6: P010 적재 + coupon_eligible 백필 ══"
  env_common; cd $REPO
  python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P010.json
  python scripts/neo4j_load/09_coupon_eligibility.py --merchants data/coupon | tee $LOG/coupon_backfill.txt
}

s7_llm() {  # LLM 서버 (A100×2 TP2) — nohup 상주
  echo "══ s7: EXAONE 서빙 (TP=2) ══"
  source $VENV/bin/activate; export HF_HOME=$BASE/hf_cache
  pkill -f vllm.entrypoints || true; sleep 3
  nohup bash $REPO/scripts/serve/serve_exaone45_awq_a100x2.sh > $LOG/llm.log 2>&1 &
  echo "  로그: $LOG/llm.log — 모델 로드 대기(최대 20분)..."
  for i in $(seq 1 120); do
    curl -sf http://localhost:$LLM_PORT/v1/models >/dev/null 2>&1 && { echo "LLM READY"; curl -s http://localhost:$LLM_PORT/v1/models | head -c 300; echo; return; }
    sleep 10
  done
  echo "❌ LLM 기동 실패 — tail $LOG/llm.log:"; tail -20 $LOG/llm.log
  echo "   AWQ schema 이슈면 FP8 폴백: MODEL=LGAI-EXAONE/EXAONE-4.5-33B-FP8 QUANT=fp8 bash s7 재실행"
  exit 1
}

s8_preflight() {  # 정책 사전점검 + LLM 헬스 (스모크)
  echo "══ s8: preflight ══"
  env_common; cd $REPO
  python scripts/sim/policy_preflight.py data/neo4j_load/policies/P010.json
  python - <<'PY'
import sys; sys.path.insert(0, 'scripts/sim')
from llm_client import call_chat
r = call_chat(None, "당신은 테스트 봇.", "'준비완료' 네 글자만 출력.", temperature=0.0, max_tokens=10)
print("LLM 스모크:", r.choices[0].message.content.strip())
PY
}

s9_run() {  # 14일 본 런 (일별 타이밍 자동 로그) — nohup
  echo "══ s9: 14일 시뮬 (start=$SIM_START days=$SIM_DAYS agents=$AGENTS workers=$WORKERS) ══"
  env_common; cd $REPO
  nohup python scripts/sim/run_simulation.py \
      --start $SIM_START --days $SIM_DAYS --limit $AGENTS --workers $WORKERS \
      > $LOG/run.log 2>&1 &
  echo "  진행: tail -f $LOG/run.log"
  echo "  일별 타이밍(장표 기입용): grep 'done in' $LOG/run.log"
}

case "${1:-all}" in
  all) s1_deps; s2_model; s3_neo4j; s4_reset; s5_seed; s6_policy; s7_llm; s8_preflight; s9_run ;;
  s1_deps|s2_model|s3_neo4j|s4_reset|s5_seed|s6_policy|s7_llm|s8_preflight|s9_run) "$1" ;;
  timing) grep -E "\[Day .*done in" $LOG/run.log ;;
  *) echo "usage: $0 [all|s1_deps|s2_model|s3_neo4j|s4_reset|s5_seed|s6_policy|s7_llm|s8_preflight|s9_run|timing]"; exit 1 ;;
esac
