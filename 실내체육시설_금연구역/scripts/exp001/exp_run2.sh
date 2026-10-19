#!/usr/bin/env bash
# 실험 러너 v2 — 순차 전후 설계용.
#   인자: NAME LIMIT DAYS POLICY START RESTORE
#     POLICY  : 정책 json 경로 | none        (none 이면 무정책 런)
#     START   : 시뮬 시작일 (예: 2025-07-14)
#     RESTORE : 덤프 경로 | skip             (skip 이면 현재 그래프에 이어서 실행)
#
# exp_run.sh 와의 차이: 시작일 지정 가능, 덤프 재적재 생략 가능.
# 순차 설계에서 2구간(정책 주간)은 반드시 RESTORE=skip 으로 띄워야 1구간이 보존된다.
set -euo pipefail
NAME="${1:-X0}"; LIMIT="${2:-200}"; DAYS="${3:-7}"
POLF="${4:-none}"; START="${5:-2025-07-14}"
RESTORE="${6:-/data/dumps/neo4j_base_day0.dump}"
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
source /data/venv/bin/activate
cd /data/exp001_repo
pkill -9 -f "run_simulation[.]py" 2>/dev/null || true; sleep 2

export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=${NEO4J_PASSWORD:-changeme}
export NEO4J_DATABASE=neo4j PYTHONIOENCODING=utf-8

if [ "$RESTORE" != "skip" ]; then
  echo "  덤프 복원: $RESTORE"
  $NEO/bin/neo4j stop >/dev/null 2>&1 || true
  mkdir -p /data/dumps_restore && cp "$RESTORE" /data/dumps_restore/neo4j.dump
  $NEO/bin/neo4j-admin database load neo4j --from-path=/data/dumps_restore \
      --overwrite-destination=true >/dev/null 2>&1
  $NEO/bin/neo4j start >/dev/null 2>&1
  for i in $(seq 1 60); do
    $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p "$NEO4J_PASSWORD" 'RETURN 1' \
      >/dev/null 2>&1 && break; sleep 3; done
  # 복원 직후에만 실행 — 이어달리기(skip)에서는 에이전트 속성을 다시 건드리면 안 된다.
  python3 /data/exp001/reconcile_income_label.py 2>&1 | sed 's/^/  [라벨] /'
  python3 /data/exp001/fix_spending_anchor.py    2>&1 | sed 's/^/  [앵커] /'
  python3 /data/exp001/load_agent_cats.py        2>&1 | sed 's/^/  [업종] /'
else
  echo "  덤프 복원 생략 — 현재 그래프에 이어서 실행"
fi

if [ "$POLF" != "none" ]; then
  python3 scripts/neo4j_load/10_load_grant_policy.py "$POLF" >/dev/null 2>&1
  echo "  정책 주입: $POLF"
else
  echo "  무정책 런"
fi

export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
export SIM_OUTPUT_DIR=/data/exp001/out_$NAME PYTHONUNBUFFERED=1
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start "$START" --days "$DAYS" \
  --limit "$LIMIT" --workers 48 > /data/exp001/run_${NAME}.log 2>&1 &
echo "  실험 $NAME 시작 PID=$! (LIMIT=$LIMIT DAYS=$DAYS START=$START OUT=$SIM_OUTPUT_DIR)"
