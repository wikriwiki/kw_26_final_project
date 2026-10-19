#!/usr/bin/env bash
# 실험 러너 — 인자: 실험명 LIMIT DAYS [POLICY_FILE]
# 매번 깨끗한 baseline 복원 + 정책 주입 후 실행.
# 출력은 NFS(/data/exp001/out_<name>)에 둔다 — 컨테이너 재생성에도 생존.
set -euo pipefail
NAME="${1:-R0}"; LIMIT="${2:-300}"; DAYS="${3:-3}"
POLF="${4:-data/neo4j_load/policies/P010.json}"
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
source /data/venv/bin/activate
cd /data/exp001_repo
pkill -9 -f run_simulation.py 2>/dev/null || true; sleep 2

# baseline 복원
$NEO/bin/neo4j stop >/dev/null 2>&1 || true
mkdir -p /data/dumps_restore && cp /data/dumps/neo4j_baseline_pre_p010_20250720.dump /data/dumps_restore/neo4j.dump
$NEO/bin/neo4j-admin database load neo4j --from-path=/data/dumps_restore --overwrite-destination=true >/dev/null 2>&1
$NEO/bin/neo4j start >/dev/null 2>&1
for i in $(seq 1 60); do $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' >/dev/null 2>&1 && break; sleep 3; done

export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass NEO4J_DATABASE=neo4j PYTHONIOENCODING=utf-8

# 정책 주입 (POLF=none 이면 무정책 baseline 런)
if [ "$POLF" != "none" ]; then
  python3 scripts/neo4j_load/10_load_grant_policy.py "$POLF" >/dev/null 2>&1
  echo "  정책 주입: $POLF"
else
  echo "  무정책 baseline 런"
fi
python3 /data/exp001/reconcile_income_label.py 2>&1 | sed 's/^/  [라벨] /'
python3 /data/exp001/fix_spending_anchor.py 2>&1 | sed "s/^/  [앵커] /"
python3 /data/exp001/load_agent_cats.py 2>&1 | sed "s/^/  [업종] /"

export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
export SIM_OUTPUT_DIR=/data/exp001/out_$NAME PYTHONUNBUFFERED=1
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-21 --days "$DAYS" --limit "$LIMIT" --workers 48 \
  > /data/exp001/run_${NAME}.log 2>&1 &
echo "  실험 $NAME 시작 PID=$! (LIMIT=$LIMIT DAYS=$DAYS OUT=$SIM_OUTPUT_DIR)"
