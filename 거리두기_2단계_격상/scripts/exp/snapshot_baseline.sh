#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EXP-001 baseline 영구 덤프 — 정책(P010) 시행 직전(2025-07-20 종료) 스냅샷
#
# 논리: 정책 시행일 = 2025-07-21(Day8). 그 직전 = Day7(07-20) 종료 시점의 그래프
#       = 순수 baseline 7일치. v8_baseline_before_p009 와 동일한 과학적 근거.
# 안전: 시뮬을 resume-safe하게 중단 → Day8(07-21) 이후 오염 노드 제거(순수성 보장)
#       → Neo4j 정지·덤프·재기동 → 시뮬 resume(남은 날 이어서).
#
# 사용: bash snapshot_baseline.sh          # Day7 완료 확인 후 실행
#       FORCE=1 bash snapshot_baseline.sh  # 완료 확인 생략
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail
NAS=/home/ubuntu/data/exp001
DUMP_DIR=/home/ubuntu/data/dumps
NEO=/data/neo4j-community-5.26.0/bin
PW=exp001pass
OUT=$DUMP_DIR/neo4j_baseline_pre_p010_20250720.dump

d20=$(grep -c '"status": "ok"' "$NAS/sim_output/metrics/day_2025-07-20.jsonl" 2>/dev/null || echo 0)
echo "Day 2025-07-20 처리 완료: ${d20} agents"
if [ "${FORCE:-0}" != "1" ] && [ "${d20}" -lt 7400 ]; then
  echo "❌ Day7(07-20)이 아직 충분히 완료되지 않음(${d20}/7500). 대기 후 재실행 (또는 FORCE=1)."
  exit 1
fi

echo "── 1) 시뮬 중단 (resume-safe) ──"
pkill -f run_simulation.py 2>/dev/null || true
sleep 8
pkill -9 -f run_simulation.py 2>/dev/null || true
sleep 3

echo "── 2a) 07-21 이후 metrics/checkpoint 제거 (그래프와 동기화: resume가 재처리하도록) ──"
# 그래프 노드만 지우고 metrics/checkpoint를 남기면 resume가 '완료'로 착각해 건너뛰어
# 정책일 데이터 공백이 생긴다. 반드시 함께 제거.
rm -f "$NAS/sim_output/metrics/day_2025-07-21.jsonl"
rm -f "$NAS/sim_output/checkpoints/done_2025-07-21.json" "$NAS/sim_output/checkpoints/failed_2025-07-21.json"
ls "$NAS/sim_output/metrics/" | tail -2

echo "── 2b) 순수성 보장: 07-21 이후 오염 노드 제거 ──"
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=$PW PYTHONIOENCODING=utf-8
python3 - <<'PY'
import sys; sys.path.insert(0,'/data/exp001_repo/scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    for lbl in ["Plan","State","Memory","Conversation"]:
        n=s.run(f"MATCH (x:{lbl}) WHERE x.day >= date('2025-07-21') RETURN count(x) AS c").single()["c"]
        if n:
            s.run(f"MATCH (x:{lbl}) WHERE x.day >= date('2025-07-21') "
                  f"CALL (x) {{ DETACH DELETE x }} IN TRANSACTIONS OF 5000 ROWS")
        print(f"  {lbl}: 07-21+ 오염 제거 {n}")
    r=s.run("MATCH (st:State) RETURN min(toString(st.day)) AS mn, max(toString(st.day)) AS mx, count(st) AS c").single()
    print(f"  baseline State 범위: {r['mn']} ~ {r['mx']} (총 {r['c']})")
PY

echo "── 3) Neo4j 정지 → 덤프 → 재기동 ──"
$NEO/neo4j stop
mkdir -p "$DUMP_DIR"
rm -f "$DUMP_DIR/neo4j.dump"
$NEO/neo4j-admin database dump neo4j --to-path="$DUMP_DIR" --overwrite-destination=true
mv "$DUMP_DIR/neo4j.dump" "$OUT"
$NEO/neo4j start
for i in $(seq 1 40); do
  $NEO/cypher-shell -a bolt://localhost:7687 -u neo4j -p $PW 'RETURN 1' >/dev/null 2>&1 && { echo "  Neo4j READY"; break; }
  sleep 3
done

echo "── 4) 시뮬 resume (Day8~ 이어서) ──"
cd /data/exp001_repo
WORKERS=${WORKERS:-64} nohup bash /data/exp001/launch_exp001.sh >> "$NAS/logs/run.log" 2>&1 &
echo "  resume PID=$!"
echo "✅ baseline 덤프 완료:"
ls -lh "$OUT"
