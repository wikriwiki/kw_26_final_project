#!/bin/bash
# Day 7 (2026-05-31 일) Night2 완료 시 sim 정지 → P009 정책 INSERT → sim restart with --days 14
# treatment: 6/1~6/7 (7일), baseline 5/25~5/31 (7일), 총 14일
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=qwen36_35b_a3b_awq

LOG="/c/Users/Administrator/sim_output_9d/inject_policy.log"
SIM_LOG="/c/Users/Administrator/sim_output_9d/sim_9d.log"
DAY7_JSONL="/c/Users/Administrator/sim_output_9d/metrics/day_2026-05-31.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
log "=== wait_day7_inject_policy 시작 (PID=$$) ==="

# Step 1: Day 7 jsonl 7500 도달 대기
log "Day 7 (2026-05-31) jsonl 7500 도달 대기..."
until [ -f "$DAY7_JSONL" ] && [ "$(wc -l < "$DAY7_JSONL" 2>/dev/null)" -ge 7500 ]; do
    sleep 60
done
log "Day 7 jsonl $(wc -l < "$DAY7_JSONL") 도달"

# Step 2: Day 7 Night2 완료 대기 (Day 8 시작 직전)
log "Day 7 Night2 완료 대기 ('Day 8 2026-06-01 processing' 신호)..."
until grep -q "Day [0-9]\+ 2026-06-01.*processing" "$SIM_LOG" 2>/dev/null; do
    sleep 30
done
log "Day 8 시작 신호 감지 — Day 7 Night2 완료됨"

# Step 3: sim process 정지 (Day 8 진행 시작했지만 초기라 안전)
log "sim 정지 — Day 8 진행 초기에 정책 주입"
ps -ef 2>/dev/null | grep -E "run_simulation|run_9d_baseline" | grep -v grep | awk '{print $2}' | xargs -r kill -9 2>/dev/null
sleep 5

# Step 4: Day 8 진행분 정리 (정책 없이 진행된 잔재 삭제)
log "Day 8 jsonl + Plan/State 잔재 정리 (정책 없이 진행된 부분)..."
rm -f /c/Users/Administrator/sim_output_9d/metrics/day_2026-06-01.jsonl
rm -f /c/Users/Administrator/sim_output_9d/checkpoints/done_2026-06-01.json
rm -f /c/Users/Administrator/sim_output_9d/checkpoints/failed_2026-06-01.json
python3 -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    p = s.run(\"MATCH (p:Plan) WHERE p.day = date('2026-06-01') DETACH DELETE p RETURN count(p) AS c\").single()['c']
    st = s.run(\"MATCH (st:State) WHERE st.day = date('2026-06-01') DETACH DELETE st RETURN count(st) AS c\").single()['c']
    print(f'Day 8 Plan 삭제: {p}, State 삭제: {st}')
" 2>&1 | tee -a "$LOG"

# Step 5: P009 정책 노드 INSERT + 25 자치구 applied_to 연결
log "P009 정책 INSERT (effective 2026-06-01 ~ 2026-06-07)..."
python3 -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    s.run('''
        MERGE (p:Policy {id: 'P009'})
        SET p.name = '소득 차등 지원금',
            p.type = 'grant',
            p.effective_from = date('2026-06-01'),
            p.effective_until = date('2026-06-07'),
            p.income_grants = '{\"중상\":100000,\"중\":250000,\"중하\":450000,\"하\":600000}',
            p.excluded_income = '[\"상\"]',
            p.target_all_seoul = true
    ''')
    n_app = s.run('''
        MATCH (p:Policy {id: 'P009'})
        MATCH (d:District)
        MERGE (p)-[:applied_to]->(d)
        RETURN count(d) AS n
    ''').single()['n']
    print(f'P009 INSERT 완료, applied_to {n_app} District')
    # 검증
    r = s.run(\"MATCH (p:Policy {id:'P009'}) RETURN p.effective_from AS f, p.effective_until AS u, p.income_grants AS ig\").single()
    print(f'정책 검증: from={r[\"f\"]}, until={r[\"u\"]}, grants={r[\"ig\"]}')
" 2>&1 | tee -a "$LOG"

# Step 6: sim restart with --days 14 (5/25~6/7), resume guard로 Day 1~7 skip
log "sim restart with --days 14 (resume guard로 Day 1~7 skip, Day 8부터 정책 적용)"
nohup bash -c '
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=qwen36_35b_a3b_awq
export SIM_OUTPUT_DIR="C:/Users/Administrator/sim_output_9d"
echo "[$(date)] === 14D SIM START (baseline 7 + treatment 7) ===" >> "C:/Users/Administrator/sim_output_9d/sim_9d.log"
python3 -u scripts/sim/run_simulation.py --start 2026-05-25 --days 14 --workers 32 >> "C:/Users/Administrator/sim_output_9d/sim_9d.log" 2>&1
echo "[$(date)] === 14D SIM EXIT code=$? ===" >> "C:/Users/Administrator/sim_output_9d/sim_9d.log"
' > /dev/null 2>&1 &
RESTART_PID=$!
log "sim restart PID=$RESTART_PID"

log "=== DONE — Day 8(6/1)부터 P009 정책 적용 시작, 6/7까지 ==="
