#!/bin/bash
# 30,000 페르소나 적재 + 1일 시뮬 (월요일 2026-06-15) + speculative decoding
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=exaone35_78b_awq

LOG="/c/Users/Administrator/sim_persona_compare/run_30k.log"
mkdir -p "$(dirname $LOG)"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

T0=$(date +%s)
log "===== 30000 + speculative 시뮬 시작 ====="

log "[Step 1] vLLM HTTP 응답 대기"
until curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "EXAONE"; do
    sleep 30
done
log "vLLM 준비 완료"

log "[Step 2] jsonl → agents_final.json 변환"
python3 -c "
import json
rows = [json.loads(l) for l in open('output/personas/full_30000_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl', encoding='utf-8')]
json.dump(rows, open('data/neo4j_load/agents/agents_final.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'변환: {len(rows)} agents')
" 2>&1 | tee -a "$LOG"

log "[Step 3] 04_agents 적재"
PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/04_agents.py 2>&1 | tail -5 | tee -a "$LOG"

log "[Step 4] 05_anchors 적재"
PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/05_anchors.py 2>&1 | tail -5 | tee -a "$LOG"

log "[Step 5] 07_initial_awareness 적재"
PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/07_initial_awareness.py 2>&1 | tail -5 | tee -a "$LOG"

log "[Step 6] 08_initial_state 적재"
PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/08_initial_state.py 2>&1 | tail -5 | tee -a "$LOG"

PYTHONIOENCODING=utf-8 python3 -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    for L in ['Agent','State']:
        n = s.run(f'MATCH (n:{L}) RETURN count(n) AS n').single()['n']
        print(f'  {L}={n}')
" 2>&1 | tee -a "$LOG"

log "[Step 7] 1일 시뮬 시작 (2026-06-15 월) — speculative decoding 적용"
SIM_T0=$(date +%s)
LLM_MODE=exaone35_78b_awq SIM_OUTPUT_DIR="C:/Users/Administrator/sim_persona_30000_spec" \
    python3 -u scripts/sim/run_simulation.py \
    --start 2026-06-15 --days 1 --workers 32 \
    >> "$LOG" 2>&1
SIM_T1=$(date +%s)
SIM_ELAPSED=$((SIM_T1 - SIM_T0))
log "★ 시뮬 소요: ${SIM_ELAPSED}s = $((SIM_ELAPSED/60))m $((SIM_ELAPSED%60))s"

log "[Step 8] Neo4j dump"
PYTHONIOENCODING=utf-8 python3 -u scripts/sim/_backup_current_state.py 2>&1 | tail -5 | tee -a "$LOG"

T1=$(date +%s)
TOTAL=$((T1 - T0))
log "===== 30000 + spec 완료: 총 ${TOTAL}s = $((TOTAL/60))m $((TOTAL%60))s ====="
