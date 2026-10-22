#!/bin/bash
# 7,500 + 30,000 페르소나 각각 1일 시뮬 (월요일 2026-06-15) + 시간 측정
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=exaone40_32b_awq

LOG="/c/Users/Administrator/sim_persona_compare/run.log"
mkdir -p "$(dirname $LOG)"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

AGENTS_JSON_PATH="data/neo4j_load/agents/agents_final.json"
AGENTS_BAK="data/neo4j_load/agents/agents_final.json.bak_orig"

# 원본 백업 1회
if [ ! -f "$AGENTS_BAK" ] && [ -f "$AGENTS_JSON_PATH" ]; then
    cp "$AGENTS_JSON_PATH" "$AGENTS_BAK"
fi

run_sim_phase() {
    local LABEL=$1
    local PERSONA_JSONL=$2
    local OUT_DIR=$3

    log "===== $LABEL 시작 ====="
    T0=$(date +%s)

    log "[$LABEL Step 1] Agent/Plan/State cleanup"
    PYTHONIOENCODING=utf-8 python3 -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    for label in ['Conversation','Memory','Plan','State','Agent']:
        total = 0
        while True:
            r = s.run(f'MATCH (n:{label}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS n').single()['n']
            total += r
            if r == 0: break
        print(f'  {label}: {total}')
" 2>&1 | tee -a "$LOG"

    log "[$LABEL Step 2a] jsonl → agents_final.json 변환"
    python3 -c "
import json
rows = [json.loads(l) for l in open('$PERSONA_JSONL', encoding='utf-8')]
json.dump(rows, open('$AGENTS_JSON_PATH', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'  변환: {len(rows)} agents')
" 2>&1 | tee -a "$LOG"

    log "[$LABEL Step 2b] 04_agents 적재"
    PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/04_agents.py 2>&1 | tail -10 | tee -a "$LOG"

    log "[$LABEL Step 2c] 05_anchors 적재"
    PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/05_anchors.py 2>&1 | tail -10 | tee -a "$LOG"

    log "[$LABEL Step 2d] 07_initial_awareness 적재"
    PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/07_initial_awareness.py 2>&1 | tail -5 | tee -a "$LOG"

    log "[$LABEL Step 2e] 08_initial_state 적재"
    PYTHONIOENCODING=utf-8 python3 scripts/neo4j_load/08_initial_state.py 2>&1 | tail -5 | tee -a "$LOG"

    log "[$LABEL] Agent 적재 검증"
    PYTHONIOENCODING=utf-8 python3 -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    for L in ['Agent','State']:
        n = s.run(f'MATCH (n:{L}) RETURN count(n) AS n').single()['n']
        print(f'  {L}={n}')
" 2>&1 | tee -a "$LOG"

    log "[$LABEL Step 3] 1일 시뮬 시작 (2026-06-15 월)"
    SIM_T0=$(date +%s)
    LLM_MODE=exaone40_32b_awq SIM_OUTPUT_DIR="$OUT_DIR" \
        python3 -u scripts/sim/run_simulation.py \
        --start 2026-06-15 --days 1 --workers 32 \
        >> "$LOG" 2>&1
    SIM_T1=$(date +%s)
    SIM_ELAPSED=$((SIM_T1 - SIM_T0))
    log "[$LABEL] ★ 시뮬 소요: ${SIM_ELAPSED}s = $((SIM_ELAPSED/60))m $((SIM_ELAPSED%60))s"

    log "[$LABEL Step 4] Neo4j dump"
    PYTHONIOENCODING=utf-8 python3 -u scripts/sim/_backup_current_state.py 2>&1 | tail -5 | tee -a "$LOG"

    T1=$(date +%s)
    TOTAL=$((T1 - T0))
    log "===== $LABEL 완료: 총 ${TOTAL}s = $((TOTAL/60))m $((TOTAL%60))s ====="
    log ""
}

log "===== 페르소나 비교 시뮬 시작 (월요일 2026-06-15, 1일) ====="

run_sim_phase "7500" \
    "output/personas/full_7500_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl" \
    "C:/Users/Administrator/sim_persona_7500"

run_sim_phase "30000" \
    "output/personas/full_30000_exaone40/A_rank_coupling_bdc_nvidia_v2.jsonl" \
    "C:/Users/Administrator/sim_persona_30000"

log "===== 전체 완료 ====="
log "★ 시간 요약:"
grep "★ 시뮬 소요" "$LOG"
