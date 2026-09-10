#!/bin/bash
# 조합 A (workers 64 + max-model-len 8192 + max-num-seqs 384) — 30분 측정 후 종료
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=exaone40_32b_awq

PY='/c/ProgramData/Anaconda3/python.exe'
LOG="/c/Users/Administrator/sim_persona_compare/run_combo_a.log"
mkdir -p "$(dirname $LOG)"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

log "===== 조합 A 시뮬 (workers 64, max-len 8192) 시작 ====="

log "[Step 1] vLLM HTTP 응답 대기 (최대 30분)"
T0=$(date +%s)
until curl -sS -m 5 http://localhost:8000/v1/models 2>/dev/null | grep -q "EXAONE"; do
    NOW=$(date +%s)
    if [ $((NOW - T0)) -gt 1800 ]; then
        log "⚠️ vLLM 30분 timeout — 중단"
        exit 1
    fi
    sleep 20
done
log "vLLM 준비 완료"

log "[Step 2] Neo4j cleanup"
"$PY" -c "
import sys; sys.path.insert(0,'scripts/neo4j_load')
from _common import driver_session
with driver_session() as s:
    for label in ['Conversation','Memory','Plan','State','Agent']:
        total = 0
        while True:
            r = s.run(f'MATCH (n:{label}) WITH n LIMIT 5000 DETACH DELETE n RETURN count(*) AS n').single()['n']
            total += r
            if r == 0: break
        if total > 0: print(f'  {label}: {total}')
" 2>&1 | tee -a "$LOG"

log "[Step 3] 15,000 페르소나 (Drive Stream 손상으로 30k 대신 full/ 사용) → agents_final.json"
"$PY" -c "
import json
rows = [json.loads(l) for l in open('output/personas/full/A_rank_coupling_bdc_nvidia_v2.jsonl', encoding='utf-8')]
json.dump(rows, open('data/neo4j_load/agents/agents_final.json', 'w', encoding='utf-8'), ensure_ascii=False)
print(f'변환: {len(rows)}')
" 2>&1 | tee -a "$LOG"

log "[Step 4] 04/05/07/08 적재"
"$PY" scripts/neo4j_load/04_agents.py 2>&1 | tail -3 | tee -a "$LOG"
"$PY" scripts/neo4j_load/05_anchors.py 2>&1 | tail -3 | tee -a "$LOG"
"$PY" scripts/neo4j_load/07_initial_awareness.py 2>&1 | tail -3 | tee -a "$LOG"
"$PY" scripts/neo4j_load/08_initial_state.py 2>&1 | tail -3 | tee -a "$LOG"

log "[Step 5] 시뮬 시작 — workers 64"
SIM_T0=$(date +%s)
LLM_MODE=exaone40_32b_awq SIM_OUTPUT_DIR="C:/Users/Administrator/sim_combo_a" \
    nohup "$PY" -u scripts/sim/run_simulation.py \
    --start 2026-06-15 --days 1 --workers 64 \
    >> "$LOG" 2>&1 &
SIM_PID=$!
log "sim PID=$SIM_PID"

log "[Step 6] 30분 대기 + 속도 측정"
sleep 1800

JSONL="/c/Users/Administrator/sim_combo_a/metrics/day_2026-06-15.jsonl"
if [ -f "$JSONL" ]; then
    P=$(wc -l < "$JSONL")
    SIM_T1=$(date +%s)
    ELAPSED=$((SIM_T1 - SIM_T0))
    RATE=$("$PY" -c "print(round($P*60/$ELAPSED,2))")
    log "★ 30분 측정 결과:"
    log "  처리량: $P agents"
    log "  실시간 소요: ${ELAPSED}s = $((ELAPSED/60))m"
    log "  평균 속도: ${RATE}/min"
    log "  ETA 30,000 = $("$PY" -c "print(round(30000/$RATE/60,1))")h"
else
    log "⚠️ jsonl 없음 — sim 시작 fail"
fi

log "[Step 7] sim 종료"
kill -TERM $SIM_PID 2>&1 | tee -a "$LOG"
sleep 5

log "[Step 8] vLLM 종료"
wsl.exe -e bash -c 'ps aux | grep -iE "vllm|EngineCore" | grep -v grep | awk "{print \$2}" | xargs -r kill -9 2>&1' 2>&1

log "===== 완료 ====="
