#!/bin/bash
# 6/1 Night2 단독 redo (fixed FETCH_POLICY_CYPHER bug) → main sim restart (--days 14, resume guard)
cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=qwen36_35b_a3b_awq
export SIM_OUTPUT_DIR="C:/Users/Administrator/sim_output_9d"

REDO_LOG="/c/Users/Administrator/sim_output_9d/redo_6_1_night2.log"
SIM_LOG="/c/Users/Administrator/sim_output_9d/sim_9d.log"

# ─── 1) 6/1 Night2 redo ───────────────────────────────
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === REDO 6/1 NIGHT2 START ===" | tee -a "$REDO_LOG"
python3 -u scripts/sim/_redo_6_1_night2.py >> "$REDO_LOG" 2>&1
REDO_EXIT=$?
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === REDO EXIT=$REDO_EXIT ===" | tee -a "$REDO_LOG"
if [ $REDO_EXIT -ne 0 ]; then
    echo "REDO FAILED — abort sim restart" | tee -a "$REDO_LOG"
    exit 1
fi

# ─── 2) Main sim restart (resume guard로 5/25~6/1 skip, 6/2부터 진행) ───
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === MAIN SIM RESTART (--start 2026-05-25 --days 14) ===" >> "$SIM_LOG"
python3 -u scripts/sim/run_simulation.py --start 2026-05-25 --days 14 --workers 32 >> "$SIM_LOG" 2>&1
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === SIM EXIT $? ===" >> "$SIM_LOG"
