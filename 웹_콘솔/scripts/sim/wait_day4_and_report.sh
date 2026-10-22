#!/bin/bash
# Day 4 (2026-05-28) jsonl 7500 도달 시 baseline 보고서 자동 생성
# 시뮬 계속 진행, 보고서는 백그라운드
cd "/g/내 드라이브/Kw/final_project"
set -a
. data/neo4j_load/.env
set +a
export LLM_MODE=qwen36_35b_a3b_awq

DAY4_JSONL="/c/Users/Administrator/sim_output_9d/metrics/day_2026-05-28.jsonl"
LOG="/c/Users/Administrator/sim_output_9d/baseline_report.log"
OUT="output/sim_9d_baseline/report/FINAL_REPORT_4D_BASELINE.md"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== wait_day4_and_report 시작 (PID=$$) ==="

# Day 4 jsonl 7500 도달 대기
until [ -f "$DAY4_JSONL" ] && [ "$(wc -l < "$DAY4_JSONL" 2>/dev/null)" -ge 7500 ]; do
    sleep 30
done

DAY4_LINES=$(wc -l < "$DAY4_JSONL")
log "Day 4 jsonl 7500 도달 ($DAY4_LINES lines)"

mkdir -p output/sim_9d_baseline/report

log "baseline 보고서 생성 시작 (start=2026-05-25, days=4)"
python3 -u scripts/sim/generate_baseline_report.py \
    --start 2026-05-25 --days 4 \
    --baseline \
    --out "$OUT" 2>&1 | tee -a "$LOG"

log "=== DONE === md=$OUT html=${OUT%.md}.html"
