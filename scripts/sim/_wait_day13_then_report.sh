#!/bin/bash
# 6/6 (Day 13) 완료 대기 → 최종 보고서 자동 생성
# Day 14 sim은 계속 진행 (간섭 없음)

cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=qwen36_35b_a3b_awq

LOG="/c/Users/Administrator/sim_output_9d/wait_day13_report.log"
SIM_LOG="/c/Users/Administrator/sim_output_9d/sim_9d.log"
DAY13_JSONL="/c/Users/Administrator/sim_output_9d/metrics/day_2026-06-06.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== wait_day13_then_report 시작 (PID=$$) ==="

# Step 1: 6/6 jsonl 7500 도달 대기 (Day phase 완료)
log "Step 1: Day 13 (6/6) jsonl 7500 도달 대기..."
until [ -f "$DAY13_JSONL" ] && [ "$(wc -l < "$DAY13_JSONL" 2>/dev/null)" -ge 7500 ]; do
    sleep 120
done
log "Day 13 jsonl $(wc -l < "$DAY13_JSONL") 도달"

# Step 2: 6/6 Night2 완료 대기 (Day 14 시작 신호)
log "Step 2: 6/6 Night2 완료 대기 (Day 14 2026-06-07 processing 신호)..."
until grep -q "Day [0-9]\+ 2026-06-07.*processing" "$SIM_LOG" 2>/dev/null; do
    sleep 60
done
log "Day 14 시작 신호 감지 — 6/6 Night2 완료됨"

# Step 3: 최종 보고서 생성 (5/25~6/6, 13 days, P009 from 6/1)
log "Step 3: 최종 보고서 생성 (5/25~6/6, P009 from 6/1)..."
OUT_MD="docs/FINAL_REPORT_6_6.md"
python3 -u scripts/sim/generate_final_report.py \
    --start 2026-05-25 \
    --days 13 \
    --policy-from 2026-06-01 \
    --out "$OUT_MD" \
    2>&1 | tee -a "$LOG"

if [ -f "$OUT_MD" ]; then
    log "✅ 보고서 생성 완료: $OUT_MD"
    log "   HTML: ${OUT_MD%.md}.html"
    log "   차트: ${OUT_MD%.md}.d/"
else
    log "⚠️ 보고서 파일 없음 — 생성 실패 의심"
fi

log "=== DONE — Day 14 sim 계속 진행 중 ==="
