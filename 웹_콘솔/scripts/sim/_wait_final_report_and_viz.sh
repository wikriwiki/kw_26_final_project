#!/bin/bash
# 6/7 (Day 14) 완료 → 14일치 최종 보고서 + 시각화 HTML 자동 생성

cd "/g/내 드라이브/Kw/final_project"
set -a; . data/neo4j_load/.env; set +a
export LLM_MODE=qwen36_35b_a3b_awq

LOG="/c/Users/Administrator/sim_output_9d/auto_final_viz.log"
SIM_LOG="/c/Users/Administrator/sim_output_9d/sim_9d.log"
DAY_JSONL="/c/Users/Administrator/sim_output_9d/metrics/day_2026-06-07.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== auto_final_report_and_viz 시작 (PID=$$) ==="

log "Step 1: 6/7 jsonl 7500 도달 대기..."
until [ -f "$DAY_JSONL" ] && [ "$(wc -l < $DAY_JSONL 2>/dev/null)" -ge 7500 ]; do
    sleep 60
done
log "6/7 jsonl $(wc -l < $DAY_JSONL) 도달"

log "Step 2: 6/7 Night2 완료 대기..."
# Night2 패턴 카운트 기준
INITIAL=$(grep -c "\[Night2\] Conversation +" "$SIM_LOG" 2>/dev/null || echo 0)
log "초기 Night2 완료 카운트 $INITIAL — 새 Night2 완료 대기"
until [ "$(grep -c "\[Night2\] Conversation +" $SIM_LOG 2>/dev/null)" -gt "$INITIAL" ]; do
    sleep 60
done
log "Night2 완료 감지"
sleep 60

log "Step 3: 14일치 최종 보고서 생성..."
mkdir -p /c/Users/Administrator/sim_output_9d/reports
python3 -u scripts/sim/generate_final_report.py \
    --start 2026-05-25 \
    --days 14 \
    --policy-from 2026-06-01 \
    --skip-interview \
    --out /c/Users/Administrator/sim_output_9d/reports/FINAL_REPORT_14D.md \
    2>&1 | tee -a "$LOG"

if [ -f /c/Users/Administrator/sim_output_9d/reports/FINAL_REPORT_14D.html ]; then
    cp /c/Users/Administrator/sim_output_9d/reports/FINAL_REPORT_14D.md docs/ 2>&1
    cp /c/Users/Administrator/sim_output_9d/reports/FINAL_REPORT_14D.html docs/ 2>&1
    mkdir -p docs/FINAL_REPORT_14D.d
    cp -r /c/Users/Administrator/sim_output_9d/reports/FINAL_REPORT_14D.d/* docs/FINAL_REPORT_14D.d/ 2>&1
    log "✅ 14일치 보고서 docs/ 복사 완료"
else
    log "⚠️ 14일치 보고서 HTML 생성 실패"
fi

log "Step 4: 시각화 데이터 export (Neo4j → JSON)..."
python3 -u scripts/sim/export_visualization.py --start 2026-05-25 --days 14 2>&1 | tee -a "$LOG"

log "Step 5: standalone HTML 빌드..."
python3 -u scripts/sim/build_standalone_html.py 2>&1 | tee -a "$LOG"

VIZ_OUT="output/sim/visualization/sim_standalone.html"
if [ -f "$VIZ_OUT" ]; then
    SZ=$(du -h "$VIZ_OUT" | cut -f1)
    log "✅ 시각화 HTML 생성: $VIZ_OUT ($SZ)"
else
    log "⚠️ 시각화 HTML 생성 실패"
fi

log "=== DONE ==="
