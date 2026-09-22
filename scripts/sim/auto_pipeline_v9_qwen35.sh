#!/bin/bash
# v9 qwen35: Qwen3.5-9B-AWQ 모델 + workers=80 으로 sim 재개.
# 단일 인스턴스 락(/tmp/sim_v9.lock)으로 중복 실행 차단.
set -u

LOCK="/tmp/sim_v9.lock"
if [ -f "$LOCK" ]; then
    OWNER=$(cat "$LOCK" 2>/dev/null)
    if [ -n "$OWNER" ] && kill -0 "$OWNER" 2>/dev/null; then
        echo "이미 실행 중: PID=$OWNER" >&2
        exit 1
    fi
    rm -f "$LOCK"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

WD="/g/내 드라이브/Kw/final_project"
cd "$WD"
LOG="$WD/logs/auto_pipeline_v9_simonly.log"
mkdir -p "$WD/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== START Qwen3.5-9B-AWQ (PID=$$) ==="

# STEP 3: 시뮬 (Qwen3.5-9B-AWQ, workers=80)
log "=== STEP 3: 시뮬 (5/25~5/27, Qwen3.5-9B-AWQ, workers=80) ==="
export LLM_MODE=${LLM_MODE:-qwen35_9b_awq}
export PYTHONUNBUFFERED=1

ATTEMPT=0
MAX_ATTEMPTS=10
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "시뮬 attempt $ATTEMPT/$MAX_ATTEMPTS 시작 (qwen35)"
    SIMLOG="$WD/logs/sim_v9_qwen35_attempt${ATTEMPT}.log"

    python3 -u "$WD/scripts/sim/run_simulation.py" \
        --start 2026-05-25 \
        --days 3 \
        --workers 80 \
        > "$SIMLOG" 2>&1
    RC=$?
    log "시뮬 종료 (exit=$RC, attempt $ATTEMPT)"

    LAST_JSONL="/c/Users/Administrator/sim_output/metrics/day_2026-05-27.jsonl"
    if [ -f "$LAST_JSONL" ]; then
        LASTLINES=$(wc -l < "$LAST_JSONL")
        if [ "$LASTLINES" -ge 14560 ]; then
            log "5/27 완료 ($LASTLINES) — 시뮬 정상 종료"
            break
        fi
    fi

    if [ "$RC" -eq 0 ]; then
        log "exit=0 but jsonl incomplete — 분석 진행"
        break
    fi

    log "WARN: silent kill 의심 — 30초 후 resume"
    sleep 30
done

# STEP 4
log "=== STEP 4: visited Memory 복구 ==="
python3 "$WD/scripts/sim/recover_visited_memory.py" \
    --start 2026-05-25 --days 3 \
    >> "$LOG" 2>&1
log "복구 완료"

# STEP 5
log "=== STEP 5: 보고서·시각화 ==="
LLM_MODE=${LLM_MODE:-qwen35_9b_awq} python3 "$WD/scripts/sim/generate_final_report.py" \
    --start 2026-05-25 --days 3 --policy-from 2026-05-27 \
    --out "$WD/output/sim/report/FINAL_REPORT_3D_P009.md" \
    --skip-interview \
    >> "$LOG" 2>&1
log "1차 보고서 완료"

LLM_MODE=${LLM_MODE:-qwen35_9b_awq} python3 "$WD/scripts/sim/generate_final_report.py" \
    --start 2026-05-25 --days 3 --policy-from 2026-05-27 \
    --out "$WD/output/sim/report/FINAL_REPORT_3D_P009_FULL.md" \
    >> "$LOG" 2>&1
log "전체 보고서 완료"

python3 "$WD/scripts/sim/analyze_policy_spend.py" \
    --start 2026-05-25 --days 3 \
    --out "$WD/output/sim/report/POLICY_SPEND_ANALYSIS.md" \
    >> "$LOG" 2>&1
log "정책 사용 분석 완료"

python3 "$WD/scripts/sim/export_visualization.py" \
    --start 2026-05-25 --days 3 \
    >> "$LOG" 2>&1
python3 "$WD/scripts/sim/build_standalone_html.py" \
    >> "$LOG" 2>&1
log "시각화 완료"

log "=== ALL DONE Qwen3.5-9B-AWQ ==="
