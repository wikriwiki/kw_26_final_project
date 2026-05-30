#!/bin/bash
# v9 simonly: 페르소나 이미 적재된 상태에서 시뮬+사후 처리만 자동.
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

log "=== START (PID=$$) ==="

# ============================================================
# STEP 3: 시뮬 시작 + 자동 resume 루프
# ============================================================
log "=== STEP 3: 시뮬 시작 (5/25~5/28, workers=48) ==="
export LLM_MODE=qwen8b
export PYTHONUNBUFFERED=1

ATTEMPT=0
MAX_ATTEMPTS=10
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "시뮬 attempt $ATTEMPT/$MAX_ATTEMPTS 시작"
    SIMLOG="$WD/logs/sim_v9_attempt${ATTEMPT}.log"

    python3 -u "$WD/scripts/sim/run_simulation.py" \
        --start 2026-05-25 \
        --days 4 \
        --workers 48 \
        > "$SIMLOG" 2>&1
    RC=$?
    log "시뮬 종료 (exit=$RC, attempt $ATTEMPT)"

    LAST_JSONL="/c/Users/Administrator/sim_output/metrics/day_2026-05-28.jsonl"
    if [ -f "$LAST_JSONL" ]; then
        LASTLINES=$(wc -l < "$LAST_JSONL")
        if [ "$LASTLINES" -ge 14560 ]; then
            log "5/28 완료 ($LASTLINES) — 시뮬 정상 종료"
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

# ============================================================
# STEP 4: visited Memory 사후 복구
# ============================================================
log "=== STEP 4: visited Memory 사후 복구 ==="
python3 "$WD/scripts/sim/recover_visited_memory.py" \
    --start 2026-05-25 --days 4 \
    >> "$LOG" 2>&1
log "복구 완료"

# ============================================================
# STEP 5: 보고서·시각화
# ============================================================
log "=== STEP 5: 보고서·시각화 빌드 ==="
LLM_MODE=qwen8b python3 "$WD/scripts/sim/generate_final_report.py" \
    --start 2026-05-25 --days 4 --policy-from 2026-05-27 \
    --out "$WD/output/sim/report/FINAL_REPORT_4D_P009.md" \
    --skip-interview \
    >> "$LOG" 2>&1
log "1차 보고서(skip-interview) 완료"

LLM_MODE=qwen8b python3 "$WD/scripts/sim/generate_final_report.py" \
    --start 2026-05-25 --days 4 --policy-from 2026-05-27 \
    --out "$WD/output/sim/report/FINAL_REPORT_4D_P009_FULL.md" \
    >> "$LOG" 2>&1
log "전체 보고서(인터뷰 포함) 완료"

python3 "$WD/scripts/sim/analyze_policy_spend.py" \
    --start 2026-05-25 --days 4 \
    --out "$WD/output/sim/report/POLICY_SPEND_ANALYSIS.md" \
    >> "$LOG" 2>&1
log "정책 사용 분석 완료"

python3 "$WD/scripts/sim/export_visualization.py" \
    --start 2026-05-25 --days 4 \
    >> "$LOG" 2>&1
python3 "$WD/scripts/sim/build_standalone_html.py" \
    >> "$LOG" 2>&1
log "시각화 빌드 완료"

log "=== ALL DONE ==="
