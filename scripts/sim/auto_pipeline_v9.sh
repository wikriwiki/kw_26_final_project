#!/bin/bash
# 페르소나 완료 → Neo4j 적재 → 시뮬 시작 → 자동 resume → 분석
# 사용자 자는 동안 자동으로 다 처리.
set -u

WD="/g/내 드라이브/Kw/final_project"
cd "$WD"
LOG="$WD/logs/auto_pipeline_v9.log"
mkdir -p "$WD/logs"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

# ============================================================
# Step 1: 페르소나 생성 완료 대기 (파일 라인 수 14000+ 도달 신호)
# ============================================================
log "=== STEP 1: 페르소나 생성 완료 대기 ==="
PERSONA_OUT="$WD/output/personas/full/A_rank_coupling_bdc_nvidia_v2.jsonl"

while true; do
    if [ -f "$PERSONA_OUT" ]; then
        LINES=$(wc -l < "$PERSONA_OUT" 2>/dev/null || echo 0)
        if [ "$LINES" -ge 14000 ]; then
            # 파일 사이즈가 60초 안에 안 변하면 write 끝난 것
            SIZE1=$(stat -c%s "$PERSONA_OUT" 2>/dev/null || echo 0)
            sleep 60
            SIZE2=$(stat -c%s "$PERSONA_OUT" 2>/dev/null || echo 0)
            if [ "$SIZE1" = "$SIZE2" ]; then
                log "페르소나 완료: $LINES 명 (file stable at $SIZE2 bytes)"
                break
            fi
        fi
    fi
    sleep 60
done

# ============================================================
# Step 2: Neo4j 적재 — Agent.personality_lifestyle_raw 업데이트
# ============================================================
log "=== STEP 2: Neo4j 적재 ==="
python3 "$WD/scripts/sim/load_persona_to_neo4j.py" \
    --input "$PERSONA_OUT" \
    --backup-old \
    --bump-seq \
    >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
    log "FATAL: Neo4j 적재 실패. 종료."
    exit 1
fi
log "Neo4j 적재 완료"

# ============================================================
# Step 3: 시뮬 시작 + 자동 resume 루프
# ============================================================
log "=== STEP 3: 시뮬 시작 (5/25~5/28, workers=32) ==="
export LLM_MODE=qwen8b

ATTEMPT=0
MAX_ATTEMPTS=10
while [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; do
    ATTEMPT=$((ATTEMPT + 1))
    log "시뮬 attempt $ATTEMPT/$MAX_ATTEMPTS 시작"
    SIMLOG="$WD/logs/sim_v9_attempt${ATTEMPT}.log"

    python3 "$WD/scripts/sim/run_simulation.py" \
        --start 2026-05-25 \
        --days 4 \
        --workers 32 \
        > "$SIMLOG" 2>&1
    RC=$?
    log "시뮬 종료 (exit=$RC, attempt $ATTEMPT)"

    # 5/28 (Day 3) jsonl 14,560 도달했나 — 마지막 날 완료 신호
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
# Step 4: 사후 복구 — day_idx==0 visited Memory 누락 보정
# ============================================================
log "=== STEP 4: visited Memory 사후 복구 ==="
python3 "$WD/scripts/sim/recover_visited_memory.py" \
    --start 2026-05-25 --days 4 \
    >> "$LOG" 2>&1
log "복구 완료"

# ============================================================
# Step 5: 분석 보고서 자동 빌드
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
