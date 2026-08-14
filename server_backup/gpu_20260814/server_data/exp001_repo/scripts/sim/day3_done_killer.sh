#!/bin/bash
# Day 3 (5/27) jsonl 14,881 도달하면 즉시 python kill.
# Night2 skip → bash auto_pipeline의 LASTLINES 체크 통과 → STEP 4·5 자동 진행.
# 6/1 12:00 마감을 위함.
set -u

WD="/g/내 드라이브/Kw/final_project"
LOG="$WD/logs/day3_killer.log"
JSONL="/c/Users/Administrator/sim_output/metrics/day_2026-05-27.jsonl"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== Day3 killer 시작 (PID=$$) ==="

while true; do
    if [ -f "$JSONL" ]; then
        L=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
        if [ "$L" -ge 14881 ]; then
            log "Day 3 jsonl $L lines 도달 — python kill (Night2 skip)"
            # python3.13.exe (cygwin/Windows) kill — bash auto_pipeline은 RC 받고 LAST_JSONL 14560 이상 확인 → break → STEP 4
            taskkill.exe //F //IM python3.13.exe 2>&1 | tail -3 | tee -a "$LOG"
            log "kill 신호 발송 완료"
            break
        fi
        log "Day 3 jsonl $L / 14881"
    else
        log "Day 3 jsonl 아직 없음"
    fi
    sleep 120
done

log "=== Day3 killer 종료 ==="
