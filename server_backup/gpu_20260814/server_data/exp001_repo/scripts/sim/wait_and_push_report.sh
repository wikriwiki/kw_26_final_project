#!/bin/bash
# sim_resume_qwen36의 "ALL DONE" 신호 잡으면 보고서·시각화 git push.
set -u

WD="/g/내 드라이브/Kw/final_project"
LOG="$WD/logs/auto_pipeline_v9_simonly.log"
PUSH_LOG="$WD/logs/auto_push_report.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PUSH_LOG"; }

log "=== wait_and_push 시작 (PID=$$) ==="

# tail -F + ALL DONE 신호 잡음
tail -F -n 0 "$LOG" 2>/dev/null | while read line; do
    if echo "$line" | grep -qE "ALL DONE|보고서.*완료|시각화.*완료"; then
        log "신호 잡힘: $line"
        # 빌드 파일 fs 동기화 대기
        sleep 30

        cd "$WD"
        log "보고서·시각화 push 시작"

        # 보고서 + 시각화 파일들 stage
        git add output/sim/report/ 2>&1 >> "$PUSH_LOG"
        git add output/sim/metrics/ 2>&1 >> "$PUSH_LOG" || true
        git add output/sim/_archive_20260501_to_07/ 2>&1 >> "$PUSH_LOG" || true
        # 새 무시 파일 추가 — 보고서 빌드 산출물
        find output/sim/report -name "*.md" -o -name "*.html" -o -name "*.png" -o -name "*.json" 2>/dev/null | head -30 | xargs -I {} git add {} 2>&1 >> "$PUSH_LOG"

        git status --short 2>&1 | head -20 >> "$PUSH_LOG"

        git commit -m "report(P009): 3-day sim final report + dashboard

Qwen3.6-35B-A3B-AWQ + workers=32 + P009 grant 정책 적용 (2026-05-27 1일).
- FINAL_REPORT_3D_P009_FULL.md / .html (인터뷰 포함)
- POLICY_SPEND_ANALYSIS.md
- 정책 적용 그룹 income bucket(중상/중/중하/하)별 DID 분석
- 정책 미주입 그룹(income=상)은 자연 비교군

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>" 2>&1 >> "$PUSH_LOG"

        git push origin agent_persona 2>&1 >> "$PUSH_LOG"
        log "push 완료"

        # 종료
        pkill -P $$ tail 2>/dev/null
        exit 0
    fi
done
