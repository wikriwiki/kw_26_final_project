#!/usr/bin/env bash
# 본런이 채점까지 끝낸 뒤: 보존 검사 → 덤프 복원 대조 → 인터뷰 → 보고서.
#
# 목표가 요구하는 세 가지(그래프 보존 · 메모리 누수 없이 1대1 인터뷰 가능 · 결제원장
# 보존)를 **검사로** 확인하고, 그 결과를 검증지표 표와 같은 보고서에 싣는다. 하나라도
# 실패하면 보고서는 만들되 결론에 실패를 적고, 이 스크립트도 실패로 끝난다.
#
# 따로 있는 이유: bash 는 스크립트를 이어 읽는다. 실행 중인 러너를 덮지 않는다.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
BASE=${FINISH_BASE:-/data/multipolicy_v53_20260928/p012m}
DAYS=${P012M_DAYS:-7}
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
FLOG=${FINISH_LOG:-/data/p012m_finish.log}
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$FLOG"; }
trap 'log "FAILED at line $LINENO"' ERR
verdict=0

test -s "$BASE/score/score_full.json"

# 가구 구성 지도가 있으면 다시 채점한다 — K18 을 '자료없음' 대신 구성 대리로 맞댄다.
# (러너의 채점은 지도 없이 돌았다. 러너 파일은 실행 중이라 고치지 않았다.)
if [[ -s "$BASE/roster_family_type.json" ]]; then
  log '=== 0. 가구 구성 지도로 다시 채점 (K18)'
  python scripts/report/score_p012_two_arm.py --dir "$BASE/score"     --family-map "$BASE/roster_family_type.json"     --json-out "$BASE/score/score_full.json" | tee "${FINISH_SCORE_TXT:-/data/p012m.score.txt}"
fi

log '=== 1. 보존 검사 (그래프 · 메모리 · 원장, 두 팔)'
if python scripts/report/verify_p012_preservation.py --base "$BASE" --days "$DAYS" \
     --json-out "$BASE/preservation_check.json" | tee "$BASE/preservation_check.txt"; then
  log '  통과'
else
  log '  **보존 검사 실패** — 계속하되 결론에 적는다'; verdict=1
fi

log '=== 2. 덤프를 실제로 복원해 dossier 와 대조'
# 시험용 스위치: 복원은 Neo4j 를 내리므로 다른 런이 돌 때는 건너뛴다. 실제 마무리에서는 켜지 않는다.
if [[ ${FINISH_SKIP_RESTORE:-0} == 1 ]]; then
  log '  (시험 모드 — 복원 건너뜀)'
elif bash tools/verify_graph_restore_20260929.sh "$BASE"; then
  log '  두 팔 모두 일치'
else
  log '  **복원 대조 실패**'; verdict=1
fi

log '=== 3. 1대1 인터뷰 — 명부 전원 가능 여부'
mkdir -p "$BASE/dossier"
for arm in on off; do cp -f "$BASE/$arm/dossier.jsonl" "$BASE/dossier/$arm.dossier.jsonl"; done
if python scripts/report/interview_agents.py --dir "$BASE/dossier" --out "$BASE/interviews" \
     --check-all > "$BASE/interview_check.txt"; then
  log "  $(grep '인터뷰 \*\*가능' "$BASE/interview_check.txt" | sed 's/^ *//')"
else
  log '  **인터뷰 불가 인원이 있다**'; cat "$BASE/interview_check.txt" | tee -a "$FLOG"; verdict=1
fi

IV=''
if [[ ${FINISH_SKIP_INTERVIEWS:-0} == 1 ]]; then
  log '  (시험 모드 — 인터뷰 건너뜀)'
elif curl -fsS -m 5 -o /dev/null http://localhost:8000/v1/models; then
  log '=== 4. 인터뷰 (반응 5분위 x 소비수준 3분단, 칸마다 1명 — 비반응자 포함)'
  LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5 \
    python scripts/report/interview_agents.py --dir "$BASE/dossier" --per-cell 1 \
      --out "$BASE/interviews" | tee "${FINISH_IV_TXT:-/data/p012m.interviews.txt}"
  IV="$BASE/interviews/interviews.jsonl"
else
  log '  모델 서버가 없다 — 인터뷰는 건너뛴다(가능 여부 검사는 위에서 했다)'
fi

log '=== 5. 보고서'
python scripts/report/build_p012_validity_report.py \
  --score "$BASE/score/score_full.json" \
  --roster-manifest "$BASE/roster.manifest.json" \
  --dossier-manifest "$BASE/on/dossier.jsonl.manifest.json" \
  --preservation "$BASE/preservation_check.json" \
  --restore-dir "$BASE/restore_check" \
  --interview-check "$BASE/interview_check.txt" \
  ${IV:+--interviews "$IV"} \
  --out "$BASE/P012_VALIDITY.md"
test -s "$BASE/P012_VALIDITY.md"
if [[ $verdict == 0 ]]; then
  log "DONE — 보존 3종 통과 · 보고서 $BASE/P012_VALIDITY.md"
else
  log "DONE WITH FAILURES — 보고서의 보존 절을 먼저 읽어라: $BASE/P012_VALIDITY.md"
fi
exit $verdict
