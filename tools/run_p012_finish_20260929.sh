#!/usr/bin/env bash
# 본런이 채점까지 끝낸 뒤 인터뷰와 보고서를 얹는다.
#
# 왜 따로 있는가: bash 는 스크립트를 이어 읽는다. 실행 중인 러너 파일을 덮으면
# 남은 부분을 엉뚱한 위치에서 읽어 깨진다. 그래서 러너를 건드리지 않고 이 파일로 잇는다.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928/p012m
NEO=/data/neo4j-community-5.26.0
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a /data/p012m.log; }

# 채점까지 끝났는지 확인 — 아니면 손대지 않는다.
test -s "$BASE/score/score_full.json"
for arm in on off; do
  test -s "$BASE/$arm/dossier.jsonl"
  test -s "$BASE/$arm/dossier.jsonl.manifest.json"
done

if ! curl -fsS -m 5 -o /dev/null http://localhost:8000/v1/models; then
  log '모델 서버가 없다 — 인터뷰를 건너뛰고 보고서만 만든다'
  IV=''
else
  log '=== 인터뷰 (칸마다 1명 · 비반응자 포함)'
  mkdir -p "$BASE/dossier"
  for arm in on off; do cp -f "$BASE/$arm/dossier.jsonl" "$BASE/dossier/$arm.dossier.jsonl"; done
  LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5 \
    python scripts/report/interview_agents.py --dir "$BASE/dossier" --per-cell 1 \
      --out "$BASE/interviews" | tee /data/p012m.interviews.txt
  IV="$BASE/interviews/interviews.jsonl"
fi

log '=== 보고서'
python scripts/report/build_p012_validity_report.py \
  --score "$BASE/score/score_full.json" \
  --roster-manifest "$BASE/roster.manifest.json" \
  --dossier-manifest "$BASE/on/dossier.jsonl.manifest.json" \
  ${IV:+--interviews "$IV"} \
  --out "$BASE/P012_VALIDITY.md"
test -s "$BASE/P012_VALIDITY.md"
log "DONE — $BASE/P012_VALIDITY.md"
