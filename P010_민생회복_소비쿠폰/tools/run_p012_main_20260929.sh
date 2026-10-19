#!/usr/bin/env bash
# P012 본런 — 31일 창의 동결본 정책으로 두 팔을 순서대로 돌리고 보존까지 한다.
#
# 순서가 중요하다. 이 파일이 존재하는 이유가 그것이다.
#   1) on 팔 실행    -> 2) on 보존(원장·dossier·그래프 덤프) -> 3) off 팔 실행
#   4) off 보존       -> 5) 채점
# 보존을 먼저 하지 않고 off 팔로 넘어가면 **off 팔의 그래프 복원이 on 팔의 기억을
# 덮는다.** 검수 런에서 실제로 확인했다 — 라이브 그래프에 앞선 케이스의 State 가
# 하나도 남지 않았다.
#
# 창(P012M_DAYS)과 정책 파일은 함께 움직인다. 31일이면 동결본, 7일이면 문턱·한도를
# 7/31 로 줄인 압축월 파일이다. 둘을 섞으면 문턱이 창에 비해 너무 높아 아무도 못
# 넘고 정책이 조용히 사라진다.
#
# 창을 줄이는 값: 표준오차는 창 길이에 거의 무관하므로(7일 7.80 vs 31일 5.32) 판정력은
# 유지되고 시간만 준다. 치르는 값은 **실측과의 눈금**이다 — 파일럿 원장을 잘라 재면
# 총소비 추정이 7일 +18.9% -> 14일 +15.5% -> 21일 +13.8% -> 31일 +11.3% 로 움직이고
# 실측 +11.25% 는 한 달치다. 압축월이 그 차이를 메운다는 것은 **설계 논증이고 측정된
# 사실이 아니다.** 보고서에 그대로 적어야 한다.
set -Eeuo pipefail
umask 077

REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928/p012m
ROSTER=${P012M_ROSTER:-/data/cohort/p012_main_3000.json}
DAYS=${P012M_DAYS:-7}
export P012M_ROSTER="$ROSTER" P012M_DAYS="$DAYS"
cd "$REPO"

log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a /data/p012m.log; }
NEO=/data/neo4j-community-5.26.0
# tmux 세션이 끝나면 그 세션의 자식으로 뜬 Neo4j 도 함께 내려간다(검수 런에서 두 번
# 확인: "shutdown initiated by request"). 보존은 그래프를 읽어야 하므로 그 앞에서
# 살아 있는지 보고, 없으면 올린다.
ensure_neo4j() {
  if curl -fsS -m 5 -o /dev/null http://localhost:7474; then return 0; fi
  log 'Neo4j 가 내려가 있다 — 올린다'
  "$NEO/bin/neo4j" start >/dev/null 2>&1 || true
  for _ in $(seq 1 60); do
    curl -fsS -m 5 -o /dev/null http://localhost:7474 && return 0
    sleep 2
  done
  log 'Neo4j 를 올리지 못했다'; return 1
}
trap 'log "FAILED at line $LINENO — 그래프/출력은 그대로 둔다"' ERR

test -s "$ROSTER"
test -s "$ROSTER.manifest.json"
python - "$ROSTER.manifest.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding='utf-8'))
assert m.get("schema") == "demographic_matched_cohort_v1", m.get("schema")
assert m["worst_margin_error"] <= 0.02, m["worst_margin_error"]
# 소득이 맞춰지지 않았다는 사실은 지우지 않는다 — 보고서에 그대로 옮긴다.
assert m.get("income_calibrated") is False
print("roster ok: %d명 · 주변분포 최대오차 %.5f · 소득 미교정"
      % (m["citizens"], m["worst_margin_error"]))
PY

for arm in on off; do
  log "=== $arm 팔 시작 (명부 $(basename "$ROSTER") · 창 ${DAYS}일)"
  bash tools/run_multi_v53_short_arm_20260928.sh p012m "$arm" \
    > "/data/p012m_${arm}.launch.log" 2>&1
  echo 0 > "$BASE/$arm/launch.exitcode"
  cp "/data/p012m_${arm}.launch.log" "$BASE/$arm/launch.log"
  log "=== $arm 팔 실행 완료 — 보존 시작 (덮이기 전에 뽑는다)"
  ensure_neo4j
  bash tools/preserve_multi_v53_short_arm_20260928.sh p012m "$arm" \
    > "/data/p012m_${arm}.preserve.log" 2>&1
  # 보존이 남겨야 하는 것 — 하나라도 없으면 다음 팔로 넘어가지 않는다.
  test -s "$BASE/$arm/sector.ledger.jsonl"
  test -s "$BASE/$arm/cashback.ledger.jsonl"
  test -s "$BASE/$arm/dossier.jsonl"
  test -s "$BASE/$arm/dossier.jsonl.manifest.json"
  test -s "$BASE/$arm/graph_backup/SHA256SUMS"
  (cd "$BASE/$arm/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  printf 'verified_on=%s\n' "$(date -Is)" > "$BASE/$arm/external_copy_verified.txt"
  log "=== $arm 팔 보존 완료 (원장·dossier·그래프 덤프 검증)"
done

ensure_neo4j
log "=== 채점"
mkdir -p "$BASE/score"
for arm in on off; do
  cp "$BASE/$arm/sector.ledger.jsonl" "$BASE/score/$arm.sector.ledger.jsonl"
  cp "$BASE/$arm/cashback.ledger.jsonl" "$BASE/score/$arm.cashback.ledger.jsonl"
done
cp "$BASE/roster.json" "$BASE/score/roster.json"
PYTHONIOENCODING=utf-8 python scripts/report/score_p012_two_arm.py \
  --dir "$BASE/score" --json-out "$BASE/score/score_full.json" \
  | tee /data/p012m.score.txt
log "DONE — /data/p012m.score.txt 와 $BASE/score/score_full.json"
