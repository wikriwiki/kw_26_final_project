#!/usr/bin/env bash
# 본런 스크립트가 지원금 있는 7일을 마친 뒤 보존 단계에서 멈췄다(결제원장 추출기가
# 고정 소득표를 요구했다 — 고쳤다). 그 스크립트가 하려던 일을 같은 순서로 잇는다.
#
#   1) 지원금 있는 시뮬레이션 보존 (그래프가 아직 그 7일을 들고 있다 — 덮이기 전에)
#   2) 보존 확인 — 원장·기억 모음·그래프 덤프 체크섬. 하나라도 없으면 멈춘다
#   3) 지원금 없는 시뮬레이션 7일 실행 (같은 명부·같은 날짜)
#   4) 그 보존과 확인
#   5) 채점 → 마무리(보존 검사·덤프 복원 대조·인터뷰·보고서)
#
# 3) 은 그래프를 초기 상태로 되돌린다. 그래서 2) 가 통과하기 전에는 절대 넘어가지 않는다.
set -Eeuo pipefail
umask 077
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928/p012m
NEO=/data/neo4j-community-5.26.0
export P012M_ROSTER=${P012M_ROSTER:-/data/cohort/p012_main_3000.json}
export P012M_DAYS=${P012M_DAYS:-7}
cd "$REPO"
source /data/venv/bin/activate
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
LOG=/data/p012m_continue.log
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$LOG"; }
trap 'log "FAILED at line $LINENO — 그래프/출력은 그대로 둔다"' ERR

ensure_neo4j() {
  if curl -fsS -m 5 -o /dev/null http://localhost:7474; then return 0; fi
  log 'Neo4j 가 내려가 있다 — 세션 밖에서 올린다'
  setsid nohup "$NEO/bin/neo4j" start >/dev/null 2>&1 < /dev/null || true
  for _ in $(seq 1 60); do
    curl -fsS -m 5 -o /dev/null http://localhost:7474 && return 0
    sleep 2
  done
  log 'Neo4j 를 올리지 못했다'; return 1
}

preserve_and_check() {
  local arm=$1
  if [[ -s "$BASE/$arm/graph_backup/SHA256SUMS" ]]; then
    # 덤프는 이미 떴고 묶음 단계에서 멈췄던 경우 — 덤프를 다시 뜨지 않고 꼬리만 한다.
    log "=== $arm 덤프는 이미 있다 — 묶음·체크섬만 다시 한다"
    PRESERVE_TAIL_ONLY=1 bash tools/preserve_multi_v53_short_arm_20260928.sh p012m "$arm" \
      >> "/data/p012m_${arm}.preserve.log" 2>&1
  else
    log "=== $arm 보존 시작 (덮이기 전에 뽑는다)"
    ensure_neo4j
    bash tools/preserve_multi_v53_short_arm_20260928.sh p012m "$arm" \
      > "/data/p012m_${arm}.preserve.log" 2>&1
  fi
  test -s "$BASE/$arm/sector.ledger.jsonl"
  test -s "$BASE/$arm/cashback.ledger.jsonl"
  test -s "$BASE/$arm/dossier.jsonl"
  test -s "$BASE/$arm/dossier.jsonl.manifest.json"
  test -s "$BASE/$arm/graph_backup/SHA256SUMS"
  (cd "$BASE/$arm/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  test -s "$BASE/${arm}_artifacts.sha256"
  (cd "$BASE" && sha256sum -c "${arm}_artifacts.sha256" >/dev/null)
  test -s "$BASE/$arm/outputs.sha256"
  sha256sum -c "$BASE/$arm/outputs.sha256" >/dev/null
  python - "$BASE/$arm/dossier.jsonl.manifest.json" "$P012M_DAYS" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
assert not m["state_day_gaps"], m["state_day_gaps"][:1]
assert m["totals"]["states"] == m["agents"] * int(sys.argv[2]), m["totals"]
print("  기억 모음: %d명 · 상태 %d · 기억 %d · 계획항목 %d"
      % (m["agents"], m["totals"]["states"], m["totals"]["memories"], m["totals"]["plan_items"]))
PY
  printf 'verified_on=%s\n' "$(date -Is)" > "$BASE/$arm/external_copy_verified.txt"
  log "=== $arm 보존 완료 (원장·기억 모음·그래프 덤프 체크섬 확인)"
}

log "이어하기 시작 — 지원금 있는 7일은 이미 끝났다"

# 1·2) 지원금 있는 쪽 — 이미 덤프가 있으면 다시 하지 않는다(보존 스크립트도 중복을 거부한다)
# '보존 완료' 는 묶음까지 확인된 뒤에 적히는 external_copy_verified.txt 로 판단한다 —
# 덤프만 있는 반쪽 상태를 완료로 보지 않는다.
if [[ -s "$BASE/on/external_copy_verified.txt" ]]; then
  log "on 은 이미 보존돼 있다 — 체크섬만 다시 본다"
  (cd "$BASE/on/graph_backup" && sha256sum -c SHA256SUMS >/dev/null)
  (cd "$BASE" && sha256sum -c on_artifacts.sha256 >/dev/null)
else
  test "$(cat "$BASE/on/launch.exitcode")" = 0
  test -s "$BASE/on/summary.json"
  preserve_and_check on
fi

# 3·4) 지원금 없는 쪽
if [[ -s "$BASE/off/external_copy_verified.txt" ]]; then
  log "off 도 이미 보존돼 있다 — 건너뛴다"
elif [[ -s "$BASE/off/graph_backup/SHA256SUMS" ]]; then
  log "off 는 덤프까지 끝났다 — 다시 돌리지 않고 묶음만 한다"
  preserve_and_check off
else
  log "=== off 시작 (명부 $(basename "$P012M_ROSTER") · ${P012M_DAYS}일)"
  bash tools/run_multi_v53_short_arm_20260928.sh p012m off > /data/p012m_off.launch.log 2>&1
  echo 0 > "$BASE/off/launch.exitcode"
  cp /data/p012m_off.launch.log "$BASE/off/launch.log"
  log "=== off 실행 완료"
  preserve_and_check off
fi

# 5) 채점
ensure_neo4j
log "=== 채점"
mkdir -p "$BASE/score"
for arm in on off; do
  cp "$BASE/$arm/sector.ledger.jsonl" "$BASE/score/$arm.sector.ledger.jsonl"
  cp "$BASE/$arm/cashback.ledger.jsonl" "$BASE/score/$arm.cashback.ledger.jsonl"
done
cp "$BASE/roster.json" "$BASE/score/roster.json"
PYTHONIOENCODING=utf-8 python scripts/report/score_p012_two_arm.py \
  --dir "$BASE/score" --json-out "$BASE/score/score_full.json" > /data/p012m.score.txt
log "=== 채점 완료 — 마무리로 넘긴다"
