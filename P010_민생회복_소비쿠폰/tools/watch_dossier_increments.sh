#!/usr/bin/env bash
# 날마다 그 하루치 기억·스케줄·지출을 읽기 전용으로 빼 둔다.
#
# 왜 필요한가: 그래프 덤프는 팔이 **끝난 뒤**에만 찍힌다. 런이 중간에 죽으면
# 그때까지 쌓인 기억이 그래프에만 있고, 그 그래프는 다음 팔의 복원으로 덮인다.
#
# **그래프가 지금 이 팔의 것인지 확인하고서만 읽는다.** 두 팔은 같은 그래프 자리를
# 차례로 쓴다. on 팔 감시기가 off 팔 복원 뒤에 돌면, off 팔 그래프를 on 팔 이름으로
# 내려받는다 — 오류 없이 틀린 데이터가 된다. 그래서:
#   · on 팔: off 팔의 graph_restored.marker 가 생기면 즉시 멈춘다
#   · off 팔: 자기 graph_restored.marker 가 생기기 전에는 읽지 않는다
#
#   bash tools/watch_dossier_increments.sh p012m on 7
set -Eeuo pipefail
umask 077
case_id=${1:?case id}
arm=${2:?arm}
days=${3:?days}
[[ $arm == on || $arm == off ]]
REPO=/data/pilot_repo_20260927
BASE=/data/multipolicy_v53_20260928/$case_id
ARM=$BASE/$arm
OUT=$ARM/daily
START=${START:-2021-10-01}
mkdir -p "$OUT"
cd "$REPO"
source /data/venv/bin/activate
source <(grep '^export NEO4J_URI=' "$REPO/tools/run_p013_ruler.sh" | head -1)
export PYTHONPATH="$REPO" PYTHONIOENCODING=utf-8
log() { printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "$OUT/watch.log"; }

# 이 팔의 그래프가 지금 올라와 있는가.
graph_is_mine() {
  if [[ $arm == on ]]; then
    [[ -s "$ARM/graph_restored.marker" && ! -e "$BASE/off/graph_restored.marker" ]]
  else
    [[ -s "$ARM/graph_restored.marker" ]]
  fi
}

log "감시 시작: $case_id/$arm · $days일 · $OUT"
done_count=0
for _ in $(seq 1 20000); do
  if [[ $done_count -ge $days ]]; then break; fi
  if [[ $arm == on && -e "$BASE/off/graph_restored.marker" ]]; then
    log "off 팔 복원이 시작됐다 — on 팔 그래프가 아니므로 멈춘다 (증분 $done_count/$days)"
    exit 0
  fi
  if ! graph_is_mine; then sleep 60; continue; fi
  for i in $(seq 0 $((days - 1))); do
    day=$(date -d "$START + $i days" +%F)
    if [[ ! -s "$ARM/day_${day}.json" ]]; then continue; fi
    if [[ -s "$OUT/dossier_${day}.jsonl" ]]; then continue; fi
    if [[ ! -s "$BASE/roster.json" ]]; then continue; fi
    # 읽기 직전에 한 번 더 — 확인과 읽기 사이에 복원이 끼어들 수 있다.
    if ! graph_is_mine; then break; fi
    if python scripts/report/export_agent_dossier.py \
         --roster "$BASE/roster.json" --start "$day" --end "$day" --arm "$arm" \
         --out "$OUT/dossier_${day}.jsonl" >> "$OUT/watch.log" 2>&1; then
      # 읽는 사이에 복원이 시작됐으면 그 증분은 믿을 수 없다 — 버린다.
      if ! graph_is_mine; then
        rm -f "$OUT/dossier_${day}.jsonl" "$OUT/dossier_${day}.jsonl.manifest.json"
        log "  $day 읽는 도중 그래프가 바뀌었다 — 증분을 버린다"
        break
      fi
      sha256sum "$OUT/dossier_${day}.jsonl" >> "$OUT/SHA256SUMS"
      log "  $day 증분 저장"
    else
      rm -f "$OUT/dossier_${day}.jsonl"
      log "  $day 증분 실패 — 다시 시도한다"
    fi
  done
  done_count=$(ls "$OUT"/dossier_*.jsonl 2>/dev/null | wc -l || true)
  sleep 60
done
log "감시 끝: 증분 $done_count/$days 일"
