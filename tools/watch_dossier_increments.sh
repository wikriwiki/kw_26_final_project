#!/usr/bin/env bash
# 날마다 그 하루치 기억·스케줄·지출을 읽기 전용으로 빼 둔다.
#
# 왜 필요한가: 그래프 덤프는 팔이 **끝난 뒤**에만 찍힌다. 런이 중간에 죽으면
# 그때까지 쌓인 기억이 그래프에만 있고, 그 그래프는 다음 팔의 복원으로 덮인다.
# 한 번 그렇게 잃었다. 그래서 하루가 끝나는 대로 그 하루를 파일로 내려 둔다.
#
# 시뮬을 멈추지 않는다 — Neo4j 를 읽기만 하고, 하루당 몇 초다.
#
#   bash tools/watch_dossier_increments.sh p012m on 7 &
set -Eeuo pipefail
umask 077
case_id=${1:?case id}
arm=${2:?arm}
days=${3:?days}
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

log "감시 시작: $case_id/$arm · $days일 · $OUT"
done_count=0
for _ in $(seq 1 20000); do
  # set -e 에서 `[[ ... ]] && break` 는 조건이 거짓일 때 1 을 돌려 셸을 끝낸다.
  if [[ $done_count -ge $days ]]; then break; fi
  for i in $(seq 0 $((days - 1))); do
    day=$(date -d "$START + $i days" +%F)
    # 하루가 완전히 끝났다는 증거는 day_<날>.json 이다(런너가 검증 후 쓴다).
    [[ -s "$ARM/day_${day}.json" ]] || continue
    if [[ -s "$OUT/dossier_${day}.jsonl" ]]; then continue; fi
    if [[ ! -s "$BASE/roster.json" ]]; then sleep 20; continue; fi
    if python scripts/report/export_agent_dossier.py \
         --roster "$BASE/roster.json" --start "$day" --end "$day" --arm "$arm" \
         --out "$OUT/dossier_${day}.jsonl" >> "$OUT/watch.log" 2>&1; then
      sha256sum "$OUT/dossier_${day}.jsonl" >> "$OUT/SHA256SUMS"
      log "  $day 증분 저장"
    else
      # 그래프가 잠깐 바쁠 수 있다. 지우고 다음 바퀴에 다시 시도한다.
      rm -f "$OUT/dossier_${day}.jsonl"
      log "  $day 증분 실패 — 다시 시도한다"
    fi
  done
  done_count=$(ls "$OUT"/dossier_*.jsonl 2>/dev/null | wc -l || true)
  sleep 60
done
log "감시 끝: 증분 $done_count/$days 일"
