#!/usr/bin/env bash
# 시도 예산 소진으로 한 갈래가 멈췄을 때 (2026-10-09) — 그 런의 실행기와 남은 갈래 프로세스를 멈추고
# 시도 예산만 올려(AB_MAX_ATTEMPTS, 실행 지문 밖 운영 값) 같은 실행 스크립트로 이어 돌린다.
# 끝난 날·끝난 사람은 저장본을 쓰고 건너뛴다. 정책 전 주·끝난 날 데이터는 건드리지 않는다.
# 사용: recover_exhausted.sh <p012|p013> [예산=24]
set -euo pipefail
c=$1; budget=${2:-24}
case $c in
  p012) pair=1; roster=p012_ab3w_2000.json; repo=/data/repo_ab3w_20261008;;
  p013) pair=2; roster=p013_main_2000.json; repo=/data/repo_ab3w_20261007c;;
  *) echo "모르는 런 $c" >&2; exit 2;;
esac
B=/data/ab3w/${c}_main
grep -q "시도 예산" "$B/orchestrate.log" || { echo "$c: 시도 예산 소진 기록이 없다 — 하지 않는다" >&2; exit 1; }
# 그 런의 실행기·하위 셸·run_simulation 만 고른다(pkill -f 금지 — 다른 런을 물 수 있다)
pids=()
for p in $(pgrep -f "^bash tools/run_ab3w.sh"); do
  tr "\0" "\n" < /proc/$p/environ 2>/dev/null | grep -qx "AB_CASE=$c" && pids+=("$p")
done
sims=$(pgrep -f "run_simulation.py .*--roster $B/roster.json" || true)
echo "멈출 실행기 ${pids[*]:-없음} · run_simulation ${sims:-없음}"
# 밤 단계 중이면 기다린다(밤 단계 쓰기 중 중단 회피): run log 마지막 줄에 Night 가 있으면 끝날 때까지
for s in $sims; do
  for i in $(seq 1 120); do
    log=$(ls -t $B/*/day_*_attempt*.run.log | head -1)
    tail -3 "$log" | grep -qiE "night|밤" || break
    sleep 10
  done
done
[[ ${#pids[@]} -gt 0 ]] && kill -TERM "${pids[@]}" 2>/dev/null || true
[[ -n "$sims" ]] && kill -TERM $sims 2>/dev/null || true
for i in $(seq 1 60); do pgrep -f "run_simulation.py .*--roster $B/roster.json" >/dev/null || break; sleep 2; done
pgrep -f "run_simulation.py .*--roster $B/roster.json" && { echo "아직 남은 프로세스가 있다 — 멈춘다" >&2; exit 1; }
cp -a $B/orchestrate.log $B/orchestrate.before_recover_$(date +%Y%m%d%H%M%S).log
export AB_MAX_ATTEMPTS=$budget
bash /data/ab3w/launch_main_repo.sh "$c" "$pair" "$roster" "$repo"
echo "다시 띄움: $c 예산 $budget"
