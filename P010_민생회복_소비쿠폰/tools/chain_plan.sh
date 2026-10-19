#!/usr/bin/env bash
# arm A 러너가 끝나면 → /data/repo 에 3층 고침을 심고 → 본런을 건다.
# 사전등록 experiments/plan_channel/prereg.md
#
# **러너를 기다리지 프로세스를 죽이지 않는다.** arm A 의 파이프를 건드리면
# arm A 까지 죽는다. B 팔은 표식으로 이미 건너뛴다.
set -uo pipefail
LOG=/data/chain_plan.log
say(){ echo "[$(date +%F' '%H:%M:%S)] $*" | tee -a $LOG; }
WAIT_PID="${1:-1396804}"

say "arm A 러너(pid $WAIT_PID)가 끝나기를 기다린다"
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
say "러너 종료. 시뮬 프로세스가 남아 있지 않은지 본다"
while pgrep -f "run_simulation.py" > /dev/null; do sleep 30; done

say "백업"
for f in scripts/sim/consumption.py scripts/sim/run_simulation.py \
         scripts/sim/score_policy.py scripts/sim/prompts/__init__.py \
         data/experiments/scoring_table.json; do
  cp "/data/repo/$f" "/data/repo/$f.bak_preplan" 2>/dev/null
done

say "심는다 (닻이 하나라도 없으면 아무것도 안 심고 멈춘다)"
if ! /data/venv/bin/python /tmp/patch_split_anchor.py /data/repo 2>&1 | tee -a $LOG; then
  say "심기 실패 — 본런 안 함"; exit 3
fi

say "문법 확인"
if ! /data/venv/bin/python /tmp/_syntax_check.py 2>&1 | tee -a $LOG; then
  say "문법 깨짐 — 되돌린다"
  for f in consumption run_simulation score_policy; do
    cp "/data/repo/scripts/sim/$f.py.bak_preplan" "/data/repo/scripts/sim/$f.py" 2>/dev/null
  done
  exit 3
fi

say "본런 시작"
bash /data/run_plan_p012.sh 2>&1 | tee -a $LOG
say "=== CHAIN_PLAN_DONE ==="
