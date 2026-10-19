#!/usr/bin/env bash
# 계획이 끝날 때까지 돌린다. 서버가 죽으면 다시 띄우고 멈춘 자리에서 잇는다.
#
# xgrammar 가 300~350칸마다 서버를 죽인다. 17:01 에 max_threads=1 로 LogFatalError 를
# 없앴더니 이번엔 segfault 가 났다. 서버를 살리는 일은 상류의 몫이고, 우리 쪽에서
# 할 수 있는 것은 **죽어도 잃지 않는 것**이다.
#
#   plan_until_done.sh <config> <src> <out> <preflight> [최대시도]
set -uo pipefail
CFG=$1; SRC=$2; OUT=$3; PRE=$4; MAX=${5:-12}
TOK=/data/hf_cache/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/31e6a965d0661bbe4a8b895e22a77f8271772ba0
MODEL=LGAI-EXAONE/EXAONE-4.5-33B-AWQ
cd /data/validation_v3/repo

alive() {
  timeout 60 curl -s -X POST http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1\"}],\"max_tokens\":2}" \
    2>/dev/null | grep -q choices
}
ensure() {
  alive && return 0
  echo "[$(date +%T)] 서버를 띄운다"
  setsid nohup bash /data/start_sglang_reasoning.sh > /data/sglang_boot.err 2>&1 < /dev/null &
  for i in $(seq 1 40); do sleep 15; alive && { echo "[$(date +%T)] 준비됨"; return 0; }; done
  return 1
}

for try in $(seq 1 "$MAX"); do
  ensure || { echo "[$(date +%T)] 서버를 못 띄웠다"; exit 1; }
  FLAG=""; [ -d "$OUT" ] && FLAG="--resume"
  echo "[$(date +%T)] 시도 $try/$MAX $FLAG"
  /data/venv_sgl/bin/python scripts/sim/validate_action_planner.py \
    --config "$CFG" --source "$SRC" --out "$OUT" --tokenizer "$TOK" \
    --exclude "$PRE" $FLAG >> "${OUT%/}.log" 2>&1
  rc=$?
  if [ -f "$OUT/summary.json" ]; then
    echo "[$(date +%T)] 계획 완료 (시도 $try)"; exit 0
  fi
  if [ -f "$OUT/ABORTED_SERVER_GONE.json" ]; then
    echo "[$(date +%T)] 서버가 사라져 멈췄다 — 다시 띄우고 잇는다"
    for pid in $(ps -eo pid,args | grep -F "sglang.launch_server" | grep -v grep | awk '{print $1}'); do
      kill "$pid" 2>/dev/null
    done
    sleep 20
    continue
  fi
  echo "[$(date +%T)] 계획이 다른 이유로 끝났다 (rc=$rc) — 멈춘다"; exit "$rc"
done
echo "[$(date +%T)] $MAX 번 시도하고도 못 끝냈다"; exit 1
