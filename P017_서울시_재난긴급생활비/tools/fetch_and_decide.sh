#!/usr/bin/env bash
# 돌고 있는 라운드의 채점 파일을 서버에서 받아 **최종 프롬프트 판정**까지 한 번에.
#
#   bash tools/fetch_and_decide.sh
#
# 아직 안 끝난 라운드는 "아직 채점 파일이 없다" 로 나온다 — 그것도 답이다.
# 판정 규칙은 각 라운드의 사전등록 그대로이고 여기서 새로 정하지 않는다
# (scripts/report/final_prompt_decision.py 의 머리말 참조).
set -uo pipefail
cd "$(dirname "$0")/.."

KEY=${SIM_KEY:-/c/Users/Administrator/.ssh/outofmemory.pem}
HOST=${SIM_HOST:-outofmemory@123.37.28.167}
PORT=${SIM_PORT:-10022}
OUT=output/answerkey
mkdir -p "$OUT"

say(){ echo "[$(date +%H:%M)] $*"; }

say "서버에서 채점 파일을 받는다"
for SRC in \
  "/data/v50_answerkey/score_v50_v5.json" \
  "/data/v50_answerkey/score_v50_v45.json" \
  "/data/answerkey_round2/score_r2_v5.json" \
  "/data/answerkey_round2/score_r2_v45.json" \
  "/data/answerkey_round3/score_r3_v5.json" \
  "/data/answerkey_round3/score_r3_v51.json" ; do
  N=$(basename "$SRC")
  if scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
        "$HOST:$SRC" "$OUT/$N" 2>/dev/null; then
    echo "  ← $N"
  else
    echo "  · $N  아직 없음"
  fi
done

echo
say "진행 상황"
ssh -i "$KEY" -p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 "$HOST" \
  'for A in /data/answerkey_round2/r2_v5 /data/answerkey_round2/r2_v45 \
            /data/answerkey_round3/r3_v5 /data/answerkey_round3/r3_v51; do
     python3 -c "
import json,os,sys
p=\"$A/summary.json\"
n=os.path.basename(\"$A\")
if os.path.exists(p):
    s=json.load(open(p,encoding=\"utf-8\"))[\"summary\"]
    print(\"  %-8s %d일 · 오류 %d\" % (n, len(s), sum(r[\"err\"] for r in s)))
else:
    print(\"  %-8s 아직\" % n)
"
   done' 2>/dev/null || echo "  (서버 확인 실패)"

echo
say "등록된 규칙으로 판정한다"
# Windows 의 python 은 스토어 스텁으로 잡히는 일이 있다 — 런처를 먼저 찾는다
PY=$(command -v py || command -v python3 || command -v python)
PYTHONIOENCODING=utf-8 "$PY" scripts/report/final_prompt_decision.py --dir "$OUT"
