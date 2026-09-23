#!/usr/bin/env bash
# 런이 하나 끝날 때마다 **보고를 전부 같은 자료로** 다시 만든다.
#
#   bash tools/refresh_reports.sh            # 지금 저장소 자료로 다시 만든다
#   bash tools/refresh_reports.sh --fetch    # 서버의 새 채점 파일을 먼저 받아 온다
#
# ## 왜 묶는가
#
# 지금은 스크립트 넷을 따로 돌린다. 하나만 돌리고 나머지를 잊으면 **표와 그림이
# 서로 다른 런을 말한다.** 실제로 부호 적중표가 거리두기를 n=200 런으로 읽는
# 동안 다른 문서는 n=500 을 인용하고 있었다.
#
# 넷은 같은 채점표를 읽고 같은 런 고르기(sign_scoreboard.READINGS)를 쓴다.
# 그러므로 한 번에 돌리면 서로 어긋날 수 없다.
set -uo pipefail
cd "$(dirname "$0")/.."

KEY=${SIM_KEY:-/c/Users/Administrator/.ssh/outofmemory.pem}
HOST=${SIM_HOST:-outofmemory@123.37.28.167}
PORT=${SIM_PORT:-10022}
PY=$(command -v py || command -v python3 || command -v python)
say(){ echo "[$(date +%H:%M)] $*"; }

if [ "${1:-}" = "--fetch" ]; then
  say "서버에서 새 채점 파일을 받는다"
  mkdir -p output/rounds
  for SRC in \
    "/data/p013_ruler/score_ruler_a.json" \
    "/data/p013_ruler/score_ruler_b.json" \
    "/data/scope_round/score_scope_off.json" \
    "/data/scope_round/score_scope_on.json" \
    "/data/p016/score_p016_v5.json" ; do
    N=$(basename "$SRC")
    if scp -q -i "$KEY" -P "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
          "$HOST:$SRC" "output/rounds/$N" 2>/dev/null; then
      echo "  <- $N"
    else
      echo "  ·  $N  아직 없음"
    fi
  done
  echo
  echo "  주의: 받아 온 파일은 **아직 채점표에 옮겨지지 않았다.**"
  echo "        채점표에 블록을 더해야 아래 보고에 반영된다 — 그 옮겨 적기는"
  echo "        사람이 한다. 자동으로 하면 어느 런을 읽는지가 코드에 숨는다."
  echo
fi

fail=0
run(){
  local label=$1; shift
  say "$label"
  if PYTHONIOENCODING=utf-8 "$PY" "$@"; then :; else fail=1; echo "  **실패: $***"; fi
  echo
}

# ① 잴 준비가 됐는가 — 이것이 빨간불이면 아래 숫자는 믿을 게 못 된다
run "① 지표가 채점 가능한가 · 정책이 잴 준비가 됐는가" \
    scripts/report/audit_indicator_coverage.py --quiet
run "② 정책이 모델에게 닿는가" \
    scripts/report/audit_policy_delivery.py

# ② 성적
run "③ 부호 적중 — 전 정책" scripts/report/sign_scoreboard.py
run "④ 빗나간 이유 — 프롬프트가 고칠 자리는 어디인가" scripts/report/why_it_misses.py
run "⑤ 프롬프트가 읽히는 자리" scripts/report/steerability_map.py
run "⑥ 정답지와의 거리 페이지" scripts/report/build_convergence_page.py

say "끝"
if [ $fail -ne 0 ]; then
  echo "  **하나 이상 실패했다 — 위 출력을 볼 것.**"
  exit 1
fi
echo "  output/report/convergence.html 를 아티팩트로 다시 올리면 팀이 보는 것도 갱신된다."
