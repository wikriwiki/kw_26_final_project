#!/usr/bin/env bash
# 서버 쪽 정기점검 — 세션과 무관하게 3시간마다 health_check 결과를 쌓아둔다.
#
# 대화 세션은 턴이 있을 때만 동작하므로 "다음 점검 05:15" 같은 약속은 지켜지지 않는다.
# 이 스크립트는 서버에서 계속 돌며 이력을 남긴다. 어느 시점에 확인하든
# 그동안 무슨 일이 있었는지 tail 한 번으로 알 수 있다.
set -uo pipefail
LOG=/data/exp001/health_history.log
ALERT=/data/exp001/health_alert.log
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8

pick_run() {
  # 지금 돌고 있는 런 이름을 인자에서 읽는다 (없으면 마지막 out_ 디렉터리)
  local a
  a=$(ps -eo args | grep '[r]un_simulation.py' | head -1)
  case "$a" in
    *"--start 2025-07-21"*) [ -d /data/exp001/out_POL7500H ] && echo POL7500H && return ;;
    *"--start 2025-07-14"*) [ -d /data/exp001/out_BASE7500H ] && echo BASE7500H && return ;;
  esac
  basename "$(ls -dt /data/exp001/out_* | head -1)" | sed 's/^out_//'
}

while true; do
  RUN=$(pick_run)
  {
    echo "════════ $(date '+%m-%d %H:%M') · RUN=$RUN ════════"
    python3 /data/exp001/health_check.py "$RUN" 2>&1 | grep -v -i deprecat
  } >> $LOG

  # [이상] 이 잡히면 따로 모아둔다 — 놓치지 않기 위해
  tail -80 $LOG | grep '\[이상\]' | while read -r l; do
    echo "$(date '+%m-%d %H:%M') [$RUN] $l" >> $ALERT
  done

  # 시뮬·체인이 모두 끝났으면 마지막 한 번 더 남기고 종료
  # 시뮬이 잠깐 안 떠 있는 순간(구간 전환·모델 로딩)을 종료로 오판하지 않도록
  # 연속 3회(9시간) 모두 비어 있을 때만 종료한다.
  if ! pgrep -f "run_simulation[.]py" >/dev/null && ! pgrep -f "chain_p[0-9a-z]*[.]sh" >/dev/null; then
    IDLE=$((${IDLE:-0}+1))
  else
    IDLE=0
  fi
  if [ "${IDLE:-0}" -ge 3 ]; then
    echo "[$(date '+%m-%d %H:%M')] 시뮬·체인 종료 — 점검 감시 종료" >> $LOG
    break
  fi
  sleep 10800   # 3시간
done
