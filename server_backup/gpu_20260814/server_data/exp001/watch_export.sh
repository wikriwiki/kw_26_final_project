#!/usr/bin/env bash
# 하루가 끝날 때마다 결제원장을 자동으로 빼둔다.
#
# 완료 신호는 timing/day_<DAY>.json 이다 — 이 파일은 그날 전원 처리가 끝나야 쓰인다
# (metrics/day_*.jsonl 은 처리 중에도 계속 커지므로 완료 신호로 쓸 수 없다).
# 읽기 전용 조회라 Neo4j를 멈추지 않으며 시뮬과 병행 가능하다.
set -uo pipefail
LOG=/data/exp001/watch_export.log
RUNS="BASE7500 POL7500 BASE7500H BASE7500H_r2 POL7500H"
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8

echo "[$(date '+%m-%d %H:%M:%S')] 일자별 원장 내보내기 감시 시작 ($RUNS)" >> $LOG

while true; do
  for RUN in $RUNS; do
    T="/data/exp001/out_$RUN/timing"
    [ -d "$T" ] || continue
    for f in "$T"/day_*.json; do
      [ -e "$f" ] || continue
      DAY=$(basename "$f" .json); DAY=${DAY#day_}
      OUTF="/data/exp001/out_$RUN/events_${DAY}.jsonl"
      [ -e "$OUTF" ] && continue                 # 이미 내보냄
      python3 /data/exp001/export_day.py "$RUN" "$DAY" >> $LOG 2>&1 \
        || echo "[$(date '+%m-%d %H:%M:%S')] 실패 $RUN $DAY" >> $LOG
    done
  done
  # 시뮬과 체인이 모두 끝났고 남은 일자도 없으면 종료
  if ! pgrep -f "run_simulation[.]py" >/dev/null && ! pgrep -f "chain_p2[.]sh" >/dev/null; then
    echo "[$(date '+%m-%d %H:%M:%S')] 시뮬·체인 종료 확인 — 감시 종료" >> $LOG
    break
  fi
  sleep 300
done
