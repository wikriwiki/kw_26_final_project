#!/usr/bin/env bash
# FINAL 완주 대기 → 결제 이벤트 내보내기 → 무정책 baseline(200명 7일) 기동
# exp_run.sh 가 덤프를 재적재하므로 내보내기가 반드시 먼저 끝나야 한다.
set -uo pipefail
LOG=/data/exp001/chain_base.log
echo "[$(date +%H:%M:%S)] FINAL 완주 대기" > $LOG
while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] FINAL 종료 — $(ls /data/exp001/out_FINAL/metrics | wc -l)일" >> $LOG
source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass PYTHONIOENCODING=utf-8
python3 /data/exp001/export_run.py FINAL >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] 내보내기 완료 — baseline 기동" >> $LOG
bash /data/exp001/exp_run.sh BASE 200 7 none >> $LOG 2>&1
echo "[$(date +%H:%M:%S)] baseline 기동 완료" >> $LOG
