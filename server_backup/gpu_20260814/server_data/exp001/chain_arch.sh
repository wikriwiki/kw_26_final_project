#!/usr/bin/env bash
# BASE(무정책 baseline) 완주 → 결제원장 내보내기 → 그래프 덤프 보관.
# FINAL에서 덤프를 놓친 사고 재발 방지: 다음 런이 덤프를 재적재하기 전에 반드시 뜬다.
set -uo pipefail
LOG=/data/exp001/chain_arch.log
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ARCH=/data/exp001/archive; mkdir -p "$ARCH"

echo "[$(date +%H:%M:%S)] BASE 완주 대기" > $LOG
while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] BASE 종료 — $(ls /data/exp001/out_BASE/metrics | wc -l)일" >> $LOG

source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export PYTHONIOENCODING=utf-8
python3 /data/exp001/export_run.py BASE >> $LOG 2>&1

echo "[$(date +%H:%M:%S)] BASE 그래프 덤프 시작" >> $LOG
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" \
    --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/BASE_nopolicy_7d_200agents.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1

# 산출물 한곳에 모으기 (로컬 다운로드 대상)
cp -f /data/exp001/out_BASE/events.jsonl   "$ARCH/BASE_events.jsonl"   2>>$LOG
cp -f /data/exp001/out_FINAL/events.jsonl  "$ARCH/FINAL_events.jsonl"  2>>$LOG
cp -f /data/exp001/out_FINAL/poi_summary.json "$ARCH/FINAL_poi_summary.json" 2>>$LOG
tar czf "$ARCH/metrics_FINAL_BASE.tar.gz" \
    -C /data/exp001 out_FINAL/metrics out_BASE/metrics 2>>$LOG

echo "[$(date +%H:%M:%S)] 보관 완료" >> $LOG
ls -lh "$ARCH" >> $LOG 2>&1
