#!/usr/bin/env bash
# 2단계 — 7,500명 순차 전후 설계.
#   1구간 BASE7500 : 2025-07-14 ~ 07-20, 무정책      (약 57시간)
#   2구간 POL7500  : 2025-07-21 ~ 07-27, P010 주입   (약 57시간, 덤프 재적재 없이 이어달림)
# 각 구간이 끝날 때마다 결제원장 내보내기 + 그래프 덤프를 뜬다(FINAL 덤프 소실 재발 방지).
set -uo pipefail
LOG=/data/exp001/chain_p2.log
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ARCH=/data/exp001/archive; mkdir -p "$ARCH"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

say "2단계 시작 — 7,500명 순차 전후"

# ── 1구간: Day 0 덤프에서 출발해 무정책 7일 ────────────────────────────────
say "Day 0 덤프 복원"
$NEO/bin/neo4j stop >> $LOG 2>&1
mkdir -p /data/dumps_restore
cp /data/dumps/neo4j_base_day0.dump /data/dumps_restore/neo4j.dump
$NEO/bin/neo4j-admin database load neo4j --from-path=/data/dumps_restore \
    --overwrite-destination=true >> $LOG 2>&1
$NEO/bin/neo4j start >> $LOG 2>&1
for i in $(seq 1 60); do
  $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' \
    >/dev/null 2>&1 && break; sleep 3; done

source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export NEO4J_DATABASE=neo4j PYTHONIOENCODING=utf-8

# 덤프에 구코드로 만든 7/14~ 이력이 들어 있으면 지우고 새로 만든다.
$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  "MATCH (pl:Plan) RETURN min(pl.day) AS mn, max(pl.day) AS mx, count(DISTINCT pl.day) AS d" >> $LOG 2>&1
$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  "MATCH (pl:Plan) WHERE pl.day >= date('2025-07-14') DETACH DELETE pl" >> $LOG 2>&1
say "기존 7/14 이후 Plan 정리 완료"

cd /data/exp001_repo
python3 /data/exp001/reconcile_income_label.py >> $LOG 2>&1
python3 /data/exp001/fix_spending_anchor.py    >> $LOG 2>&1
python3 /data/exp001/load_agent_cats.py        >> $LOG 2>&1
say "라벨·앵커·업종 적재 완료"

export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
export SIM_OUTPUT_DIR=/data/exp001/out_BASE7500 PYTHONUNBUFFERED=1
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-14 --days 7 \
  --limit 7500 --workers 192 > /data/exp001/run_BASE7500.log 2>&1 &
say "1구간 BASE7500 기동 PID=$! (무정책 7/14~07/20)"

while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "1구간 종료 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"

python3 /data/exp001/export_run.py BASE7500 >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" \
    --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/BASE7500_nopolicy_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
for i in $(seq 1 60); do
  $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' \
    >/dev/null 2>&1 && break; sleep 3; done
cp -f "$SIM_OUTPUT_DIR/events.jsonl" "$ARCH/BASE7500_events.jsonl" 2>>$LOG
say "1구간 보관 완료"

# ── 2구간: 덤프 재적재 없이 P010 주입 후 이어달림 ──────────────────────────
python3 scripts/neo4j_load/10_load_grant_policy.py \
    data/neo4j_load/policies/P010.json >> $LOG 2>&1
say "P010 주입 (그래프 유지 — 1구간 이력 보존)"

export SIM_OUTPUT_DIR=/data/exp001/out_POL7500
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-21 --days 7 \
  --limit 7500 --workers 192 > /data/exp001/run_POL7500.log 2>&1 &
say "2구간 POL7500 기동 PID=$! (P010 7/21~07/27)"

while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "2구간 종료 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"

python3 /data/exp001/export_run.py POL7500 >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" \
    --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/POL7500_p010_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
cp -f "$SIM_OUTPUT_DIR/events.jsonl" "$ARCH/POL7500_events.jsonl" 2>>$LOG
tar czf "$ARCH/metrics_7500.tar.gz" -C /data/exp001 \
    out_BASE7500/metrics out_POL7500/metrics 2>>$LOG
say "2단계 전체 완료"
ls -lh "$ARCH" >> $LOG 2>&1
