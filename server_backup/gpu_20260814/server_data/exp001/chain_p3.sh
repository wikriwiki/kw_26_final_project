#!/usr/bin/env bash
# 3단계 — 광역상권(Huff 허브) 복원판 전체 재실행.
#
# 2단계까지의 모든 런은 output/stats/ 가 서버에 없어 mobility._load() 가 조용히 빈 구조로
# degrade 했고(try/except: pass), suggest_hubs() 가 항상 0개를 반환했다. 즉 에이전트에게
# 광역상권 후보가 한 번도 제시되지 않았다. 이 체인은 그 데이터를 제자리에 넣고 같은
# 순차 전후 설계를 처음부터 다시 돌린다.
#
#   1구간 BASE7500H : 2025-07-14 ~ 07-20, 무정책
#   2구간 POL7500H  : 2025-07-21 ~ 07-27, P010 (덤프 재적재 없이 이어달림)
set -uo pipefail
LOG=/data/exp001/chain_p3.log
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ARCH=/data/exp001/archive; mkdir -p "$ARCH"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

say "3단계 대기 — 2단계 완주까지"
while pgrep -f "run_simulation[.]py" >/dev/null || pgrep -f "chain_p2[.]sh" >/dev/null; do
  sleep 120
done
say "2단계 종료 확인"

# ── 광역상권 데이터 제자리 배치 ────────────────────────────────────────────
mkdir -p /data/exp001_repo/output/stats
cp -f /data/exp001/stage_stats/*.json /data/exp001_repo/output/stats/
say "output/stats 배치 — $(ls /data/exp001_repo/output/stats/*.json | wc -l)개"

source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export NEO4J_DATABASE=neo4j PYTHONIOENCODING=utf-8
cd /data/exp001_repo

# 실제로 허브가 살아났는지 확인 — 0개면 중단한다(꺼진 채 60시간 돌리는 사고 방지)
NHUB=$(python3 -c "
import sys; sys.path.insert(0,'/data/exp001_repo/scripts')
from sim import mobility
d=mobility._load()
print(len(d['all_hubs']))" 2>/dev/null)
say "허브 풀 ${NHUB:-0}개"
if [ "${NHUB:-0}" -lt 100 ]; then
  say "중단 — 허브가 로드되지 않았다"; exit 1
fi

# ── 1구간: Day 0 덤프 → 무정책 7일 ────────────────────────────────────────
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

$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  "MATCH (pl:Plan) WHERE pl.day >= date('2025-07-14') DETACH DELETE pl" >> $LOG 2>&1
python3 /data/exp001/reconcile_income_label.py >> $LOG 2>&1
python3 /data/exp001/fix_spending_anchor.py    >> $LOG 2>&1
python3 /data/exp001/load_agent_cats.py        >> $LOG 2>&1
say "그래프 준비 완료"

export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
export SIM_OUTPUT_DIR=/data/exp001/out_BASE7500H PYTHONUNBUFFERED=1
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-14 --days 7 \
  --limit 7500 --workers 192 > /data/exp001/run_BASE7500H.log 2>&1 &
say "1구간 BASE7500H 기동 PID=$! (무정책·광역상권 ON)"

while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "1구간 종료 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"
python3 /data/exp001/export_run.py BASE7500H >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" \
    --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/BASE7500H_nopolicy_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
for i in $(seq 1 60); do
  $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' \
    >/dev/null 2>&1 && break; sleep 3; done
cp -f "$SIM_OUTPUT_DIR/events.jsonl" "$ARCH/BASE7500H_events.jsonl" 2>>$LOG
say "1구간 보관 완료"

# ── 경계일 야간 정산 보정 ──────────────────────────────────────────────────
# run_simulation 은 각 런 첫날에 전날 야간 정산을 건너뛴다(의도된 동작). 순차 설계에서는
# 1구간 마지막 날(07-20)이 그 대상이 되어 기억·KNOWS_POI 가 통째로 사라진다.
# 2단계에서 실제로 발생했으므로(기억 0건·KNOWS_POI 갱신 0건) 2구간 시작 전에 보정한다.
python3 /data/exp001/consolidate_day.py 2025-07-20 >> $LOG 2>&1
say "경계일(07-20) 야간 정산 보정"

# ── 2구간: 덤프 재적재 없이 P010 주입 후 이어달림 ──────────────────────────
python3 scripts/neo4j_load/10_load_grant_policy.py \
    data/neo4j_load/policies/P010.json >> $LOG 2>&1
say "P010 주입 (그래프 유지)"

export SIM_OUTPUT_DIR=/data/exp001/out_POL7500H
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-21 --days 7 \
  --limit 7500 --workers 192 > /data/exp001/run_POL7500H.log 2>&1 &
say "2구간 POL7500H 기동 PID=$! (P010·광역상권 ON)"

while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "2구간 종료 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"
python3 /data/exp001/export_run.py POL7500H >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" \
    --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/POL7500H_p010_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
cp -f "$SIM_OUTPUT_DIR/events.jsonl" "$ARCH/POL7500H_events.jsonl" 2>>$LOG
tar czf "$ARCH/metrics_7500H.tar.gz" -C /data/exp001 \
    out_BASE7500H/metrics out_POL7500H/metrics 2>>$LOG
say "3단계 전체 완료"
