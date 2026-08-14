#!/usr/bin/env bash
# 3단계 복구 재개 — 컨테이너 재생성으로 /data 심볼릭·Java·프로세스가 날아간 뒤 이어달리기.
#
# 살아남은 것: NFS(/home/ubuntu/data) 전체 — 그래프 07-14~17 완전 + 07-18 부분(2,256명)
# 복구한 것: /data 심볼릭, openjdk-17, Neo4j, SGLang
#
# 재개 절차
#   1) 07-18 부분 Plan 제거 (중복 방지)
#   2) 07-17 경계 정산 보정 — 07-18 부터 재개하면 그 전날 야간 정산이 skip 되므로 필수
#   3) 07-18 ~ 07-20 (3일) 무정책
#   4) 07-20 경계 정산 → P010 주입 → 07-21 ~ 07-27 (7일)
set -uo pipefail
LOG=/data/exp001/chain_p3r.log
NEO=/data/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ARCH=/data/exp001/archive; mkdir -p "$ARCH"
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

source /data/venv/bin/activate
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD=exp001pass
export NEO4J_DATABASE=neo4j PYTHONIOENCODING=utf-8
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5 PYTHONUNBUFFERED=1
cd /data/exp001_repo

say "3단계 복구 재개 시작"

# SGLang 준비 대기
for i in $(seq 1 90); do
  curl -s -m 3 http://localhost:8000/health >/dev/null 2>&1 && break; sleep 10
done
curl -s -m 5 http://localhost:8000/health >/dev/null 2>&1 || { say "중단 — SGLang 응답 없음"; exit 1; }
say "SGLang 준비됨"

# 광역상권이 켜져 있는지 확인 (꺼진 채 도는 사고 방지)
NHUB=$(python3 -c "
import sys; sys.path.insert(0,'/data/exp001_repo/scripts')
from sim import mobility
print(len(mobility._load()['all_hubs']))" 2>/dev/null)
say "허브 풀 ${NHUB:-0}개"
[ "${NHUB:-0}" -lt 100 ] && { say "중단 — 허브 미로드"; exit 1; }

# 1) 끊긴 07-18 부분분 제거
$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  "MATCH (pl:Plan) WHERE pl.day >= date('2025-07-18') DETACH DELETE pl" >> $LOG 2>&1
say "07-18 이후 부분 Plan 제거"

# 2) 07-17 경계 정산 (재개 첫날이 07-18 이라 그 전날 야간 정산이 skip 된다)
python3 /data/exp001/consolidate_day.py 2025-07-17 >> $LOG 2>&1
say "07-17 경계 정산 보정"

# 3) 07-18 ~ 07-20 무정책
export SIM_OUTPUT_DIR=/data/exp001/out_BASE7500H_r2
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-18 --days 3 \
  --limit 7500 --workers 192 > /data/exp001/run_BASE7500H_r2.log 2>&1 &
say "1구간 잔여(07-18~20) 기동 PID=$!"
while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "1구간 완주 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"

python3 /data/exp001/export_run.py BASE7500H_r2 >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/BASE7500H_nopolicy_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
for i in $(seq 1 60); do
  $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' >/dev/null 2>&1 && break; sleep 3; done
say "1구간 보관 완료"

# 4) 경계 정산 → P010 주입 → 정책 7일
python3 /data/exp001/consolidate_day.py 2025-07-20 >> $LOG 2>&1
say "07-20 경계 정산 보정"
python3 scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P010.json >> $LOG 2>&1
say "P010 주입"

export SIM_OUTPUT_DIR=/data/exp001/out_POL7500H
rm -rf "$SIM_OUTPUT_DIR"; mkdir -p "$SIM_OUTPUT_DIR"
nohup python3 scripts/sim/run_simulation.py --start 2025-07-21 --days 7 \
  --limit 7500 --workers 192 > /data/exp001/run_POL7500H.log 2>&1 &
say "2구간 POL7500H 기동 PID=$!"
while pgrep -f "run_simulation[.]py" >/dev/null; do sleep 60; done
say "2구간 완주 — $(ls $SIM_OUTPUT_DIR/metrics | wc -l)일"

python3 /data/exp001/export_run.py POL7500H >> $LOG 2>&1
$NEO/bin/neo4j stop >> $LOG 2>&1
$NEO/bin/neo4j-admin database dump neo4j --to-path="$ARCH" --overwrite-destination=true >> $LOG 2>&1
mv -f "$ARCH/neo4j.dump" "$ARCH/POL7500H_p010_7d.dump" 2>>$LOG
$NEO/bin/neo4j start >> $LOG 2>&1
tar czf "$ARCH/metrics_7500H.tar.gz" -C /data/exp001 \
    out_BASE7500H/metrics out_BASE7500H_r2/metrics out_POL7500H/metrics 2>>$LOG
say "3단계 전체 완료"
