#!/usr/bin/env bash
# 복구 3단계 — Neo4j 설정·복원·기동, SGLang 서버 기동
set -euo pipefail
NAS=/home/ubuntu/data
NEO=$NAS/neo4j-community-5.26.0
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
exec >>"$NAS/recover3.log" 2>&1
echo "===== 3단계 시작 $(date -u +%FT%TZ) ====="

echo "--- Neo4j 설정 ---"
CONF=$NEO/conf/neo4j.conf
grep -q "^server.memory.heap.max_size" $CONF || cat >> $CONF <<'C'
server.memory.heap.initial_size=8g
server.memory.heap.max_size=16g
server.memory.pagecache.size=16g
server.default_listen_address=0.0.0.0
dbms.security.auth_enabled=true
C
echo "  conf OK"

echo "--- 비밀번호 초기화 ---"
$NEO/bin/neo4j-admin dbms set-initial-password exp001pass 2>&1 | tail -2 || echo "  (이미 설정됨)"

echo "--- 덤프 복원 ---"
mkdir -p $NAS/dumps_restore
cp $NAS/dumps/neo4j_baseline_pre_p010_20250720.dump $NAS/dumps_restore/neo4j.dump
$NEO/bin/neo4j-admin database load neo4j --from-path=$NAS/dumps_restore --overwrite-destination=true 2>&1 | tail -3

echo "--- Neo4j 기동 ---"
$NEO/bin/neo4j start 2>&1 | tail -3
for i in $(seq 1 80); do
  if $NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass 'RETURN 1' >/dev/null 2>&1; then
    echo "  Neo4j 응답 OK ($((i*3))초)"; break; fi
  sleep 3
done
$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  'MATCH (a:Agent) RETURN count(a) AS agents' 2>&1 | tail -3
$NEO/bin/cypher-shell -a bolt://localhost:7687 -u neo4j -p exp001pass \
  'MATCH (p:POI) RETURN count(p) AS pois' 2>&1 | tail -3

echo "--- SGLang 서버 기동 ---"
source $NAS/venv_sgl/bin/activate
export HF_HOME=$NAS/models
MODEL=$NAS/models/hub/models--LGAI-EXAONE--EXAONE-4.5-33B-AWQ/snapshots/31e6a965d0661bbe4a8b895e22a77f8271772ba0
nohup python -m sglang.launch_server --model-path "$MODEL" --served-model-name LGAI-EXAONE/EXAONE-4.5-33B-AWQ \
  --tp 2 --port 30000 --host 0.0.0.0 --mem-fraction-static 0.88 --context-length 8192 \
  > $NAS/sglang.log 2>&1 &
echo "  SGLang PID=$!"
echo "RECOVER3_LAUNCHED $(date -u +%FT%TZ)"
