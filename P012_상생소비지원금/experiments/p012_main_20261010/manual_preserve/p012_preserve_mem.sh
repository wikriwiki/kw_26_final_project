#!/usr/bin/env bash
# P012 보존 단계가 업종 원장 노출 검사(신청 판단으로 비신청자는 정책이 안 보임)에서 멈춘 뒤, 기억·그래프부터 손으로 보존한다(2026-10-10).
# 실행기 run_ab3w.sh 의 preserve_arm 과 같은 명령. 정책 원장·external_copy_verified 는 여기서 쓰지 않는다.
set -euo pipefail
source /data/ab3w/neo4j_credentials.sh
BASE=/data/ab3w/p012_main; REPO=/data/repo_ab3w_20261008
cd "$REPO"; source /data/venv/bin/activate
declare -A NEO=([on]=/data/neo4j-community-5.26.0 [off]=/data/neo4j2-community-5.26.0)
declare -A BOLT=([on]=7687 [off]=7688) HTTP=([on]=7474 [off]=7475)
for arm in on off; do
  dir=$BASE/$arm
  export NEO4J_URI=bolt://localhost:${BOLT[$arm]}
  if [[ $arm == on ]]; then export NEO4J_PASSWORD=$NEO4J_PASSWORD_ON; else export NEO4J_PASSWORD=$NEO4J_PASSWORD_OFF; fi
  if [[ ! -s $dir/dossier.jsonl.manifest.json ]]; then
    python scripts/report/export_agent_dossier.py --roster "$BASE/roster.json" --start 2021-09-24 --end 2021-10-07 --arm "$arm" --out "$dir/dossier.jsonl"
  fi
  python tools/ab3w_dossier_check.py "$dir/dossier.jsonl.manifest.json" 14
  mkdir -p "$dir/graph_backup"
  if [[ ! -s $dir/graph_backup/SHA256SUMS ]]; then
    neo=${NEO[$arm]}
    "$neo/bin/neo4j" stop >/dev/null 2>&1 || true
    for i in $(seq 1 60); do "$neo/bin/neo4j" status >/dev/null 2>&1 || break; sleep 2; done
    "$neo/bin/neo4j-admin" database dump neo4j --to-path="$dir/graph_backup" --overwrite-destination=true
    "$neo/bin/neo4j" version > "$dir/graph_backup/neo4j_version.txt"
    (cd "$dir/graph_backup" && sha256sum neo4j.dump neo4j_version.txt > SHA256SUMS)
    setsid nohup "$neo/bin/neo4j" start >/dev/null 2>&1 < /dev/null || true
    for i in $(seq 1 90); do curl -fsS -m 5 -o /dev/null "http://localhost:${HTTP[$arm]}" && break; sleep 2; done
  fi
  (cd "$dir/graph_backup" && sha256sum -c SHA256SUMS)
  echo "[$(date -Is)] 손 보존 $arm: 기억 모음·그래프 덤프 끝"
done
