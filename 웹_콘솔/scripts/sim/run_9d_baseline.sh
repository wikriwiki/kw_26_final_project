#!/bin/bash
# 9일 baseline 시뮬 (무정책) — 7,500 agent, Qwen3.6-35B-A3B-AWQ, workers 32
# 출력: C:/Users/Administrator/sim_output_9d (Google Drive OSError 회피)
# resume: run_simulation 내부 jsonl status=ok + done_checkpoint 가드
cd "/g/내 드라이브/Kw/final_project"
export LLM_MODE=qwen36_35b_a3b_awq
export SIM_OUTPUT_DIR="C:/Users/Administrator/sim_output_9d"
# Neo4j 접속을 환경변수로 export → .env 파일에 의존 X (Google Drive 일시 unmount 안전)
set -a
. data/neo4j_load/.env
set +a
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 9D BASELINE START (7500 agent, days=9, workers=32) ==="
python3 -u scripts/sim/run_simulation.py --start 2026-05-25 --days 9 --workers 32
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === 9D BASELINE EXIT code=$? ==="
