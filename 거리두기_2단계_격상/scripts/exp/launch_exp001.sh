#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EXP-001 실전 런 — 민생회복 소비쿠폰(P010) 14일 · EXAONE-4.5-33B-AWQ(SGLang TP2)
#
# 산출물은 NAS(/home/ubuntu/data/exp001)로 직접 기록 → 컨테이너 소실에도 보존되고,
# 재시작 시 checkpoint/metrics로 resume된다. Neo4j·모델은 빠른 로컬(/data)에 둔다.
#
# 사전: Neo4j 기동 + P010 적재 + Day0 시드 완료(setup_gpulive_exp001.sh s3~s6),
#       SGLang 서버 기동(scripts/serve/serve_exaone45_sglang_a100x2.sh, :8000).
# 사용: WORKERS=64 nohup bash scripts/exp/launch_exp001.sh > $NAS/logs/run.log 2>&1 &
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

REPO="${REPO:-/data/exp001_repo}"
VENV="${VENV:-/data/venv}"
NAS="${NAS:-/home/ubuntu/data/exp001}"
SIM_START="${SIM_START:-2025-07-14}"
SIM_DAYS="${SIM_DAYS:-14}"
AGENTS="${AGENTS:-7500}"
WORKERS="${WORKERS:-64}"

mkdir -p "$NAS/sim_output" "$NAS/logs"
source "$VENV/bin/activate"
export NEO4J_URI=bolt://localhost:7687 NEO4J_USER=neo4j NEO4J_PASSWORD="${NEO4J_PW:-exp001pass}" NEO4J_DATABASE=neo4j
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODE=exaone_4_5
export SIM_OUTPUT_DIR="$NAS/sim_output" PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8
export BACKUP_DIR="${BACKUP_DIR:-$NAS/backup}"
export DAWN_PERSONA_CACHE="${DAWN_PERSONA_CACHE:-1}"
export DAWN_POLICY_CACHE="${DAWN_POLICY_CACHE:-1}"
# 정책 POI 정렬 가점은 결과를 사전 유도하므로 기본 비활성. 사용 가능 표시는 유지.
export POLICY_POI_SORT_BOOST="${POLICY_POI_SORT_BOOST:-0}"
cd "$REPO"
exec python scripts/sim/run_simulation.py \
  --start "$SIM_START" --days "$SIM_DAYS" --limit "$AGENTS" --workers "$WORKERS"
