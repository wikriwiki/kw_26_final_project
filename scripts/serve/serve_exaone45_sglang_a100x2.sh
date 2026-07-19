#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# EXP-001 실제 서빙 스택 — EXAONE-4.5-33B-AWQ, SGLang(TP2), A100 80GB × 2
#
# 왜 vLLM이 아니라 SGLang인가:
#   vLLM(0.11~0.18)은 이 모델의 compressed-tensors(pack-quantized int4) 스키마를
#   로드하지 못하는 이력이 있어(serve_exaone45_awq_a100x2.sh는 그 시도용), EXP-001은
#   EXAONE-4.5 지원 SGLang 포크로 서빙했다. 이 스크립트가 실전에서 돈 버전이다.
#
# 사전: scripts/deploy/install_sglang_exaone45.sh 로 /data/venv_sgl 구성.
# 주의: 컨테이너 /dev/shm 이 64MB로 작으면 TP2 NCCL이 실패한다.
#       NCCL_CUMEM_ENABLE=1 로 cuMem 채널을 써서 shm 한계를 우회한다.
# ═══════════════════════════════════════════════════════════════════════
set -euo pipefail

VENV="${VENV:-/data/venv_sgl}"
MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"

source "$VENV/bin/activate"
rm -f /dev/shm/nccl-* 2>/dev/null || true
export NCCL_CUMEM_ENABLE=1

echo "[serve-sglang] model=$MODEL tp=$TP port=$PORT shm우회=NCCL_CUMEM_ENABLE=1"
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" --host 0.0.0.0 \
  --tp-size "$TP" \
  --attention-backend triton \
  --trust-remote-code
