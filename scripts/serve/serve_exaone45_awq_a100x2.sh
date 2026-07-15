#!/usr/bin/env bash
# vLLM server — EXAONE-4.5-33B-AWQ, A100 80GB × 2 (tensor parallel)
#
# EXP-001 (GPU LIVE): 국산 모델 제약 → LG EXAONE 채택. "GPU 모두 사용" 요구 → TP=2.
# 주의: 과거 vllm 0.11에서 이 AWQ의 quantization schema 미지원 이력(llm_client.py 참조).
#       → 서버에서 최신 vllm(>=0.12) 설치 후 본 스크립트 우선 시도.
#       실패 시 폴백(동일 모델 FP8, 국산 유지):
#         MODEL=LGAI-EXAONE/EXAONE-4.5-33B-FP8 QUANT=fp8 bash scripts/serve/serve_exaone45_awq_a100x2.sh
#
# served model name = HF id 그대로 (llm_client가 spec.hf_id로 호출하므로 반드시 일치).
set -euo pipefail

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
QUANT="${QUANT:-awq}"            # awq | awq_marlin | fp8 (A100=sm80: marlin 가속 가능)
PORT="${PORT:-8000}"             # llm_client 자동감지: SGLang 30000 / vLLM 8000
TP="${TP:-2}"                    # A100 × 2 전부 사용
GPU_UTIL="${GPU_UTIL:-0.92}"
MAX_LEN="${MAX_LEN:-8192}"

echo "[serve] model=$MODEL quant=$QUANT tp=$TP port=$PORT gpu_util=$GPU_UTIL"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --quantization "$QUANT" \
  --tensor-parallel-size "$TP" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --trust-remote-code
