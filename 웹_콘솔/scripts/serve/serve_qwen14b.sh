#!/bin/bash
# =============================================================================
# vLLM 서버 — Qwen3-14B-AWQ (중간 사이즈, RTX 5090 32GB 여유 fit)
# =============================================================================
# 32B 대비 추론 ~2배 빠름. 한국어 품질 유지.
# 32B와 동일한 vLLM 인터페이스라 llm_client.py에서 LLM_MODE=qwen14b로 그대로 호환.
#
# 사전: conda activate vllm
# 실행: bash serve_qwen14b.sh
# 테스트:
#   curl http://localhost:8000/v1/models
# =============================================================================

MODEL="${MODEL:-Qwen/Qwen3-14B-AWQ}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

echo "============================================"
echo " vLLM 서버 시작"
echo " 모델: $MODEL  (AWQ 양자화)"
echo " 포트: $PORT"
echo " GPU 메모리 사용률: $GPU_UTIL"
echo " 최대 컨텍스트: $MAX_MODEL_LEN tokens"
echo "============================================"

exec vllm serve "$MODEL" \
    --quantization awq_marlin \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --enable-prefix-caching \
    --trust-remote-code
