#!/bin/bash
# Qwen3-8B-AWQ (Qwen 공식 AWQ)
# Qwen3.5-9B-AWQ swap 실패 시 fallback (같은 8B family라 분석 mix 영향 최소)

MODEL="Qwen/Qwen3-8B-AWQ"
PORT=8000
GPU_UTIL=0.92
MAX_MODEL_LEN=16384

echo "============================================"
echo " vLLM Qwen3-8B-AWQ (공식 AWQ) 시작"
echo " 모델: $MODEL"
echo " 포트: $PORT"
echo " GPU util: $GPU_UTIL / max-len: $MAX_MODEL_LEN"
echo "============================================"

vllm serve "$MODEL" \
    --quantization awq_marlin \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --enable-prefix-caching \
    --trust-remote-code
