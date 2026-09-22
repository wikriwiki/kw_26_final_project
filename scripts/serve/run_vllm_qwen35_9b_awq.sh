#!/bin/bash
# Qwen3.5-9B-AWQ (community QuantTrio 빌드)
# VRAM ~5GB → KV cache 여유 → workers 80+ 가능

MODEL="QuantTrio/Qwen3.5-9B-AWQ"
PORT=8000
GPU_UTIL=0.92
MAX_MODEL_LEN=16384

echo "============================================"
echo " vLLM Qwen3.5-9B-AWQ 시작"
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
