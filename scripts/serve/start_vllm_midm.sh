#!/bin/bash
# Midm 2.0 Base Instruct vLLM 기동 스크립트
# 최적화: max-model-len 6144, prefix-caching, workers 32 → 20 가정
set -e
source $HOME/venv_sim/bin/activate
export HF_HOME=$HOME/.cache/huggingface
NV=$HOME/venv_sim/lib/python3.11/site-packages/nvidia
export CUDA_HOME=$NV/cu13
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$NV/cu13/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cuda_runtime/lib:$NV/cuda_nvrtc/lib:$NV/curand/lib:$NV/cufft/lib:$NV/cuda_cupti/lib:$LD_LIBRARY_PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=TORCH_SDPA
export FLASHINFER_DISABLE=1

# v3 baseline 변경점:
# - max-model-len 8192 → 6144 (잘림 위험 <1% 측정 확인, KV cache 25% 여유)
# - --enable-prefix-caching 명시 (시스템 prompt + 페르소나 prefix 캐시 적중)
# - gpu-memory-utilization 0.92 → 0.95 (KV cache 추가 확보)
exec python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL:-K-intelligence/Midm-2.0-Base-Instruct}" \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 6144 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --trust-remote-code \
    --disable-log-requests \
    --served-model-name midm-2.0-base-instruct K-intelligence/Midm-2.0-Base-Instruct
