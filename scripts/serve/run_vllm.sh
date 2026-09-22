#!/bin/bash
# =============================================================================
# vLLM 서버 시작 스크립트 — Qwen3-32B-AWQ (단일 모델 단일화)
# =============================================================================
#
# 페르소나 생성·스케줄 Stage 1·2·Night 의도 분류·정책 NL 추출 모두 이 모델 단독.
# 결정: 2026-05-04 (Gemma/EXAONE 후보 폐기). project_vllm_model_decision 메모 참조.
#
# 사전 준비:
#   conda activate vllm  (vllm>=0.19.0 + openai sdk)
#
# 실행:
#   bash run_vllm.sh
#
# 모델 로딩 확인:
#   curl http://localhost:8000/v1/models
#
# 테스트:
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{"model":"Qwen/Qwen3-32B-AWQ",
#          "messages":[{"role":"user","content":"안녕하세요"}],
#          "max_tokens":50}'
# =============================================================================

MODEL="Qwen/Qwen3-32B-AWQ"
PORT=8000
GPU_UTIL=0.92
MAX_MODEL_LEN=8192

echo "============================================"
echo " vLLM 서버 시작"
echo " 모델: $MODEL  (AWQ 양자화)"
echo " 포트: $PORT"
echo " GPU 메모리 사용률: $GPU_UTIL"
echo " 최대 컨텍스트: $MAX_MODEL_LEN tokens"
echo " 옵션: prefix-caching 활성 (Stage 1·2 SYSTEM/persona 공유)"
echo "============================================"

vllm serve "$MODEL" \
    --quantization awq_marlin \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --port "$PORT" \
    --enable-prefix-caching \
    --trust-remote-code
