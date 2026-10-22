#!/bin/bash
# vLLM 서버 — LGAI-EXAONE/EXAONE-4.5-33B-AWQ (text-only)
#
# - AWQ 4-bit quant → RTX 5090 32GB 단일 GPU fit (가중치 ~17GB + KV cache 여유)
# - 멀티모달 비활성 (text-to-text only): --task generate + LIMIT_MM_PER_PROMPT 0
# - OpenAI 호환 endpoint: http://localhost:8000/v1
#
# llm_client.py 호환: VLLM_FALLBACK_URL = http://localhost:8000/v1
#
# 실행:
#   bash scripts/serve/run_vllm_exaone45_33b.sh
# 헬스체크:
#   curl -s http://localhost:8000/v1/models | jq

set -e

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# RTX 5090 (32GB) — AWQ 33B
#  - max-model-len 8192 (시뮬 프롬프트 ~3-5k + 응답 ~2k 여유)
#  - gpu-memory-utilization 0.92 — KV cache 충분 (multi-pass 시뮬)
#  - swap-space 4GiB — 긴 batch 대비
#  - 멀티모달 완전 비활성: --task generate (멀티모달 어댑터 로드 X)
#  - dtype half (AWQ 자체 4-bit)
exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --task generate \
    --host "$HOST" \
    --port "$PORT" \
    --quantization awq_marlin \
    --dtype half \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --swap-space 4 \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-requests \
    --served-model-name exaone-4.5-33b-awq
