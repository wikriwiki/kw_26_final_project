#!/bin/bash
# SGLang 서버 — LGAI-EXAONE/EXAONE-4.5-33B-AWQ (text-only)
#
# OpenAI 호환 endpoint: http://localhost:30000/v1
# llm_client.py 자동 감지: SGLang(30000) → vLLM(8000) 순서
#
# 실행:
#   bash scripts/serve/serve_exaone45_33b.sh
# 헬스체크:
#   curl -s http://localhost:30000/v1/models | jq

set -e

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
PORT="${PORT:-30000}"
HOST="${HOST:-0.0.0.0}"

# RTX 5090 32GB — AWQ 33B
exec python -m sglang.launch_server \
    --model-path "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --quantization awq_marlin \
    --dtype half \
    --context-length 8192 \
    --mem-fraction-static 0.90 \
    --trust-remote-code \
    --served-model-name exaone-4.5-33b-awq \
    --disable-radix-cache
