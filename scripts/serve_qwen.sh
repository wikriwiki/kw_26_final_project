#!/usr/bin/env bash
# Launch sglang server with Qwen3.5-4B (dev / fast-iteration mode).
#
# Prerequisites (one-time, separate venv recommended):
#   uv venv .venv-qwen
#   source .venv-qwen/bin/activate
#   uv pip install 'sglang[all]>=0.4'
#
# Hardware: A100 80GB single GPU (tp-size 1). Adjust via env vars.
#
# Tuning knobs (set via env):
#   LLM_PORT             — server port (default 30000)
#   LLM_CONTEXT_LENGTH   — max context (default 32768; full model is 262144)
#   LLM_MEM_FRACTION     — KV cache fraction (default 0.88)
#   LLM_MAX_RUNNING      — server-side concurrency (default 256 for 4B model)

set -euo pipefail

PORT="${LLM_PORT:-30000}"
CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${LLM_MEM_FRACTION:-0.88}"
MAX_RUNNING="${LLM_MAX_RUNNING:-256}"

exec python -m sglang.launch_server \
    --model-path Qwen/Qwen3.5-4B \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size 1 \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype fp8_e5m2 \
    --enable-radix-cache \
    --max-running-requests "$MAX_RUNNING" \
    --context-length "$CONTEXT_LENGTH" \
    --reasoning-parser qwen3
