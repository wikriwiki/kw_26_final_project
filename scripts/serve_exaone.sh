#!/usr/bin/env bash
# Launch sglang server with EXAONE-4.5-33B-FP8 (competition / production mode).
#
# IMPORTANT: EXAONE-4.5 needs a FORK of sglang + transformers (not yet upstreamed
# as of 2026-05). Pin commits when you find a stable combination.
#
# Prerequisites (one-time, SEPARATE venv from Qwen — fork conflicts with upstream):
#   uv venv .venv-exaone
#   source .venv-exaone/bin/activate
#   uv pip install 'git+https://github.com/lkm2835/sglang.git@add-exaone4_5#subdirectory=python&egg=sglang[all]'
#   uv pip install 'git+https://github.com/nuxlear/transformers.git@add-exaone4_5-v5.3.0.dev0'
#
# Hardware: A100 80GB single GPU (tp-size 1). If the fork rejects tp-size 1,
# fall back to 2× A100 40GB with LLM_TP_SIZE=2.
#
# Tuning knobs (env):
#   LLM_PORT             — server port (default 30000)
#   LLM_CONTEXT_LENGTH   — max context (default 32768; full model is 262144)
#   LLM_MEM_FRACTION     — KV fraction (default 0.88 — ~44GB KV on 80GB GPU)
#   LLM_MAX_RUNNING      — server-side concurrency (default 96 for 33B)
#   LLM_TP_SIZE          — tensor parallel size (default 1)

set -euo pipefail

PORT="${LLM_PORT:-30000}"
CONTEXT_LENGTH="${LLM_CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${LLM_MEM_FRACTION:-0.88}"
MAX_RUNNING="${LLM_MAX_RUNNING:-96}"
TP_SIZE="${LLM_TP_SIZE:-1}"

exec python -m sglang.launch_server \
    --model-path LGAI-EXAONE/EXAONE-4.5-33B-FP8 \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tp-size "$TP_SIZE" \
    --mem-fraction-static "$MEM_FRACTION" \
    --kv-cache-dtype fp8_e5m2 \
    --enable-radix-cache \
    --max-running-requests "$MAX_RUNNING" \
    --context-length "$CONTEXT_LENGTH"
