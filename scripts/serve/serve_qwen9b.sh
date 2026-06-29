#!/usr/bin/env bash
# SGLang server — Qwen3.5-9B (개발/디버깅용)
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.85}"
MAX_LEN="${MAX_LEN:-8192}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --tp-size 1 \
  --port "$PORT" \
  --host 0.0.0.0 \
  --mem-fraction-static "$MEM_FRAC" \
  --context-length "$MAX_LEN" \
  --enable-metrics \
  --trust-remote-code
