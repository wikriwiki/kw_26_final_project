#!/usr/bin/env bash
# SGLang server — Qwen3-32B-AWQ (기존 기본 모델)
# A100 80GB 1장 가정. RadixAttention prefix cache 활성화.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-32B-AWQ}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.88}"
MAX_LEN="${MAX_LEN:-8192}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization awq \
  --tp-size 1 \
  --port "$PORT" \
  --host 0.0.0.0 \
  --mem-fraction-static "$MEM_FRAC" \
  --context-length "$MAX_LEN" \
  --enable-metrics \
  --trust-remote-code
