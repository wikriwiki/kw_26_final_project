#!/usr/bin/env bash
# SGLang server — EXAONE-4.5-33B-FP8 (국내 대회용)
# A100 80GB 1장: FP8 33B ≈ 33GB + KV cache 여유.
set -euo pipefail

MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-FP8}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.88}"
MAX_LEN="${MAX_LEN:-8192}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization fp8 \
  --tp-size 1 \
  --port "$PORT" \
  --host 0.0.0.0 \
  --mem-fraction-static "$MEM_FRAC" \
  --context-length "$MAX_LEN" \
  --enable-metrics \
  --trust-remote-code
