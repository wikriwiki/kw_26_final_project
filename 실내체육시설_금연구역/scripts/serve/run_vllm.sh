#!/usr/bin/env bash
# Optional vLLM LG EXAONE entry; the project default remains SGLang.
# Prepare a compatible vLLM environment separately before using this entry.
# AWQ weights use compressed-tensors metadata; do not force an AWQ kernel name.
set -euo pipefail

DEFAULT_MODEL="LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
MODEL="${MODEL:-$DEFAULT_MODEL}"
case "$MODEL" in
  LGAI-EXAONE/*) ;;
  *) echo "MODEL must use the official LGAI-EXAONE namespace." >&2; exit 2 ;;
esac
if [[ "$MODEL" == "$DEFAULT_MODEL" ]]; then
  MODEL_REVISION="${MODEL_REVISION:-31e6a965d0661bbe4a8b895e22a77f8271772ba0}"
else
  : "${MODEL_REVISION:?Set MODEL_REVISION explicitly when changing the LG model.}"
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TP="${TP:-1}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_LEN="${MAX_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

echo "[serve] model=$MODEL revision=$MODEL_REVISION tp=$TP port=$PORT"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --tensor-parallel-size "$TP" \
  --dtype auto \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --enable-prefix-caching \
  --language-model-only
