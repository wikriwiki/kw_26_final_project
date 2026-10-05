#!/usr/bin/env bash
# Canonical existing SGLang stack: EXAONE 4.5 AWQ, A100 80GB x2.
# Keep /data/venv_sgl, the persistent cache and NCCL shared-memory workaround.
# Checkpoint quantization metadata is compressed-tensors; no AWQ kernel override.
set -euo pipefail

VENV="${VENV:-/data/venv_sgl}"
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
PORT="${PORT:-8000}"
# Preserve host.docker.internal access for the existing Docker app deployment.
# The separate Vast launcher binds privately to 127.0.0.1.
HOST="${HOST:-0.0.0.0}"
TP="${TP:-2}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"

source "$VENV/bin/activate"
echo "[serve-sglang] model=$MODEL revision=$MODEL_REVISION tp=$TP port=$PORT"
exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --revision "$MODEL_REVISION" \
  --served-model-name "$MODEL" \
  --port "$PORT" --host "$HOST" \
  --tp-size "$TP" \
  --attention-backend triton \
  --trust-remote-code
