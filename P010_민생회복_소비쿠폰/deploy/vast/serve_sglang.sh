#!/usr/bin/env bash
# Keep inference private to this instance; SSH provides remote access.
set -euo pipefail
umask 077
MODEL="${MODEL:-LGAI-EXAONE/EXAONE-4.5-33B-AWQ}"
[[ "$MODEL" == LGAI-EXAONE/EXAONE-4.5-* ]] || { printf '%s\n' 'Only LG EXAONE 4.5 models are allowed' >&2; exit 2; }
if [[ "$MODEL" == LGAI-EXAONE/EXAONE-4.5-33B-AWQ ]]; then
  MODEL_REVISION="${MODEL_REVISION:-31e6a965d0661bbe4a8b895e22a77f8271772ba0}"
fi
: "${MODEL_REVISION:?Set the matching LG model revision}"
if [[ "$MODEL" == LGAI-EXAONE/EXAONE-4.5-33B-AWQ && -n "${QUANTIZATION:-}" ]]; then
  printf '%s\n' 'EXAONE AWQ uses compressed-tensors metadata; leave QUANTIZATION empty' >&2
  exit 2
fi
PORT="${PORT:-8000}"
SGLANG_VENV="${SGLANG_VENV:-/workspace/venv_sgl}"
SERVER_PYTHON="$SGLANG_VENV/bin/python"
export HF_HOME="${HF_HOME:-/workspace/huggingface}"
args=(--model-path "$MODEL" --host 127.0.0.1 --port "$PORT"
  --served-model-name "$MODEL" --dtype auto
  --context-length "${MAX_MODEL_LEN:-16384}"
  --max-running-requests "${MAX_NUM_SEQS:-4}"
  --tp-size "${TENSOR_PARALLEL_SIZE:-1}"
  --mem-fraction-static "${GPU_MEMORY_UTILIZATION:-0.88}"
  --attention-backend "${ATTENTION_BACKEND:-triton}"
  --revision "$MODEL_REVISION" --random-seed 42 --trust-remote-code)
if [[ -n "${QUANTIZATION:-}" ]]; then
  args+=(--quantization "$QUANTIZATION")
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  printf '%q ' "$SERVER_PYTHON" -m sglang.launch_server "${args[@]}"
  printf '\n'
  exit 0
fi
command -v nvidia-smi >/dev/null
[ -x "$SERVER_PYTHON" ] || { printf '%s\n' 'Run deploy/vast/bootstrap.sh first' >&2; exit 2; }
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
mkdir -p output/experiments/no_smoking_zone/runtime
# Record only the non-secret launch arguments, never dump the environment.
"$SERVER_PYTHON" - "$MODEL_REVISION" "${args[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path('deploy/vast').resolve()))
from validate_server import server_environment
config = server_environment()
config.update(model_revision=sys.argv[1], argv=[sys.executable, '-m', 'sglang.launch_server', *sys.argv[2:]])
path = Path('output/experiments/no_smoking_zone/runtime/server-config.json')
path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
PY
exec "$SERVER_PYTHON" -m sglang.launch_server "${args[@]}"
