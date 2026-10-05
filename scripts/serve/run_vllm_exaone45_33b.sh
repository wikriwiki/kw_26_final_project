#!/usr/bin/env bash
# Optional vLLM EXAONE 4.5 AWQ entry; default serving uses SGLang.
set -euo pipefail
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/run_vllm.sh" "$@"
