#!/usr/bin/env bash
# Run inside the pinned PyTorch base after uploading the snapshot.
set -euo pipefail
umask 077
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
command -v nvidia-smi >/dev/null
command -v timeout >/dev/null
command -v git >/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'
python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("CUDA:", torch.version.cuda)'
SGLANG_VENV="${SGLANG_VENV:-/workspace/venv_sgl}"
# The existing project installer pins the LG-supported fork and compatibility overrides.
VENV="$SGLANG_VENV" timeout --signal=TERM --kill-after=30 1800 bash scripts/deploy/install_sglang_exaone45.sh
python3 -m venv --system-site-packages .venv-no-smoking
.venv-no-smoking/bin/python -m pip install -r deploy/vast/requirements-runtime.txt
mkdir -p output/experiments/no_smoking_zone/runtime
.venv-no-smoking/bin/python scripts/sim/prompt_budget.py --download output/experiments/no_smoking_zone/runtime/tokenizer
"$SGLANG_VENV/bin/python" deploy/vast/validate_server.py > output/experiments/no_smoking_zone/runtime/server-environment.json
"$SGLANG_VENV/bin/python" -m pip freeze > output/experiments/no_smoking_zone/runtime/server-packages.txt
.venv-no-smoking/bin/python -m pip freeze > output/experiments/no_smoking_zone/runtime/client-packages.txt
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > output/experiments/no_smoking_zone/runtime/gpu.csv
.venv-no-smoking/bin/python scripts/experiments/no_smoking_zone.py --help
printf '%s\n' 'Bootstrap finished. Run the fixture smoke test before starting live inference.'
