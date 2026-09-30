#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/no-smoking-results
# Same lifetime lock and 15s checks as the night-progress controller, plus the GPU-pool layer.
nohup /workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python \
    /workspace/no-smoking-runtime-hotfix-v22-gpu-pool/resume_runtime.py \
    >>/workspace/no-smoking-results/integration-main-v22-1154-recovery-controller.log 2>&1 \
    </dev/null &
