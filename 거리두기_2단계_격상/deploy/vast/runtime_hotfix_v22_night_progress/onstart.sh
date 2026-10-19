#!/usr/bin/env bash
set -euo pipefail
mkdir -p /workspace/no-smoking-results
# The Python controller holds a process lock for its lifetime and checks every 15s.
nohup /workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python \
    /workspace/no-smoking-runtime-hotfix-v22-night-progress/resume_runtime.py \
    >>/workspace/no-smoking-results/integration-main-v22-1154-recovery-controller.log 2>&1 \
    </dev/null &
