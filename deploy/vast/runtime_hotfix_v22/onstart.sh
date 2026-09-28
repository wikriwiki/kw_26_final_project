#!/usr/bin/env bash
# Vast runs this after a container start. The launcher verifies the frozen
# source, restores the exact model settings, resumes once, then starts the
# backup watchdog without premature Vast stop. No passwords are printed here.
set -euo pipefail
root=/workspace/no-smoking-results
mkdir -p "$root"
(
    exec 9>"$root/integration-main-v22-1154-onstart.lock"
    flock -n 9 || exit 0
    exec /workspace/no-smoking-project-v22-perf-final/.venv-no-smoking/bin/python \
        /workspace/no-smoking-runtime-hotfix-v22-skip/resume_runtime.py \
        >>"$root/integration-main-v22-1154-onstart.log" 2>&1
) </dev/null >/dev/null 2>&1 &
