"""GPU-pool routing layer on top of the pinned v22 night-progress runtime.

Generation code, prompts, validation, retry budgets and persistence are the
unchanged night-progress chain. This layer only points the LLM client of the
simulator processes at the local inference pool proxy when that proxy is up.
If the proxy is absent or disabled the process keeps the Vast-only endpoint.
"""
import hashlib
import importlib.util
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

PREVIOUS = Path('/workspace/no-smoking-runtime-hotfix-v22-night-progress/sitecustomize.py')
PREVIOUS_SHA256 = 'b22ba7cc53837a3c9d0194c50a3e4f06f84319f95eeb870add8f588cf976db92'
DIRECT_URL = 'http://127.0.0.1:8000/v1'
POOL_ORIGIN = 'http://127.0.0.1:30000'
DISABLE_FLAG = Path('/workspace/no-smoking-gpu-pool/DISABLED')
ROUTE_LOG = Path('/workspace/no-smoking-results/integration-main-v22-1154-gpu-pool-routing.jsonl')


def _pool_reference():
    try:
        with urllib.request.urlopen(POOL_ORIGIN + '/pool/status', timeout=3) as response:
            status = json.load(response)
        local = [b for b in status['backends'] if b.get('local')]
        if len(local) == 1 and local[0]['healthy'] and status.get('reference_identity_sha256'):
            return status['reference_identity_sha256']
    except Exception:
        pass
    return None


def _route():
    # Only processes that were handed the frozen direct endpoint are simulator
    # processes; the supervisor and backup tools never carry this variable.
    if os.environ.get('SGLANG_BASE_URL') != DIRECT_URL:
        return
    reference = None if DISABLE_FLAG.exists() else _pool_reference()
    if reference:
        os.environ['SGLANG_BASE_URL'] = POOL_ORIGIN + '/v1'
        os.environ['NO_SMOKING_GPU_POOL_REFERENCE_SHA256'] = reference
    try:
        with ROUTE_LOG.open('a', encoding='utf-8') as out:
            out.write(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                                  'orig_argv': getattr(sys, 'orig_argv', [])[1:3],
                                  'decision': 'pool' if reference else 'direct',
                                  'base_url': os.environ['SGLANG_BASE_URL'],
                                  'pool_reference_identity_sha256': reference,
                                  'disabled_flag': DISABLE_FLAG.exists()}) + '\n')
    except OSError:
        pass


if os.environ.get('NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX') == '1':
    try:
        if hashlib.sha256(PREVIOUS.read_bytes()).hexdigest() != PREVIOUS_SHA256:
            raise ValueError('Night-progress runtime entry point changed')
        spec = importlib.util.spec_from_file_location('_previous_night_progress', PREVIOUS)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _route()
    except SystemExit:
        raise
    except Exception as exc:
        # An invalid deployment must not silently continue with unpatched code.
        raise SystemExit(f'GPU-pool layer installation failed: {type(exc).__name__}: {exc}') from exc
