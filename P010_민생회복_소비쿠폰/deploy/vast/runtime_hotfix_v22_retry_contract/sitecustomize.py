"""Verified, day-gated extension of the deployed v22 runtime layers."""
import hashlib
import importlib.util
import os
from pathlib import Path

MANIFEST_SHA256 = '7f0c6f2d4ed7b956b1371898b3736db511150625ae71db1f6044819520166b99'
PREVIOUS_SHA256 = '7dd2a8a425724ba5c8c658c04a0fdfd9190ec3b21d06be8b23350c9844eb97d5'

if os.environ.get('NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX') == '1':
    try:
        previous = Path('/workspace/no-smoking-runtime-hotfix-v22-grounding/sitecustomize.py')
        if hashlib.sha256(previous.read_bytes()).hexdigest() != PREVIOUS_SHA256:
            raise ValueError('Prior grounding runtime changed')
        spec = importlib.util.spec_from_file_location('_prior_v22_grounding', previous)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = Path(__file__).resolve().parent
        import json
        raw = (root / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
            raise ValueError('Retry manifest changed')
        runtime = root / 'runtime.py'
        if hashlib.sha256(runtime.read_bytes()).hexdigest() != json.loads(raw)['files']['runtime.py']:
            raise ValueError('Retry runtime changed')
        spec = importlib.util.spec_from_file_location('_v22_retry_contract', runtime)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.install(root, MANIFEST_SHA256)
    except Exception as exc:
        # Python normally ignores sitecustomize exceptions and would silently
        # continue with unpatched code. An invalid deployment must not do that.
        raise SystemExit(f'Retry-contract installation failed: {type(exc).__name__}: {exc}') from exc
