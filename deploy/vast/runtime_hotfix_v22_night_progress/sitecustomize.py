"""Pinned extension of the existing Day6 retry-contract runtime."""
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

MANIFEST_SHA256 = '19f2a2eaf2587b8ee85da0e5f14278877a86fd4de543346bd31bb325383455eb'
if os.environ.get('NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX') == '1':
    try:
        previous = Path('/workspace/no-smoking-runtime-hotfix-v22-retry-contract/sitecustomize.py')
        if hashlib.sha256(previous.read_bytes()).hexdigest() != '732eaab121623997c6f55a007160f1af237c1a18196210221c42d313fb502f74':
            raise ValueError('Previous retry contract changed')
        spec = importlib.util.spec_from_file_location('_previous_retry_contract', previous)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = Path(__file__).resolve().parent
        spec = importlib.util.spec_from_file_location('_night_progress_runtime', root / 'runtime.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules['_night_progress_runtime'] = module
        module.install(root, MANIFEST_SHA256)
    except Exception as exc:
        raise SystemExit(f'Night progress installation failed: {type(exc).__name__}: {exc}') from exc
