#!/usr/bin/env python3
"""Recovery controller for the GPU-pool layer.

Loads the pinned night-progress controller unchanged and adds two things:
the supervisor is started with the GPU-pool layer first on PYTHONPATH, and the
inference pool proxy keeper is restarted if it is not running. Validation,
locking, resume decisions and completion handling are the original code.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

BASE = Path('/workspace/no-smoking-runtime-hotfix-v22-night-progress/resume_runtime.py')
BASE_SHA256 = 'c798e9f8d0ff15cad7178639f846ffe3c4c20925833f4fc1039004c1c85701d6'
LAYER = Path('/workspace/no-smoking-runtime-hotfix-v22-gpu-pool')
MANIFEST_SHA256 = '3ce5ea010f86f7ce0fc695ca0807d4ed539d65450dac974d0e0bce143da6bb0a'
POOL = Path('/workspace/no-smoking-gpu-pool')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_layer():
    raw = (LAYER / 'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('GPU-pool manifest changed')
    for name, expected in json.loads(raw)['files'].items():
        path = (LAYER / name).resolve()
        if not path.is_relative_to(LAYER) or sha256(path) != expected:
            raise ValueError(f'GPU-pool runtime changed: {name}')
    return hashlib.sha256(raw).hexdigest()


def load_base():
    if sha256(BASE) != BASE_SHA256:
        raise ValueError('Night-progress controller changed')
    spec = importlib.util.spec_from_file_location('_night_progress_controller', BASE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Subprocess:
    """subprocess facade for the base controller: only the supervisor launch differs."""

    def __init__(self, base, manifest_sha256):
        self._base = base
        self._manifest_sha256 = manifest_sha256
        self._supervisor = str(base.PROJECT / 'deploy/vast/run_shared.py')

    def __getattr__(self, name):
        return getattr(subprocess, name)

    def Popen(self, args, *pos, **kwargs):
        if isinstance(args, (list, tuple)) and len(args) >= 2 and str(args[1]) == self._supervisor:
            if validate_layer() != self._manifest_sha256:
                raise ValueError('GPU-pool manifest changed after controller start')
            env = dict(kwargs['env'])
            env['PYTHONPATH'] = f"{LAYER}:{env['PYTHONPATH']}"
            kwargs['env'] = env
            self._base.event('gpu_pool_layer_attached', manifest_sha256=self._manifest_sha256)
        return subprocess.Popen(args, *pos, **kwargs)


def ensure_pool(base):
    """Best effort: the experiment never depends on the proxy being up."""
    try:
        if (POOL / 'DISABLED').exists() or base.process_pid(str(POOL / 'gpu_pool_proxy.py').encode()):
            return
        with (POOL / 'keeper.log').open('ab') as out:
            subprocess.Popen(['bash', str(POOL / 'keeper.sh')], cwd=POOL, stdin=subprocess.DEVNULL,
                             stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        base.event('gpu_pool_keeper_started')
    except Exception as exc:
        base.event('gpu_pool_keeper_error', error=f'{type(exc).__name__}: {exc}'[:300])


def main(dry_run=False):
    base = load_base()
    manifest_sha256 = validate_layer()
    base.subprocess = _Subprocess(base, manifest_sha256)
    recover_once = base.recover_once

    def recover_with_pool():
        ensure_pool(base)
        return recover_once()
    if not dry_run:
        base.recover_once = recover_with_pool
        base.event('gpu_pool_controller_starting', manifest_sha256=manifest_sha256,
                   base_controller_sha256=BASE_SHA256)
    base.main(dry_run)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    main(parser.parse_args().dry_run)
