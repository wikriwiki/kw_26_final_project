"""Bounded source provenance and atomic local artifacts. Never reads secrets."""
import os
from functools import lru_cache
from pathlib import Path
import tempfile

from evidence_contract import canonical, digest


@lru_cache(maxsize=1)
def source_fingerprint():
    root = Path(__file__).resolve().parent
    # Include simulator adapters and prompt modules, not only the entry point.
    # Input graph/model weights still require a separate deployment manifest.
    paths = sorted(root.rglob('*.py'))
    return digest({p.relative_to(root).as_posix(): p.read_text(encoding='utf-8') for p in paths})


def execution_fingerprint():
    settings = ('LLM_MODE','SIM_ENVIRONMENT','SIM_PROMPT_VARIANT','CONSUMPTION_MODEL',
                'EXP_PAYMENT_CHOICE','EXP_ELIGIBLE_SHARE','EXP_GRANT_USE',
                'POLICY_BACKTEST_DETERMINISTIC')
    return digest({'source':source_fingerprint(), 'settings':{k:os.environ.get(k) for k in settings}})


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as output:
            output.write(canonical(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)
