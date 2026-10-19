"""The checkpoint audit must catch a broken balance, including graph-external spend."""
from __future__ import annotations

import gzip
import importlib.util
import io
import json
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
module_path = ROOT / 'scripts/report/audit_p012_checkpoint_accounting.py'
spec = importlib.util.spec_from_file_location('audit_p012_checkpoint_accounting', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _archive(path, kind, rows):
    content = gzip.compress(''.join(json.dumps(row) + '\n' for row in rows).encode())
    with tarfile.open(path, 'w:gz') as tar:
        info = tarfile.TarInfo(f'graph/checkpoint_{kind}.jsonl.gz')
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


def test_detects_missing_online_debit_in_current_balance(tmp_path):
    before = tmp_path / 'before.tar.gz'
    after = tmp_path / 'after.tar.gz'
    _archive(before, 'state', [{'aid': 'a', 'day': '2021-10-15',
                                'state': {'balance': 500, 'month_spent': 200}}])
    spend = {'aid': 'a', 'day': '2021-10-16',
             'spend': {'actual_spent': 100, 'spent_from_policy': '{}'}}
    state = {'aid': 'a', 'day': '2021-10-16',
             'state': {'balance': 430, 'month_spent': 320,
                       'income_today': 50, 'online_spent': 20}}
    with tarfile.open(after, 'w:gz') as tar:
        for kind, rows in (('state', [state]), ('spend', [spend])):
            content = gzip.compress(''.join(json.dumps(row) + '\n' for row in rows).encode())
            info = tarfile.TarInfo(f'graph/checkpoint_{kind}.jsonl.gz')
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    good = module.audit(before, after, '2021-10-15', '2021-10-16')
    assert good['balance_mismatch_count'] == good['month_mismatch_count'] == 0
    assert good['gross'] == 100

    # If the stored State omitted the graph-external debit, the audit must fail.
    state['state']['balance'] = 450
    with tarfile.open(after, 'w:gz') as tar:
        for kind, rows in (('state', [state]), ('spend', [spend])):
            content = gzip.compress(''.join(json.dumps(row) + '\n' for row in rows).encode())
            info = tarfile.TarInfo(f'graph/checkpoint_{kind}.jsonl.gz')
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    broken = module.audit(before, after, '2021-10-15', '2021-10-16')
    assert broken['balance_mismatch_count'] == 1
