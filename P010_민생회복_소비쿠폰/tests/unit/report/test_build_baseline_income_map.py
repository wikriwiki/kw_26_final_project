"""Neutral-period calibration must reject policy exposure and disclose gaps."""
from __future__ import annotations

import importlib.util
import io
import json
import tarfile
from datetime import date
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
module_path = ROOT / 'scripts/report/build_baseline_income_map.py'
spec = importlib.util.spec_from_file_location('build_baseline_income_map', module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _archive(path, policy_hit=0):
    days = {
        '2021-10-01': [
            {'aid': 'a', 'status': 'ok', 'policy_hits': policy_hit,
             'cm_today_total': 20, 'cm_online_total': 30,
             'cm_today_total_incl_online': 50},
            {'aid': 'b', 'status': 'ok', 'policy_hits': 0,
             'cm_today_total': 40, 'cm_online_total': 20,
             'cm_today_total_incl_online': 60},
        ],
        '2021-10-02': [
            {'aid': 'a', 'status': 'ok', 'policy_hits': 0,
             'cm_today_total': 30, 'cm_online_total': 30,
             'cm_today_total_incl_online': 60},
            {'aid': 'b', 'status': 'error'},
        ],
    }
    with tarfile.open(path, 'w:gz') as tar:
        for day, rows in days.items():
            raw = ''.join(json.dumps(row) + '\n' for row in rows).encode()
            info = tarfile.TarInfo(f'metrics/day_{day}.jsonl')
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))


def test_freezes_only_policy_free_completed_rows_and_records_missing(tmp_path):
    path = tmp_path / 'days.tar.gz'
    _archive(path)
    with pytest.raises(ValueError, match='Too many missing'):
        module.build(path, date(2021, 10, 1), date(2021, 10, 2), 2)
    result = module.build(path, date(2021, 10, 1), date(2021, 10, 2), 2, 1)
    assert result['daily_income_by_aid'] == {'a': 55, 'b': 60}
    assert result['missing_days_by_aid'] == {'b': ['2021-10-02']}
    assert result['minimum_completed_days'] == 1


def test_policy_exposure_cannot_enter_baseline(tmp_path):
    path = tmp_path / 'days.tar.gz'
    _archive(path, policy_hit=1)
    with pytest.raises(ValueError, match='Policy exposure'):
        module.build(path, date(2021, 10, 1), date(2021, 10, 2), 2, 1)
