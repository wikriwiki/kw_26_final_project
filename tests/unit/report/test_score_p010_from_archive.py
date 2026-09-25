"""P010 원장 추정의 표본 단위를 검증한다."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[3] / "scripts/report/score_p010_from_archive.py"
    spec = importlib.util.spec_from_file_location("score_p010_from_archive", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P010 = _module()


def test_같은_시민의_반복관측을_독립_표본으로_세지_않는다():
    pairs = ([('a', 0.0, 1.0)] * 50 + [('b', 1.0, 1.0)] * 50)
    lo, hi = P010.boot_ci(pairs, n=2000, seed=7)
    assert lo == pytest.approx(0.0)
    assert hi == pytest.approx(1.0)


def test_정책결제액_가중과_시민_수를_별도로_기록한다():
    rows = [
        ('2025-07-21', {'aid': 'a', 'cm_mpc_new_share': 0.0, 'policy_spend_today': 10}),
        ('2025-07-22', {'aid': 'a', 'cm_mpc_new_share': 1.0, 'policy_spend_today': 30}),
        ('2025-07-21', {'aid': 'b', 'cm_mpc_new_share': 1.0, 'policy_spend_today': 20}),
    ]
    result = P010.weighted_mpc(rows)
    assert result['mpc'] == pytest.approx(50 / 60)
    assert (result['agents'], result['cells'], result['days']) == (2, 3, 2)
