"""MPC 채점은 원장 비율을 읽고 총지출 차이로 대체하지 않는다."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
import score_policy  # noqa: E402
from score_policy import (  # noqa: E402
    cashback_calendar_aligned, cashback_metric_sample, cashback_month_coverage,
    metric_values,
    score_mpc_from_metrics,
)


def test_mpc는_영수증_차이로_계산하지_않는다():
    off, on = metric_values("mpc_amount", [], [], [], [])
    assert off is None and on is None


def test_mpc는_해당_on일의_원장만_사용한다(tmp_path):
    for day, rows in {
        "2025-07-21": [('a', 1.0, 100), ('b', 1.0, 100)],
        "2025-07-22": [('a', 0.0, 10), ('b', 1.0, 30)],
    }.items():
        (tmp_path / f"day_{day}.jsonl").write_text(''.join(
            json.dumps({'aid': a, 'status': 'ok', 'cm_mpc_new_share': m,
                        'policy_spend_today': w}) + '\n' for a, m, w in rows),
            encoding='utf-8')
    score = score_mpc_from_metrics(tmp_path, ["2025-07-22"])
    assert score['mean'] == pytest.approx(0.75)
    assert score['n'] == 2 and score['n_cells'] == 2
    assert score['unit'] == 'ratio' and score['bootstrap_unit'] == 'aid'
    assert not score['measurement_complete']  # 구 원장에는 최종 결제 귀속 증거가 없다
    with pytest.raises(ValueError, match='관측일 누락'):
        score_mpc_from_metrics(tmp_path, ["2025-07-22", "2025-07-23"])


def test_최종_결제와_응답의_전액_대조가_될때만_mpc_측정완전(tmp_path):
    rows = [
        {'aid': 'a', 'status': 'ok', 'cm_mpc_new_share': 0.2,
         'policy_spend_today': 100, 'cm_mpc_paid_won': 100,
         'cm_mpc_unresolved_won': 0, 'cm_mpc_coverage': 1},
        {'aid': 'b', 'status': 'ok', 'cm_mpc_new_share': 0.4,
         'policy_spend_today': 100, 'cm_mpc_paid_won': 100,
         'cm_mpc_unresolved_won': 0, 'cm_mpc_coverage': 1},
    ]
    (tmp_path / 'day_2025-07-22.jsonl').write_text(
        ''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    score = score_mpc_from_metrics(tmp_path, ['2025-07-22'])
    assert score['measurement_complete']
    assert score['measurement_coverage'] == 1
    assert score['mean'] == pytest.approx(0.3)


def test_mpc_원장_인자_없이_채점하면_명시적_오류():
    result = subprocess.run(
        [sys.executable, str(ROOT / 'scripts/sim/score_policy.py'),
         '--policy', 'P010', '--off', '2025-07-15:2025-07-16',
         '--on', '2025-07-22:2025-07-23'],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
        env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    assert result.returncode != 0
    assert '--metrics-dir' in result.stderr


def test_캐시백_평균과_상한비율은_수령자를_분모로_쓴다():
    cb = {
        'a': {'reached': 1, 'cashback': 100_000, 'capped': 1},
        'b': {'reached': 1, 'cashback': 10_000, 'capped': 0},
        'c': {'reached': 0, 'cashback': 0, 'capped': 0},
        'd': {'reached': 0, 'cashback': 0, 'capped': 0},
    }
    amounts, n_recipients = cashback_metric_sample(cb, 'cashback_per_capita')
    caps, _ = cashback_metric_sample(cb, 'cap_reach_rate')
    reached, _ = cashback_metric_sample(cb, 'threshold_reach_rate')
    assert (sum(amounts) / len(amounts), n_recipients) == (55_000, 2)
    assert sum(caps) / len(caps) == 0.5
    assert sum(reached) / len(reached) == 0.5


def test_캐시백_실측은_온전한_달의_정책노출이_필요하다():
    assert not cashback_calendar_aligned('2021-10-28', '2021-10-01')
    assert not cashback_calendar_aligned('2021-10-31', '2021-10-15')
    assert not cashback_calendar_aligned('2021-10-31', None)
    assert cashback_calendar_aligned('2021-10-31', '2021-10-01')


def test_월말_캐시백_채점은_시민별_한달_관측을_확인한다(monkeypatch):
    class FakeResult:
        def __init__(self, complete):
            self.complete = complete

        def single(self):
            return {'n_complete': self.complete}

    class FakeSession:
        complete = 0

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def run(self, query, **kwargs):
            assert kwargs['first'] == '2021-10-01'
            assert kwargs['last'] == '2021-10-31'
            assert kwargs['expected_days'] == 31
            assert kwargs['aids'] == ['a', 'b']
            return FakeResult(self.complete)

    session = FakeSession()
    monkeypatch.setattr(score_policy, 'driver_session', lambda: session)
    assert cashback_month_coverage('2021-10-31', []) == (False, 0)
    session.complete = 1
    assert cashback_month_coverage('2021-10-31', ['a', 'b']) == (False, 1)
    session.complete = 2
    assert cashback_month_coverage('2021-10-31', ['a', 'b']) == (True, 2)
