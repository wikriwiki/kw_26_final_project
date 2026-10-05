"""월별 적립 실적은 월 경계를 넘지 않는다."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/sim"))
from dawn_context import monthly_state_for_today  # noqa: E402
from plan_writer import NIGHT_STATE_CYPHER  # noqa: E402


def test_월초_프롬프트는_잔액은_이어받고_월누적만_초기화한다():
    previous = {'balance': 123_000, 'month_spent': 600_000,
                'sangsaeng_month_spent': 300_000, 'policy_lc': '{"P012":true}'}
    current = monthly_state_for_today(previous, date(2021, 10, 1))
    assert current['balance'] == 123_000
    assert current['policy_lc'] == '{"P012":true}'
    assert current['month_spent'] == 0
    assert current['sangsaeng_month_spent'] == 0
    assert previous['sangsaeng_month_spent'] == 300_000
    assert monthly_state_for_today(previous, date(2021, 10, 2)) is previous


def test_야간_state_저장도_두_월누적을_초기화한다():
    assert NIGHT_STATE_CYPHER.count('CASE WHEN date($today).day = 1 THEN 0') == 2
    assert 's.month_spent = prev_month_spent +' in NIGHT_STATE_CYPHER
    assert 's.sangsaeng_month_spent = prev_sangsaeng_month_spent +' in NIGHT_STATE_CYPHER
