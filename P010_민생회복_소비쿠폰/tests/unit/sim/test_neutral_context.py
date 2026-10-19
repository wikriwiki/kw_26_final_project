from datetime import date
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from neutral_context import initial_state, render


def test_current_eligible_spending_is_subset_of_common_total():
    p = {'daily_wd': 100000, 'daily_we': 50000}
    off = initial_state(p, date(2021, 10, 25), 700000)
    on = initial_state(p, date(2021, 10, 25), 700000, ('NEW', 280000))
    assert off['month_spent'] > off['sangsaeng_month_spent'] > 0
    assert all(on[k] == value for k, value in off.items())
    assert on['grant_remaining'] == {'NEW': 280000}
    assert 'grant_remaining' not in off


def test_impossible_pre_history_rejected():
    with pytest.raises(ValueError, match='exceeds'):
        initial_state({'daily_wd': 100}, date(2021, 10, 25), 100000)


def test_neutral_status_keeps_rules_without_pace_or_old_footer():
    blocks = {'persona': '평일 재택 10h', 'policy': '하루 목표 페이스\n- 판단 원칙: 지출을 늘린다',
              'policy_facts': '정책 법정 조건', 'zones': '위치 A\n평일엔 주로 생활권, 기타', 'state': '잔액'}
    raw = dict(blocks)
    out = render(blocks, today=date(2021, 10, 25), day_type='weekday', zones=['A'],
                 cashback_status={'eligible_month_spent_won': 10, 'total_month_spent_won': 20, 'refund_rate': .1})
    assert '정책 법정 조건' in out and 'refund_rate' in out
    assert all(s not in out for s in ['페이스', '판단 원칙', '마지막 점검', '평일엔 주로 생활권'])
    assert '집 체류 10h' in out and 'zone:A' in out
    assert blocks == raw


def test_renderer_refuses_contradictory_state():
    with pytest.raises(ValueError, match='Contradictory'):
        render({'persona': '', 'policy': '', 'zones': ''}, today=date(2021, 10, 25), day_type='weekday', zones=[],
               cashback_status={'eligible_month_spent_won': 20, 'total_month_spent_won': 10})
