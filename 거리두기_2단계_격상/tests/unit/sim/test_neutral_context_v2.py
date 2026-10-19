from datetime import date
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from neutral_context_v2 import render


def test_legacy_behavior_and_unobserved_eligibility_not_reused():
    blocks={'policy':'대상(소비 규모 기준) | 약 3일치','policy_facts':'별도 제도의 법정 조건',
            'persona':'평일 재택 10h','zones':'- [광역상권] A, 2.0km ← 주말 나들이·여가 등에 적합\n평일엔 주로 생활권',
            'state':'현재 잔액'}
    statuses=[{'id':'NEW','available_wallet_won':1000,'scenario_assumptions':['참여 자격은 관측값이 아닌 시나리오 가정']}]
    result=render(blocks,today=date(2026,9,21),day_type='weekday',zones=['A'],personal_policy_status=statuses)
    assert all(word not in result for word in ['소비 규모 기준','3일치','나들이','평일엔 주로'])
    assert all(word in result for word in ['법정 조건','available_wallet_won','시나리오 가정','2.0km','집 체류'])
    assert '3일치' in blocks['policy']


def test_status_cannot_omit_assumptions_channel():
    with pytest.raises(ValueError,match='separate'):
        render({},today=date(2026,9,21),day_type='weekday',zones=[],personal_policy_status=[{'id':'NEW'}])
