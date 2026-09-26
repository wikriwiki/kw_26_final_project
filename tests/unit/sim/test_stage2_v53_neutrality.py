"""The neutral Stage2 contract must receive policy facts without wallet defaults."""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

from scripts.sim import stage2_poi


def _render(policies):
    event = SimpleNamespace(
        time="12:00", anchor="zone:11680670", category="식사",
        sub_category="한식", intent="점심",
    )
    candidate = {
        "poi_id": "C_1", "name": "식당", "known": False, "km": 0.1,
        "avg_satisfaction": None, "visit_count": 0, "price_band": 2,
    }
    return stage2_poi.build_stage2_prompt(
        [event], {0: [candidate]},
        persona={"lifestyle": "직장인", "daily_wd": 40000, "daily_we": 50000,
                 "policy_budget_summary": "옛 지원금 결제 유도 문구"},
        state={"balance": 100000, "grant_remaining": '{"PX": 20000}'},
        active_policies=policies, today=date(2021, 10, 25), neutral=True,
    )


def test_v53_stage2_system_removes_legacy_wallet_and_merchant_defaults(monkeypatch):
    monkeypatch.setenv("SIM_PROMPT_VARIANT", "v53")
    text = stage2_poi.active_stage2_system()
    assert text == stage2_poi.SYSTEM_S2_NEUTRAL
    for leak in ('P009', '[쿠폰]', '지원금은 개인 잔액', '여러 주치가 되는 금액이면',
                 '굳이 자기 돈 쓸 것 없이', '이 돈이 있어서 비로소'):
        assert leak not in text
    assert '정책 블록의 적용 조건' in text
    assert '나중에 돌려받는 혜택이나 가격 할인액' in text
    monkeypatch.setenv("SIM_PROMPT_VARIANT", "v5")
    assert stage2_poi.active_stage2_system() == stage2_poi.SYSTEM_S2


def test_v53_stage2_gets_only_actual_policy_facts_and_status():
    no_policy = _render([])
    assert '(활성 정책 없음)' in no_policy
    assert '옛 지원금 결제 유도 문구' not in no_policy

    policy = {
        "id": "PX", "type": "grant", "name": "가상의 지원금",
        "from_": "2021-10-01", "until_": "2021-10-31",
        "description": "적격 매장에서만 결제할 수 있다.",
        "poi_restricted": True, "eligible_marker": "[적격]",
    }
    rendered = _render([policy])
    assert 'PX' in rendered
    assert '[적격] 표시 POI에서만 사용' in rendered
    assert '정책지갑 잔액 20,000원' in rendered
    assert '옛 지원금 결제 유도 문구' not in rendered
