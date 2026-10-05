"""KW26 이식(2026-10-05) — doinggyu 엔진에 옮긴 고침이 기본값에서는 현행과 같고, 켜면 의도대로 움직인다."""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
sys.path.insert(0, str(ROOT / "scripts"))


def _events():
    return [{"category": "식사", "sub_category": "한식", "poi_id": "a", "planned_amount": 12000,
             "price_factor": 1.0, "coupon_eligible": True},
            {"category": "마트", "sub_category": "슈퍼마켓", "poi_id": "b", "planned_amount": 30000,
             "price_factor": 1.0, "coupon_eligible": True}]


def test_direction_sentences_are_gone_and_facts_stay():
    import mechanisms
    importlib.reload(mechanisms)
    wallet = mechanisms._PRINCIPLE["wallet"]
    cash = mechanisms._PRINCIPLE["cashback"]
    assert "소비 자체를 새로 만들라는" not in wallet and "결제 건마다" in wallet
    assert "늘려주지 않는다" not in cash and "다음 달에 돌려받는 것이다" in cash
    src = (ROOT / "scripts/sim/dawn_context.py").read_text(encoding="utf-8")
    code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    assert "소비 자체를 새로 만들라는" not in code and "소비 예산을 늘려주지 않는다" not in code


def test_eligible_channel_default_off_and_median_dong_is_identity(monkeypatch):
    import consumption as C
    importlib.reload(C)
    assert C.EXP_ELIGIBLE_CHANNEL is False
    table, med = C._dong_share()
    assert table, "dong_eligible_share.json 이 읽혀야 한다"
    # 중앙값 동의 인정 몫은 SHARE_BASE — 현행 눈금 그대로
    mid = min(table, key=lambda k: abs(table[k] - med))
    s, src = C.eligible_share_for_dong(mid)
    assert src == "bdc_dong" and abs(s - C.SHARE_BASE * table[mid] / med) < 1e-9
    assert C.eligible_share_for_dong("99999999") == (C.SHARE_BASE, "share_base_constant")
    assert C._dong_of("AGT_11140550_M_70대이상_001") == "11140550"


def test_fingerprint_sees_every_exp_setting(monkeypatch):
    import experience_provenance as EP
    monkeypatch.delenv("EXP_PLAN_DRIVES_TOTAL", raising=False)
    a = EP.execution_fingerprint()
    monkeypatch.setenv("EXP_PLAN_DRIVES_TOTAL", "1")
    b = EP.execution_fingerprint()
    assert a != b


def test_fingerprint_unchanged_without_extra_exp(monkeypatch):
    import experience_provenance as EP
    for k in list(os.environ):
        if k.startswith("EXP_"):
            monkeypatch.delenv(k, raising=False)
    settings = ('LLM_MODE','SIM_ENVIRONMENT','SIM_PROMPT_VARIANT','CONSUMPTION_MODEL',
                'EXP_PAYMENT_CHOICE','EXP_ELIGIBLE_SHARE','EXP_GRANT_USE',
                'POLICY_BACKTEST_DETERMINISTIC','SIM_INTERVIEW_EVIDENCE',
                'SIM_PROMPT_TOKEN_GUARD','SIM_MODEL_CONTEXT_LENGTH')
    from evidence_integrity import digest
    old = {'source': EP.source_fingerprint(), 'settings': {k: os.environ.get(k) for k in settings}}
    if os.environ.get("NO_SMOKING_CONTEXT") is None:
        assert EP.execution_fingerprint() == digest(old)


def test_no_skip_setting_defaults_keep_current_budget(monkeypatch):
    monkeypatch.delenv("EXP_NO_SKIP", raising=False)
    monkeypatch.delenv("EXP_AGENT_DAY_MAX_ATTEMPTS", raising=False)
    src = (ROOT / "scripts/sim/run_simulation.py").read_text(encoding="utf-8")
    assert 'int(os.environ.get("EXP_AGENT_DAY_MAX_ATTEMPTS", "6"))' in src
    assert 'os.environ.get("EXP_NO_SKIP", "0") == "1"' in src
    assert src.count("raise AgentDayExhausted(") == 2
    assert "max_rounds = AGENT_DAY_MAX_ATTEMPTS" in src


def test_month_reset_and_income_reach_the_state_query():
    src = (ROOT / "scripts/sim/plan_writer.py").read_text(encoding="utf-8")
    assert "coalesce(prev.month_spent, 0) * $month_carry" in src
    assert "coalesce(prev.sangsaeng_month_spent, 0) * $month_carry" in src
    assert "prev_balance + $today_income" in src
    assert 'os.environ.get("EXP_MONTH_RESET", "0") == "1" and today.day == 1' in src
