"""계획 통로의 기준선을 평일·주말로 갈라 찾는다 — 같은 계획액이 요일종류에 맞는 기준과 비교된다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
import consumption as C  # noqa: E402


def _events():
    return [{"category": "식사", "poi_id": "C_1", "actual_spent": 20000, "policy_spend": {}},
            {"category": "마트", "poi_id": "C_2", "actual_spent": 20000, "policy_spend": {}}]


def _run(monkeypatch, tmp_path, baseline, is_weekend):
    f = tmp_path / "base.json"
    f.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(C, "EXP_PLAN_DRIVES_TOTAL", True)
    monkeypatch.setattr(C, "PLAN_BASELINE_FILE", str(f))
    monkeypatch.setattr(C, "_PLAN_BASELINE", None)
    meta = C.apply_consumption_model(
        _events(), daily=30000, income_tier="중", tendency="평범", balance=500000,
        llm_propensity=None, aid="A", is_weekend=is_weekend)
    monkeypatch.setattr(C, "_PLAN_BASELINE", None)
    return meta["today_total"]


def test_weekday_and_weekend_baselines_are_used_separately(monkeypatch, tmp_path):
    # 계획액 40,000원. 평일 기준 40,000 -> 배수 1.0, 주말 기준 20,000 -> 배수 2.0(상한)
    base = {"A|wd": 40000, "A|we": 20000}
    wd = _run(monkeypatch, tmp_path, base, False)
    we = _run(monkeypatch, tmp_path, base, True)
    assert we > wd * 1.5      # 같은 날 앵커·계획이라도 주말 기준선이 낮으면 배수가 크다


def test_single_baseline_still_works(monkeypatch, tmp_path):
    a = _run(monkeypatch, tmp_path, {"A": 40000}, False)
    b = _run(monkeypatch, tmp_path, {"A": 40000}, True)
    assert a > 0 and b > 0


def test_no_baseline_file_leaves_accounting_unchanged(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "EXP_PLAN_DRIVES_TOTAL", False)
    m1 = C.apply_consumption_model(_events(), daily=30000, income_tier="중", tendency="평범",
                                   balance=500000, llm_propensity=None, aid="A", is_weekend=True)
    m2 = C.apply_consumption_model(_events(), daily=30000, income_tier="중", tendency="평범",
                                   balance=500000, llm_propensity=None, aid="A")
    assert m1["today_total"] == m2["today_total"]
