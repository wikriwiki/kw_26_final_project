"""소비 기준액이 없는 시민 — 원장에서 빼지 않고, 문턱을 지어내지 않는다.

명부 3,000명 중 2명은 BDC 소비 기준액(s_daily_wd/we)이 없다. 시뮬에서도 예산이 없어
지출이 0 이다. 이들을 빼면 '명부 전원 보존' 검사가 깨지고, 0 을 기준액으로 넣으면
문턱 0 → 1원만 써도 적립되는 가짜 값이 된다. 그래서 '없음' 으로 남기고 캐시백 0.
단 기준액이 없는데 적립 대상 지출이 있으면 캐시백을 정할 근거가 없으므로 멈춘다.
"""
from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import export_cashback_month as cashback  # noqa: E402


def policy():
    return {"id": "P012", "type": "cashback", "effective_from": "2021-10-01",
            "effective_until": "2021-11-30", "benefit_rate": 0.1,
            "threshold_ratio": 1.03, "cap_per_agent": 100000}


def test_no_anchor_is_reported_not_invented():
    agent = {"aid": "x", "daily_wd": None, "daily_we": None, "sangsaeng_base_daily": None}
    assert cashback.monthly_anchor(agent, 0.268) == (None, "no_anchor")


def _run(tmp_path, monkeypatch, b_eligible_per_day):
    days = cashback.month_days("2021-10")
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    for day in days:
        rows = [{"aid": aid, "status": "ok",
                 "execution_fingerprint": "fp", "experience_run_id": "on-run",
                 "s2_timing": {"n_llm_calls": 1, "attempts": [{"status": "ok"}]}}
                for aid in ("a", "b")]
        (metrics_dir / f"day_{day}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (tmp_path / f"cohort_{day}.json").write_text(json.dumps({
            "agent_ids": ["a", "b"], "execution_fingerprint": "fp",
            "run_id": "on-run", "prompt_variant": "v51",
            "system_prompt_sha256": "a" * 64}), encoding="utf-8")
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy()), encoding="utf-8")

    class Session:
        def run(self, query, **params):
            if query == cashback.POLICY_QUERY:
                return [{"policy": policy()}]
            if query == cashback.AGENT_QUERY:
                return [{"aid": "a", "daily_wd": 100, "daily_we": 100,
                         "sangsaeng_base_daily": None},
                        {"aid": "b", "daily_wd": None, "daily_we": None,
                         "sangsaeng_base_daily": None}]
            idx = days.index(params["day"]) + 1
            b = b_eligible_per_day * idx
            if query == cashback.STATE_QUERY:
                return [{"aid": "a", "eligible_cumulative": 100 * idx,
                         "self_month_cumulative": 100 * idx, "online_spent": 0},
                        {"aid": "b", "eligible_cumulative": b,
                         "self_month_cumulative": b, "online_spent": 0}]
            if query == cashback.SPEND_QUERY:
                out = [{"aid": "a", "spent": 100, "eligible": True}]
                if b_eligible_per_day:
                    out.append({"aid": "b", "spent": b_eligible_per_day, "eligible": True})
                return out
            raise AssertionError("unexpected query")

    monkeypatch.setattr(cashback, "driver_session", lambda: nullcontext(Session()))
    out = tmp_path / "on.jsonl"
    cashback.export(month="2021-10", arm="on", policy_id="P012", policy_file=path,
                    base_ratio=0.268, roster=["a", "b"], metrics_dir=metrics_dir, out=out)
    return out


def test_no_anchor_citizen_who_spent_nothing_stays_in_ledger(tmp_path, monkeypatch):
    out = _run(tmp_path, monkeypatch, 0)
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    b = [r for r in rows if r["aid"] == "b"]
    assert len(b) == 31                       # 빠지지 않았다
    assert all(r["anchor_won"] is None and r["threshold_won"] is None for r in b)
    assert all(r["anchor_source"] == "no_anchor" for r in b)
    assert all(r["cashback_accrued_won"] == 0 for r in b)
    a_last = [r for r in rows if r["aid"] == "a"][-1]
    assert a_last["cashback_accrued_won"] > 0  # 다른 사람 계산은 그대로
    man = json.loads((tmp_path / "on.jsonl.manifest.json").read_text(encoding="utf-8"))
    assert man["no_anchor_citizens"] == ["b"]
    assert man["rows"] == 62


def test_no_anchor_citizen_with_eligible_spending_stops_export(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="no spending anchor spent at eligible"):
        _run(tmp_path, monkeypatch, 50)
    assert not (tmp_path / "on.jsonl").exists()
