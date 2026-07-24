"""정책 프롬프트·10분위 지급·병목 계측 회귀 테스트."""
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SIM_DIR = ROOT / "scripts" / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import dawn_context  # noqa: E402
import plan_writer  # noqa: E402
import timing_metrics  # noqa: E402


def _p010() -> dict:
    return json.loads(
        (ROOT / "data" / "neo4j_load" / "policies" / "P010.json").read_text(
            encoding="utf-8"
        )
    )


def test_memory_query_limits_before_optional_poi_joins():
    """최근 기억 전체를 POI와 조인하지 않고 Top-N만 조인해야 한다."""
    query = dawn_context.MEMORY_CYPHER
    assert query.index("LIMIT $top_n") < query.index("OPTIONAL MATCH (m)-[:ABOUT_POI]")
    assert "$memory_since" in query
    assert "$today" in query


def test_p010_uses_complete_ten_decile_map():
    policy = _p010()
    grants = policy["decile_grants"]
    assert set(grants) == {str(i) for i in range(1, 11)}
    assert grants == {
        "1": 400_000,
        "2": 300_000,
        **{str(i): 150_000 for i in range(3, 11)},
    }
    assert policy["grant_key"] == "spend_decile"
    assert policy["income_grants"] == {}


def test_decile_grant_has_priority_and_exclusion_is_honored():
    policy = _p010()
    assert plan_writer._grant_for_single_policy("상", policy, spend_decile=1) == 400_000
    assert plan_writer._grant_for_single_policy("중", policy, spend_decile=2) == 300_000
    assert plan_writer._grant_for_single_policy("하", policy, spend_decile=10) == 150_000

    excluded = {**policy, "excluded_deciles": ["10"]}
    assert plan_writer._grant_for_single_policy("하", excluded, spend_decile=10) == 0


def test_policy_prompt_keeps_consumption_autonomous_but_states_payment_priority():
    policy = _p010()
    row = {
        "id": policy["id"],
        "name": policy["name"],
        "type": policy["type"],
        "description": policy["description"],
        "from_": policy["effective_from"],
        "until_": policy["effective_until"],
        "regions": ["강남구", "종로구"],
        "target_l1s": [],
        "poi_restricted": True,
        "decile_grants": policy["decile_grants"],
        "excluded_deciles": policy["excluded_deciles"],
        "grant_key": "spend_decile",
    }
    facts = dawn_context._format_policy_facts([row])
    status = dawn_context._format_policy_status(
        [row],
        persona={"income": "중", "spend_decile": 7, "daily_wd": 30_000},
        state={
            "grant_received": json.dumps({"P010": 150_000}),
            "grant_remaining": json.dumps({"P010": 120_000}),
            "grant_days_since": json.dumps({"P010": 2}),
        },
    )
    text = facts + "\n" + status

    assert "소비 7분위" in text
    assert "지급액 150,000원" in text
    assert "정책지갑 잔액 120,000원" in text
    assert "소비 필요·시점·총액·POI" in text
    assert "정책지갑을 자기자금보다 먼저 결제" in text
    assert "소비 자체를 새로 만들라는 뜻이 아니다" in text
    for directed in (
        "무조건 이득",
        "남기면 손해",
        "평소보다 씀씀이",
        "미뤄온 소비",
        "쿠폰 매장을 우선",
    ):
        assert directed not in text


def test_policy_prompt_shows_differential_amount_for_low_spending_decile():
    policy = _p010()
    row = {
        "id": policy["id"],
        "name": policy["name"],
        "type": policy["type"],
        "description": policy["description"],
        "from_": policy["effective_from"],
        "until_": policy["effective_until"],
        "regions": ["서울특별시"],
        "target_l1s": [],
        "poi_restricted": True,
        "decile_grants": policy["decile_grants"],
        "excluded_deciles": [],
        "grant_key": "spend_decile",
    }
    first = dawn_context._format_policy_status(
        [row],
        persona={"income": "하", "spend_decile": 1, "daily_wd": 30_000},
        state={},
    )
    second = dawn_context._format_policy_status(
        [row],
        persona={"income": "하", "spend_decile": 2, "daily_wd": 30_000},
        state={},
    )

    assert "소비 1분위" in first and "지급액 400,000원" in first
    assert "소비 2분위" in second and "지급액 300,000원" in second


def test_timing_report_separates_llm_review_and_cache_metrics():
    rows = [
        {
            "status": "ok",
            "timing_t_dawn": 2.0,
            "timing_t_s1": 8.0,
            "timing_t_s2": 5.0,
            "dawn_timing": {
                "t_memory": 1.0,
                "n_memory_returned": 4,
                "persona_cache_hit": False,
                "policy_cache_hit": True,
            },
            "s1_timing": {"t_llm": 7.0, "n_llm_calls": 1},
            "s2_timing": {
                "t_llm_initial": 2.0,
                "t_review_lookup": 0.4,
                "t_llm_review": 1.5,
                "n_llm_calls": 2,
            },
            "cm_policy_requested_total": 0,
            "cm_policy_allocated_total": 18_000,
            "cm_policy_eligible_spend_total": 18_000,
            "cm_policy_eligible_event_count": 1,
            "cm_policy_liquidity_relief": 2_000,
            "grant_expired_today": 3_000,
        },
        {
            "status": "ok",
            "timing_t_dawn": 1.0,
            "timing_t_s1": 4.0,
            "timing_t_s2": 3.0,
            "dawn_timing": {
                "t_memory": 0.5,
                "n_memory_returned": 2,
                "persona_cache_hit": True,
                "policy_cache_hit": True,
            },
            "s1_timing": {"t_llm": 3.0, "n_llm_calls": 1},
            "s2_timing": {
                "t_llm_initial": 1.5,
                "t_review_lookup": 0.0,
                "t_llm_review": 0.0,
                "n_llm_calls": 1,
            },
            "cm_policy_requested_total": 5_000,
            "cm_policy_allocated_total": 7_000,
            "cm_policy_eligible_spend_total": 10_000,
            "cm_policy_eligible_event_count": 2,
            "cm_policy_liquidity_relief": 0,
            "grant_expired_today": 0,
        },
        {"status": "error"},
    ]

    report = timing_metrics.build_timing_report(rows)

    assert report["agents_ok"] == 2
    assert report["agents_error"] == 1
    assert report["cache"]["persona_hit_rate"] == 0.5
    assert report["cache"]["policy_hit_rate"] == 1.0
    assert report["timings"]["stage2.t_llm_review"]["total"] == 1.5
    assert report["timings"]["stage2.t_review_lookup"]["total"] == 0.4
    assert report["counters"]["stage2.n_llm_calls"]["avg"] == 1.5
    assert report["policy_payment"] == {
        "llm_requested_total": 5_000,
        "system_allocated_total": 25_000,
        "eligible_spend_total": 28_000,
        "eligible_event_count": 3,
        "payment_coverage": round(25_000 / 28_000, 6),
        "agents_using_policy": 2,
        "agent_usage_rate": 1.0,
        "liquidity_relief_total": 2_000,
        "expired_wallet_total": 3_000,
    }


def test_review_second_pass_remains_enabled_and_instrumented():
    source = (SIM_DIR / "stage2_poi.py").read_text(encoding="utf-8")
    assert "review_lookup_requests" in source
    assert "lookup_reviews_batch(valid_lookup_ids[:8], max_reviews=3)" in source
    assert 'call_kind = "review" if review_lookup_used' in source
    assert '"t_llm_review": 0.0' in source
    assert "continue  # 다음 iteration에서 prompt_now에 첨부됨" in source


def test_grant_day_prompt_state_is_updated_before_stage1():
    """지급 당일 Stage1이 어제 잔액 0원을 보지 않도록 현재 정책지갑을 먼저 반영한다."""
    source = (SIM_DIR / "run_simulation.py").read_text(encoding="utf-8")
    received_update = source.index('ctx.state["grant_received"] = merged_grant_received')
    remaining_update = source.index('ctx.state["grant_remaining"] = grant_avail_today')
    stage1_call = source.index("s1, m1 = call_stage1(aid, today, ctx=ctx)")
    assert received_update < stage1_call
    assert remaining_update < stage1_call


def test_legacy_path_uses_common_policy_wallet_settlement():
    source = (SIM_DIR / "run_simulation.py").read_text(encoding="utf-8")
    legacy_marker = source.index("# legacy는 총소비액을 건드리지 않되")
    settlement_call = source.index(
        "_settlement = settle_policy_spend_priority(",
        legacy_marker,
    )
    validator_call = source.index("policy_spend_corrected = validate_policy_spend(")
    assert legacy_marker < settlement_call < validator_call
