"""analyze_repeat_visits 의 순수 함수 6종 검증.

Neo4j fetch 는 모킹하지 않는다 — compute_* 함수들이 PlanEvent 리스트 받는 pure
function 이라 합성 입력으로 직접 테스트 가능.
"""

from __future__ import annotations

from datetime import date

import pytest

from scripts.sim.analyze_repeat_visits import (
    PlanEvent,
    compute_all_metrics,
    compute_category_diversity,
    compute_district_event_share,
    compute_revisit_intervals,
    compute_same_day_repeat_rate,
    compute_top_poi_concentration,
    compute_unique_poi_per_day,
    render_markdown,
)


def _e(aid, day_str, poi, l1="식사", sub="한식", district="11680"):
    return PlanEvent(
        aid=aid, day=date.fromisoformat(day_str),
        poi_id=poi, l1=l1, sub=sub, district=district,
    )


# ---------------------------------------------------------------------------
# 1. unique_poi_per_day
# ---------------------------------------------------------------------------
def test_unique_poi_all_different():
    events = [_e("A", "2026-05-01", f"P{i}") for i in range(3)]
    r = compute_unique_poi_per_day(events)
    assert r["mean"] == 1.0
    assert r["n"] == 1


def test_unique_poi_partial_repeat():
    # agent A 하루 3 이벤트 중 2개가 P1, 1개 P2 → 2/3
    events = [_e("A", "2026-05-01", p) for p in ["P1", "P1", "P2"]]
    r = compute_unique_poi_per_day(events)
    assert r["mean"] == pytest.approx(2 / 3, abs=1e-3)


def test_unique_poi_across_days():
    # 같은 POI 라도 다른 날이면 각 날의 unique ratio 는 1.0
    events = [_e("A", "2026-05-01", "P1"), _e("A", "2026-05-02", "P1")]
    r = compute_unique_poi_per_day(events)
    assert r["mean"] == 1.0
    assert r["n"] == 2


def test_unique_poi_empty():
    r = compute_unique_poi_per_day([])
    assert r["n"] == 0
    assert r["mean"] is None


# ---------------------------------------------------------------------------
# 2. same_day_repeat_rate
# ---------------------------------------------------------------------------
def test_same_day_repeat_detected():
    events = [
        _e("A", "2026-05-01", "P1"),
        _e("A", "2026-05-01", "P1"),    # 같은 날 같은 POI
        _e("B", "2026-05-01", "P1"),
        _e("B", "2026-05-01", "P2"),    # 다른 POI
    ]
    r = compute_same_day_repeat_rate(events)
    assert r["agent_days_with_2plus_commerce"] == 2
    assert r["with_repeat"] == 1
    assert r["rate"] == 0.5


def test_same_day_no_repeat_at_all():
    events = [
        _e("A", "2026-05-01", "P1"),
        _e("A", "2026-05-01", "P2"),
        _e("A", "2026-05-01", "P3"),
    ]
    r = compute_same_day_repeat_rate(events)
    assert r["rate"] == 0.0


def test_same_day_single_event_skipped():
    """이벤트 1개뿐인 agent-day 는 분모에서 제외 (반복 정의 안 됨)."""
    events = [_e("A", "2026-05-01", "P1")]
    r = compute_same_day_repeat_rate(events)
    assert r["agent_days_with_2plus_commerce"] == 0
    assert r["rate"] is None


# ---------------------------------------------------------------------------
# 3. revisit_intervals
# ---------------------------------------------------------------------------
def test_revisit_intervals_basic():
    events = [
        _e("A", "2026-05-01", "P1"),
        _e("A", "2026-05-02", "P1"),    # 1일 간격
        _e("A", "2026-05-05", "P1"),    # 3일 간격
        _e("A", "2026-05-10", "P2"),
        _e("A", "2026-05-25", "P2"),    # 15일 간격
    ]
    r = compute_revisit_intervals(events)
    assert r["n"] == 3
    assert r["buckets"]["1일"] == 1
    assert r["buckets"]["2-3일"] == 1
    assert r["buckets"]["15일+"] == 1


def test_revisit_intervals_same_day_excluded():
    """0일(같은 날)은 same_day_repeat 와 중복이라 제외."""
    events = [_e("A", "2026-05-01", "P1"), _e("A", "2026-05-01", "P1")]
    r = compute_revisit_intervals(events)
    assert r["n"] == 0


def test_revisit_intervals_no_revisit():
    events = [_e("A", "2026-05-01", "P1"), _e("A", "2026-05-02", "P2")]
    r = compute_revisit_intervals(events)
    assert r["n"] == 0


# ---------------------------------------------------------------------------
# 4. top_poi_concentration
# ---------------------------------------------------------------------------
def test_top_concentration_full_share():
    # P0..P9 각 10번, P10..P19 각 1번 → top10 점유율 = 100/110 ≈ 0.909
    events = []
    for i in range(10):
        for _ in range(10):
            events.append(_e(f"A{i}", "2026-05-01", f"P{i}"))
    for i in range(10, 20):
        events.append(_e(f"X{i}", "2026-05-01", f"P{i}"))
    r = compute_top_poi_concentration(events)
    assert r["total_visits"] == 110
    assert r["unique_pois"] == 20
    assert r["top10_share"] == pytest.approx(100 / 110, abs=1e-3)


def test_top_concentration_empty():
    r = compute_top_poi_concentration([])
    assert r["total"] == 0


# ---------------------------------------------------------------------------
# 5. category_diversity
# ---------------------------------------------------------------------------
def test_category_diversity_basic():
    events = [
        _e("A", "2026-05-01", "P1", sub="한식"),
        _e("A", "2026-05-02", "P2", sub="카페"),
        _e("A", "2026-05-03", "P3", sub="한식"),   # 중복
        _e("B", "2026-05-01", "P1", sub="한식"),
    ]
    r = compute_category_diversity(events)
    assert r["n_agents"] == 2
    # A: {한식, 카페} = 2, B: {한식} = 1
    assert r["mean_unique_subs"] == 1.5


def test_category_diversity_ignores_null_sub():
    events = [_e("A", "2026-05-01", "P1", sub=None)]
    r = compute_category_diversity(events)
    assert r["n"] == 0


# ---------------------------------------------------------------------------
# 6. district_event_share
# ---------------------------------------------------------------------------
def test_district_share_basic():
    events = [
        _e("A", "2026-05-01", "P1", district="11680"),
        _e("A", "2026-05-01", "P2", district="11680"),
        _e("B", "2026-05-01", "P3", district="11680"),
        _e("C", "2026-05-01", "P4", district="11440"),
    ]
    r = compute_district_event_share(events)
    # 11680: events=3, agents={A,B}=2, per_agent=1.5
    assert r["11680"]["events"] == 3
    assert r["11680"]["agents"] == 2
    assert r["11680"]["events_per_agent"] == 1.5
    assert r["11440"]["events"] == 1
    assert r["11440"]["agents"] == 1


def test_district_share_skips_null():
    events = [_e("A", "2026-05-01", "P1", district=None)]
    r = compute_district_event_share(events)
    assert r == {}


# ---------------------------------------------------------------------------
# Integration — compute_all_metrics + render_markdown
# ---------------------------------------------------------------------------
def test_all_metrics_smoke():
    events = [
        _e("A", "2026-05-01", "P1"),
        _e("A", "2026-05-01", "P1"),     # same-day repeat
        _e("A", "2026-05-02", "P1"),     # 1-day revisit
        _e("B", "2026-05-01", "P2", sub="카페"),
    ]
    m = compute_all_metrics(events)
    assert m["same_day_repeat_rate"]["rate"] == 1.0    # A 만 분모, A 가 반복
    assert m["revisit_interval_dist"]["n"] == 1
    assert "P1" in {e.poi_id for e in events}
    md = render_markdown(m, date(2026, 5, 1), 2)
    assert "POI 반복 방문 분석" in md
    assert "풀 분할 효과 직접 측정" in md
    assert "단골 집중도" in md
