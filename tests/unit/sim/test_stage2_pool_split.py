"""Stage 2 후보 풀 분할 (옵션 A) — 같은 (dong, sub_cat) 이벤트 N개에 같은 POI 가
여러 풀에 동시 출현하지 않는지 검증.

¹-E 이후: 분할 직전에 desire 점수로 재정렬 → 단골 1순위가 아닌 desire 1순위가
가장 이른 이벤트에 할당된다. 합성 cand 는 모두 desire 동일(미인지+novelty)이라
입력 순서가 유지됨 (stable sort)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from scripts.sim import stage2_poi
from scripts.sim.stage1_intent import Stage1Event


TODAY = date(2026, 5, 15)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ev(time: str, anchor: str, l1: str, sub: str, intent: str = "x") -> Stage1Event:
    return Stage1Event(time=time, anchor=anchor, category=l1, sub_category=sub, intent=intent)


def _cands(poi_ids: list[str]) -> list[dict]:
    """기본: 모두 미인지·미방문 (desire 동일, sort stable 로 입력 순서 유지)."""
    return [
        {"poi_id": p, "name": p, "known": False, "visit_count": 0,
         "avg_satisfaction": None, "affinity": 0.0, "source": None, "km": 0.5,
         "last_visit": None, "v30": 0,
         "cat_tau": 3.0, "cat_drop": 0.85, "cat_sat_n": 12}
        for p in poi_ids
    ]


def _cand(poi_id: str, **fields) -> dict:
    """단일 cand — desire 입력 일부 override 가능."""
    base = {
        "poi_id": poi_id, "name": poi_id, "known": False, "visit_count": 0,
        "avg_satisfaction": None, "affinity": 0.0, "source": None, "km": 0.5,
        "last_visit": None, "v30": 0,
        "cat_tau": 3.0, "cat_drop": 0.85, "cat_sat_n": 12,
    }
    base.update(fields)
    return base


@pytest.fixture
def persona() -> dict:
    return {"home_dong_code": "11680101", "work_dong_code": "11680102"}


@pytest.fixture
def stub_fetchers(monkeypatch):
    """build_stage2_candidates 시리즈를 호출 인자별 모의 풀로 대체.

    사용: stub_fetchers.set(dong, sub_cat, [poi_ids])
          stub_fetchers.set_l1_dong(dong, l1, [poi_ids])
          stub_fetchers.set_l1_district(district, l1, [poi_ids])
    """
    class Stub:
        def __init__(self):
            self.sub_pools: dict[tuple[str, str], list[str] | list[dict]] = {}
            self.l1_dong_pools: dict[tuple[str, str], list[str]] = {}
            self.l1_dist_pools: dict[tuple[str, str], list[str]] = {}
            self.calls: list[tuple] = []

        def set(self, dong, sub, ids_or_cands):
            """ids_or_cands: list[str] (기본 cand) 또는 list[dict] (custom)."""
            self.sub_pools[(dong, sub)] = ids_or_cands

        def set_l1_dong(self, dong, l1, ids):
            self.l1_dong_pools[(dong, l1)] = ids

        def set_l1_district(self, dist, l1, ids):
            self.l1_dist_pools[(dist, l1)] = ids

    stub = Stub()

    def _materialize(items):
        if not items:
            return []
        if isinstance(items[0], dict):
            return list(items)
        return _cands(items)

    def fake_sub(aid, dong, sub, limit):
        stub.calls.append(("sub", dong, sub, limit))
        return _materialize(stub.sub_pools.get((dong, sub), []))[:limit]

    def fake_l1_dong(aid, dong, l1, limit):
        stub.calls.append(("l1_dong", dong, l1, limit))
        return _materialize(stub.l1_dong_pools.get((dong, l1), []))[:limit]

    def fake_l1_dist(aid, dist, l1, limit):
        stub.calls.append(("l1_dist", dist, l1, limit))
        return _materialize(stub.l1_dist_pools.get((dist, l1), []))[:limit]

    monkeypatch.setattr(stage2_poi, "build_stage2_candidates", fake_sub)
    monkeypatch.setattr(stage2_poi, "build_stage2_candidates_l1_dong", fake_l1_dong)
    monkeypatch.setattr(stage2_poi, "build_stage2_candidates_l1_district", fake_l1_dist)
    return stub


# ---------------------------------------------------------------------------
# 단일 이벤트 — 분할 없음, 기존 동작 유지
# ---------------------------------------------------------------------------
def test_single_event_no_split(persona, stub_fetchers):
    stub_fetchers.set("11680101", "한식", [f"P{i}" for i in range(20)])
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    assert len(out[0]) == 15                           # k_per_event 그대로
    assert stats.get("cand_sub_match") == 1
    assert stats.get("pool_split_groups", 0) == 0
    # 단일 이벤트일 땐 풀 크기 = k_per_event
    last_call = [c for c in stub_fetchers.calls if c[0] == "sub"][-1]
    assert last_call[3] == 15


# ---------------------------------------------------------------------------
# 두 이벤트 같은 (dong, sub) — 분할로 겹침 없음
# ---------------------------------------------------------------------------
def test_two_events_same_group_split_no_overlap(persona, stub_fetchers):
    stub_fetchers.set("11680101", "한식", [f"P{i}" for i in range(30)])
    events = [
        _ev("08:00", "zone:11680101", "식사", "한식", "아침"),
        _ev("12:00", "zone:11680101", "식사", "한식", "점심"),
    ]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    ids_morning = {c["poi_id"] for c in out[0]}
    ids_lunch = {c["poi_id"] for c in out[1]}

    # 핵심: 두 풀이 disjoint
    assert ids_morning.isdisjoint(ids_lunch)
    # 각자 15개씩 받음 (라운드로빈으로 30개를 N=2로)
    assert len(out[0]) == 15
    assert len(out[1]) == 15
    # 단골 1순위(P0) 는 가장 이른 이벤트(아침)에 가야 함
    assert out[0][0]["poi_id"] == "P0"
    assert out[1][0]["poi_id"] == "P1"
    # 그룹 단위 카운트 (n=2 만큼 누적)
    assert stats["cand_sub_match"] == 2
    assert stats["pool_split_groups"] == 1
    assert stats["pool_split_events"] == 2
    # 풀 fetch 는 그룹 1회 (n×k = 30)
    sub_calls = [c for c in stub_fetchers.calls if c[0] == "sub"]
    assert len(sub_calls) == 1
    assert sub_calls[0][3] == 30


# ---------------------------------------------------------------------------
# 세 이벤트 같은 그룹 — round-robin 3-way
# ---------------------------------------------------------------------------
def test_three_events_same_group_round_robin(persona, stub_fetchers):
    stub_fetchers.set("11680101", "카페", [f"C{i}" for i in range(45)])
    events = [
        _ev("09:00", "zone:11680101", "카페", "카페", "모닝커피"),
        _ev("13:00", "zone:11680101", "카페", "카페", "점심후"),
        _ev("16:00", "zone:11680101", "카페", "카페", "오후"),
    ]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    sets = [{c["poi_id"] for c in out[i]} for i in range(3)]
    # 세 풀 모두 disjoint
    assert sets[0].isdisjoint(sets[1])
    assert sets[1].isdisjoint(sets[2])
    assert sets[0].isdisjoint(sets[2])
    # 각자 15개씩 (45 / 3)
    assert all(len(out[i]) == 15 for i in range(3))
    # 라운드로빈 검증: 0,3,6,... → bucket 0
    assert [c["poi_id"] for c in out[0][:3]] == ["C0", "C3", "C6"]
    assert [c["poi_id"] for c in out[1][:3]] == ["C1", "C4", "C7"]
    assert [c["poi_id"] for c in out[2][:3]] == ["C2", "C5", "C8"]


# ---------------------------------------------------------------------------
# 다른 (dong, sub) 그룹은 독립
# ---------------------------------------------------------------------------
def test_different_groups_independent(persona, stub_fetchers):
    stub_fetchers.set("11680101", "한식", ["A1", "A2", "A3"])
    stub_fetchers.set("11680101", "카페", ["B1", "B2", "B3"])
    events = [
        _ev("12:00", "zone:11680101", "식사", "한식"),
        _ev("15:00", "zone:11680101", "카페", "카페"),
    ]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    assert {c["poi_id"] for c in out[0]} == {"A1", "A2", "A3"}
    assert {c["poi_id"] for c in out[1]} == {"B1", "B2", "B3"}
    # 분할 일어나지 않음 (각 그룹 n=1)
    assert stats.get("pool_split_groups", 0) == 0


# ---------------------------------------------------------------------------
# 풀이 작을 때 — 일부 이벤트는 후보 부족
# ---------------------------------------------------------------------------
def test_pool_too_small_partial_buckets(persona, stub_fetchers):
    # 3개 이벤트에 5개 풀밖에 없음
    stub_fetchers.set("11680101", "한식", ["P0", "P1", "P2", "P3", "P4"])
    events = [_ev(f"{8+i:02d}:00", "zone:11680101", "식사", "한식") for i in range(3)]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    # 5개를 round-robin: bucket 0 → P0,P3 / bucket 1 → P1,P4 / bucket 2 → P2
    assert [c["poi_id"] for c in out[0]] == ["P0", "P3"]
    assert [c["poi_id"] for c in out[1]] == ["P1", "P4"]
    assert [c["poi_id"] for c in out[2]] == ["P2"]
    # 여전히 disjoint
    s0, s1, s2 = ({c["poi_id"] for c in out[i]} for i in range(3))
    assert s0.isdisjoint(s1) and s1.isdisjoint(s2) and s0.isdisjoint(s2)


# ---------------------------------------------------------------------------
# Fallback 체인 — sub 비어있으면 L1_dong, 그것도 비면 L1_district
# 분할은 fallback 결과에도 동일 적용
# ---------------------------------------------------------------------------
def test_fallback_l1_dong_then_split(persona, stub_fetchers):
    # sub_cat 매칭 0개, L1_dong 으로 fallback
    stub_fetchers.set_l1_dong("11680101", "식사", [f"P{i}" for i in range(20)])
    events = [
        _ev("08:00", "zone:11680101", "식사", "한식"),
        _ev("12:00", "zone:11680101", "식사", "한식"),
    ]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    s0 = {c["poi_id"] for c in out[0]}
    s1 = {c["poi_id"] for c in out[1]}
    assert s0.isdisjoint(s1)
    assert stats["cand_fallback_l1_dong"] == 2
    assert stats["pool_split_groups"] == 1


def test_fallback_l1_district_when_nothing_in_dong(persona, stub_fetchers):
    # sub + L1_dong 둘 다 빈 풀, 자치구 fallback
    stub_fetchers.set_l1_district("11680", "식사", [f"P{i}" for i in range(10)])
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    assert len(out[0]) == 10
    assert stats["cand_fallback_l1_district"] == 1


def test_all_empty_increments_counter(persona, stub_fetchers):
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]
    stats: dict = {}
    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)
    assert out[0] == []
    assert stats["cand_all_empty"] == 1
    assert stats.get("pool_split_groups", 0) == 0


# ---------------------------------------------------------------------------
# 스킵 케이스 — INTERNAL_CATS / pinned_poi / sub_cat None / dong None
# ---------------------------------------------------------------------------
def test_internal_cats_skipped(persona, stub_fetchers):
    events = [
        Stage1Event(time="08:00", anchor="residence", category="집", sub_category=None, intent="기상"),
        Stage1Event(time="09:00", anchor="workplace", category="직장", sub_category=None, intent="출근"),
    ]
    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15)
    assert out[0] == []
    assert out[1] == []


def test_pinned_poi_skipped(persona, stub_fetchers):
    ev = Stage1Event(
        time="12:00", anchor="zone:11680101", category="식사",
        sub_category="한식", intent="약속", pinned_poi="C_pinned",
    )
    out = stage2_poi.fetch_candidates_for_events("A", [ev], persona, TODAY, k_per_event=15)
    assert out[0] == []


# ---------------------------------------------------------------------------
# 회귀: 같은 날 4개 이벤트 — 모든 쌍이 disjoint
# ---------------------------------------------------------------------------
def test_four_events_all_pairs_disjoint(persona, stub_fetchers):
    stub_fetchers.set("11680101", "한식", [f"P{i}" for i in range(60)])
    events = [_ev(f"{7+i*3:02d}:00", "zone:11680101", "식사", "한식") for i in range(4)]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15, stats=stats)

    sets = [{c["poi_id"] for c in out[i]} for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[i].isdisjoint(sets[j]), f"event {i} and {j} share POIs"
    assert stats["pool_split_groups"] == 1
    assert stats["pool_split_events"] == 4


# ---------------------------------------------------------------------------
# ¹-E: desire 점수가 단골 정렬을 뒤집는다 — 어제 간 단골 < 새 가게
# ---------------------------------------------------------------------------
def test_recently_visited_regular_demoted_by_desire(persona, stub_fetchers):
    """어제 방문한 단골(high affinity)이 미방문 새 가게보다 desire 가 낮아져 후보 풀에서 뒤로 밀림."""
    yesterday = TODAY - timedelta(days=1)
    cands = [
        # 단골 — 어제 방문, 30일 4회 (식사 sat_n=12 라 포화 아님). recency≈0.39 → desire 낮음
        _cand("REG_YESTERDAY", known=True, affinity=0.8, avg_satisfaction=0.7,
              last_visit=yesterday, v30=4),
        # 미방문 새 가게 — recency=1, novelty 보너스
        _cand("NEW_PLACE", known=False, affinity=0.0, avg_satisfaction=None,
              last_visit=None, v30=0),
    ]
    stub_fetchers.set("11680101", "한식", cands)
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15)
    # 새 가게가 1순위 — desire 점수가 단골보다 높음
    assert out[0][0]["poi_id"] == "NEW_PLACE"
    assert out[0][1]["poi_id"] == "REG_YESTERDAY"
    # desire 필드가 부여됨
    assert "desire" in out[0][0]
    assert out[0][0]["desire"] > out[0][1]["desire"]
    # 어제 방문 단골의 days_since_visit = 1
    reg_cand = next(c for c in out[0] if c["poi_id"] == "REG_YESTERDAY")
    assert reg_cand["days_since_visit"] == 1


def test_long_ago_regular_still_preferred(persona, stub_fetchers):
    """14일 전 방문한 단골은 recency 거의 회복 → 미방문 가게보다 desire 높음."""
    long_ago = TODAY - timedelta(days=14)
    cands = [
        _cand("REG_OLD", known=True, affinity=0.8, avg_satisfaction=0.8,
              last_visit=long_ago, v30=1),
        _cand("NEW_PLACE", known=False, affinity=0.0, avg_satisfaction=None,
              last_visit=None, v30=0),
    ]
    stub_fetchers.set("11680101", "한식", cands)
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15)
    # 단골이 다시 1순위
    assert out[0][0]["poi_id"] == "REG_OLD"


def test_saturated_regular_demoted(persona, stub_fetchers):
    """30일 안에 sat_n 훌쩍 넘게 방문 (포화) → desire 매우 낮음."""
    long_ago = TODAY - timedelta(days=14)
    cands = [
        _cand("OVER_VISITED", known=True, affinity=0.9, avg_satisfaction=0.9,
              last_visit=long_ago, v30=25, cat_sat_n=12),    # saturation ≈ 0
        _cand("NEW_PLACE", known=False, affinity=0.0, avg_satisfaction=None,
              last_visit=None, v30=0),
    ]
    stub_fetchers.set("11680101", "한식", cands)
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, TODAY, k_per_event=15)
    assert out[0][0]["poi_id"] == "NEW_PLACE"


# ---------------------------------------------------------------------------
# ¹-F: 프롬프트 포맷에 desire / 방문일 노출 + system 룰
# ---------------------------------------------------------------------------
def test_prompt_includes_desire_and_recency_labels(persona, stub_fetchers):
    yesterday = TODAY - timedelta(days=1)
    cands = [
        _cand("REG_YESTERDAY", known=True, affinity=0.8, avg_satisfaction=0.7,
              last_visit=yesterday, v30=4),
        _cand("NEW_PLACE", known=False),
    ]
    stub_fetchers.set("11680101", "한식", cands)
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]

    cands_by_order = stage2_poi.fetch_candidates_for_events(
        "A", events, persona, TODAY, k_per_event=15,
    )
    prompt = stage2_poi.build_stage2_prompt(events, cands_by_order)
    assert "욕구" in prompt
    assert "어제" in prompt           # REG_YESTERDAY
    assert "안 가봄" in prompt        # NEW_PLACE
    assert "(30일 4회)" in prompt


def test_system_prompt_mentions_desire_rule():
    assert "욕구" in stage2_poi.SYSTEM_S2
    assert "0.2" in stage2_poi.SYSTEM_S2
