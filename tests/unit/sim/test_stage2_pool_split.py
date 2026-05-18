"""Stage 2 후보 풀 분할 (옵션 A) — 같은 (dong, sub_cat) 이벤트 N개에 같은 POI 가
여러 풀에 동시 출현하지 않는지 검증."""

from __future__ import annotations

import pytest

from scripts.sim import stage2_poi
from scripts.sim.stage1_intent import Stage1Event


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _ev(time: str, anchor: str, l1: str, sub: str, intent: str = "x") -> Stage1Event:
    return Stage1Event(time=time, anchor=anchor, category=l1, sub_category=sub, intent=intent)


def _cands(poi_ids: list[str]) -> list[dict]:
    return [
        {"poi_id": p, "name": p, "known": False, "visit_count": 0,
         "avg_satisfaction": None, "affinity": 0.0, "source": None, "km": 0.5}
        for p in poi_ids
    ]


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
            self.sub_pools: dict[tuple[str, str], list[str]] = {}
            self.l1_dong_pools: dict[tuple[str, str], list[str]] = {}
            self.l1_dist_pools: dict[tuple[str, str], list[str]] = {}
            self.calls: list[tuple] = []

        def set(self, dong, sub, ids):
            self.sub_pools[(dong, sub)] = ids

        def set_l1_dong(self, dong, l1, ids):
            self.l1_dong_pools[(dong, l1)] = ids

        def set_l1_district(self, dist, l1, ids):
            self.l1_dist_pools[(dist, l1)] = ids

    stub = Stub()

    def fake_sub(aid, dong, sub, limit):
        stub.calls.append(("sub", dong, sub, limit))
        return _cands(stub.sub_pools.get((dong, sub), [])[:limit])

    def fake_l1_dong(aid, dong, l1, limit):
        stub.calls.append(("l1_dong", dong, l1, limit))
        return _cands(stub.l1_dong_pools.get((dong, l1), [])[:limit])

    def fake_l1_dist(aid, dist, l1, limit):
        stub.calls.append(("l1_dist", dist, l1, limit))
        return _cands(stub.l1_dist_pools.get((dist, l1), [])[:limit])

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

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

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

    assert len(out[0]) == 10
    assert stats["cand_fallback_l1_district"] == 1


def test_all_empty_increments_counter(persona, stub_fetchers):
    events = [_ev("12:00", "zone:11680101", "식사", "한식")]
    stats: dict = {}
    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)
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
    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15)
    assert out[0] == []
    assert out[1] == []


def test_pinned_poi_skipped(persona, stub_fetchers):
    ev = Stage1Event(
        time="12:00", anchor="zone:11680101", category="식사",
        sub_category="한식", intent="약속", pinned_poi="C_pinned",
    )
    out = stage2_poi.fetch_candidates_for_events("A", [ev], persona, k_per_event=15)
    assert out[0] == []


# ---------------------------------------------------------------------------
# 회귀: 같은 날 4개 이벤트 — 모든 쌍이 disjoint
# ---------------------------------------------------------------------------
def test_four_events_all_pairs_disjoint(persona, stub_fetchers):
    stub_fetchers.set("11680101", "한식", [f"P{i}" for i in range(60)])
    events = [_ev(f"{7+i*3:02d}:00", "zone:11680101", "식사", "한식") for i in range(4)]
    stats: dict = {}

    out = stage2_poi.fetch_candidates_for_events("A", events, persona, k_per_event=15, stats=stats)

    sets = [{c["poi_id"] for c in out[i]} for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert sets[i].isdisjoint(sets[j]), f"event {i} and {j} share POIs"
    assert stats["pool_split_groups"] == 1
    assert stats["pool_split_events"] == 4
