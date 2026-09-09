"""Behavior and work-count regressions for indexed simulation operations."""
import copy
import importlib
import random
import sys
from datetime import date
from unittest.mock import patch

import pytest

from tests.unit.sim import performance_reference as reference

# CLI entrypoints intentionally import their local _common. Other suites also
# load persona/_common under that name; emulate the standalone CLI resolution
# during import without changing either module's runtime contract.
with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.neo4j_load._common")}):
    from scripts.sim import export_visualization as export
    from scripts.sim import night_interaction as night
    social = importlib.import_module("scripts.neo4j_load.06_social")


@pytest.mark.parametrize("size", [0, 1, 2, 5, 21, 22, 50, 100, 300])
def test_social_pairs_and_rng_state_match(size):
    for seed in range(8):
        rng = random.Random(seed)
        members = [f"a{i}" for i in range(size)]
        work = {"w": members, "empty": []}
        home = {"h": list(reversed(members)), "subset": members[::3]}
        # Duplicates can result from multiple graph anchors. They must retain
        # their sampling probability and all copies of self must be excluded.
        if members:
            work["duplicate"] = rng.choices(members, k=30)
            home["duplicate"] = rng.choices(members, k=25)
        before = copy.deepcopy((work, home))
        a, b = random.Random(seed), random.Random(seed)
        expected = reference.social_pairs(work, home, a)
        actual = social.build_social_pairs(work, home, b)
        assert actual == expected
        assert list(actual) == list(expected)  # same set insertion/iteration
        assert a.getstate() == b.getstate()
        assert (work, home) == before


def test_social_other_members_preserves_all_duplicate_exclusions():
    values = ["a", "b", "b", "c", "a", "d", "a"]
    for member in values:
        view = social._OtherMembers(values, [i for i, x in enumerate(values) if x == member])
        expected = [x for x in values if x != member]
        assert list(view) == expected
        assert view[::-1] == expected[::-1]
        assert view[-1] == expected[-1]
        with pytest.raises(IndexError):
            _ = view[len(view)]


class CountingList(list):
    def __init__(self, values):
        super().__init__(values)
        self.visited = 0

    def __iter__(self):
        for item in super().__iter__():
            self.visited += 1
            yield item


def test_social_large_groups_do_not_rescan_members_per_agent():
    members = CountingList([f"a{i}" for i in range(1000)])
    social.build_social_pairs({"work": members}, {"home": members}, random.Random(42))
    assert members.visited == 4 * len(members)  # index + sample loop per group


@pytest.mark.parametrize("fractional", [False, True])
def test_exposure_matches_cartesian_multiplicity_and_missing_values(fractional):
    rng = random.Random(4096)
    hours = [None, 0, 1, 12, 13, 23, 24]
    if fractional:
        hours += [12.25, 12.5]
    for _ in range(200):
        visits = {a: [(rng.choice([None, "d1", "d2"]), rng.choice(hours))
                      for _ in range(rng.randrange(35))] for a in ["a", "b"]}
        data = {"visits": visits}
        index = {a: night._count_visits(v) for a, v in visits.items()}
        for a, b in [("a", "b"), ("b", "a"), ("missing", "a"), ("a", "a")]:
            expected = reference.exposure(a, b, data)
            assert night.calc_exposure(a, b, data) == expected
            assert night.calc_exposure(a, b, data, visit_counts=index) == expected


def test_indexed_exposure_does_not_rescan_raw_visits():
    visits = {a: CountingList([("d", 12)] * 300) for a in ["a", "b"]}
    index = {a: night._count_visits(v) for a, v in visits.items()}
    for _ in range(100):
        assert night.calc_exposure("a", "b", {"visits": visits}, visit_counts=index) == 1.0
    assert all(v.visited == 300 for v in visits.values())


@pytest.mark.parametrize("stochastic", [False, True])
def test_night_selected_pairs_identical_with_original_scorer(monkeypatch, stochastic):
    rng = random.Random(8)
    ids = [f"a{i:02}" for i in range(40)]
    data = {"visits": {a: [("d", rng.randrange(10, 15)) for _ in range(6)] for a in ids},
            "outed": set(ids), "knows": {(a, b): "colleague" for a, b in zip(ids, ids[1:])},
            "conv_history": {}, "state": {}, "info_count": {a: i for i, a in enumerate(ids)}}
    monkeypatch.setattr(night, "fetch_all", lambda day: data)
    optimized = night.select_interaction_pairs(date(2021, 9, 6), seed=71, verbose=False, stochastic=stochastic)
    monkeypatch.setattr(night, "calc_exposure", lambda a, b, data, **kw: reference.exposure(a, b, data))
    original = night.select_interaction_pairs(date(2021, 9, 6), seed=71, verbose=False, stochastic=stochastic)
    assert optimized == original


def make_events(seed, n_agents=10, n_days=10):
    rng = random.Random(seed)
    days = [f"2021-09-{i+1:02}" for i in range(n_days)]
    agents = {}
    for a in range(n_agents):
        events = []
        for day in days:
            for order in range(rng.randrange(0, 9)):
                # Deliberately non-monotonic hours: preserve the original
                # prefix walk over order rather than silently sorting by time.
                events.append({"day": day, "ord": order, "time": f"{rng.randrange(24):02}:30",
                               "lon": rng.choice([0, None, 126.9]), "lat": 37.5,
                               "cat": rng.choice([None, "식사"]), "intent": f"{a}/{day}/{order}",
                               "sat": rng.random(), "anchor": "residence", "spent": order * 1000})
        agents[str(a)] = events
    return agents, days


@pytest.mark.parametrize("seed", range(8))
def test_timeline_frames_match_order_gaps_and_coordinate_fallback(seed):
    agents, days = make_events(seed)
    for chosen_days in [days, days[::2], list(reversed(days)), [days[0], days[0]], []]:
        before = copy.deepcopy(agents)
        assert export.build_timeline_frames(agents, chosen_days) == reference.timeline_frames(agents, chosen_days)
        assert agents == before


def test_timeline_scans_each_raw_event_once():
    agents, days = make_events(91)
    agents = {a: CountingList(events) for a, events in agents.items()}
    export.build_timeline_frames(agents, days)
    assert all(events.visited == len(events) for events in agents.values())
