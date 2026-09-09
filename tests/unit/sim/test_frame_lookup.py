import copy
import random

import pytest

from scripts.sim.visualization_3d import derive
from tests.unit.sim.test_visualization_3d_derive import AGENTS, TIMELINE, MEMORIES


@pytest.mark.parametrize("seed", range(12))
def test_sparse_frame_lookup_matches_original_ties_duplicates_and_missing_days(seed):
    rng = random.Random(seed)
    timeline = [{"day": str(rng.randrange(10)), "hour": rng.randrange(24)} for _ in range(80)]
    exact = derive._frame_index_by_day_hour(timeline)
    indexed = derive._FrameLookup(exact)
    for day in [str(i) for i in range(12)]:
        assert derive._has_frame_day(indexed, day) == derive._has_frame_day(exact, day)
        for hour in range(-1, 26):
            assert derive._frame_for_day_hour(indexed, day, hour) == derive._frame_for_day_hour(exact, day, hour)
    assert dict(indexed) == exact


def test_equidistant_hours_choose_frame_index_not_clock_hour():
    indexed = derive._FrameLookup({("day", 16): 0, ("day", 8): 1})
    assert indexed.nearest("day", 12) == 0


def test_build_viz_meta_is_identical_without_index(monkeypatch):
    agents, timeline, memories = copy.deepcopy((AGENTS, TIMELINE, MEMORIES))
    events = {"AGT_A": [{"day": "2026-05-01", "time": "10:00", "spent": 1000,
                          "lon": 126.9, "lat": 37.5, "poi_id": "C_1", "poi_name": "테스트카페"}]}
    actual = derive.build_viz_meta(agents, timeline, memories, events)
    # Preserve original helper behavior by passing a plain dict from the
    # frame-index factory; no stale or globally cached state may leak.
    original_lookup = derive._FrameLookup

    class PlainLookup(original_lookup):
        def __new__(cls, exact):
            return dict(exact)

    monkeypatch.setattr(derive, "_FrameLookup", PlainLookup)
    expected = derive.build_viz_meta(agents, timeline, memories, events)
    assert actual == expected


def test_index_has_no_external_mapping_mutation_or_global_cache():
    mapping = {("d", 12): 9}
    first = derive._FrameLookup(mapping)
    mapping[("d", 12)] = 20
    second = derive._FrameLookup(mapping)
    assert first.nearest("d", 12) == 9
    assert second.nearest("d", 12) == 20
    assert derive._FrameLookup({}).nearest("missing", 12) == 0
