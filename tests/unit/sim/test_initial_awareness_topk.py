import importlib
import math
import random
import sys
from unittest.mock import patch

import pytest

with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.neo4j_load._common")}):
    awareness = importlib.import_module("scripts.neo4j_load.07_initial_awareness")


@pytest.mark.parametrize("limit", [-1, 0, 1, 30, 40, 1000])
def test_nearest_pois_preserve_order_including_tied_coordinates(limit):
    rng = random.Random(19)
    pool = [{"id": str(i), "lon": 126.9 + rng.randrange(10) / 1000,
             "lat": 37.5 + rng.randrange(10) / 1000} for i in range(500)]
    expected = sorted(pool, key=lambda p: awareness.haversine_km(p["lon"], p["lat"], 126.91, 37.51))[:limit]
    actual = awareness.nearest_pois(pool, 126.91, 37.51, limit)
    assert actual == expected
    assert all(a is b for a, b in zip(actual, expected))


def test_nonfinite_keys_preserve_original_sort_and_key_evaluation_order(monkeypatch):
    calls = []

    def key(lon, lat, anchor_lon, anchor_lat):
        calls.append(lon)
        return lon

    monkeypatch.setattr(awareness, "haversine_km", key)
    pool = [{"id": str(i), "lon": value, "lat": 0} for i, value in enumerate([1.0, math.nan, 0.0, math.inf, 2.0])]
    expected = sorted(pool, key=lambda p: p["lon"])[:3]
    actual = awareness.nearest_pois(pool, 0, 0, 3)
    assert [p["id"] for p in actual] == [p["id"] for p in expected]
    assert len(calls) == len(pool)
    assert calls[0] == 1 and math.isnan(calls[1]) and calls[2:] == [0, math.inf, 2]


def test_empty_pool_and_invalid_coordinate_behavior():
    assert awareness.nearest_pois([], 0, 0, 30) == []
    with pytest.raises(TypeError):
        awareness.nearest_pois([{"id": "bad", "lon": None, "lat": None}], 0, 0, 30)
