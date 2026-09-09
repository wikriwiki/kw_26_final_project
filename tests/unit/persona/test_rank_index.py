"""Compare indexed rank assignment with the original filtering implementation."""
import importlib
import random
import sys
from unittest.mock import patch

import pytest

from scripts.persona.rank_index import RankMatcher

with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.persona._common")}):
    original = importlib.import_module("scripts.persona.build_rank_coupling")


@pytest.mark.parametrize("seed", range(12))
def test_rank_choices_preserve_ties_duplicates_and_exhausted_fallbacks(seed):
    rng = random.Random(seed)
    rows = [{"uuid": rng.choice([None, "", f"u{i}", "duplicate"]),
             "sex": rng.choice(["여자", "남자"]), "age": rng.choice([25, 35, 45, 55, 65]),
             "district": rng.choice(["마포구", "종로구"]),
             "education_level": rng.choice(["고등학교", "대학원"])} for i in range(120)]
    # Every SES score has many ties. all_sorted and merged cell order differ.
    pool_index = original.index_nvidia_pool(rows)
    all_sorted = sorted(rows, key=original.ses_proxy)
    matcher = RankMatcher(pool_index, all_sorted, original._AGE_NEIGHBORS, original.ses_proxy)
    used = set()
    levels = set()
    for _ in range(300):
        cell = (rng.choice(["마포구", "종로구", "missing"]), rng.choice(["F", "M", "?"]),
                rng.choice(["10대", "20대", "30대", "40대", "50대", "60대", "70대이상", "unknown"]))
        percentile = rng.choice([0, 0.25, 0.5, 0.75, 1, rng.random()])
        expected, expected_level = original.pick_nvidia_by_rank(pool_index, all_sorted, cell, percentile, used)
        actual, level = matcher.pick(cell, percentile)
        assert actual is expected
        assert level == expected_level
        levels.add(level)
        if expected.get("uuid"):
            used.add(expected["uuid"])
        matcher.mark_used(actual.get("uuid"))
        assert matcher.used == used
    assert "any_emergency" in levels


def test_rank_index_late_pool_creation_and_complete_exhaustion():
    rows = [{"uuid": f"u{i}", "sex": "여자", "age": 35, "district": "마포구"} for i in range(20)]
    pool_index = original.index_nvidia_pool(rows)
    matcher = RankMatcher(pool_index, rows, original._AGE_NEIGHBORS, original.ses_proxy)
    used = set()
    for i in range(30):
        # Materialize progressively broader pools after earlier selections.
        cell = [("마포구", "F", "30대"), ("missing", "F", "30대"),
                ("missing", "F", "20대"), ("missing", "F", "unknown"),
                ("missing", "?", "unknown")][i // 6]
        expected = original.pick_nvidia_by_rank(pool_index, rows, cell, 0.5, used)
        actual = matcher.pick(cell, 0.5)
        assert actual == expected
        used.add(expected[0]["uuid"])
        matcher.mark_used(actual[0]["uuid"])
        matcher.mark_used(actual[0]["uuid"])  # repeated exclusion is idempotent


def test_rank_index_empty_input_keeps_original_error():
    matcher = RankMatcher({}, [], original._AGE_NEIGHBORS, original.ses_proxy)
    with pytest.raises(IndexError):
        matcher.pick(("missing", "F", "30대"), 0.5)


def test_full_persona_build_matches_original_assignment(monkeypatch):
    # Exercise the real quant generation, assembly, serialization fields, and
    # final diverse sampling, with self-contained population inputs.
    profiles = {f"11110515_F_{age}": {"location": {"gu": "마포구", "dong": "동"},
                                    "consumption": {"weekday_spending_level": 5, "weekend_spending_level": 4}}
                for age in ["20대", "30대", "40대"]}
    monkeypatch.setattr(original, "load_bdc_stats", lambda: {
        "profiles": profiles, "allocation": {key: 15 for key in profiles}, "deciles": {}})
    monkeypatch.setattr(original, "load_nvidia_seoul", lambda: [
        {"uuid": f"u{i}", "sex": "여자", "age": 25 + (i % 3) * 10, "district": "마포구",
         "occupation": "사무원", "persona": str(i)} for i in range(40)])
    actual = original.build(seed=76, limit=10)

    class OriginalMatcher:
        def __init__(self, pool_index, all_sorted, *args):
            self.pool_index, self.all_sorted, self.used = pool_index, all_sorted, set()

        def pick(self, cell, percentile):
            return original.pick_nvidia_by_rank(self.pool_index, self.all_sorted, cell, percentile, self.used)

        def mark_used(self, uuid):
            if uuid:
                self.used.add(uuid)

    monkeypatch.setattr(original, "RankMatcher", OriginalMatcher)
    assert actual == original.build(seed=76, limit=10)
