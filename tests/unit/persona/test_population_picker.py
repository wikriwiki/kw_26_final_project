import importlib
import random
import sys
from unittest.mock import patch

import pytest

with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.persona._common")}):
    conditional = importlib.import_module("scripts.persona.build_conditional")


@pytest.mark.parametrize("seed", range(10))
def test_population_picker_matches_each_draw_and_rng_state(seed):
    rng = random.Random(seed)
    index = {(gu, sex, age): [(f"{gu}/{i}", rng.choice([0.0, 0.01, 100.0, 1e16]), {"index": i})
                             for i in range(20)]
             for gu in ["a", "b"] for sex in ["F", "M"] for age in ["20", "30"]}
    picker = conditional.PopulationPicker(index)
    old, new = random.Random(seed), random.Random(seed)
    for _ in range(200):
        cell = (rng.choice(["a", "b", "missing"]), rng.choice(["F", "M", "?"]), rng.choice(["20", "30"]))
        assert picker.pick(cell, new) == conditional.pick_dong_by_population(index, cell, old)
        assert old.getstate() == new.getstate()


def test_conditional_build_output_matches_original(monkeypatch):
    profiles = {f"1111051{i}_F_30대": {"location": {"gu": f"gu{i}", "dong": str(i)},
                                        "demographics": {"population": i + 1}}
                for i in range(4)}
    monkeypatch.setattr(conditional, "load_bdc_stats", lambda: {"profiles": profiles, "deciles": {}})
    monkeypatch.setattr(conditional, "load_nvidia_seoul", lambda: [
        {"uuid": str(i), "sex": "여자", "age": 35, "district": f"gu{i % 6}"} for i in range(60)])
    actual = conditional.build(seed=83, limit=15)
    monkeypatch.setattr(conditional.PopulationPicker, "pick",
                        lambda self, cell, rng: conditional.pick_dong_by_population(self.profile_index, cell, rng))
    assert actual == conditional.build(seed=83, limit=15)


def test_invalid_population_and_custom_rng_preserve_contract():
    cell = ("gu", "F", "30")
    index = {cell: [("one", 0.0, {})]}
    with pytest.raises(ValueError):
        conditional.PopulationPicker(index).pick(cell, random.Random(1))

    class CustomRandom(random.Random):
        def choices(self, population, weights=None, *, k=1):
            assert weights == [0.0]
            return [population[0]]

    assert conditional.PopulationPicker(index).pick(cell, CustomRandom(1)) == ("one", 0.0, {}, "gu_sex_age")
