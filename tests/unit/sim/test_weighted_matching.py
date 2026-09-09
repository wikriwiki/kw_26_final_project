"""Exact choices (including RNG state), not just distribution comparisons."""
import importlib
import itertools
import math
import random
import sys
from unittest.mock import patch

import pytest

from scripts.sim.weighted_matching import _WeightTree, indexed_softmax_select

with patch.dict(sys.modules, {"_common": importlib.import_module("scripts.neo4j_load._common")}):
    from scripts.sim.night_interaction import _softmax_select, _softmax_select_scan


def scored_graph(seed, nodes=70, edges=350, quantized=True):
    rng = random.Random(seed)
    pairs = rng.sample(list(itertools.combinations(range(nodes), 2)), edges)
    return [{"aid_a": f"a{a}", "aid_b": f"a{b}",
             "score": rng.randrange(1, 11) / 10 if quantized else rng.random(),
             "exposure": rng.random()} for a, b in pairs]


@pytest.mark.parametrize("temperature", [0.5, -0.5, 0.001, 1e100, 1e-300])
@pytest.mark.parametrize("limit", [1, 2, 5])
def test_seeded_matching_is_identical(temperature, limit):
    for seed in range(20):
        rows = scored_graph(seed, quantized=bool(seed % 2))
        before = [dict(row) for row in rows]
        old_rng, new_rng = random.Random(seed + 71), random.Random(seed + 71)
        expected = _softmax_select_scan(rows, limit, temperature, old_rng)
        stats = {}
        actual = indexed_softmax_select(rows, limit, temperature, new_rng, _softmax_select_scan, stats=stats)
        assert actual == expected
        assert all(a is b for a, b in zip(actual, expected))
        assert old_rng.getstate() == new_rng.getstate()
        assert rows == before
        assert stats["tree_draws"] + stats["reference_draws"] == len(actual)
        assert stats["removed"] <= len(rows)


def _disjoint_rows(n=300, score=0.5):
    return [{"aid_a": f"a{i}", "aid_b": f"b{i}", "score": score} for i in range(n)]


@pytest.mark.parametrize("uniform", [0.0, 1 / 300, math.nextafter(1 / 300, 0),
                                     math.nextafter(1 / 300, 1), 0.5, math.nextafter(1.0, 0)])
def test_boundary_draws_replay_original_without_extra_rng_call(uniform):
    rows = _disjoint_rows()
    counts = [0, 0]

    def rng_for(which):
        rng = random.Random(1)

        def draw():
            counts[which] += 1
            return uniform

        rng.random = draw
        return rng

    expected = _softmax_select_scan(rows, 1, 0.5, rng_for(0))
    stats = {}
    actual = indexed_softmax_select(rows, 1, 0.5, rng_for(1), _softmax_select_scan, stats=stats)
    assert actual == expected
    assert counts == [len(rows), len(rows)]
    assert stats["reference_draws"] > 0


def test_underflowed_weights_reappear_when_maximum_is_removed():
    rows = _disjoint_rows()
    for i, row in enumerate(rows):
        row["score"] = [0.0, -740.0, -750.0, -1500.0][i % 4]
    old, new = random.Random(4), random.Random(4)
    stats = {}
    assert indexed_softmax_select(rows, 1, 1.0, new, _softmax_select_scan, stats=stats) == _softmax_select_scan(rows, 1, 1.0, old)
    assert new.getstate() == old.getstate()
    assert stats["rebuilds"] == 4


@pytest.mark.parametrize("kind", ["duplicate", "self", "rng", "choices", "zero", "nan", "infinity", "zero_limit"])
def test_unusual_private_inputs_keep_original_behavior(kind):
    rows = _disjoint_rows()
    temperature, limit = 0.5, 2
    if kind == "duplicate":
        rows.append(dict(rows[0]))
    if kind == "self":
        rows[0]["aid_b"] = rows[0]["aid_a"]
    if kind == "zero":
        temperature = 0.0
    if kind == "nan":
        rows[0]["score"] = float("nan")
    if kind == "infinity":
        temperature = float("inf")
    if kind == "zero_limit":
        limit = 0

    class LastChoice(random.Random):
        def choices(self, population, weights=None, *, k=1):
            return [population[-1]] * k

    rng_type = LastChoice if kind == "rng" else random.Random
    a, b = rng_type(8), rng_type(8)
    if kind == "choices":
        a.choices = b.choices = lambda population, **kwargs: [population[-1]]
    try:
        expected = _softmax_select_scan(rows, limit, temperature, a)
    except Exception as error:
        with pytest.raises(type(error), match=str(error)):
            _softmax_select(rows, limit, temperature, b)
    else:
        assert _softmax_select(rows, limit, temperature, b) == expected
    assert a.getstate() == b.getstate()


def test_equal_score_graph_uses_one_weight_build():
    rows = _disjoint_rows(1024)
    stats = {}
    actual = indexed_softmax_select(rows, 1, 0.5, random.Random(91), _softmax_select_scan, stats=stats)
    assert len(actual) == 1024
    assert stats["rebuilds"] == 1
    assert stats["weight_evaluations"] == len(rows)
    assert stats["tree_draws"] == len(rows)
    assert stats["removed"] == len(rows)


def test_weight_tree_recomputes_sums_and_ignores_zero_leaves():
    weights = [1.0, 1e-300, 0.0, 2.0, 0.0]
    tree = _WeightTree(weights)
    assert tree.choose(0) == 0
    assert tree.choose(1.1) == 3
    tree.remove(0)
    assert tree.choose(0) == 1
    assert tree.prefix(3) == 1e-300
    tree.remove(3)
    assert tree.total == 1e-300
