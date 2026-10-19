"""유사도 공식이 말한 대로 움직이는지 — 같으면 1, 반대 부호면 0, 순위 상관과 순열 p."""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import similarity_p012 as sim  # noqa: E402


def test_symmetric_similarity_bounds():
    assert sim.sym_similarity(11.25, 11.25) == 1.0
    assert sim.sym_similarity(-5.0, 5.0) == 0.0          # 부호가 반대면 0
    assert sim.sym_similarity(-1.0, 30.0) == 0.0
    assert math.isclose(sim.sym_similarity(5.0, 10.0), 2 / 3)   # 절반이면 0.67
    assert sim.sym_similarity(10.0, 5.0) == sim.sym_similarity(5.0, 10.0)   # 대칭
    assert sim.sym_similarity(0.0, 0.0) == 1.0


def test_rank_ties_are_averaged():
    assert sim.rankdata([3.0, 1.0, 3.0, 2.0]) == [3.5, 1.0, 3.5, 2.0]


def test_spearman_and_exact_permutation_p():
    t = [36.23, 14.27, 13.82, 13.33, 6.3, 2.8, 11.25, 2.85, 4.62]
    assert math.isclose(sim.spearman(t, t), 1.0)
    assert math.isclose(sim.spearman(t, [-x for x in t]), -1.0)
    # 완전히 같은 순서는 9! 가지 중 하나 — 정확 p = 1/9!
    assert math.isclose(sim.spearman_perm_p(t, t), 1 / math.factorial(9))
    # 완전히 뒤집힌 순서는 모든 줄 세우기가 그 이상이다
    assert sim.spearman_perm_p(t, [-x for x in t]) == 1.0


def test_metrics_combines_effects_and_rank_directions():
    m = sim.metrics([10.0, -2.0], [10.0, 5.0], [3.0], [8.4])
    assert math.isclose(m["magnitude_similarity"], 0.5)      # (1 + 0) / 2
    assert math.isclose(m["direction_agreement"], 2 / 3)     # 효과 2개 + 순위 1개 중 2개
    assert math.isclose(m["mae_pp"], (0 + 7) / 2)
