"""통제 반올림 — 각 칸은 기대치의 내림/올림, 행 합과 열 합은 목표와 정확히 같다. 합성 표만 쓴다."""
import importlib.util
import math
import random
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "demographic_cohort", Path(__file__).resolve().parents[2] / "tools/freeze_demographic_matched_cohort.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _table(seed, rows=40, cols=10, total=200):
    rnd = random.Random(seed)
    w = {("r%02d" % i, "c%d" % j): rnd.random() ** 3 for i in range(rows) for j in range(cols)}
    s = sum(w.values())
    return {k: total * v / s for k, v in w.items()}, total


def _check(cells, total, rnd):
    rsum, csum = {}, {}
    for (r, c), v in cells.items():
        rsum[r] = rsum.get(r, 0.0) + v
        csum[c] = csum.get(c, 0.0) + v
    rt = MODULE.largest_remainder({k: v / total for k, v in rsum.items()}, total)
    ct = MODULE.largest_remainder({k: v / total for k, v in csum.items()}, total)
    out, opened = MODULE.controlled_round(cells, rt, ct, rnd)
    assert opened == 0
    for k, v in cells.items():
        assert out[k] in (math.floor(v), math.ceil(v)), (k, v, out[k])
    for r, t in rt.items():
        assert sum(out[(r, c)] for c in ct) == t
    for c, t in ct.items():
        assert sum(out[(r, c)] for r in rt) == t


def test_rows_and_columns_both_hit_their_rounded_totals():
    for seed in range(20):
        cells, total = _table(seed)
        _check(cells, total, None)
        _check(cells, total, random.Random(seed))


def test_shuffled_search_does_not_favour_the_first_columns():
    # 모든 칸 기대치가 같은 0.5 이면, 섞어 고를 때 첫 열이 다른 열보다 늘 먼저 채워지지 않는다
    cells = {("r%02d" % i, "c%d" % j): 0.5 for i in range(40) for j in range(10)}
    rt = {"r%02d" % i: 5 for i in range(40)}
    ct = {"c%d" % j: 20 for j in range(10)}
    picks_first_half = 0
    for seed in range(30):
        out, _ = MODULE.controlled_round(cells, rt, ct, random.Random(seed))
        early_rows = ["r%02d" % i for i in range(20)]
        picks_first_half += sum(out[(r, "c0")] for r in early_rows)
    # 섞지 않으면 앞 행들이 c0 를 독차지한다(20/20). 섞으면 평균 절반 근처.
    assert 0.3 < picks_first_half / (30 * 20) < 0.7
