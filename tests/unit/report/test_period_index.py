"""Preserve period membership and float addition order while batching groups."""
import random

import pytest

from scripts.report import analytics


def original_period_cell(scan, days, categories):
    cell = analytics._empty_cell()
    wanted = None if categories is None else {str(c) for c in categories}
    for (day, l1), value in scan["by_day_l1"].items():
        if day not in days or (wanted is not None and l1 not in wanted):
            continue
        for key in cell:
            cell[key] += value[key]
    return cell


@pytest.mark.parametrize("seed", range(8))
def test_period_index_preserves_membership_and_addition_order(seed):
    rng = random.Random(seed)
    # Deliberately different magnitudes and row order expose sum reassociation.
    rows = [(str(d), str(c)) for d in range(15) for c in range(8)]
    rng.shuffle(rows)
    scan = {"by_day_l1": {key: {field: rng.choice([1e16, 0.01, -1e16, rng.random()])
                               for field in analytics._empty_cell()} for key in rows}}
    days = ["12", "7", "1", "7", "missing"]
    indexed = analytics._period_cells_by_category(scan, days)
    for category in [None, [], ["1"], ["2", "4", "2"], ["missing"]]:
        assert analytics._period_cell(scan, days, category) == original_period_cell(scan, days, category)
    for category, value in indexed.items():
        assert value == original_period_cell(scan, days, [category])
    assert analytics._period_cells_by_category(scan, []) == {}


def test_category_did_scans_table_four_times_independent_of_category_count():
    class CountingDict(dict):
        calls = 0

        def items(self):
            self.calls += 1
            return super().items()

    table = CountingDict({(str(day), str(cat)): {**analytics._empty_cell(), "amt": 1000.0, "events": 1.0}
                          for day in range(10) for cat in range(50)})
    scan = {"by_day_l1": table, "by_l1": {str(c): {**analytics._empty_cell(), "amt": 10000.0}
                                            for c in range(50)}}
    result = analytics.did_by_category(scan, {"pre": [str(i) for i in range(5)],
                                              "post": [str(i) for i in range(5, 10)]}, ["0"], ["1"])
    assert len(result) == 50
    assert table.calls == 4  # two control totals, two category indexes
    assert all(row["pre_events"] == row["post_events"] == 5 for row in result)
