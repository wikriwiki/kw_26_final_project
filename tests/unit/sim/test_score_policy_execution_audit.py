"""codex/fe039529-execution-audit에서 가져온 채점 회귀 검사."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))

from score_policy import daterange, metric_values, paired  # noqa: E402


def test_category_entry_and_exit_remain_in_paired_sample():
    off = [
        {"aid": "A", "d": "d1", "amt": 100, "l1": "식사"},
        {"aid": "B", "d": "d1", "amt": 100, "l1": "카페"},
        {"aid": "C", "d": "d1", "amt": 0, "l1": "집"},
    ]
    on = [
        {"aid": "A", "d": "d2", "amt": 100, "l1": "카페"},
        {"aid": "B", "d": "d2", "amt": 50, "l1": "식사"},
        {"aid": "C", "d": "d2", "amt": 0, "l1": "집"},
    ]
    for row in off + on:
        row.update(kdi=None, sub=None)

    before, after = metric_values("sector_spend:카페", off, on, ["d1"], ["d2"])

    assert before == {"A": 0, "B": 100, "C": 0}
    assert after == {"A": 100, "B": 0, "C": 0}
    assert paired(before, after) == [100, -100, 0]


def test_paired_order_is_stable():
    assert paired({"B": 3, "A": 1}, {"A": 4, "B": 8}) == [3, 5]


def test_invalid_date_window_rejected():
    with pytest.raises(ValueError, match="ascending"):
        daterange("2021-10-02:2021-10-01")


def test_disjoint_policy_results_cannot_rank_candidates(tmp_path):
    table = json.loads(
        (ROOT / "data" / "experiments" / "scoring_table.json").read_text(encoding="utf-8")
    )
    policies = [
        (name, spec)
        for name, spec in table.items()
        if not name.startswith("_") and isinstance(spec, dict) and spec.get("indicators")
    ][:2]
    assert len(policies) == 2

    for index, (name, spec) in enumerate(policies):
        indicator = spec["indicators"][0]
        result = {
            "label": f"candidate{index}",
            "policy": name,
            "results": [{**indicator, "hit": True, "got": indicator["expect"], "mean": 1}],
        }
        (tmp_path / f"{index}.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "sim" / "rank_candidates.py"),
            "--dir",
            str(tmp_path),
            "--stage",
            "3",
        ],
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 2
