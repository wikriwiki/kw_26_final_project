"""The convergence page must not plot incomparable effects on one scale."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "build_convergence_page", ROOT / "scripts/report/build_convergence_page.py")
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)


def test_incomparable_numbers_have_no_gap_or_shared_truth_marker():
    row = {"pct": 12.6, "truth": 7.3, "ci_pct": None,
           "comparable": False, "state": "measured", "expect_txt": "증가",
           "n": 200, "dir_ok": True, "crosses": False, "note": "", "suspect": False}
    assert '직접 크기 비교 불가' in page._nums(row)
    assert 'class="gap"' not in page._nums(row)
    assert 'class="mk tru"' not in page._bar(row)
    row["comparable"] = True
    assert 'class="gap"' in page._nums(row)
    assert 'class="mk tru"' in page._bar(row)


def test_historical_runs_have_no_direct_magnitude_comparison():
    rows = page.collect()
    assert rows
    assert not any(row.get("comparable") for row in rows)
    assert all('class="gap"' not in page._nums(row) for row in rows)
    suspect = [row for row in rows if row['suspect'] and row['dir_ok'] is not None]
    assert suspect
    assert all('방향 일치' not in page._nums(row) for row in suspect)
