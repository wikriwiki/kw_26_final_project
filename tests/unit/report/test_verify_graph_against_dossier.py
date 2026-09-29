"""덤프를 복원한 그래프가 dossier 와 사람 단위로 같은지 — 비교 논리를 못 박는다.

실제 복원은 런이 끝난 뒤에만 할 수 있다(Community 는 DB 하나다). 그래서 그래프 조회를
가짜로 바꿔, 같으면 통과하고 하나라도 다르면 잡는지를 여기서 확인한다.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.report.verify_graph_against_dossier import compare, expected, sample

D = Path(__file__).resolve().parents[3] / "data/experiments/p012t_dossier/on.dossier.jsonl"


def recs():
    return [json.loads(x) for x in D.read_text(encoding="utf-8").splitlines() if x.strip()]


def faithful(rs):
    table = {r["aid"]: expected(r) for r in rs}
    return lambda aid, s, e: dict(table[aid])


def test_a_graph_that_matches_the_dossier_passes():
    rs = recs()
    assert compare(rs, ("2021-10-01", "2021-10-07"), faithful(rs)) == []


def test_one_lost_memory_is_caught():
    rs = recs()
    good = faithful(rs)
    victim = rs[5]["aid"]

    def lossy(aid, s, e):
        got = good(aid, s, e)
        if aid == victim:
            got["mem"] -= 1          # 그래프에서 기억 하나가 사라졌다
        return got
    bad = compare(rs, ("2021-10-01", "2021-10-07"), lossy)
    assert [b["aid"] for b in bad] == [victim]


def test_a_missing_state_day_is_caught():
    rs = recs()
    good = faithful(rs)
    victim = rs[0]["aid"]

    def gap(aid, s, e):
        got = good(aid, s, e)
        if aid == victim:
            got["days"] = got["days"][:-1]
        return got
    assert [b["aid"] for b in compare(rs, ("2021-10-01", "2021-10-07"), gap)] == [victim]


def test_sampling_spans_the_memory_distribution_not_only_the_top():
    rs = recs()
    picked = sample(rs, 10)
    counts = sorted(len(r.get("memories") or []) for r in rs)
    got = sorted(len(r.get("memories") or []) for r in picked)
    assert got[0] == counts[0] and got[-1] == counts[-1]   # 양 끝을 모두 포함한다
    assert len({r["aid"] for r in picked}) == 10
