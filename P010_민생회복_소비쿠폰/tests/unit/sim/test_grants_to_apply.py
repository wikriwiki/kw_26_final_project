"""지급 게이트 — `plan_writer.grants_to_apply`

**이 시험들이 지키려는 사고**: `p013_ruler` 에서 전 분위 280,000원짜리 grant 정책이
**한 푼도 지급되지 않았다.** `grant_applied_today` 가 모든 날 모든 사람 0 이었고,
그 사실을 몇 달 몰랐다. 게이트가 `process_one` 안에 인라인이라 시험할 수 없었다.

가장 중요한 시험은 `test_effective_from_이_Date_객체여도_지급된다` 다. 옛 게이트는
`str(today) != pol["effective_from"]` 로 **문자열 비교**를 했다. 날짜가 문자열이
아닌 형태로 들어오면 **항상 불일치**라 `continue` 로 빠져 지급이 영영 일어나지
않는다 — 조용히, 오류 없이.

`experiments/plan_channel/P013_evidence_is_weaker.md`
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "sim"))

from plan_writer import as_date, grants_to_apply  # noqa: E402

P013 = {"id": "P013", "type": "grant", "effective_from": "2020-05-13",
        "grant_key": "spend_decile",
        "decile_grants": {str(i): 280000 for i in range(1, 11)}}


class _Neo4jDate:
    """neo4j.time.Date 흉내 — `to_native()` 를 준다."""

    def __init__(self, d):
        self._d = d

    def to_native(self):
        return self._d

    def __str__(self):
        return self._d.isoformat()


# ---------------------------------------------------------------- 날짜 변환

@pytest.mark.parametrize("v", [
    "2020-05-13",
    date(2020, 5, 13),
    "2020-05-13T00:00:00",
    _Neo4jDate(date(2020, 5, 13)),
])
def test_어떤_형태로_와도_같은_날이_된다(v):
    assert as_date(v) == date(2020, 5, 13)


@pytest.mark.parametrize("v", [None, "", "아무말", 12345])
def test_못_바꾸면_None(v):
    assert as_date(v) is None


# ---------------------------------------------------------------- 게이트

def test_시행일에_지급된다():
    got = grants_to_apply([P013], date(2020, 5, 13), {}, "하", 2)
    assert got == {"P013": 280000}


def test_effective_from_이_Date_객체여도_지급된다():
    """**이것이 그 사고다.** 문자열 비교였으면 여기서 0 이 된다."""
    pol = dict(P013, effective_from=_Neo4jDate(date(2020, 5, 13)))
    assert grants_to_apply([pol], date(2020, 5, 13), {}, "하", 2) == {"P013": 280000}


def test_today_가_문자열이어도_지급된다():
    assert grants_to_apply([P013], "2020-05-13", {}, "하", 2) == {"P013": 280000}


def test_시행일이_아니면_안_준다():
    for d in (date(2020, 5, 12), date(2020, 5, 14)):
        assert grants_to_apply([P013], d, {}, "하", 2) == {}


def test_이미_받았으면_다시_안_준다():
    """resume 멱등 — 두 번 주면 지갑이 부풀어 정책 효과가 조작된다."""
    assert grants_to_apply([P013], date(2020, 5, 13), {"P013": 280000}, "하", 2) == {}


def test_grant_가_아닌_정책은_건너뛴다():
    cash = {"id": "P012", "type": "cashback", "effective_from": "2020-05-13"}
    assert grants_to_apply([cash], date(2020, 5, 13), {}, "하", 2) == {}


def test_분위가_없으면_0원이라_안_실린다():
    assert grants_to_apply([P013], date(2020, 5, 13), {}, "하", None) == {}


def test_여러_정책이_같은_날_시행되면_다_준다():
    p2 = dict(P013, id="P099")
    got = grants_to_apply([P013, p2], date(2020, 5, 13), {}, "하", 2)
    assert got == {"P013": 280000, "P099": 280000}


def test_정책이_없으면_빈_dict():
    assert grants_to_apply(None, date(2020, 5, 13)) == {}
    assert grants_to_apply([], date(2020, 5, 13)) == {}


def test_소득_기반_정책도_그대로_동작한다():
    pol = {"id": "P009", "type": "grant", "effective_from": "2020-05-13",
           "grant_key": "income", "income_grants": {"하": 500000, "중": 300000}}
    assert grants_to_apply([pol], date(2020, 5, 13), {}, "하")["P009"] == 500000
    assert grants_to_apply([pol], date(2020, 5, 13), {}, "상") == {}
