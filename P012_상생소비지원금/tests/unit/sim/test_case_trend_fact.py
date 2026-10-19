"""후보 2 — 확진 수준에 **기준**을 같이 준다 (EXP_CASE_TREND).

설계: `experiments/case_trend/design_note.md` (후보 1 탐침 결과를 보기 전에 적었다)

모델은 손에 쥔 돈에 세게, 제약에 약하게 움직인다(EM-3 +12.6% 대 실측 +7.3% /
DS-1 −7.7% 대 −14.1%). 제약 쪽에서 받는 세상 정보는 "신규 확진 112명(7일 평균
110명)" 뿐인데, **110명이 많은지 적은지 그 줄만으로는 모른다.**

같은 원자료에서 2주 전 7일 평균을 세어 배수를 적는다. 새 자료를 끌어오지 않고,
방향도 말하지 않는다.

### 이것이 사실인가 넛지인가

불편한 점을 먼저 적는다 — 이 배수는 무정책 창(1.9배)과 정책 창(2.6배)에서
다르고 그 차이가 정답지 방향으로 민다. 그래서 **반증 조건을 설계에 먼저
등록했다**: 정책이 없는 시점 위약에서 소비가 움직이면 이 줄은 상태가 아니라
방향을 주입한 것이다. 그 판정은 라운드에서 한다 — 여기서는 **그 줄이 사실인지,
방향 어휘가 없는지, 기본이 꺼짐인지**만 본다.
"""
from datetime import date
from pathlib import Path
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from environments import build_environment   # noqa: E402

OFF_DAY = date(2020, 11, 17)      # 거리두기 무정책 창
ON_DAY = date(2020, 11, 24)       # 정책 창


def facts(day, flag):
    old = os.environ.get('EXP_CASE_TREND')
    os.environ['EXP_CASE_TREND'] = flag
    try:
        return build_environment('covid_2021', day).get('facts') or []
    finally:
        if old is None:
            os.environ.pop('EXP_CASE_TREND', None)
        else:
            os.environ['EXP_CASE_TREND'] = old


@pytest.mark.parametrize('day', [OFF_DAY, ON_DAY])
def test_off_by_default_so_every_past_run_is_reproducible(day):
    assert not any('2주 전' in f for f in facts(day, '0'))


@pytest.mark.parametrize('day', [OFF_DAY, ON_DAY])
def test_on_adds_exactly_one_fact_and_changes_nothing_else(day):
    off, on = facts(day, '0'), facts(day, '1')
    assert len(on) == len(off) + 1
    added = [f for f in on if f not in off]
    assert len(added) == 1 and '2주 전' in added[0]
    assert [f for f in on if f != added[0]] == off, '있던 사실이 바뀌었다'


def test_the_multiple_is_actually_computed_from_the_same_source():
    """배수가 맞는가 — 틀리면 거짓을 주입하는 것이다."""
    import re
    for day, want in ((OFF_DAY, 1.9), (ON_DAY, 2.6)):
        line = [f for f in facts(day, '1') if '2주 전' in f][0]
        base = int(re.search(r'평균은 ([\d,]+)명', line).group(1).replace(',', ''))
        mult = float(re.search(r'그 ([\d.]+)배', line).group(1))
        now = int(re.search(r'최근 \d+일 평균 ([\d,]+)명',
                            [f for f in facts(day, '1') if '신규 확진' in f][0])
                  .group(1).replace(',', ''))
        assert abs(mult - now / base) < 0.05, (day, mult, now, base)
        assert abs(mult - want) < 0.05, '기대한 배수와 다르다: %s' % line


@pytest.mark.parametrize('bad', [
    '나가지', '자제', '줄이', '삼가', '위험하', '조심', '덜 ', '더 써',
    '-14.1', '14.1%', '4.2%',
])
def test_it_states_a_level_not_a_direction(bad):
    for day in (OFF_DAY, ON_DAY):
        line = [f for f in facts(day, '1') if '2주 전' in f][0]
        assert bad not in line, '방향이나 정답이 샌다: %r in %r' % (bad, line)


def test_it_stays_silent_when_there_is_no_two_week_history():
    """자료가 없으면 지어내지 않는다 — 없는 기준을 만들면 거짓이다."""
    early = date(2020, 2, 10)      # 원자료 시작(2020-02-06) 바로 뒤
    assert not any('2주 전' in f for f in facts(early, '1'))


def test_the_two_candidates_are_separate_switches():
    """후보 1 과 후보 2 를 한 스위치로 묶으면 어느 줄이 무엇을 했는지 못 가른다."""
    old1 = os.environ.get('EXP_SCOPE_FACT')
    os.environ['EXP_SCOPE_FACT'] = '1'
    try:
        assert not any('2주 전' in f for f in facts(ON_DAY, '0')), \
            '후보 1 을 켰는데 후보 2 가 따라 켜졌다'
    finally:
        if old1 is None:
            os.environ.pop('EXP_SCOPE_FACT', None)
        else:
            os.environ['EXP_SCOPE_FACT'] = old1
