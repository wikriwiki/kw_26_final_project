"""v46·v47 이 셋을 한꺼번에 넣어 갈라 읽을 수 없게 만들었다. v48·v49 가 하나씩 맡는다.

v47 결과: 겨냥한 둘(explanation 6→3 · anchor_category 5→1)은 고쳐졌고 겨냥하지 않은
하나(time 11→29)가 그보다 크게 망가졌다. 마무리 예시와 예시 정렬 중 어느 것이 무엇을
했는지 말할 수 없었다. 이 두 후보가 그것을 가른다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v45, v46, v47, v48, v49  # noqa: E402

V45, V46, V47 = v45.SYSTEM_PROMPT, v46.SYSTEM_PROMPT, v47.SYSTEM_PROMPT
V48, V49 = v48.SYSTEM_PROMPT, v49.SYSTEM_PROMPT
EV = re.compile(r'\{"time":"(\d\d):(\d\d)","anchor":"([^"]+)"')


def events(text):
    a = text.index('{"events": [')
    return EV.findall(text[a:text.index('\n ],\n "daily_propensity"', a)])


def test_v48_closes_the_day_like_v47_does():
    assert events(V48)[-1][2] == 'residence'
    assert '"time":"22:30","anchor":"residence","category":"집"' in V48


def test_v48_does_not_sort_the_example():
    """정렬이 간격 위반을 11 → 25 로 늘린 것으로 보인다. 그것은 넣지 않는다."""
    mins45 = [int(h) * 60 + int(m) for h, m, _ in events(V45)]
    mins48 = [int(h) * 60 + int(m) for h, m, _ in events(V48)]
    assert mins48[:len(mins45)] == mins45          # 앞부분 순서가 v45 그대로다
    assert mins48 != sorted(mins48)                # 정렬되지 않았다


def test_v48_does_not_add_the_time_rule():
    assert V48.count('최소 20분 뒤') == 0
    assert '이벤트 간 최소 체류 20분' in V48


def test_v49_adds_only_the_time_rule():
    assert '최소 20분 뒤' in V49
    assert '시각은 하루 동안 뒤로 가지 않는다' in V49


def test_v49_touches_no_example():
    assert events(V49) == events(V45)
    assert len(re.findall(r'\{"time":', V49)) == len(re.findall(r'\{"time":', V45))


def test_the_three_changes_are_now_separable():
    """v48 ∪ v49 의 변경이 합쳐져 v46+v47 의 변경 집합을 이룬다 — 정렬만 빼고."""
    assert V48 != V45 and V49 != V45 and V48 != V49
    assert '최소 20분 뒤' in V46 and '최소 20분 뒤' in V47     # v46 이 문장을 넣었다
    assert events(V47)[-1][2] == 'residence'                   # v47 이 마무리를 넣었다
    mins47 = [int(h) * 60 + int(m) for h, m, _ in events(V47)]
    assert mins47 == sorted(mins47)                            # v46 이 정렬했다


# 프롬프트 전문을 파라미터로 넘기면 pytest 가 그것을 시험 id 로 써서 환경변수
# 한도(32767자)를 넘긴다. 이름으로 넘기고 본문은 표에서 찾는다.
BODIES = {'v48': V48, 'v49': V49}


@pytest.mark.parametrize('name', sorted(BODIES))
def test_policy_neutrality_and_format_are_kept(name):
    text = BODIES[name]
    for word in ('캐시백', '문턱', '쿠폰', '바우처'):
        assert word not in text
    for marker in ('zone:', 'workplace', 'daily_propensity', '[출력 형식]'):
        assert text.count(marker) == V45.count(marker)


@pytest.mark.parametrize('mod, attr, msg', [
    (v48, 'CLOSER_ANCHOR', '예시 배열의 끝'),
    (v49, 'DWELL_OLD', '체류 규칙 줄'),
])
def test_build_refuses_when_v45_moves_under_it(mod, attr, msg):
    original = getattr(mod, attr)
    try:
        setattr(mod, attr, '이 문장은 v45 에 없다')
        with pytest.raises(ValueError, match=msg):
            mod.build()
    finally:
        setattr(mod, attr, original)


@pytest.mark.parametrize('name', ['v48', 'v49'])
def test_they_are_selectable(name):
    import prompts
    assert name in prompts.list_variants()
