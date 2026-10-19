"""예시는 계약을 처음부터 끝까지 시연해야 한다 — 시작만 보여 주고 끝을 빼면 안 된다.

계약은 events[0] 과 events[-1] 이 모두 residence 이기를 요구하는데 v46 까지의 예시는
집에서 시작해 마트에서 끝났다. 모델은 마무리를 즉흥으로 지어내야 했고, v41 에서 남은
'reason' 오타 3건이 전부 그 자리(하루 마지막 쪽 residence 이벤트)에 있었다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v46, v47  # noqa: E402

V46 = v46.SYSTEM_PROMPT
V47 = v47.SYSTEM_PROMPT
EV = re.compile(r'\{"time":"(\d\d):(\d\d)","anchor":"([^"]+)"')


def events(text):
    a = text.index('{"events": [')
    b = text.index('\n ],\n "daily_propensity"', a)
    return EV.findall(text[a:b])


def test_v46_really_did_leave_the_day_open():
    """전제 확인 — 고칠 것이 있었다."""
    ev = events(V46)
    assert ev[0][2] == 'residence'
    assert ev[-1][2] != 'residence'


def test_the_example_now_starts_and_ends_at_home():
    ev = events(V47)
    assert ev[0][2] == 'residence'
    assert ev[-1][2] == 'residence'


def test_the_closing_event_demonstrates_the_things_that_go_wrong_there():
    """오타·앵커·업종이 전부 이 자리에서 났다. 한 이벤트가 셋을 동시에 시연한다."""
    assert '"time":"22:30","anchor":"residence","category":"집"' in V47
    assert '"reasoning":"장 본 것을 정리하고' in V47
    assert '"trigger":"none"}' in V47


def test_the_closing_event_obeys_the_spacing_rule_too():
    mins = [int(h) * 60 + int(m) for h, m, _ in events(V47)]
    assert all(b - a >= 20 for a, b in zip(mins, mins[1:]))
    assert mins == sorted(mins)


def test_only_one_event_was_added():
    assert len(events(V47)) == len(events(V46)) + 1


def test_no_existing_event_changed():
    for t, m, a in events(V46):
        assert '{"time":"%s:%s","anchor":"%s"' % (t, m, a) in V47


def test_build_checks_its_own_output():
    """예시가 계약을 어기면 조용히 내보내지 말고 멈춘다."""
    original = v47.CLOSER_NEW
    try:
        v47.CLOSER_NEW = ('  ...,\n  {"time":"19:25","anchor":"residence","category":"집",'
                          '"intent":"x","reasoning":"y","trigger":"none"}\n'
                          ' ],\n "daily_propensity": 0.72}')
        with pytest.raises(ValueError, match='20분 미만'):
            v47.build()
    finally:
        v47.CLOSER_NEW = original


def test_build_refuses_when_v46_moves_under_it():
    original = v47.CLOSER_ANCHOR
    try:
        v47.CLOSER_ANCHOR = '이 문장은 v46 에 없다'
        with pytest.raises(ValueError, match='예시 배열의 끝'):
            v47.build()
    finally:
        v47.CLOSER_ANCHOR = original


@pytest.mark.parametrize('word', ['캐시백', '문턱', '쿠폰', '바우처'])
def test_policy_neutrality_is_kept(word):
    assert word not in V47


@pytest.mark.parametrize('marker', ['zone:', 'workplace', 'daily_propensity', '[출력 형식]'])
def test_format_scaffolding_is_untouched(marker):
    assert V47.count(marker) == V46.count(marker)


def test_it_is_selectable_by_name():
    import prompts
    assert 'v47' in prompts.list_variants()
    assert prompts.get('v47').SYSTEM_PROMPT == V47
