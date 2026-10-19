"""v51·v52 는 그리스디에서 남은 결함 둘을 **따로** 겨냥한다.

temperature 0 에서 v45 가 남긴 23칸의 정체를 원문에서 확인했다.

    zone 9칸         anchor='zone:<code>' 에 category='직장' — 코드는 허용 목록 안이다
    explanation 7칸  저녁 집 이벤트에서 reasoning 키가 통째로 빠진다
    time 7칸         대부분 09:00 의 15분 간격 — v49 가 이미 건드렸고 재현되지 않았다

셋을 한 후보에 묶으면 어느 것이 먹혔는지 못 가린다. v45 의 교훈 ① 이 그것이다 —
넷을 고쳤는데 앞의 둘이 +95칸, 뒤의 둘이 −17칸이었다. 그래서 하나씩 간다.

이 검사들이 지키는 것은 **두 후보가 v45 에서 딱 한 자리만 다르다는 것**이다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v45, v51, v52  # noqa: E402

V45 = v45.SYSTEM_PROMPT
BODIES = {'v51': v51.SYSTEM_PROMPT, 'v52': v52.SYSTEM_PROMPT}

# 기전 어휘는 v45 에서 0 이 됐다. 후보가 그것을 되살리면 안 된다.
MECHANISM_WORDS = ('캐시백', '적립 실적', '문턱', 'grant_kept_share', '쿠폰', '바우처')

# 정답지의 방향·크기를 본문에 넣는 것이 금지의 핵심이다. 형식 규칙은 사양이지만
# 결과값은 사양이 아니다.
ANSWER_KEY_LEAKS = ('14.1', '4.2%', '20.82', '11.1%p', '47,880', '0.21',
                    '줄어든다', '늘어난다')


@pytest.mark.parametrize('name', sorted(BODIES))
@pytest.mark.parametrize('word', MECHANISM_WORDS)
def test_no_mechanism_word_comes_back(name, word):
    assert word not in BODIES[name], '%s 에 %r 이 되살아났다' % (name, word)


@pytest.mark.parametrize('name', sorted(BODIES))
@pytest.mark.parametrize('leak', ANSWER_KEY_LEAKS)
def test_no_answer_key_value_enters_the_prompt(name, leak):
    """실측값도, 방향을 지시하는 말도 모델에게 주지 않는다."""
    assert leak not in BODIES[name], '%s 에 정답지가 새어 들어갔다: %r' % (name, leak)


@pytest.mark.parametrize('name', sorted(BODIES))
def test_each_candidate_changes_exactly_one_place(name):
    """한 자리만 다르다 — 그래야 결과를 그 자리에 귀속할 수 있다."""
    import difflib
    changed = [l for l in difflib.unified_diff(V45.split('\n'),
                                               BODIES[name].split('\n'), lineterm='', n=0)
               if l[:1] in '+-' and l[:3] not in ('+++', '---')]
    # v51 은 두 줄을 덧붙이고, v52 는 한 줄을 고쳐 쓴다. 어느 쪽도 4줄을 넘지 않는다.
    assert 0 < len(changed) <= 4, '%s 의 변경이 한 자리를 넘는다: %d줄' % (name, len(changed))


@pytest.mark.parametrize('name', sorted(BODIES))
@pytest.mark.parametrize('marker', ['zone:', 'trigger', 'residence',
                                    '[출력 형식]', 'daily_propensity'])
def test_format_scaffolding_is_untouched(name, marker):
    """v10 이 예시를 없애 66.7% → 7.3% 로 무너진 자리다. 뼈대는 건드리지 않는다."""
    assert BODIES[name].count(marker) == V45.count(marker)


def test_only_v51_says_anchor_one_more_time():
    """v51 의 새 줄은 anchor='workplace' 를 한 번 쓴다 — 그것이 이 후보의 전부다.

    'anchor' 를 뼈대 검사에서 빼는 대신, 늘어난 양을 정확히 못 박는다.
    한 번을 넘으면 규칙을 더한 것이고, 그것은 이 라운드의 설계가 아니다.
    """
    assert BODIES['v51'].count('anchor') == V45.count('anchor') + 1
    assert BODIES['v52'].count('anchor') == V45.count('anchor')


@pytest.mark.parametrize('name', sorted(BODIES))
def test_the_event_examples_are_all_still_there(name):
    assert len(re.findall(r'\{"time":', BODIES[name])) == 14


def test_v51_draws_the_boundary_where_the_attractor_is():
    """'직장 동' 과 'zone 코드' 를 한 줄에서 묶던 자리에 경계를 긋는다."""
    assert "근무·회의·출퇴근 자체는" in BODIES['v51']
    assert "anchor='workplace' 이고, 동 코드로 적지 않는다" in BODIES['v51']
    # 규칙을 새로 더하는 것이 아니라 기존 줄에 붙인다.
    assert BODIES['v51'].count('직장 동 근처 외출') == V45.count('직장 동 근처 외출') == 1


def test_v51_does_not_touch_the_reasoning_rule():
    """v52 의 자리를 건드리면 두 후보의 판정이 섞인다."""
    assert '하루의 마지막 이벤트(귀가·취침)까지 똑같이 쓴다' in BODIES['v51']


def test_v52_widens_the_reasoning_requirement_past_the_last_event():
    """빠진 것은 마지막 이벤트가 아니라 그 앞의 '취침 준비' 였다."""
    assert '취침 준비' in BODIES['v52']
    assert '하루의 마지막 이벤트(귀가·취침)까지 똑같이 쓴다' not in BODIES['v52']
    assert '**모든 이벤트는 time · reasoning · trigger 필드를 반드시 포함**' in BODIES['v52']


def test_v52_does_not_touch_the_anchor_rules():
    assert '근무·회의·출퇴근 자체는' not in BODIES['v52']


def test_neither_candidate_touches_the_time_rule():
    """time 7칸은 v49 가 이미 건드렸고 재현되지 않았다. 이번 라운드에서는 손대지 않는다."""
    for name in BODIES:
        assert BODIES[name].count('이벤트 간 최소 체류 20분') == V45.count('이벤트 간 최소 체류 20분') == 1


def test_the_frozen_bodies_did_not_move():
    """v5 는 대조군이고 p010 은 동결이다. 이 라운드가 그것을 건드리면 안 된다."""
    import hashlib

    from prompts.candidates import make_system_prompt
    assert hashlib.sha256(make_system_prompt('v5').encode()).hexdigest()[:16] == '250311b63adbc4f6'
    assert hashlib.sha256(V45.encode()).hexdigest()[:16] == 'd36a1763095459b2'
