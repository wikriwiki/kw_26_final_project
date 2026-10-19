"""프롬프트는 자기가 요구하는 계약을 스스로 지켜야 한다.

v41 관문에서 v45 의 남은 실패 21건 중 10건이 time 이었고 전부 간격이었다.
원인 둘 다 자기모순이다 — 규칙이 검사되는 것과 다른 말을 하고, 예시가 그 규칙을 어긴다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v45, v46  # noqa: E402

V45 = v45.SYSTEM_PROMPT
V46 = v46.SYSTEM_PROMPT


def example_times(text):
    a = text.index('{"events": [')
    return [int(t[:2]) * 60 + int(t[3:])
            for t in re.findall(r'\{"time":"(\d\d:\d\d)"', text[a:])]


def test_the_v45_example_really_did_break_its_own_contract():
    """전제 확인 — 고칠 것이 있었다는 것부터 고정한다."""
    t = example_times(V45)
    assert any(b - a < 20 for a, b in zip(t, t[1:]))


def test_the_v46_example_obeys_the_contract_it_illustrates():
    t = example_times(V46)
    assert t == sorted(t), '예시 시각이 순증하지 않는다'
    assert all(b - a >= 20 for a, b in zip(t, t[1:])), '예시 안에 20분 미만 간격이 있다'


def test_sorting_kept_every_event():
    assert sorted(example_times(V45)) == example_times(V46)
    assert len(re.findall(r'\{"time":', V46)) == len(re.findall(r'\{"time":', V45)) == 14


def test_no_event_content_changed():
    """순서만 바꾼다. 한 이벤트의 글자는 그대로여야 한다."""
    for probe in ('"intent":"출근길 음료"', '"intent":"아이 신발 사기"',
                  '"intent":"수강료 결제"', '"sub_category":"약국"'):
        assert probe in V46 and probe in V45


def test_the_rule_says_what_is_actually_checked():
    """검사기는 시작 시각 사이의 간격을 잰다. '체류' 는 다른 것이다."""
    assert '다음 이벤트의 `time` 은 앞 이벤트의 `time` 보다' in V46
    assert '시각은 하루 동안 뒤로 가지 않는다' in V46


def test_the_old_dwell_sentence_is_kept_not_replaced():
    """체류 20분도 진짜 규칙이다. 지우는 것이 아니라 옆에 검사되는 말을 놓는다."""
    assert '이벤트 간 최소 체류 20분' in V46


def test_the_checker_and_the_prompt_now_agree():
    """검사기 원문을 읽어 같은 숫자를 쓰는지 본다."""
    src = (ROOT / 'scripts/sim/validate_prompt_v3.py').read_text(encoding='utf-8')
    assert 'minute - previous < 20' in src
    assert '최소 20분 뒤' in V46


@pytest.mark.parametrize('marker', ['zone:', 'anchor', 'trigger', 'residence',
                                    'workplace', 'daily_propensity', '[출력 형식]'])
def test_format_scaffolding_is_untouched(marker):
    assert V46.count(marker) == V45.count(marker)


@pytest.mark.parametrize('word', ['캐시백', '문턱', '쿠폰', '바우처'])
def test_policy_neutrality_is_kept(word):
    assert word not in V46


def test_build_refuses_when_v45_moves_under_it():
    original = v46.DWELL_OLD
    try:
        v46.DWELL_OLD = '이 문장은 v45 에 없다'
        with pytest.raises(ValueError, match='체류 규칙 줄'):
            v46.build()
    finally:
        v46.DWELL_OLD = original


def test_it_is_selectable_by_name():
    import prompts
    assert 'v46' in prompts.list_variants()
    assert prompts.get('v46').SYSTEM_PROMPT == V46


def test_it_only_added_one_rule_sentence():
    """빼는 후보다. 더한 문장은 검사되는 말 한 줄뿐이다."""
    assert len(V46) - len(V45) < 200
