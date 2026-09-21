"""v42 는 원문이 실제로 틀린 세 곳만 고친다 — 강조를 더하지 않고 모순을 없앤다.

세 결함은 저장된 384개 응답에서 셌다(experiments/WHAT_v5_ACTUALLY_GETS_WRONG.md).
그 숫자가 이 파일의 근거이므로, 고침이 그 자리를 실제로 겨냥하는지 여기서 고정한다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v40, v42                      # noqa: E402
from prompts.candidates import make_system_prompt  # noqa: E402

V5 = make_system_prompt('v5')
V40 = v40.SYSTEM_PROMPT
V42 = v42.SYSTEM_PROMPT


def test_it_builds_on_v40_not_on_v5():
    """오염 제거와 형식 고침을 따로 읽으려면 v42 가 v40 위에 서야 한다."""
    assert len(V42) > len(V40)
    for old, _ in v42.REWRITES:
        assert old in V40


@pytest.mark.parametrize('word', ['캐시백', '문턱', 'grant_kept_share'])
def test_the_contamination_stays_removed(word):
    assert word not in V42


@pytest.mark.parametrize('marker', ['zone:', 'trigger', 'daily_propensity', '[출력 형식]'])
def test_format_scaffolding_is_not_disturbed(marker):
    assert V42.count(marker) == V40.count(marker)


def test_the_event_examples_are_untouched():
    """v10 이 없애서 7.3% 로 무너진 자리다."""
    assert len(re.findall(r'\{"time":', V42)) == len(re.findall(r'\{"time":', V40)) == 14


def test_the_meal_contradiction_is_gone():
    """31건의 원인 — '식사 = 끼니' 와 '집에서 먹으면 집' 이 같은 글에 있었다."""
    assert '**식사** = 식당·한식·양식·중식·일식·분식 등 끼니.' not in V42
    assert '식당에서 사 먹는' in V42
    assert '가게에서 돈을 쓰는 업종' in V42


def test_the_missing_commute_case_is_named():
    """24건 중 17건이 출근·퇴근이었다. 어휘에 이동이 없어 모델이 지어냈다."""
    assert '이동 자체는 이벤트가 아니다' in V42
    assert "출근은 anchor='workplace'" in V42


def test_the_field_name_is_pinned_including_the_last_event():
    """23건 전부 'reason' 오타였고, 전부 하루 마지막 쪽 residence 이벤트였다."""
    assert '`reason`·`reasonning` 은 무효' in V42
    assert '마지막 이벤트' in V42


def test_it_adds_only_the_three_declared_fixes():
    """되돌리면 v40 이 바이트 단위로 나와야 한다."""
    restored = V42
    for old, new in v42.REWRITES:
        assert new in restored, new[:40]
        restored = restored.replace(new, old, 1)
    assert restored == V40


def test_no_policy_direction_or_number_enters():
    """세 고침은 형식이다. 어떤 정책의 결과도 가리키지 않는다."""
    added = ''.join(new for _, new in v42.REWRITES)
    for word in ('늘려', '줄여', '더 쓰', '덜 쓰', '증가', '감소', '%'):
        assert word not in added, word


def test_build_refuses_when_v40_moves_under_it():
    original = v42.REWRITES
    try:
        v42.REWRITES = (('이 문장은 v40 에 없다', '무엇이든'),)
        with pytest.raises(ValueError, match='찾지 못했다'):
            v42.build()
    finally:
        v42.REWRITES = original


def test_it_is_selectable_by_name():
    import prompts
    assert 'v42' in prompts.list_variants()
    assert prompts.get('v42').SYSTEM_PROMPT == V42


def test_it_carries_the_shared_dawn_blocks():
    assert hasattr(v42, 'format_dawn_blocks')


def test_the_three_defects_are_still_forbidden_by_the_old_rules_too():
    """모순을 없앴다고 해서 원래 규칙을 지운 것은 아니다."""
    assert "anchor='residence'일 때 category='집'" in V42
    assert '절대 금지' in V42
