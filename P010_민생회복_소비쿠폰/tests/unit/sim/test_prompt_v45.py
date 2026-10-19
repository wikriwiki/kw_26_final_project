"""v45 의 당시 어휘 제거 기록; 남은 사용처·지갑 편향은 v53에서 고친다.

v40 이 캐시백만 걷어냈을 때 지갑형 어휘가 그대로 남아 있었다. 한쪽만 걷어내면
이 검사는 나열한 단어만 막았으며 실질 정책 중립성을 증명하지 못했다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v42, v45                      # noqa: E402
from prompts.candidates import make_system_prompt  # noqa: E402

V5 = make_system_prompt('v5')
V42 = v42.SYSTEM_PROMPT
V45 = v45.SYSTEM_PROMPT

# 기전마다 그 기전에만 있는 말. 어느 것도 공통 본문에 있으면 안 된다.
MECHANISM_WORDS = ('캐시백', '적립 실적', '문턱', 'grant_kept_share', '쿠폰', '바우처')


@pytest.mark.parametrize('word', MECHANISM_WORDS)
def test_no_mechanism_survives_in_the_shared_body(word):
    assert word not in V45, '%r 이 v45 에 남아 있다' % word


def test_the_asymmetry_is_gone():
    """v42 는 캐시백만 0 이고 지갑형은 남아 있었다. 그것이 이 후보의 이유다."""
    assert V42.count('캐시백') == 0
    assert V42.count('쿠폰') > 0 and V42.count('바우처') > 0
    assert V45.count('쿠폰') == 0 and V45.count('바우처') == 0


def test_a_persons_own_balance_is_not_a_mechanism_word():
    """'개인 잔액' 은 본인 돈이다. 이것까지 지우면 판단 근거를 잃는다."""
    assert '개인 잔액' in V45


@pytest.mark.parametrize('marker', ['zone:', 'anchor', 'trigger', 'residence',
                                    'workplace', '[출력 형식]', 'daily_propensity'])
def test_format_scaffolding_is_untouched(marker):
    assert V45.count(marker) == V42.count(marker)


def test_the_event_examples_are_all_still_there():
    assert len(re.findall(r'\{"time":', V45)) == len(re.findall(r'\{"time":', V42)) == 14


def test_the_citation_rule_points_at_computed_numbers_of_any_shape():
    """런타임은 지갑형이면 남은 금액, 캐시백이면 문턱까지 거리를 계산해 준다.
    '잔액' 이라고만 적으면 캐시백 쪽 숫자를 가리키지 못한다."""
    assert '정책 블록에 적힌 숫자와 조건을 그대로 인용' in V45
    assert '조건까지 남은 거리와 남은 일수' in V45
    assert '정책 사용 시 잔액·할인율·카테고리를 명시' not in V45


def test_the_marker_is_not_hardcoded():
    """eligible_marker 는 정책마다 다르다. 하드코딩하면 남의 표시를 쓴다."""
    assert '[쿠폰] 표시' not in V45
    assert '정책이 정해 둔 표시' in V45


def test_the_policy_trigger_still_exists():
    """기전 열거는 뺐지만 policy trigger 자체는 남아야 한다."""
    assert '"trigger":"policy"' in V45
    assert 'policy(오늘 활성인 정책 조건)' in V45


def test_it_adds_only_the_declared_rewrites():
    """되돌리면 v42 가 바이트 단위로 나와야 한다."""
    restored = V45
    for old, new in v45.REWRITES:
        assert new in restored, new[:40]
        restored = restored.replace(new, old, 1)
    assert restored == V42


def test_no_direction_or_number_enters():
    added = ''.join(new for _, new in v45.REWRITES)
    for word in ('늘려', '줄여', '더 쓰', '덜 쓰', '증가', '감소'):
        assert word not in added, word


def test_build_refuses_when_v42_moves_under_it():
    original = v45.REWRITES
    try:
        v45.REWRITES = (('이 문장은 v42 에 없다', '무엇이든'),)
        with pytest.raises(ValueError, match='찾지 못했다'):
            v45.build()
    finally:
        v45.REWRITES = original


def test_it_is_selectable_by_name():
    import prompts
    assert 'v45' in prompts.list_variants()
    assert prompts.get('v45').SYSTEM_PROMPT == V45


def test_the_ladder_is_strictly_nested():
    """v5 → v40 → v42 → v45. 각 단계가 앞 단계 위에 서야 효과를 가를 수 있다."""
    import prompts
    v40s = prompts.get('v40').SYSTEM_PROMPT
    assert len(v40s) < len(V5)          # 뺐다
    assert len(V42) > len(v40s)         # 형식 규칙을 더했다
    assert V45 != V42
