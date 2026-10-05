"""v40 은 v5 에서 정책 특화만 걷어낸다 — 형식을 지탱하는 것은 한 글자도 건드리지 않는다.

v10 은 이 구분 없이 예시를 통째로 없앴고 원문 계약 통과율이 66.7% 에서 7.3% 로
무너졌다. 장소 위반 167건이 전부 `zone:` 접두사 누락이었다. 아래 검사들이 그 사고를
다시 내지 못하게 막는다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from prompts import v40                                   # noqa: E402
from prompts.candidates import make_system_prompt          # noqa: E402

V5 = make_system_prompt('v5')
V40 = v40.SYSTEM_PROMPT

# 캐시백 기전에만 있는 말. 다른 정책의 입력에 들어가면 오염이다.
# ('2분기 월평균' 은 v5 본문이 아니라 정책 JSON 본문에 있다. 여기 넣었다가 전제
#  검사에 걸렸다 — 프롬프트에 없는 말을 프롬프트에서 뺐다고 적으면 안 된다.)
CASHBACK_WORDS = ('캐시백', '적립 실적', '문턱', 'grant_kept_share')

# 형식을 지탱하는 것들. 개수가 하나라도 줄면 v10 의 사고가 재현될 수 있다.
FORMAT_MARKERS = ('zone:', 'anchor', 'residence', 'workplace', 'trigger',
                  'daily_propensity', '[출력 형식]', '[카테고리 어휘]',
                  '[anchor 규칙 — 매우 중요]', 'sub_category', 'reasoning')


@pytest.mark.parametrize('word', CASHBACK_WORDS)
def test_no_cashback_vocabulary_survives(word):
    assert word in V5, '전제가 깨졌다 — v5 에 %r 이 없다' % word
    assert word not in V40, '%r 이 v40 에 남아 있다' % word


@pytest.mark.parametrize('marker', FORMAT_MARKERS)
def test_every_format_marker_is_preserved_exactly(marker):
    assert V40.count(marker) == V5.count(marker), (
        '%r 개수가 달라졌다: v5 %d · v40 %d' % (marker, V5.count(marker), V40.count(marker)))


def test_the_event_examples_are_all_still_there():
    """v10 이 없앤 것이 이것이고, 그래서 anchor 형식이 무너졌다."""
    assert len(re.findall(r'\{"time":', V40)) == len(re.findall(r'\{"time":', V5)) == 14


def test_every_example_anchor_still_carries_its_prefix():
    """167건의 위반이 전부 `11680521` 처럼 접두사가 빠진 것이었다."""
    anchors = re.findall(r'"anchor":"([^"]+)"', V40)
    assert anchors
    for a in anchors:
        assert a in ('residence', 'workplace') or a.startswith('zone:'), a


def test_the_winning_propensity_block_is_untouched():
    """2차 선별에서 v5 가 1위였다. 이 후보는 그 블록을 고치지 않는다."""
    from prompts.candidates import ANCHOR_HEAD, ANCHOR_TAIL
    a = V5.index(ANCHOR_HEAD); b = V5.index(ANCHOR_TAIL, a) + len(ANCHOR_TAIL)
    assert V5[a:b] in V40


def test_the_field_is_gone_from_the_output_schema_too():
    """설명만 빼고 필드를 남기면 모델이 안내 없이 그 값을 지어낸다."""
    assert '"grant_kept_share"' not in V40
    assert '"daily_propensity"' in V40


def test_v40_differs_from_v5_only_where_declared():
    """표에 없는 글자는 v5 와 바이트 단위로 같다 — 되돌리면 v5 가 나와야 한다."""
    restored = V40
    for old, new in v40.REWRITES:
        assert new in restored, new[:40]
        restored = restored.replace(new, old)
    head = V5.index(v40.DROP_SECTION_HEAD)
    tail = V5.index(v40.DROP_SECTION_NEXT, head)
    section = V5[head:tail]
    restored = restored.replace(v40.DROP_SECTION_NEXT, section + v40.DROP_SECTION_NEXT, 1)
    assert restored == V5


def test_build_refuses_when_v5_moves_under_it():
    """v5 본문이 바뀌면 조용히 다른 것을 만들지 말고 멈춘다."""
    original = v40.REWRITES
    try:
        v40.REWRITES = (('이 문장은 v5 에 없다', '무엇이든'),)
        with pytest.raises(ValueError, match='찾지 못했다'):
            v40.build()
    finally:
        v40.REWRITES = original


def test_it_is_selectable_by_name():
    import prompts
    assert 'v40' in prompts.list_variants()
    assert prompts.get('v40').SYSTEM_PROMPT == V40


def test_it_carries_the_shared_dawn_blocks():
    assert hasattr(v40, 'format_dawn_blocks')


def test_the_neutral_example_still_shows_a_policy_trigger():
    """정책 유인을 연기하는 장면은 뺐지만, policy trigger 예시 자체는 남아야 한다."""
    assert '"trigger":"policy"' in V40


def test_it_got_shorter_not_longer():
    """문장을 더하는 후보가 아니다. 빼는 후보다."""
    assert len(V40) < len(V5)
