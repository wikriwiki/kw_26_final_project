"""정책 블록의 기전 라벨 — 한글 문장에 영문 코드가 섞이면 안 된다.

`_LEGACY_LABEL` 에 키가 없으면 `label()` 이 타입 문자열을 그대로 돌려준다.
그래서 모델이 이런 줄을 읽고 있었다.

    - P014 [price_discount] 서울사랑상품권 | 2020-09-21~2020-10-11 | …

P012 는 `[캐시백]`, P013 은 `[지원금]` 인데 P014·P015 만 영문이었다. 하필
부호 적중이 가장 낮은 정책들이다(P014 는 세 라운드에서 1 → 0 → 1).

라벨은 **제도가 무엇인지만** 말한다. 무엇을 하라는 말은 넣지 않는다 —
그것을 넣으면 정답을 프롬프트에 주는 것이고, 순환 검증이 된다.
"""
from pathlib import Path
import re
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from mechanisms import label  # noqa: E402

# 저장소에 실제로 실린 정책이 쓰는 기전 전부
USED_TYPES = ('grant', 'cashback', 'price_discount', 'sector_voucher', 'facility')

# 환경이 실어 오는 규제 기전
ENV_TYPES = ('hours_limit', 'gathering_limit')

# 라벨에 들어가면 안 되는 말 — 행동의 방향을 지시하는 어휘
DIRECTION_WORDS = ('늘리', '줄이', '더 쓰', '덜 쓰', '옮기', '이동', '증가', '감소')


@pytest.mark.parametrize('ptype', USED_TYPES + ENV_TYPES)
def test_label_is_korean_not_a_raw_code(ptype):
    """영문 식별자가 한글 문장 한가운데로 새지 않는다."""
    got = label(ptype, '정책이름')
    inside = got[got.find('[') + 1:got.find(']')]
    assert inside != ptype, '%s 의 라벨이 없어 영문 코드가 그대로 나간다: %r' % (ptype, got)
    assert not re.search(r'[A-Za-z_]', inside), '라벨에 영문이 남아 있다: %r' % got


@pytest.mark.parametrize('ptype', USED_TYPES + ENV_TYPES)
def test_label_does_not_tell_the_agent_what_to_do(ptype):
    """라벨은 제도의 이름이지 행동 지시가 아니다."""
    got = label(ptype, '정책이름')
    for w in DIRECTION_WORDS:
        assert w not in got, '%s 의 라벨이 방향을 지시한다(%r): %r' % (ptype, w, got)


def test_frozen_policies_render_exactly_as_before():
    """P010 은 동결이다. grant·cashback 라벨이 바뀌면 그 렌더가 달라진다."""
    assert label('grant', '민생회복 소비쿠폰 1차') == '[지원금] 민생회복 소비쿠폰 1차'
    assert label('cashback', '상생소비지원금') == '[캐시백] 상생소비지원금'


def test_every_policy_in_the_repo_has_a_label():
    """새 정책을 넣고 라벨을 빼먹으면 여기서 잡는다."""
    import io
    import json
    bad = []
    for p in sorted((ROOT / 'data/neo4j_load/policies').glob('*.json')):
        d = json.load(io.open(p, encoding='utf-8'))
        t = d.get('type')
        if not t:
            continue
        got = label(t, d.get('name') or '')
        if t in got:
            bad.append('%s (type=%s)' % (p.name, t))
    assert not bad, '라벨 없는 정책: %s' % ', '.join(bad)
