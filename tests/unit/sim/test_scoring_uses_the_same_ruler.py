"""채점이 **모델이 본 것과 같은 자**로 재는가.

P015 의 HO-3 이 무효가 된 이유가 이것이다 — 8대 소비쿠폰을 채점하면서 적격
판정을 **상생소비지원금 기준**으로 했다. 같은 결함이 P013(긴급재난지원금)에도
남아 있었다. 정책에 `eligibility` 명세가 없으면 채점기가 DB 백필값
(`p.sangsaeng_eligible`)을 그대로 썼는데, 그것은 다른 정책의 자다.

### 왜 규칙을 옮겨 적지 않았는가

범용 평가기(`eligibility.Rules`)는 exclude 모드에서 **업종코드가 있으면 세분류
제외를 건너뛴다**(코드가 있고 코드 제외목록에 없으면 그 자리에서 적격 확정).
그래프의 POI 95.5% 가 업종코드를 갖고 있으므로, `is_coupon_eligible` 의
세분류 제외(시계·귀금속·부동산·법무·회계세무)가 옮겨 적은 명세에서는 죽는다.

그래서 **같은 함수를 부른다.** 어긋날 자리가 없다.
"""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from coupon_eligibility import is_coupon_eligible   # noqa: E402
from eligibility import Rules                        # noqa: E402

POLDIR = ROOT / 'data/neo4j_load/policies'


def pol(pid):
    return json.load(io.open(POLDIR / ('%s.json' % pid), encoding='utf-8'))


# (상호명, 세분류, 업종코드, 쿠폰룰이 말하는 적격 여부)
CASES = [
    ('김밥천국 역삼점', '분식', 'I56194', True),
    ('이마트 성수점', '종합소매', 'G47111', False),
    ('이마트24 R성수점', '편의점', 'G47121', True),
    ('홈플러스 익스프레스 문래점', '슈퍼마켓', 'G47112', False),
    ('롯데백화점 본점', '기타상품', 'G47190', False),
    ('스타벅스 강남대로점', '카페', 'I56221', False),
    ('메가엠지씨커피 신림점', '카페', 'I56221', True),
    ('동네정육식당', '정육', 'G47214', True),
    ('금은방 순금나라', '시계·귀금속', 'G47420', False),   # 세분류 제외
    ('서울부동산공인중개사', '부동산', 'L68221', False),    # 세분류 제외
    ('황금성 단란주점', '일반주점', 'I56211', False),
    ('호프의전설', '호프', 'I56219', True),
    (None, '한식', 'I56111', True),
]


@pytest.mark.parametrize('name,sub,code,expect', CASES)
def test_the_coupon_rule_itself(name, sub, code, expect):
    assert is_coupon_eligible(name, sub)[0] is expect


def test_transcribing_the_rule_into_a_spec_would_have_diverged():
    """옮겨 적었다면 어디서 틀렸을지 보여 둔다 — 다시 시도하지 않도록."""
    spec = {'mode': 'exclude',
            'exclude': {'subs': {'luxury': ['시계·귀금속'],
                                 'nonconsumption': ['부동산', '법무', '회계·세무']},
                        'name_regex': '이마트(?!24)|홈플러스|백화점|스타벅스|단란주점'}}
    r = Rules(spec)
    # 상호명으로 걸리는 것은 옮겨 적어도 맞는다
    assert r.eligible('이마트 성수점', '종합소매', None, 'G47111')[0] is False
    # **세분류로만 걸리는 것은 업종코드가 있으면 통과해 버린다**
    assert r.eligible('금은방 순금나라', '시계·귀금속', None, 'G47420')[0] is True
    assert is_coupon_eligible('금은방 순금나라', '시계·귀금속')[0] is False
    # 코드가 없을 때만 세분류가 산다
    assert r.eligible('금은방 순금나라', '시계·귀금속', None, None)[0] is False


def test_the_scorer_picks_the_runtime_rule_for_restricted_wallets():
    """사용처 제한이 있고 명세가 없는 정책은 런타임 함수로 채점해야 한다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sp2', ROOT / 'scripts/sim/score_policy.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    rows = [{'pname': n, 'sub': s, 'l1': None, 'upjong_l3': c, 'elig': True}
            for n, s, c, _ in CASES]
    why = m.apply_policy_eligibility(rows, 'data/neo4j_load/policies/P013.json')
    assert 'is_coupon_eligible' in why, why
    for row, (n, s, c, expect) in zip(rows, CASES):
        assert row['elig'] is expect, (n, s, row['elig'], expect)


def test_p012_keeps_the_db_value():
    """상생소비지원금은 DB 백필값이 곧 그 정책의 자다. 건드리면 안 된다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sp3', ROOT / 'scripts/sim/score_policy.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert pol('P012').get('poi_restricted') is False
    rows = [{'pname': '이마트 성수점', 'sub': '종합소매', 'l1': None,
             'upjong_l3': 'G47111', 'elig': True}]
    why = m.apply_policy_eligibility(rows, 'data/neo4j_load/policies/P012.json')
    assert 'DB 백필값' in why, why
    assert rows[0]['elig'] is True, '상생 채점의 elig 를 덮어썼다'
