"""등록되지 않은 어떤 정책이 들어와도 받아 준다 — 그리고 거짓을 말하지 않는다.

기전마다 모듈을 만드는 구조는 새 정책이 올 때마다 코드를 요구했고, 그래서 두
자리에서 조용히 틀린 말이 나갔다.

    label()      "[interest_subsidy] 소상공인 이자지원"  ← 한글에 영문 식별자
    principle()  지갑 원칙이 떨어졌다 ← **없는 지갑을 사실처럼 말한다**

둘째가 더 나쁘다. 라벨은 어색할 뿐이지만 이쪽은 거짓을 주입한다.

범용 모듈은 **대체**지 추가가 아니다. 기존 기전에 얹으면 같은 사실이 두 번
들어가고(P012 에서 실제로 겹쳤다), P010 처럼 동결된 정책의 렌더가 달라진다.
"""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

import dawn_context as dc                              # noqa: E402
from mechanisms import generic, get, label, principle  # noqa: E402

UNKNOWN = ('interest_subsidy', 'tax_credit', 'rent_support', 'point_reward', 'coupon_book')

# 방향을 지시하는 말. 어떤 정책이든 프롬프트가 이것을 말하면 정답을 주는 것이다.
DIRECTION = ('늘리', '줄이', '더 쓰', '덜 쓰', '옮기', '증가', '감소', '아껴')


def fake(ptype, **kw):
    row = {'id': 'P999', 'name': '어떤 정책', 'type': ptype,
           'description': '조건은 이 문장이 말한다.',
           'effective_from': '2026-01-01', 'effective_until': '2026-06-30',
           'target_districts': ['서울특별시'], 'benefit_categories': []}
    row.update(kw)
    row['from_'] = row['effective_from']
    row['until_'] = row['effective_until']
    row['regions'] = row['target_districts']
    row['target_l1s'] = row['benefit_categories']
    return row


@pytest.mark.parametrize('ptype', UNKNOWN)
def test_an_unknown_policy_gets_a_module(ptype):
    """None 을 돌려주면 그 정책은 사실·개인 상태 줄을 하나도 못 받는다."""
    assert get(ptype) is generic


@pytest.mark.parametrize('ptype', UNKNOWN)
def test_an_unknown_policy_is_never_told_it_has_a_wallet(ptype):
    """이것이 이 파일이 있는 이유다 — 없는 지갑을 사실처럼 말하면 안 된다."""
    p = principle([ptype])
    assert '정책지갑' not in p, '%s 가 지갑 원칙을 받는다' % ptype
    assert '잔액' not in p


@pytest.mark.parametrize('ptype', UNKNOWN)
def test_the_generic_principle_gives_no_direction(ptype):
    p = principle([ptype])
    for w in DIRECTION:
        assert w not in p, '범용 원칙이 방향을 말한다: %r' % w


@pytest.mark.parametrize('ptype', UNKNOWN)
def test_no_english_identifier_reaches_the_model(ptype):
    txt = dc._format_policy_facts([fake(ptype)])
    assert ptype not in txt, '영문 기전 코드가 정책 블록에 남는다'


def test_it_carries_the_declared_limits():
    """한도는 뜻이 하나뿐이다 — 그것은 옮긴다."""
    row = fake('point_reward', cap_per_agent=30000, poi_restricted=True,
               eligible_marker='[포인트]')
    assert '1인 누적 한도 30,000원' in generic.facts(row)
    line = generic.status('P999', row, {}, {})
    assert '30,000원' in line and '[포인트]' in line


def test_it_does_not_guess_what_a_rate_means():
    """rate 0.2 가 깎는 것인지 돌려주는 것인지는 필드만 봐서 모른다.

    추측해 적으면 틀린 사실이 들어간다. 그 설명은 description 이 한다.
    """
    row = fake('mystery', benefit_rate=0.2, discount_rate=0.2)
    joined = ' '.join(generic.facts(row))
    assert '할인' not in joined and '환급' not in joined and '캐시백' not in joined


def test_it_never_invents_a_balance():
    """지갑이 없는 정책에 잔액을 적으면 없는 돈을 만들어 준다."""
    line = generic.status('P999', fake('mystery', cap_per_agent=10000), {}, {})
    assert '잔액' not in line and '지갑' not in line


# ── 범용 모듈이 기존 정책을 건드리지 않는다 ────────────────────────────────
LEGACY = ('grant', 'cashback', 'subsidy', 'voucher', 'discount')


@pytest.mark.parametrize('ptype', LEGACY)
def test_generic_does_not_attach_to_legacy_mechanisms(ptype):
    """dawn_context 의 검증된 분기가 이미 그린다. 얹으면 사실이 두 번 들어간다."""
    assert get(ptype) is None


def test_the_frozen_policy_renders_exactly_as_before():
    """P010 은 동결이다. 이 작업으로 사실 줄이 하나라도 늘면 안 된다."""
    p = json.load(io.open(ROOT / 'data/neo4j_load/policies/P010.json', encoding='utf-8'))
    row = dict(p)
    row['from_'] = p['effective_from']; row['until_'] = p['effective_until']
    row['regions'] = p['target_districts']; row['target_l1s'] = p['benefit_categories']
    extra = [l for l in dc._format_policy_facts([row]).split('\n') if l.startswith('  · ')]
    assert extra == [], 'P010 에 사실 줄이 생겼다: %s' % extra
    assert label('grant', p['name']) == '[지원금] 민생회복 소비쿠폰 1차'


def test_p012_did_not_gain_a_duplicate_line():
    """description 이 '1인 월 최대 10만원' 을 이미 말한다. 사실 줄에 또 넣지 않는다."""
    p = json.load(io.open(ROOT / 'data/neo4j_load/policies/P012.json', encoding='utf-8'))
    row = dict(p)
    row['from_'] = p['effective_from']; row['until_'] = p['effective_until']
    row['regions'] = p['target_districts']; row['target_l1s'] = p['benefit_categories']
    extra = [l for l in dc._format_policy_facts([row]).split('\n') if l.startswith('  · ')]
    assert extra == [], 'P012 에 중복 사실 줄이 생겼다: %s' % extra
