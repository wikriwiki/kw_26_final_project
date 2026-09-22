"""정책 본문은 모델에게 끝까지 가야 한다 — 문장 한가운데서 자르지 않는다.

이 검사가 없어서 두 저장소가 갈라졌다. 검증 쪽은 절단을 제거했는데 본런 쪽은
`[:280]` 이 남아 있었고, P012 본문 358자 중 뒤의 78자가 잘려 나가고 있었다.

    잘린 문장  "캐시백은 이번 달에 미리 주는 돈이 아니라 다음 달 "   ← 여기서 끊긴다
    잃은 사실  15일에 돌려받는다 · **3% 문턱을 못 넘기면 이번 달 혜택은 사라진다**
               돌려받은 캐시백은 나중에 어디서든 쓸 수 있다

가운데 것이 이 정책의 비대칭 유인이다. 기전(3% 초과분의 10%)은 앞 280자에 남지만,
**놓치면 전부 잃는다**는 것이 사라진다.
"""
from datetime import date
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

POLICY_DIR = ROOT / 'data/neo4j_load/policies'
POLICIES = sorted(p for p in POLICY_DIR.glob('P*.json'))


def row(path):
    p = json.loads(path.read_text(encoding='utf-8'))
    d = lambda s: date(*map(int, s.split('-')))  # noqa: E731
    return {'id': p['id'], 'name': p['name'], 'type': p['type'],
            'description': p.get('description'),
            'rate': p.get('benefit_rate'), 'cap': p.get('cap_per_agent'),
            'threshold_ratio': p.get('threshold_ratio'),
            'eligible_marker': p.get('eligible_marker'),
            'mech_params': '{}',
            'poi_restricted': p.get('poi_restricted'),
            'from_': d(p['effective_from']), 'until_': d(p['effective_until']),
            'effective_from': p['effective_from'], 'effective_until': p['effective_until'],
            'income_grants': {}, 'excluded_income': [], 'decile_grants': {},
            'excluded_deciles': [], 'grant_key': p.get('grant_key'),
            'regions': p.get('target_districts') or ['서울특별시'], 'region_codes': ['11'],
            'target_l1s': []}


def rendered(path):
    from dawn_context import _format_policy_facts
    return _format_policy_facts([row(path)])


def test_there_are_policies_to_check():
    assert POLICIES, '정책 파일을 찾지 못했다'


@pytest.mark.parametrize('path', POLICIES, ids=lambda p: p.stem)
def test_the_whole_description_reaches_the_model(path):
    p = json.loads(path.read_text(encoding='utf-8'))
    desc = ' '.join(str(p.get('description') or '').split())
    if not desc:
        pytest.skip('본문 없음')
    assert desc in rendered(path), (
        '%s 본문 %d자가 온전히 들어가지 않았다' % (path.stem, len(desc)))


@pytest.mark.parametrize('path', POLICIES, ids=lambda p: p.stem)
def test_no_line_ends_mid_sentence(path):
    """자른 자리는 대개 공백이나 조사로 끝난다. 문장 부호로 끝나야 한다."""
    p = json.loads(path.read_text(encoding='utf-8'))
    desc = ' '.join(str(p.get('description') or '').split())
    if not desc:
        pytest.skip('본문 없음')
    for line in rendered(path).split('\n'):
        if line.strip().startswith('배경:'):
            body = line.split('배경:', 1)[1].strip()
            assert body.endswith(('.', '다', '요', '음', '!', '?')), (
                '%s 배경 줄이 문장 중간에서 끝난다: …%s' % (path.stem, body[-24:]))


def test_the_forfeiture_rule_survives_for_the_cashback_policy():
    """이 한 문장이 P012 의 비대칭 유인이다. 여기가 잘리면 문턱을 쫓을 이유가 약해진다."""
    path = POLICY_DIR / 'P012.json'
    assert path.exists()
    out = rendered(path)
    assert '문턱을 넘기지 못하면' in out
    assert '혜택은 사라집니다' in out


def test_the_formatter_does_not_cut_at_a_fixed_length():
    """길이 상수로 자르면 정책이 길어질 때마다 조용히 사실을 잃는다."""
    src = (ROOT / 'scripts/sim/dawn_context.py').read_text(encoding='utf-8')
    head = src.index('def _format_policy_facts')
    tail = src.index('\ndef ', head + 10)
    body = src[head:tail]
    assert '[:280]' not in body, '정책 사실 쪽에 고정 길이 절단이 다시 들어왔다'


def test_a_long_description_is_not_shortened():
    """상한이 없다는 것을 길이로 직접 확인한다."""
    from dawn_context import _format_policy_facts
    base = row(POLICY_DIR / 'P012.json')
    base['description'] = '가' * 1200
    assert '가' * 1200 in _format_policy_facts([base])
