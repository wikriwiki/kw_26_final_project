"""두 전수 점검이 **빨간불 없이** 돌아가는가.

점검 스크립트는 사람이 생각날 때 돌리면 안 본 사이에 썩는다. 실제로 그랬다 —
지표 일곱 개가 조용히 채점 불가였고, 사용처 표시는 여드레 동안 꺼져 있었다.
둘 다 스크립트를 돌렸으면 바로 보였을 것이다.

    audit_indicator_coverage   채점기가 그 지표를 계산할 수 있는가
    audit_policy_delivery      정책이 모델에게 도달하는가

앞의 것만 보면 **채점은 되는데 정책이 안 닿는** 상태를 못 잡는다. 그 상태에서는
프롬프트를 아무리 고쳐도 지표가 안 움직인다.
"""
from pathlib import Path
import importlib.util
import io
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _run(mod, argv):
    """스크립트의 main 을 돌리고 (반환값, 출력) 을 준다."""
    old_argv, old_out = sys.argv, sys.stdout
    sys.argv = argv
    sys.stdout = buf = io.StringIO()
    try:
        rc = mod.main()
    finally:
        sys.argv, sys.stdout = old_argv, old_out
    return rc, buf.getvalue()


def test_every_indicator_is_scoreable():
    m = _load('aic', 'scripts/report/audit_indicator_coverage.py')
    rc, out = _run(m, ['audit_indicator_coverage.py', '--quiet'])
    assert rc == 0, '채점 불가 지표가 있다\n' + out


def test_every_policy_reaches_the_model():
    m = _load('apd', 'scripts/report/audit_policy_delivery.py')
    rc, out = _run(m, ['audit_policy_delivery.py'])
    assert rc == 0, '정책이 모델에게 닿지 않는다\n' + out


def test_the_delivery_audit_actually_catches_a_dead_policy():
    """점검이 무엇이든 통과시키면 점검이 아니다. 고장난 정책을 넣어 확인한다."""
    m = _load('apd2', 'scripts/report/audit_policy_delivery.py')
    from mechanisms import poi_restriction

    # 지갑 없는 사용처 제한 정책인데 표시가 안 켜지는 상황을 흉내낸다
    dead = {'id': 'PDEAD', 'type': 'sector_voucher', 'poi_restricted': False}
    ids, _, _ = poi_restriction([dead], {})
    assert ids == set()

    # 그리고 룰이 실재하지 않는 업종만 가리키는 경우
    n = m.eligible_sector_count(
        {'mode': 'include', 'include': {'subs': ['있지도않은업종']}}, {'청과', '정육'})
    assert n == 0, '없는 업종을 가리키는 룰을 0개로 세지 못한다'
    n = m.eligible_sector_count(
        {'mode': 'include', 'include': {'subs': ['청과', '있지도않은업종']}}, {'청과', '정육'})
    assert n == 1


def test_the_leak_check_does_not_flag_the_policys_own_terms():
    """제도의 내용을 누설로 신고하면 아무도 이 점검을 안 보게 된다."""
    m = _load('apd3', 'scripts/report/audit_policy_delivery.py')
    pol = {'sectors': {'마트': {'mode': 'rate', 'rate': 0.2},
                       '여행사': {'mode': 'rate', 'rate': 0.3}}}
    own = m.own_numbers(pol)
    assert '20%' in own and '30%' in own
    block = {'indicators': [{'desc': '마트 20% 할인 대상 — 실측 +6.957%'}]}
    got = m.answer_numbers(block, pol)
    assert '20%' not in got, '정책이 선언한 할인율을 누설로 잡는다'
    assert '6.957%' in got, '정답지 수치를 놓친다'
