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


def test_the_steerability_map_agrees_with_the_registered_verdicts():
    """조종 가능성 표가 **이미 등록된 판정과 어긋나지 않는지** 본다.

    표가 "여기서 프롬프트가 읽힌다" 라고 한 자리에서 라운드는 "못 읽는다" 고
    판정했다면 둘 중 하나가 틀린 것이다. 문턱을 만지작거리다 등록된 판정과
    멀어지는 것을 막는다.

        DS-1    v50 이 읽었다        후보차 0.9%p > 런 이동 0.5%p
        DS-2    v50 이 안 읽었다     후보차 7.4%p < 런 이동 13.1%p
        P012-1  라운드2 가 안 읽었다  후보차 1.3%p < 런 이동 8.2%p
    """
    m = _load('smap', 'scripts/report/steerability_map.py')
    rc, out = _run(m, ['steerability_map.py'])
    assert rc == 0, out
    def row(iid):
        # 정책 이름에 공백이 있어 열 번호로 못 집는다 — 토큰으로 찾는다
        hits = [r for r in out.splitlines()
                if iid in r.split() and not r.startswith('*')]   # 각주는 뺀다
        assert len(hits) == 1, '%s 줄을 하나로 못 집었다: %d개' % (iid, len(hits))
        return hits[0]

    line = {k: row(k) for k in ('DS-1', 'DS-2', 'P012-1')}
    assert '읽힌다' in line['DS-1'], 'DS-1 은 v50 이 읽은 자리다\n' + line['DS-1']
    assert '막는다' in line['DS-2'], 'DS-2 는 런 이동에 묻혔다\n' + line['DS-2']
    assert '막는다' in line['P012-1'], 'P012-1 은 라운드2 가 못 읽었다\n' + line['P012-1']
    # 도달 관문이 열려 있어야 나머지 판정이 의미가 있다
    assert '전 정책 통과' in out
