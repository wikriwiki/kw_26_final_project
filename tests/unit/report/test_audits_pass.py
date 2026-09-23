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


def test_the_p015_transcription_matches_what_was_registered():
    """P015 의 등록된 판정을 기계가 셀 수 있게 옮겨 적었다 — **숫자가 같아야 한다.**

    원문 블록이 사람이 읽는 형식이라 표에서 통째로 빠져 있었다. 옮겨 적으면서
    다시 채점하면 유리한 쪽으로 흐른다. 원문의 '3/6 … HO-3 은 무효라 실질 3/5'
    가 그대로 나오는지 못 박는다.
    """
    import io as _io
    import json as _json
    sc = _json.loads(_io.open(ROOT / 'data/experiments/scoring_table.json',
                              encoding='utf-8').read())
    blk = sc['SECTOR_VOUCHER_2020']
    h = blk['result_policy_2026_09_17_hits']
    hits = [k for k, v in h.items() if isinstance(v, dict) and v.get('hit') is True]
    miss = [k for k, v in h.items() if isinstance(v, dict) and v.get('hit') is False]
    null = [k for k, v in h.items() if isinstance(v, dict) and v.get('hit') is None]
    assert sorted(hits) == ['HO-1', 'HO-5', 'HO-6'], hits
    assert len(hits) == 3 and len(hits) + len(miss) == 5, (hits, miss)
    assert null == ['HO-3'], null
    # 원문이 그렇게 적어 두었는지도 같이 본다
    assert 'HO-1·HO-5·HO-6 적중' in blk['result_policy_2026_09_17']['단일 런 채점']
    assert '실질 3/5' in blk['result_policy_2026_09_17']['단일 런 채점']


def test_the_sign_scoreboard_counts_placebos_by_their_own_expectation():
    """위약을 '무반응이 정답' 으로 뭉뚱그리면 lookahead 검정이 뒤집힌다."""
    m = _load('sboard', 'scripts/report/sign_scoreboard.py')
    rc, out = _run(m, ['sign_scoreboard.py'])
    assert rc == 0, out
    pl1 = [r for r in out.splitlines() if 'PL-1' in r.split()]
    assert pl1 and '증가' in pl1[0], 'PL-1 은 반응해야 적중이다(기댓값 +)\n' + str(pl1)
    pt1 = [r for r in out.splitlines() if 'PT-1' in r.split()]
    assert pt1 and '무반응' in pt1[0], 'PT-1 은 무반응이 적중이다\n' + str(pt1)
    assert '위약' in out and '실제 정책' in out


def test_a_reading_taken_with_the_wrong_ruler_is_not_counted():
    """다른 정책의 자로 잰 값이 적중으로 세어지면 성적이 부풀려진다.

    P013 의 EM-2 가 그랬다 — 적격 판정이 상생소비지원금 기준이었다. 지우지 않고
    물음표로 남기되 **분모에서 뺀다.**
    """
    m = _load('sboard2', 'scripts/report/sign_scoreboard.py')
    assert ('EMERGENCY_2020', 'result_stage3_dow_2026_09_16', 'EM-2') in m.SUSPECT
    rc, out = _run(m, ['sign_scoreboard.py'])
    assert rc == 0, out
    row = [r for r in out.splitlines() if 'EM-2' in r.split()][0]
    assert '못 셈' in row, row
    assert ' O ' not in row and ' X ' not in row, '적중/빗나감 표식이 남아 있다: ' + row
    assert '못 센 것' in out, '왜 빠졌는지 표 아래에 적혀 있어야 한다'


def test_the_miss_classifier_separates_power_from_direction():
    """X 를 뭉뚱그리면 '프롬프트를 고쳐야겠다' 로 잘못 읽는다.

    v5 의 빗나감은 대부분 **점추정이 기대 방향인데 구간이 0 을 지나는** 것이다.
    그 자리에서 후보를 가르면 잡음을 프롬프트의 성질로 적게 된다.
    """
    m = _load('why', 'scripts/report/why_it_misses.py')
    rc, out = _run(m, ['why_it_misses.py'])
    assert rc == 0, out
    assert '부호가 반대' in out and '유의하지 않다' in out
    # 지금 상태를 못 박는다 — 바뀌면 이 시험이 먼저 말한다
    import re
    n_sign = int(re.search(r'부호가 반대\)\s+(\d+)개', out).group(1))
    n_power = int(re.search(r'유의하지 않다\)\s+(\d+)개', out).group(1))
    assert n_power > n_sign, (
        '표본 문제가 방향 문제보다 많아야 한다 — 지금 %d 대 %d' % (n_power, n_sign))
    assert n_sign <= 1, '방향이 틀린 자리가 늘었다: %d개' % n_sign


def test_the_rounds_page_marks_unreadable_cells():
    """런 이동보다 작은 차이를 **흐리게 죽이지 않으면** "가까워졌으니 이겼다" 로
    읽힌다. 이 저장소가 세 번 데인 자리다.
    """
    m = _load('rounds', 'scripts/report/build_rounds_page.py')
    rc, out = _run(m, ['build_rounds_page.py'])
    assert rc == 0, out
    import io as _io
    html = _io.open(ROOT / 'output/report/rounds.html', encoding='utf-8').read()
    assert 'class="row dead"' in html, '죽인 칸이 하나도 없다'
    assert html.count('class="row dead"') == 3, '라운드마다 한 칸씩 죽어야 한다'
    # 등록한 합격선과 결과가 그림 옆에 있어야 한다
    assert '등록한 합격선' in html and '결과' in html
    for rnd in m.ROUNDS:
        assert rnd['reg'] in html and rnd['got'] in html, rnd['name']


def test_the_rounds_page_never_claims_an_unmeasured_run_shift():
    """재 본 적 없는 지표에 런 이동을 적으면 없는 근거를 만드는 것이다."""
    m = _load('rounds2', 'scripts/report/build_rounds_page.py')
    assert set(m.RUN_SHIFT) == {'DS-1', 'DS-2', 'P012-1'}
    assert set(m.READABLE) == set(m.RUN_SHIFT)
    assert m.READABLE['DS-1'] is True
    assert m.READABLE['DS-2'] is False and m.READABLE['P012-1'] is False
