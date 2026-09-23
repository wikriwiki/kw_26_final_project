"""캐시백 문턱의 **범위**를 계산된 사실로 알려 주는 한 줄 (EXP_SCOPE_FACT).

### 왜 만들었나

`why_it_misses.py` 로 갈라 보니 v5 가 방향을 틀린 자리는 구조적으로 불가능한
하나뿐이고, 남은 진짜 결함은 **동등성 밴드를 넘은 둘**이었다.

    P012-2  제외업종 무반응이 기대인데 -13.9%   밴드 251% 초과
    PL-2    위약에서 대상 아닌 업종 -4.4%       같은 모양

둘 다 같은 모양이다 — **한 업종이 오르면 다른 업종이 내려간다.** 문턱은
적립업종 지출만 세므로(`spent_elig`), 제외업종에서 줄여도 문턱은 한 푼도
가까워지지 않는다. 제도의 정의에서 바로 나오는 사실인데 모델이 스스로
세우지 못하는 것으로 보인다.

### 왜 서술이 아니라 계산인가

P010 역진의 벽에서 확인했다 — **서술 프롬프트는 무효였고 모델은 계산된
사실에만 반응했다.** 그래서 "제외업종을 줄이지 마라" 같은 지시가 아니라
문턱 산식의 성질을 적는다.

### 무엇을 하면 안 되는가

    · 방향 지시 금지      "제외업종을 늘려라" 는 정답을 주는 것이다
    · 정답지 수치 금지    실측 +2.85% 같은 값은 들어가면 안 된다
    · 측정 공식 금지      순효과·밴드·MPC 산식을 입력으로 쓰면 순환이다
"""
from pathlib import Path
import os
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

import dawn_context as dc   # noqa: E402

ROW = {'threshold_ratio': 1.03, 'rate': 0.10, 'cap': 100000}
PERSONA = {'daily_wd': 40000, 'daily_we': 55000}
STATE = {'sangsaeng_month_spent': 900000}


def render(flag):
    old = os.environ.get('EXP_SCOPE_FACT')
    os.environ['EXP_SCOPE_FACT'] = flag
    try:
        from datetime import date
        return dc._format_cashback_status('P012', ROW, PERSONA, STATE, date(2021, 10, 21))
    finally:
        if old is None:
            os.environ.pop('EXP_SCOPE_FACT', None)
        else:
            os.environ['EXP_SCOPE_FACT'] = old


def test_off_by_default_so_past_runs_are_reproducible():
    """기본은 꺼짐이다. 켜지 않은 런은 한 글자도 달라지면 안 된다."""
    assert '문턱은 적립업종 지출만 센다' not in render('0')


def test_on_adds_exactly_one_arithmetic_fact():
    t = render('1')
    assert '문턱은 적립업종 지출만 센다' in t
    assert '제외업종' in t and '가까워지지 않' in t


def test_it_only_adds_and_changes_nothing_else():
    """있던 사실이 바뀌면 이 후보가 무엇을 한 건지 알 수 없게 된다."""
    off, on = render('0'), render('1')
    head_off = off.split(' | ')
    head_on = on.split(' | ')
    assert len(head_on) == len(head_off) + 1, (head_off, head_on)
    # 새로 붙은 칸 하나를 빼면 나머지는 바이트가 같아야 한다
    added = [x for x in head_on if x not in head_off]
    assert len(added) == 1, added
    assert [x for x in head_on if x != added[0]] == head_off


@pytest.mark.parametrize('bad', [
    '2.85', '+2.85%', '20.82', '늘려', '늘려라', '줄이지 마', '쓰라', '써라',
    '많이', '더 써', '밴드', '순효과', 'MPC',
])
def test_it_does_not_leak_a_direction_or_an_answer(bad):
    assert bad not in render('1'), '정답이나 방향이 샌다: %r' % bad


def test_the_claim_is_true_of_the_mechanism():
    """적는 사실이 실제로 맞는가 — 문턱이 적립업종만 세는가.

    맞지 않으면 이 줄은 거짓을 주입하는 것이다. 렌더러가 문턱까지 남은 거리를
    `sangsaeng_month_spent`(적립업종 한정 누적)로만 계산하는지 본다.
    """
    st_more_elig = dict(STATE, sangsaeng_month_spent=STATE['sangsaeng_month_spent'] + 200000)
    from datetime import date
    a = dc._format_cashback_status('P012', ROW, PERSONA, STATE, date(2021, 10, 21))
    b = dc._format_cashback_status('P012', ROW, PERSONA, st_more_elig, date(2021, 10, 21))
    assert a != b, '적립업종 누적이 늘었는데 문턱 표시가 그대로다'
    # 적립업종과 무관한 키를 넣어도 문턱 표시는 그대로여야 한다
    st_noise = dict(STATE, month_spent_total=99999999, excluded_spent=500000)
    c = dc._format_cashback_status('P012', ROW, PERSONA, st_noise, date(2021, 10, 21))
    assert c == a, '적립업종이 아닌 지출이 문턱 계산에 섞여 들어간다'


# ---------------------------------------------------------- 업종 한정 할인권 쪽
def _sv(flag, pid='P090'):
    import json as _json
    from mechanisms import sector_voucher
    old = os.environ.get('EXP_SCOPE_FACT')
    os.environ['EXP_SCOPE_FACT'] = flag
    try:
        row = _json.load(open(ROOT / ('data/neo4j_load/policies/%s.json' % pid),
                              encoding='utf-8'))
        return sector_voucher.status(pid, row, {}, {})
    finally:
        if old is None:
            os.environ.pop('EXP_SCOPE_FACT', None)
        else:
            os.environ['EXP_SCOPE_FACT'] = old


def test_sector_voucher_is_unchanged_by_default():
    assert '혜택은 위 업종에서만' not in _sv('0')


def test_sector_voucher_gets_the_same_arithmetic_fact():
    """같은 가설이므로 **같은 스위치**로 켜진다. 기전마다 다른 플래그면 A/B 가 흐려진다."""
    t = _sv('1')
    assert '혜택은 위 업종에서만' in t and '늘지 않' in t
    assert _sv('1').startswith(_sv('0')), '있던 사실이 바뀌었다'


@pytest.mark.parametrize('pid', ['P090', 'P015', 'P016'])
@pytest.mark.parametrize('bad', ['늘려', '써라', '더 써', '많이', '+25.4', '11.6', '6.957'])
def test_sector_voucher_leaks_neither_direction_nor_answer(pid, bad):
    assert bad not in _sv('1', pid)


def test_one_switch_covers_both_mechanisms():
    """캐시백과 업종한정이 같은 가설을 시험한다 — 스위치가 갈리면 뭘 쟀는지 흐려진다."""
    on_cash, off_cash = render('1'), render('0')
    on_sv, off_sv = _sv('1'), _sv('0')
    assert on_cash != off_cash and on_sv != off_sv
    assert render('0') == off_cash and _sv('0') == off_sv


# ------------------------------------------------- 탐침이 본런과 같은 문구를 쓰는가
def test_the_probe_inserts_exactly_what_the_code_renders():
    """탐침의 문구가 코드와 다르면 **다른 것을 재고 같은 것이라 적게 된다.**

    탐침은 서버의 동결 맥락(텍스트)만 쓰므로 문구를 복사해 둔다. 그 복사가
    코드와 어긋나지 않는지 여기서 못 박는다.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'probe', ROOT / 'scripts/sim/scope_fact_probe.py')
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    rendered = render('1')
    assert pr.SCOPE_LINE in rendered, (
        '탐침이 끼우는 문구가 코드 렌더에 없다\n탐침: %s\n렌더: %s'
        % (pr.SCOPE_LINE, rendered))
    assert pr.TAIL in render('0'), '끼울 자리 표식이 렌더에 없다'


def test_the_probe_only_adds_the_one_line():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'probe2', ROOT / 'scripts/sim/scope_fact_probe.py')
    pr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pr)
    base = '- P012: 적립업종 이번달 누적 1원 | 못 넘기면 이번 달 혜택은 사라짐'
    got = pr.add_scope(base)
    assert got.count('못 넘기면') == 1
    assert pr.SCOPE_LINE in got
    assert pr.add_scope(got) == got, '두 번 넣으면 안 된다'
    assert pr.add_scope('문턱 줄이 없는 맥락') == '문턱 줄이 없는 맥락'


# ----------------------------------------------- 후보 레지스트리가 코드와 어긋나지 않는가
def _probe_mod():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'probe_reg', ROOT / 'scripts/sim/scope_fact_probe.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_every_registered_candidate_matches_what_the_code_renders():
    """후보를 자료로 적었으니 **자료가 코드와 어긋나는지** 기계가 본다.

    후보가 늘어날 때마다 사람이 대조하면 언젠가 빠뜨린다. 그러면 탐침이 A 를
    끼우고 본런이 B 를 끼운 채 같은 실험이라고 적게 된다.
    """
    pr = _probe_mod()
    import os as _os
    from datetime import date as _date
    from environments import build_environment as _env
    _old = _os.environ.get('EXP_CASE_TREND')
    _os.environ['EXP_CASE_TREND'] = '1'
    try:
        env_txt = chr(10).join(
            _env('covid_2021', _date(2020, 11, 24)).get('facts') or [])
    finally:
        if _old is None:
            _os.environ.pop('EXP_CASE_TREND', None)
        else:
            _os.environ['EXP_CASE_TREND'] = _old
    renders = {'cashback': render('1'), 'sector_voucher': _sv('1'),
               'environment': env_txt}
    for name, spec in pr.CANDIDATES.items():
        if spec.get('is_null'):
            # 영가설 줄은 코드 렌더에 **없어야** 한다 — 있으면 진짜 후보가 된다
            assert not any(spec['line'] in t for t in renders.values()),                 '영가설 줄이 렌더에 들어가 있다: %s' % name
            continue
        for key in ('line', 'tail', 'case', 'why'):
            assert spec.get(key), '%s 후보에 %s 가 없다' % (name, key)
        hit = [k for k, t in renders.items() if spec['line'] in t]
        assert hit, ('후보 %r 의 문장이 어느 렌더에도 없다 — 탐침과 본런이 다른 것을 잰다\n%s'
                     % (name, spec['line']))


def test_the_insert_is_inert_when_there_is_no_place_for_it():
    """억지로 끼우면 그 줄이 아니라 **어색한 글**에 대한 반응을 재게 된다."""
    pr = _probe_mod()
    assert pr.insert('표식이 없는 맥락', 'X', '없는표식') == '표식이 없는 맥락'
    once = pr.insert('a | 끝', 'X', '끝')
    assert once == 'a | X | 끝'
    assert pr.insert(once, 'X', '끝') == once, '두 번 끼우면 안 된다'


def test_bullet_after_inserts_into_the_environment_block():
    """환경 블록은 `- ` 줄 목록이라 파이프 방식으로는 못 끼운다(후보 2 용)."""
    pr = _probe_mod()
    ctx = ('## 사회 배경 — 오늘의 세상\n'
           '감염병 유행 중인 서울\n'
           '- 서울 신규 확진 112명 (2020-11-23 기준, 최근 7일 평균 110명)\n'
           '- 식당 매장 취식은 21시까지, 이후 포장·배달만 가능\n'
           '\n## 페르소나\n')
    got = pr.insert(ctx, '2주 전 7일 평균은 42명 — 지금은 그 2.6배',
                    '최근 7일 평균', mode='bullet_after')
    lines = got.split('\n')
    i = [k for k, l in enumerate(lines) if '최근 7일 평균' in l][0]
    assert lines[i + 1] == '- 2주 전 7일 평균은 42명 — 지금은 그 2.6배'
    assert lines[i + 2].startswith('- 식당'), '뒤 항목이 밀려나면 안 된다'
    assert pr.insert(got, '2주 전 7일 평균은 42명 — 지금은 그 2.6배',
                     '최근 7일 평균', mode='bullet_after') == got


def test_bullet_after_is_inert_without_a_marker_line():
    pr = _probe_mod()
    assert pr.insert('표식 없음', 'X', '없는표식', mode='bullet_after') == '표식 없음'
    # 표식이 있어도 `- ` 줄이 아니면 넣지 않는다 — 머리말에 끼우면 글이 망가진다
    assert pr.insert('최근 7일 평균 어쩌고', 'X', '최근 7일 평균',
                     mode='bullet_after') == '최근 7일 평균 어쩌고'


def test_an_unknown_insert_mode_fails_loudly():
    """조용히 그대로 두면 **안 끼운 채로 A/B 를 돌리게 된다.**"""
    pr = _probe_mod()
    with pytest.raises(ValueError):
        pr.insert('a | 끝', 'X', '끝', mode='엉뚱한방식')


# ------------------------------------------------- 판정은 `changed` 가 아니라 부호 검정
def test_the_sign_test_is_not_fooled_by_one_big_mover():
    """평균만 보면 한쪽으로 크게 튄 한 사람이 전체를 끌고 간다."""
    pr = _probe_mod()
    pairs = {('a%d' % i, 1): {'off': {'x': 0}, 'on': {'x': -1}} for i in range(5)}
    pairs[('big', 1)] = {'off': {'x': 0}, 'on': {'x': 100}}   # 한 명이 크게 늘었다
    t = pr.sign_test(pairs, 'x')
    assert t['mean'] > 0, '평균은 한 사람에 끌려 양수다'
    assert t['hit'] == 1 and t['miss'] == 5, '부호는 다섯 대 하나로 반대쪽이다'


def test_ties_are_reported_not_hidden():
    """동점을 분모에 넣으면 p 가 실제보다 좋아 보인다. 빼되 **몇인지 알린다.**"""
    pr = _probe_mod()
    pairs = {('t%d' % i, 1): {'off': {'x': 0}, 'on': {'x': 0}} for i in range(20)}
    pairs[('u', 1)] = {'off': {'x': 0}, 'on': {'x': 1}}
    t = pr.sign_test(pairs, 'x')
    assert t['tie'] == 20 and t['n'] == 1
    assert t['p'] == 1.0, '한 쌍으로는 아무것도 못 가른다'


def test_a_clean_split_is_detected():
    """전부 한쪽이면 갈려야 한다 — 검정이 늘 '구별 안 됨' 을 내면 쓸모가 없다."""
    pr = _probe_mod()
    pairs = {('a%d' % i, 1): {'off': {'x': 0}, 'on': {'x': 1}} for i in range(10)}
    t = pr.sign_test(pairs, 'x')
    assert t['hit'] == 10 and t['miss'] == 0
    assert t['p'] < 0.01, t


def test_pairs_keep_the_seed_so_both_seeds_survive():
    """시드를 키에서 빼면 뒤 시드가 앞 시드를 덮어 쌍이 절반으로 준다."""
    pr = _probe_mod()
    rows = []
    for seed in (1, 2):
        for side in ('off', 'on'):
            rows.append({'aid': 'A', 'seed': seed, 'side': side,
                         'raw': '{"daily_propensity": 0.%d, "category": "식사"}' % seed})
    assert pr.compare(rows)['paired'] == 2, '시드 둘이 각각 한 쌍이어야 한다'


# --------------------------------------------- 라운드 관문이 실제로 거부하는가
def test_the_round_gate_rejects_a_rejected_candidate():
    """관문은 안 쓰일 때 조용히 망가져 있다가 정작 필요한 순간에 통과시킨다.

    기각된 후보 1 의 실제 자료(최소 p=0.503)를 넣어 **거부하는지** 본다.
    라운드 런너가 쓰는 것과 같은 식이다 — 어느 지표든 p<0.05 면 통과.
    """
    import json as _json
    import io as _io
    pr = _probe_mod()
    path = ROOT / 'output/scope_probe/responses.jsonl'
    if not path.exists():
        pytest.skip('탐침 응답이 저장소에 없다')
    rows = [_json.loads(l) for l in _io.open(path, encoding='utf-8') if l.strip()]
    s = pr.compare(rows)
    best = min(s[k]['p'] for k in ('sign_propensity', 'sign_events', 'sign_excluded'))
    assert best >= 0.05, '기각된 후보인데 관문이 통과시킨다 (최소 p=%.3f)' % best


def test_the_round_gate_would_pass_a_clean_split():
    """늘 거부하기만 하면 관문이 아니라 벽이다."""
    pr = _probe_mod()
    pairs = {('a%d' % i, 1): {'off': {'n_events': 0}, 'on': {'n_events': 1}}
             for i in range(10)}
    assert pr.sign_test(pairs, 'n_events')['p'] < 0.05


def test_each_candidate_carries_its_own_expected_direction():
    """"소비성향이 내려가야 한다" 는 후보를 '증가 기대' 로 찍으면 표가 정반대로
    읽힌다. 양측 p 는 같아 판정은 안 바뀌지만 **사람이 틀리게 읽는다.**
    실제로 후보 2 에서 그랬다.
    """
    pr = _probe_mod()
    assert pr.CANDIDATES['case_trend_2020_11_24']['expect']['propensity'] == 'down'
    assert pr.CANDIDATES['scope_fact']['expect']['propensity'] == 'up'

    def rows(name, d):
        return [{'aid': 'A', 'seed': 1, 'side': s, 'candidate': name,
                 'raw': '{"daily_propensity": %s}' % v}
                for s, v in (('off', 0.5), ('on', 0.5 + d))]

    down = pr.compare(rows('case_trend_2020_11_24', -0.1))['sign_propensity']
    up = pr.compare(rows('scope_fact', -0.1))['sign_propensity']
    assert down['hit'] == 1 and down['dir'] == '감소'
    assert up['hit'] == 0 and up['dir'] == '증가'


def test_old_responses_without_a_candidate_keep_the_old_reading():
    """후보 이름이 없는 옛 자료를 다시 세었을 때 수가 달라지면 안 된다."""
    pr = _probe_mod()
    rows = [{'aid': 'A', 'seed': 1, 'side': s, 'raw': '{"daily_propensity": %s}' % v}
            for s, v in (('off', 0.5), ('on', 0.6))]
    t = pr.compare(rows)['sign_propensity']
    assert t['dir'] == '증가' and t['hit'] == 1
