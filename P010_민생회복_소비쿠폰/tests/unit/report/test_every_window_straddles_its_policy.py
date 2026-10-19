"""등록된 채점 창이 **정책 시행일을 사이에 두는가** — 전 정책 전수.

같은 실수를 이틀 사이에 두 번 했다.

    P016  시행일을 2020-11-16 으로 잘못 알고 창을 잡았다. 실제는 07-30 이라
          무정책·정책 양쪽이 모두 사업기간 안이었다. 원문 p20 을 읽고 찾았다.
    P090  P012 와 같은 창에서 비교하려고 창을 옮겼는데, 옮긴 창이 위약 정책의
          발효 기간(10-11~10-31) 안이었다. 정책 ON 대 정책 ON 을 비교하고
          그 0 을 '기전을 처리하지 못했다'로 읽었다.

둘 다 **사람이 다시 읽어야만** 찾을 수 있었다. 그래서 기계가 본다.

## 면제

면제가 필요한 설계가 실제로 있다. 면제는 **이유를 적어야** 통과한다 —
이유 없는 면제를 허용하면 이 시험은 걸리는 순간 면제되는 시험이 된다.

    P012        월 실적 문턱이라 기간 안에서 쌓인다. 월 중 시점이 달라지면
                도달자가 달라지므로 ON 대 ON 비교에 뜻이 있다.
    DISTANCING  :Policy 노드가 아니라 환경(거리두기 단계)이다. 시행일이 없다.
    PLACEBO_TIMING  정책을 아예 안 올린 런이다. 양쪽 다 무정책이 설계다.
"""
from pathlib import Path
import io
import json
import re

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCORING = ROOT / 'data/experiments/scoring_table.json'
POLDIR = ROOT / 'data/neo4j_load/policies'

# (채점표 키, 창 키) → 면제 사유. **이유 없는 면제는 없다.**
EXEMPT = {
    ('P012', 'window_stage2'):
        '월 실적 문턱이라 기간 안에서 쌓인다 — ON 대 ON 비교에 뜻이 있다',
    ('DISTANCING_2020', 'window'):
        ':Policy 노드가 아니라 환경(거리두기 단계)이다 — 시행일이 없다',
    ('DISTANCING_2020', 'window_stage2'): '같음 — 환경 레짐 변화를 잰다',
    ('DISTANCING_2020', 'window_stage3'): '같음 — 환경 레짐 변화를 잰다',
    ('PLACEBO_TIMING', 'window'):
        '정책을 아예 올리지 않은 런이다 — 양쪽 다 무정책이 설계다',
    ('EMERGENCY_2020', 'window_stage2'):
        '옛 창(stage2). stage3 로 대체됐고 그 런은 다시 쓰지 않는다',
    ('LOCAL_VOUCHER', 'window_stage2'):
        '옛 창(stage2). stage3 로 대체됐고 그 런은 다시 쓰지 않는다',
}

# 이미 찾아 기록해 둔 결함. 고쳐지면 이 목록에서 빠져야 한다.
KNOWN_BAD = {
    ('PLACEBO_FAKE', 'window'):
        'window_sits_inside_the_policy_period 에 기록됨 — 다시 돌려야 한다',
}

DATE = re.compile(r'(\d{4}-\d{2}-\d{2})')


def scoring():
    return json.loads(io.open(SCORING, encoding='utf-8').read())


def policy_of(key, blk):
    """이 채점 블록이 어느 정책 파일을 쓰는가."""
    pid = blk.get('policy_id')
    if not pid:
        m = re.search(r'(P\d+)', str(blk.get('policy_file') or ''))
        pid = m.group(1) if m else None
    if not pid:
        for w in ('window', 'window_stage3', 'window_stage2'):
            v = blk.get(w)
            if isinstance(v, dict) and v.get('policy'):
                pid = v['policy']
                break
    if not pid and (POLDIR / ('%s.json' % key)).exists():
        pid = key
    if not pid:
        return None
    p = POLDIR / ('%s.json' % pid)
    return json.load(io.open(p, encoding='utf-8')) if p.exists() else None


def windows(blk):
    """(창 키, off 날짜들, on 날짜들, 규칙 시행일). 사전형·산문형 둘 다 읽는다.

    `rule_date` 는 :Policy 노드가 없는 **환경 규칙**용이다(사적모임 인원 제한).
    정책 파일이 없다고 면제하면 그 창은 아무도 안 본다 — 대신 규칙 시행일에
    대고 같은 검사를 한다.
    """
    out = []
    for wk, v in blk.items():
        if not wk.startswith('window'):
            continue
        if isinstance(v, dict):
            off = DATE.findall(str(v.get('off') or ''))
            on = DATE.findall(str(v.get('on') or ''))
        else:
            t = str(v)
            m_off = re.search(r'무정책\s*(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})', t)
            # '무정책' 안에도 '정책' 이 있다 — 앞 글자를 막는다
            m_on = re.search(r'(?<!무)정책\s*(\d{4}-\d{2}-\d{2}):(\d{4}-\d{2}-\d{2})', t)
            off = list(m_off.groups()) if m_off else []
            on = list(m_on.groups()) if m_on else []
        rule = (v.get('rule_date') if isinstance(v, dict) else None)
        if off and on:
            out.append((wk, off, on, rule))
    return out


def all_cases():
    sc = scoring()
    cases = []
    for key, blk in sc.items():
        if not isinstance(blk, dict) or 'indicators' not in blk:
            continue
        pol = policy_of(key, blk)
        for wk, off, on, rule in windows(blk):
            cases.append(pytest.param(key, wk, off, on, pol, rule,
                                      id='%s/%s' % (key, wk)))
    return cases


CASES = all_cases()


def test_there_are_windows_to_check():
    """읽어 내지 못하면 이 시험은 조용히 통과한다. 그 일이 없게 한다."""
    assert len(CASES) >= 8, '창을 %d개밖에 못 읽었다 — 파서를 확인할 것' % len(CASES)


@pytest.mark.parametrize('key,wk,off,on,pol,rule', CASES)
def test_the_window_straddles_the_policy_start(key, wk, off, on, pol, rule):
    why = EXEMPT.get((key, wk))
    if why:
        assert len(why) > 10, '면제에는 이유가 있어야 한다: %s/%s' % (key, wk)
        pytest.skip('면제 — ' + why)
    if (key, wk) in KNOWN_BAD:
        pytest.xfail('이미 기록된 결함 — ' + KNOWN_BAD[(key, wk)])
    if rule and not pol:
        # 환경 규칙 — 시행일 하나만 있다. 끝나는 날은 검사하지 않는다.
        for d in off:
            assert d < rule, (
                '%s/%s: 무정책 창 %s 가 규칙 시행일 %s 이후다' % (key, wk, d, rule))
        for d in on:
            assert d >= rule, (
                '%s/%s: 정책 창 %s 가 규칙 시행일 %s 이전이다' % (key, wk, d, rule))
        return
    assert pol, ('%s/%s 의 정책 파일도 rule_date 도 없다 — 무엇에 대고 재는지 적어야 한다'
                 % (key, wk))
    start, end = pol['effective_from'], pol['effective_until']
    for d in off:
        assert d < start, (
            '%s/%s: 무정책 창 %s 가 시행일 %s 이후다 — 정책ON 대 정책ON 을 비교하게 된다'
            % (key, wk, d, start))
    for d in on:
        assert start <= d <= end, (
            '%s/%s: 정책 창 %s 가 시행기간(%s~%s) 밖이다' % (key, wk, d, start, end))


def test_the_recorded_defect_is_written_down_where_it_happened():
    """결함을 시험에만 적어 두면 채점표를 읽는 사람은 모른다."""
    sc = scoring()
    blk = sc['PLACEBO_FAKE']
    assert 'window_sits_inside_the_policy_period' in blk
    note = blk['window_sits_inside_the_policy_period']
    assert '정책 ON 대 정책 ON' in note['what']
    assert 'todo' in note and '10-07' in note['todo'], '어떻게 고칠지 적혀 있어야 한다'


def test_the_defective_reading_is_not_counted_in_the_scoreboard():
    """기록만 하고 계속 세면 성적이 부풀려진다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'sb', ROOT / 'scripts/report/sign_scoreboard.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for iid in ('PL-1', 'PL-2'):
        assert ('PLACEBO_FAKE', 'result_2026_09_16_fixed_plumbing', iid) in m.SUSPECT
