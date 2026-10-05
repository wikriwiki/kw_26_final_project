"""사용처 제한 표시가 **지갑 없는 정책에서도** 켜지는가.

이것이 꺼져 있으면 모델은 대상 업종과 비대상 업종을 가를 근거를 하나도 못 받는다.
프롬프트를 아무리 고쳐도 그 런은 값을 못 낸다 — 채점기가 0 을 읽는 것이 아니라
**애초에 정책이 모델에게 도달하지 않는다.**

### 어쩌다 이렇게 되어 있었는가

표시를 켜는 조건이 이랬다.

    poi_restricted 이고 **그 정책의 지갑 잔액 > 0**

P010(지원금)에서는 맞는 조건이다 — 잔액이 없으면 쓸 쿠폰이 없다. 그런데
`sector_voucher`·`price_discount` 는 **지갑을 만들지 않는다.** 결제하는 자리에서
값이 깎이는 방식이라 잔액이 영원히 0 이고, 그래서 표시가 영영 안 켜졌다.

### 무엇을 고정하는가

    ① 지갑형   잔액이 있을 때만 켠다            (P010 동결 — 동작이 변하면 안 된다)
    ② 비지갑형 정책이 살아 있으면 켠다
    ③ 판정 룰·표시 문구는 **정책이 선언한 것**을 쓴다 (코드에 정책 분기를 더하지 않는다)
    ④ 예산요약의 "이 지갑으로 낸다" 문구는 지갑형에만 붙는다 — 아니면 거짓이 된다
"""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

from mechanisms import poi_restriction  # noqa: E402

POLDIR = ROOT / 'data/neo4j_load/policies'


def pol(pid):
    return json.load(io.open(POLDIR / ('%s.json' % pid), encoding='utf-8'))


# ---------------------------------------------------------------- 지갑형(동결)
def test_wallet_policy_still_needs_a_balance():
    """P010 은 동결이다. 잔액 0 이면 예전처럼 꺼져 있어야 한다."""
    p = pol('P010')
    ids, spec, mark = poi_restriction([p], {})
    assert ids == set()
    ids, spec, mark = poi_restriction([p], {'P010': 250000})
    assert ids == {'P010'}
    # P010 은 룰도 표시도 선언하지 않는다 → 호출부의 기존 기본값으로 떨어진다
    assert spec is None and mark is None


# ------------------------------------------------------- 비지갑형(여기가 꺼져 있었다)
@pytest.mark.parametrize('pid', ['P014', 'P015', 'P016'])
def test_non_wallet_policy_turns_on_with_no_balance(pid):
    p = pol(pid)
    assert p.get('poi_restricted'), '%s 는 사용처 제한 정책이어야 한다' % pid
    ids, spec, mark = poi_restriction([p], {})     # 잔액이 없다 — 있을 수가 없다
    assert ids == {pid}, '%s 의 사용처 표시가 켜지지 않는다 — 모델이 대상을 못 본다' % pid
    assert mark == p['eligible_marker']
    assert spec == p.get('eligibility')


def test_the_declared_marker_is_used_not_a_hardcoded_one():
    """정책마다 표시가 다르다. 코드가 '[쿠폰]' 을 박아 두면 다음 정책에서 또 틀린다."""
    _, _, mark = poi_restriction([pol('P016')], {})
    assert mark == '[농할]'
    _, _, mark = poi_restriction([pol('P014')], {})
    assert mark == '[지역]'


def test_the_generic_eligibility_spec_is_handed_through():
    """판정 룰을 코드가 아니라 정책 JSON 이 정한다."""
    _, spec, _ = poi_restriction([pol('P016')], {})
    assert spec['mode'] == 'include'
    assert set(spec['include']['subs']) == {'청과', '정육', '슈퍼마켓', '식료품'}


# --------------------------------------------------------------------- 그 외
def test_cashback_is_not_a_poi_restriction():
    """P012 는 적립 업종 표시이지 사용처 제한이 아니다. 여기에 들어오면 안 된다."""
    ids, _, _ = poi_restriction([pol('P012')], {})
    assert ids == set()


def test_no_policy_means_nothing_is_on():
    assert poi_restriction([], {}) == (set(), None, None)
    assert poi_restriction(None, None) == (set(), None, None)


def test_unknown_mechanism_without_a_wallet_still_turns_on():
    """모르는 기전이 와도 지갑이 없으면 켠다 — 새 정책이 조용히 죽지 않게."""
    p = {'id': 'PZZZ', 'type': 'interest_subsidy', 'poi_restricted': True,
         'eligible_marker': '[표시]'}
    ids, _, mark = poi_restriction([p], {})
    assert ids == {'PZZZ'} and mark == '[표시]'


def test_the_frozen_p010_suffix_is_byte_identical():
    """P010 은 동결이다. 예산요약에 붙는 덧말이 한 글자도 달라지면 안 된다.

    표시 문구를 정책 선언에서 가져오게 바꾸면서, 선언이 없는 P010 이 옛 문자열로
    떨어지는지 여기서 못 박는다.
    """
    _, _, mark = poi_restriction([pol('P010')], {'P010': 250000})
    got = " (사용처 제한: %s 표시 매장에서만 사용 가능)" % (mark or "[쿠폰]")
    assert got == " (사용처 제한: [쿠폰] 표시 매장에서만 사용 가능)"


def test_the_spec_is_read_from_mech_params_too():
    """그래프에서 읽은 행은 선언이 mech_params(JSON 문자열) 안에 들어 있다.

    이것을 안 보면 파일 경로에서는 되고 **DB 경로에서만 조용히 None** 이 된다.
    런에서만 틀리고 시험에서는 통과하는 모양이라 제일 늦게 발견된다.
    """
    p = dict(pol('P016'))
    spec = p.pop('eligibility')
    p['mech_params'] = json.dumps({'eligibility': spec, 'sectors': p.get('sectors')},
                                  ensure_ascii=False)
    ids, got, mark = poi_restriction([p], {})
    assert ids == {'P016'}
    assert got == spec, 'mech_params 안의 판정 룰을 못 읽는다'
    assert mark == '[농할]'


def test_it_matches_the_implementation_the_merge_reverted():
    """`fc91872` 의 인라인 판정과 결과가 같은가 — 되돌아간 그 코드가 기준이다.

    서버의 작업 사본이 지금 그 코드로 돌고 있다. 레지스트리로 옮기면서 동작이
    달라지면 **이미 채점한 런과 다음 런이 서로 다른 코드가 된다.**
    """
    def inline(policies, balances):                 # fc91872 그대로
        _WALLET_T = {"grant", "subsidy", "voucher"}
        pids, spec, marker = set(), None, None
        for _p in (policies or []):
            if not _p.get("poi_restricted"):
                continue
            if _p.get("type") in _WALLET_T and (balances or {}).get(_p["id"], 0) <= 0:
                continue
            pids.add(_p["id"])
            if spec is None:
                _mp = _p.get("mech_params")
                try:
                    _mp = json.loads(_mp) if isinstance(_mp, str) else (_mp or {})
                except (TypeError, ValueError):
                    _mp = {}
                spec = _mp.get("eligibility")
                marker = _p.get("eligible_marker")
        return pids, spec, marker

    def as_db_row(p):
        """그래프에서 읽은 모양 — 선언이 mech_params 로 들어간다."""
        core = {'id', 'name', 'type', 'poi_restricted', 'eligible_marker'}
        row = {k: v for k, v in p.items() if k in core}
        row['mech_params'] = json.dumps(
            {k: v for k, v in p.items() if k not in core}, ensure_ascii=False)
        return row

    cases = [
        ([], {}),
        ([as_db_row(pol('P010'))], {}),
        ([as_db_row(pol('P010'))], {'P010': 250000}),
        ([as_db_row(pol('P012'))], {}),
        ([as_db_row(pol('P014'))], {}),
        ([as_db_row(pol('P015'))], {}),
        ([as_db_row(pol('P016'))], {}),
        ([as_db_row(pol('P010')), as_db_row(pol('P016'))], {'P010': 250000}),
    ]
    for policies, bal in cases:
        assert poi_restriction(policies, bal) == inline(policies, bal), (
            '되돌아간 구현과 결과가 다르다: %s' % [p.get('id') for p in policies])
