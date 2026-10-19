"""정책이 **모델에게 실제로 도달하는가** 를 전수 점검한다.

    python scripts/report/audit_policy_delivery.py

`audit_indicator_coverage.py` 는 "채점기가 그 지표를 계산할 수 있는가" 를 본다.
이 스크립트는 그 앞을 본다 — **지표를 계산할 수 있어도 정책이 모델에게 닿지
않으면 그 런은 프롬프트와 무관하게 값을 못 낸다.**

여드레 동안 그랬다. `sector_voucher`·`price_discount` 는 사용처 표시가 켜지는
조건에서 통째로 빠져 있었고(지갑 잔액 > 0 만 봤다), 에이전트에게는 자격 있는
가게가 하나도 없는 셈이었다. 위약에서 대상 업종이 기대와 반대로 −10.5% 로
줄었던 것이 그 자국이다. 채점기는 멀쩡했다 — 그래서 안 보였다.
(`experiments/MERGE_REVERTED_A_FIX.md`)

## 다섯 가지를 본다

    ① 라벨       한글 문장 한가운데 영문 식별자가 박히지 않는가
    ② 원칙       지갑 없는 정책에 "정책지갑으로 낼지" 라고 없는 지갑을 말하지 않는가
    ③ 도달       사용처 제한 정책의 표시가 켜지고, 판정 룰을 읽는가
    ④ 업종       그 판정 룰이 **실재하는 업종** 을 가리키는가 (0개면 표시가 안 붙는다)
    ⑤ 누설       정답지 수치가 모델이 읽는 글에 들어가 있지 않은가

⑤ 의 '정답지 수치' 는 손으로 적지 않는다 — 채점표의 지표 설명에서 숫자를
뽑아 쓴다. 채점표에 새 수치가 들어오면 이 점검도 저절로 그것을 본다.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'scripts' / 'sim'
POLDIR = ROOT / 'data/neo4j_load/policies'
SCORING = ROOT / 'data/experiments/scoring_table.json'

sys.path.insert(0, str(SIM))
sys.path.insert(0, str(ROOT / 'scripts'))


def _sector_names() -> set[str]:
    """그래프에서 읽은 실제 업종 이름. audit_indicator_coverage 와 같은 출처다."""
    spec = importlib.util.spec_from_file_location(
        'aic', ROOT / 'scripts/report/audit_indicator_coverage.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.L1 | m.SUB | m.KDI


# 숫자 하나만 보면 "20%" 같은 정책 자체의 조건까지 걸린다. 정답지에서 온 수치만
# 본다 — 채점표 지표 설명의 괄호 안 실측값이 그것이다.
_NUM = re.compile(r'[+\-]?\d+(?:[.,]\d+)?\s*(?:%p|%|배|원)')


_PARAM_KEYS = ('rate', 'discount_rate', 'benefit_rate', 'threshold_ratio',
               'cap', 'cap_per_agent', 'purchase_cap_monthly',
               'amount', 'rebate', 'min_amount', 'count')


def own_numbers(pol: dict) -> set[str]:
    """**정책이 스스로 말하는 수치.** 이것은 누설이 아니라 제도의 내용이다.

    P015 의 "20%·30%" 는 정답지가 아니라 `sectors` 에 선언된 할인율이다.
    이 구분이 없으면 제도를 설명한 것을 누설로 신고한다 — 그러면 아무도
    이 점검을 안 보게 된다.
    """
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in _PARAM_KEYS and isinstance(v, (int, float)):
                    if 0 < v <= 1:
                        out.add(('%g%%' % (v * 100)))
                    else:
                        out.add('%d원' % int(v))
                        out.add('{:,}원'.format(int(v)))
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(pol)
    return out


def answer_numbers(block: dict, pol: dict) -> set[str]:
    """채점표 지표 설명에 적힌 **실측 수치**. 정책 자체의 조건은 뺀다."""
    own = own_numbers(pol)
    out: set[str] = set()
    for ind in (block.get('indicators') or []):
        for m in _NUM.finditer(str(ind.get('desc') or '')):
            tok = m.group(0).replace(' ', '')
            # 한 자리 수·0 은 우연히 걸린다. 두 글자 이상 숫자만 본다.
            if len(re.sub(r'\D', '', tok)) >= 2:
                # 부호를 뗀 꼴도 같이 본다 — 채점표는 "+6.957%" 로 적지만
                # 프롬프트에 샌다면 "6.957%" 로 샐 것이다. 부호째 찾으면 놓친다.
                for t in {tok, tok.lstrip('+-')}:
                    if t and t not in own:
                        out.add(t)
    return out


def _anon_label(ptype: str) -> str:
    """익명 모드 라벨 — 실제 런(EXP_POLICY_ANONYMOUS=1)이 쓰는 표기."""
    from mechanisms import _LABEL, FALLBACK_LABEL
    return '[%s]' % (_LABEL.get((ptype or '').strip()) or FALLBACK_LABEL)


def scoring_for(pid: str, table: dict) -> tuple[str, dict]:
    """정책 id 에 붙은 채점 블록. 키가 정책 id 가 아닌 것들이 있다."""
    if pid in table:
        return pid, table[pid]
    for k, v in table.items():
        if not isinstance(v, dict):
            continue
        if v.get('policy_id') == pid or str(v.get('policy_file') or '').endswith('%s.json' % pid):
            return k, v
    return '', {}


def rendered(pol: dict) -> str:
    """에이전트가 읽는 글 — 정책 사실 줄 + 기전 고유 줄."""
    import dawn_context as dc
    from mechanisms import get as mech_get
    row = dict(pol)
    row['from_'] = pol.get('effective_from')
    row['until_'] = pol.get('effective_until')
    row['regions'] = pol.get('target_districts') or []
    row['target_l1s'] = pol.get('benefit_categories') or []
    txt = dc._format_policy_facts([row])
    mod = mech_get(pol.get('type'))
    if mod is not None and hasattr(mod, 'status'):
        try:
            txt += '\n' + mod.status(pol.get('id') or '', row, {}, {})
        except Exception as e:                       # 렌더가 터지면 그것도 결함이다
            txt += '\n**status 렌더 실패: %s**' % e
    return txt


def eligible_sector_count(spec: dict | None, names: set[str]) -> int | None:
    """판정 룰이 가리키는 **실재 업종** 의 개수. include 모드에서만 의미가 있다."""
    if not spec:
        return None
    if (spec.get('mode') or '') != 'include':
        return None                                   # exclude 는 기본 적격이라 세지 않는다
    inc = spec.get('include') or {}
    got = 0
    for key in ('subs', 'l1s'):
        for x in (inc.get(key) or []):
            if str(x).strip() in names:
                got += 1
    # 업종코드로만 지정한 경우는 이 목록으로 셀 수 없다 — 없음(None)으로 둔다
    if got == 0 and (inc.get('codes') or []):
        return None
    return got


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='', help='하나만 볼 때')
    a = ap.parse_args()

    from mechanisms import label, has_wallet, principle, poi_restriction, _PRINCIPLE

    names = _sector_names()
    table = json.loads(io.open(SCORING, encoding='utf-8').read())
    files = sorted(POLDIR.glob('P*.json'))
    if a.policy:
        files = [f for f in files if f.stem == a.policy]

    bad: list[tuple[str, str]] = []
    print('%-6s %-15s %-7s %-7s %-6s %s' % ('정책', '기전', '라벨', '원칙', '도달', '업종/누설'))
    print('-' * 92)
    for f in files:
        pol = json.loads(io.open(f, encoding='utf-8').read())
        pid, ptype = pol.get('id') or f.stem, pol.get('type') or ''
        notes: list[str] = []

        # ① 라벨 — 영문 식별자가 새지 않는가.
        # **런 설정에서 판정한다.** 모든 정답지 라운드는 EXP_POLICY_ANONYMOUS=1 이고
        # 그때 라벨은 정책 이름을 쓰지 않는다. 이름 모드에서만 새는 것은 결함이
        # 아니라 기록 사항이다 — 결함으로 올리면 실제 런과 다른 것을 고치게 된다.
        lab_anon = _anon_label(ptype)
        lab_named = label(ptype, pol.get('name'))
        ok_label = not re.search(r'[A-Za-z_]{3,}', lab_anon)
        if not ok_label:
            bad.append((pid, '익명 라벨에 영문 식별자: %s' % lab_anon))
        elif re.search(r'[A-Za-z_]{3,}', lab_named):
            notes.append('이름 모드에서만 영문: %s' % lab_named)

        # ② 원칙 — 지갑 없는 정책에 지갑 원칙을 주지 않는가
        pr = principle([ptype])
        ok_princ = has_wallet([ptype]) or pr != _PRINCIPLE['wallet']
        if not ok_princ:
            bad.append((pid, '지갑이 없는데 지갑 원칙을 준다'))

        # ③ 도달 — 사용처 제한이면 표시가 켜지고 룰을 읽는가
        if pol.get('poi_restricted'):
            bal = {pid: 250000} if has_wallet([ptype]) else {}
            ids, spec, mark = poi_restriction([pol], bal)
            ok_reach = bool(ids)
            if not ok_reach:
                bad.append((pid, '사용처 표시가 켜지지 않는다 — 모델이 대상을 못 본다'))
            reach = 'O' if ok_reach else '**X**'
            # ④ 업종 — 룰이 실재 업종을 가리키는가
            n = eligible_sector_count(spec, names)
            if n is not None:
                notes.append('적격업종 %d개' % n)
                if n == 0:
                    bad.append((pid, '판정 룰이 가리키는 업종이 그래프에 하나도 없다'))
            elif spec is None and not mark:
                notes.append('기존 쿠폰 룰로 떨어짐')
        else:
            reach = '-'

        # ⑤ 누설 — 정답지 수치가 모델이 읽는 글에 있는가
        key, block = scoring_for(pid, table)
        txt = rendered(pol)
        leaks = sorted(x for x in answer_numbers(block, pol) if x in txt.replace(' ', ''))
        if leaks:
            bad.append((pid, '정답지 수치가 프롬프트에 있다: %s' % ', '.join(leaks)))
            notes.append('**누설 %s**' % ','.join(leaks))
        elif key:
            notes.append('누설 없음(%s)' % key)
        else:
            notes.append('채점표에 없음')

        print('%-6s %-15s %-7s %-7s %-6s %s'
              % (pid, ptype, 'O' if ok_label else '**X**',
                 'O' if ok_princ else '**X**', reach, ' · '.join(notes)))

    print()
    print('정책 %d개 중 결함 %d개' % (len(files), len(bad)))
    for pid, why in bad:
        print('  %-6s %s' % (pid, why))
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
