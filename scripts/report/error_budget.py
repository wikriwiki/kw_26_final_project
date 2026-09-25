"""실측과 시뮬의 크기 비교를 감사한다.

    python scripts/report/error_budget.py

## 무엇이 검증지표인가

**실측 수치가 없으면 검증지표가 아니다.** 방향만 말하는 지표는 "맞혔다" 를
셀 수는 있어도 **오차를 계산할 수 없다.** 그러므로 오차 예산에서 뺀다.

    들어간다   추정량·기간·모집단이 확인되고 실측이 퍼센트로 있는 것 — 오차 = |시뮬 − 실측| (%p)
               실측이 두 수로 있는 순위 — 오차 = |우리 간격 − 실측 간격| (%p)
    빠진다     실측이 없는 것 (방향만 말하거나 백분율 미공개)
               단위가 달라 %p 로 맞댈 수 없는 것 (MPC 0.21 · 월 금액 47,880원)
               위약 — 정답지가 없다. 수렴지표가 아니라 내부 방어선이다

위약을 오차에서 빼는 것이 위약을 버리는 것은 아니다. **"다 오른다" 고 답하는
프롬프트를 거르는 장치**로 계속 쓰고 부호 적중표가 그것을 센다.

## 오차를 정확도로 오해하지 않기

원문 창은 주·월 단위인데 우리 창은 이틀인 경우가 많다. 대상·기간·대조 설계가
다르면 숫자끼리 뺀 값은 오차가 아니다. 원문 감사 결과(2026-09-26) 현재 보관
결과 중 직접 비교 가능한 크기 지표는 0개다. 적격한 결과가 생길 때까지 총오차를
0으로 출력하지 않는다.

각 지표의 창 불일치 정도를 `SCALE_NOTE` 에 적어 둔다. 오차가 큰 것이
프롬프트 탓인지 창 탓인지 그 줄을 보고 가른다.

## 실측이 있는데 시뮬이 값을 못 내는 것

그것은 **우리 문제**다. 오차를 "없음" 으로 두지 않고 **미측정으로 세어**
표에 남긴다 — 빼 버리면 못 재고 있다는 사실이 사라진다.
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

DASH = str.maketrans({'−': '-', '–': '-', '—': '-'})

# 창 불일치 — 오차의 구조적 하한을 읽는 데 쓴다.
SCALE_NOTE = {
    'P012-1': '원문 한 달 · 우리 이틀',
    'P012-2': '원문 한 달 · 우리 이틀 · **2026-09-24 눈금 이동** — st.online_spent 로 재측정 대기',
    'P012-5': '원문 한 달 · 우리 이틀',
    'P012-6': '기존 출력은 전체 시민·10월 일부 기간. 원문은 10~11월 수령자',
    'EM-2': '원문 19~33주 · 우리 이틀',
    'EM-3': '원문 전년 동기 대비 · 우리 같은 해 전후',
    'EM-4': '원문 19~33주 · 우리 이틀',
    'DS-1': '원문 40주 · 우리 이틀 · 실측은 한식, 우리는 식사 전체',
    'DS-2': '원문 40주 · 우리 이틀',
    'DS-6': 'hub_type 자료가 그래프에 없다',
    'C1': '원문 사업기간 4개월 · 우리 이틀 · 상품 단위 대 POI 단위',
    'C2': '같음',
    'C3': '같음',
}

# 실측·시뮬 수치는 있지만 정의가 달라 이 결과에서 오차를 낼 수 없다.
DEFINITION_PENDING = {
    'P010-1': '옛 0.216은 28일 전체 평균(등록 ON 이틀은 0.121). BOK 조사 문항·분모·기간·모집단과 최종 결제 귀속이 다르다',
    'P012-4': '47,880원은 10~11월 캐시백 수령자 평균. 옛 출력은 전체 시민·부분월이다',
    'P012-6': '21%는 10~11월 수령자 중 상한 도달률. 옛 출력은 전체 시민·부분월이다',
}


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def truth_pct(desc):
    """`(실측 +20.82%)` 에서 한 수. 없으면 None."""
    t = str(desc or '').translate(DASH)
    m = re.search(r'\(실측\s*([^)]*)\)', t)
    if not m:
        return None
    body = m.group(1)
    if '원' in body and '%' not in body:
        return None
    m2 = re.search(r'([+-]?\d+(?:\.\d+)?)\s*%', body)
    return float(m2.group(1)) if m2 else None


def truth_won(desc):
    """`(실측 47,880원)`에서 원 단위 크기만 읽는다."""
    m = re.search(r'\(실측\s*([\d,]+)\s*원', str(desc or ''))
    return float(m.group(1).replace(',', '')) if m else None


def truth_gap(desc):
    """`(실측 +10.8%p vs +3.6%p)` 에서 **간격**. 순위 지표용."""
    t = str(desc or '').translate(DASH)
    m = re.search(r'\(실측\s*([^)]*)\)', t)
    if not m:
        return None
    nums = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*%', m.group(1))
    if len(nums) < 2:
        return None
    return float(nums[0]) - float(nums[1])


def sim_gap(v):
    """순위 결과 문구에서 우리 간격. `A +3.1% vs B -7.7%` 꼴."""
    if not isinstance(v, dict):
        return None
    txt = str(v.get('note') or v.get('got') or '').translate(DASH)
    nums = re.findall(r'([+-]?\d+(?:\.\d+)?)\s*%', txt)
    if len(nums) < 2:
        return None
    return float(nums[0]) - float(nums[1])


def collect():
    sb = _load('sb', 'scripts/report/sign_scoreboard.py')
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    rows = []
    # 위약은 정답지가 없으므로 오차 예산에 넣지 않는다 — READINGS 만 본다.
    for key, name, block, _why in sb.READINGS:
        blk = sc.get(key) or {}
        res = blk.get(block) or {}
        for ind in (blk.get('indicators') or []):
            iid, expect = ind['id'], ind.get('expect')
            desc = ind.get('desc') or ''
            v = res.get(iid) if isinstance(res.get(iid), dict) else {}
            sus = (key, block, iid) in sb.SUSPECT
            row = {'policy': name, 'id': iid, 'expect': expect, 'run': block,
                   'desc': desc[:70], 'scale': SCALE_NOTE.get(iid, ''),
                   'suspect': sus, 'kind': None, 'truth': None, 'sim': None,
                   'err': None, 'why': ''}
            audit = ind.get('empirical_audit') or {}
            if audit.get('comparison') in ('different_estimand', 'unverified_source'):
                row.update(kind=('추정량불일치' if audit['comparison'] == 'different_estimand'
                                 else '원문미확인'),
                           truth=audit.get('reported_value'),
                           sim=v.get('pct') if isinstance(v.get('pct'), (int, float)) else None,
                           why=audit.get('reason') or '실측 출처·정의 확인 필요')
                rows.append(row); continue
            if iid in DEFINITION_PENDING:
                mean = v.get('mean')
                row.update(kind='정의확인',
                           truth=(truth_won(desc) if iid == 'P012-4' else truth_pct(desc)),
                           sim=(100 * mean if iid == 'P012-6' else mean)
                               if isinstance(mean, (int, float)) else None,
                           why=DEFINITION_PENDING[iid])
                rows.append(row); continue
            if iid == 'LV-1':
                row.update(kind='실측없음', why='원문의 "영향 미미"는 크기 수치가 아니다')
                rows.append(row); continue
            g = truth_gap(desc)
            t = truth_pct(desc)
            if expect == 'rank' and g is not None:
                row.update(kind='순위간격', truth=g, sim=sim_gap(v))
            elif t is not None:
                row.update(kind='퍼센트', truth=t,
                           sim=v.get('pct') if isinstance(v.get('pct'), (int, float)) else None)
            else:
                row.update(kind='실측없음', why='정답지에 수치가 없다 — 검증지표에서 뺀다')
                rows.append(row); continue
            if sus:
                row['why'] = '다른 자로 쟀다 — 다시 재기 전까지 세지 않는다'
            elif row['sim'] is None:
                row['why'] = '**실측은 있는데 시뮬이 값을 못 낸다 — 우리 문제다**'
            else:
                row['err'] = abs(row['sim'] - row['truth'])
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--json-out', default='')
    a = ap.parse_args()
    rows = collect()

    print('# 실측과의 크기 비교 가능성 감사')
    print()
    print('%-14s %-7s %-8s %9s %9s %8s  %s'
          % ('정책', '지표', '종류', '실측', '시뮬', '오차%p', '창 불일치 / 사유'))
    print('-' * 118)
    scored = [r for r in rows if r['err'] is not None]
    unmeasured = [r for r in rows if r['kind'] in ('퍼센트', '순위간격') and r['err'] is None]
    dropped = [r for r in rows if r['kind'] in ('실측없음', '정의확인', '추정량불일치', '원문미확인')]
    for r in rows:
        f = lambda x: ('%+.2f' % x) if isinstance(x, (int, float)) else '—'
        print('%-14s %-7s %-8s %9s %9s %8s  %s'
              % (r['policy'], r['id'], r['kind'], f(r['truth']), f(r['sim']),
                 ('%.2f' % r['err']) if r['err'] is not None else '—',
                 r['why'] or r['scale']))
    print()
    tot = sum(r['err'] for r in scored)
    print('## 오차 예산')
    if scored:
        print('  잰 지표 %d개 · **총 오차 %.2f%%p** · 평균 %.2f%%p'
              % (len(scored), tot, tot / len(scored)))
    else:
        print('  직접 비교 가능한 크기 지표 0개 · 총오차 **정의 불가**')
    print('  측정·자료 미완료 지표 %d개 — 실측 정의와 시뮬 출력이 모두 갖춰지지 않았다' % len(unmeasured))
    for r in unmeasured:
        print('     %-7s %s' % (r['id'], r['scale'] or r['why']))
    print('  대조 제외 지표 %d개 — 실측 부재·출처 미확인·정의/추정량 불일치' % len(dropped))
    print()
    print('가장 큰 오차부터')
    for r in sorted(scored, key=lambda x: -x['err'])[:8]:
        print('  %-7s 오차 %6.2f%%p   실측 %+.2f → 시뮬 %+.2f   %s'
              % (r['id'], r['err'], r['truth'], r['sim'], r['scale']))
    if a.json_out:
        io.open(a.json_out, 'w', encoding='utf-8', newline='\n').write(
            json.dumps({'rows': rows, 'total': tot if scored else None, 'n': len(scored)},
                       ensure_ascii=False, indent=1))
        print()
        print('→ %s' % a.json_out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
