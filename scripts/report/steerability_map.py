"""**프롬프트로 움직일 수 있는 지표가 어느 것인가** — 관문 넷을 한 표에 모은다.

    python scripts/report/steerability_map.py

목표는 "정책들의 정답에 수렴하는 프롬프트" 다. 그런데 프롬프트를 고쳐도
소용없는 자리가 여럿이고, 서로 다른 이유로 그렇다. 후보를 더 만들기 전에
**어디서 일해야 값이 나오는지**를 먼저 적는다.

## 관문 넷 — 하나라도 막히면 프롬프트는 무력하다

    ① 도달   정책이 모델에게 닿는가
             닿지 않으면 프롬프트를 아무리 고쳐도 지표가 안 움직인다.
             실제로 여드레 동안 두 기전이 막혀 있었다.
             → scripts/report/audit_policy_delivery.py

    ② 정답   정답지에 숫자가 있는가
             없으면 크기 수렴을 말할 수 없다. 부호만 채점한다.
             → experiments/WHICH_POLICIES_CAN_CONVERGE.md

    ③ 검출   지금 n 으로 그 효과를 0 과 가를 수 있는가
             못 가르면 프롬프트 차이도 못 가른다. 표본이 병목이다.
             → scripts/report/power_to_detect_truth.py

    ④ 여지   구조적으로 움직일 폭이 남아 있는가
             몫 지표의 base 가 0.77 이면 올릴 천장이 낮고, 0.06 이면 내릴 바닥이 얕다.

그리고 ⑤ 런 이동 — 같은 프롬프트를 두 번 돌렸을 때의 이동폭. 후보 간 차이가
이것보다 작으면 **프롬프트의 성질로 읽지 않는다.**

이 표는 **어느 프롬프트가 낫다**를 말하지 않는다. **어디서 재야 읽히는가**를
말한다. 판정은 각 라운드의 사전등록이 한다.
"""
from __future__ import annotations

import importlib.util
import io
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / 'data/experiments/scoring_table.json'

POLICIES = [
    ('상생소비 P012', 'P012'),
    ('긴급재난 P013', 'EMERGENCY_2020'),
    ('지역상품권 P014', 'LOCAL_VOUCHER'),
    ('거리두기', 'DISTANCING_2020'),
    ('8대쿠폰 P015', 'SECTOR_VOUCHER_2020'),
    ('농할 P016', 'P016'),
]

# 같은 프롬프트를 두 번 돌렸을 때 잰 이동폭과, 후보 둘이 실제로 벌린 차이.
# **둘 다 잰 것만 적는다** — 안 잰 지표에 숫자를 지어 넣지 않는다.
#   DS-1   런 이동 stage3→v50 0.2%p · v50→라운드3 0.5%p (둘 다 n=200) / 후보차 0.9%p
#   DS-2   런 이동 13.1%p · 2.7%p — 한 수로 적을 수 없다              / 후보차 7.4%p
#   P012-1 런 이동 8.2%p (n 이 200→500 으로 함께 바뀌어 섞여 있다)     / 후보차 1.3%p
RUN_SHIFT = {
    'DS-1': (0.5, '0.2·0.5%p'),
    'DS-2': (13.1, '13.1·2.7%p'),
    'P012-1': (8.2, '8.2%p*'),
}
CAND_DIFF = {'DS-1': 0.9, 'DS-2': 7.4, 'P012-1': 1.3}

# 몫 지표의 구조적 한계를 읽는 문턱. 값 자체가 아니라 자릿수를 본다.
CEILING_HI = 0.70      # 이미 이만큼이면 올릴 자리가 좁다
CEILING_LO = 0.08      # 이만큼뿐이면 내릴 자리가 좁다


def load_power():
    spec = importlib.util.spec_from_file_location(
        'ptt', ROOT / 'scripts/report/power_to_detect_truth.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def delivery_ok() -> tuple[bool, str]:
    """정책이 모델에게 닿는가 — 전수 점검을 그대로 돌린다."""
    spec = importlib.util.spec_from_file_location(
        'apd', ROOT / 'scripts/report/audit_policy_delivery.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    old_argv, old_out = sys.argv, sys.stdout
    sys.argv, sys.stdout = ['audit_policy_delivery.py'], io.StringIO()
    try:
        rc = m.main()
        out = sys.stdout.getvalue()
    finally:
        sys.argv, sys.stdout = old_argv, old_out
    return rc == 0, out


def latest_result(blk: dict):
    """가장 큰 n 을 가진 결과 블록. power_to_detect_truth 와 같은 고르기다."""
    best = None
    for rn, rb in blk.items():
        if not (rn.startswith('result') and isinstance(rb, dict)):
            continue
        n = max([v.get('n', 0) for v in rb.values()
                 if isinstance(v, dict) and isinstance(v.get('n'), int)] or [0])
        if best is None or n > best[0]:
            best = (n, rn, rb)
    return best


def main() -> int:
    ptt = load_power()
    sc = json.loads(io.open(SCORING, encoding='utf-8').read())
    reach_ok, reach_out = delivery_ok()

    print('프롬프트로 움직일 수 있는 지표가 어느 것인가')
    print()
    print('① 도달 — 정책이 모델에게 닿는가 :  %s'
          % ('전 정책 통과' if reach_ok else '**막힌 정책이 있다**'))
    if not reach_ok:
        print(reach_out)
    print()
    print('%-14s %-8s %-5s %-9s %-11s %-9s %s'
          % ('정책', '지표', '정답', '검출', '여지', '런 이동', '판정'))
    print('-' * 104)

    tally = {}
    for name, key in POLICIES:
        blk = sc.get(key) or {}
        inds = [i for i in (blk.get('indicators') or []) if i.get('id')]
        best = latest_result(blk)
        rb = best[2] if best else {}
        for i in inds:
            iid, expect = i['id'], i.get('expect')
            if expect == 'info':
                continue
            v = rb.get(iid) if isinstance(rb, dict) else None
            v = v if isinstance(v, dict) else {}

            # ② 정답 — 숫자가 있는가
            t = ptt.truth_pct(i.get('desc'), expect)
            if expect == 'rank':
                ans, ans_ok = '순위', True
            elif t is None:
                ans, ans_ok = '**없다**', False
            else:
                ans, ans_ok = '%+.1f%%' % t, True

            # ③ 검출 — 지금 n 으로 0 과 가를 수 있는가
            ci, n, base = v.get('ci'), v.get('n'), v.get('base')
            det, det_ok = '자료없음', None
            if isinstance(ci, list) and len(ci) == 2 and isinstance(n, int) and n > 1:
                sd = (ci[1] - ci[0]) / 3.919928 * math.sqrt(n)
                d = abs(t * base / 100.0) if (t is not None and base) else abs(v.get('mean') or 0)
                if d > 0:
                    need = (ptt.Z * sd / d) ** 2
                    det_ok = need <= n
                    det = 'n=%d 충분' % n if det_ok else '**%.1f배**' % (need / n)

            # ④ 여지 — 구조적으로 움직일 폭
            room, room_ok = '—', None
            if isinstance(base, (int, float)) and 0 < base <= 1.0:
                room_ok = CEILING_LO <= base <= CEILING_HI
                room = 'base %.3f%s' % (base, '' if room_ok else ' **좁다**')

            # ⑤ 런 이동 — 후보 간 차이가 그 안에 들어가면 프롬프트의 성질이 아니다.
            #    **검출력이 충분해도 여기서 막힌다.** P012-1 이 그 예다 — 정답지
            #    크기(+20.8%)는 n=497 로 충분히 검출되지만, 두 후보가 벌린 차이
            #    1.3%p 는 같은 프롬프트의 런 이동 8.2%p 안에 완전히 들어간다.
            shift, shift_txt = RUN_SHIFT.get(iid, (None, '—'))
            cand = CAND_DIFF.get(iid)
            shift_blocks = (shift is not None and cand is not None and cand <= shift)

            # 판정 — 막힌 것부터 적는다
            if not ans_ok:
                verdict = '수치 없음 — 부호만'
            elif det_ok is False:
                verdict = '표본이 막는다'
            elif shift_blocks:
                verdict = '런 이동이 막는다'
            elif room_ok is False:
                verdict = '구조가 막는다'
            elif det_ok is None:
                verdict = '아직 안 재봤다'
            else:
                verdict = '**여기서 프롬프트가 읽힌다**'
            tally[verdict] = tally.get(verdict, 0) + 1

            print('%-14s %-8s %-5s %-9s %-11s %-9s %s'
                  % (name, iid, ans, det, room, shift_txt, verdict))

    print()
    for k, c in sorted(tally.items(), key=lambda x: -x[1]):
        print('  %-24s %d' % (k, c))
    print()
    print('* P012-1 의 런 이동은 n 이 200→500 으로 함께 바뀌어 표본 효과가 섞여 있다.')
    print('"수치 없음" 의 이유는 정책마다 다르다 — P014 는 원문이 방향만 말하고,')
    print('P015 는 개봉했으나 백분율이 공개되어 있지 않다. 둘 다 부호로만 채점한다.')
    print('읽는 법 — 후보 간 차이가 런 이동보다 작으면 프롬프트의 성질로 적지 않는다.')
    print('이 표는 어느 프롬프트가 낫다를 말하지 않는다. 어디서 재야 읽히는가를 말한다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
