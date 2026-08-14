#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""정책런 vs 무정책 baseline 대조 — 자기보고 MPC의 독립 검증.

    실현 MPC = (정책런 소비 − baseline 소비) / 쿠폰 사용액

이 값이 에이전트가 스스로 답한 MPC(cm_mpc_new_share)와 맞으면,
"말만 그렇게 한 것이 아니라 실제로 더 썼다"가 증명된다.
BOK의 MPC도 서베이 자기보고이므로 방법론은 동일하고, 이 대조는 추가 증거다.

사용:  python3 compare_baseline.py <정책런> <baseline런> [비교일수]
"""
import sys, json, glob, statistics as st
from collections import defaultdict

POL = sys.argv[1] if len(sys.argv) > 1 else 'FINAL'
BAS = sys.argv[2] if len(sys.argv) > 2 else 'BASE'
NDAY = int(sys.argv[3]) if len(sys.argv) > 3 else 7


def load(run, nday):
    """일자별 → {agent: (총소비, 쿠폰사용액, 자기보고MPC)}"""
    out = []
    for f in sorted(glob.glob(f'/data/exp001/out_{run}/metrics/day_*.jsonl'))[:nday]:
        day = {}
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            if d.get('status') != 'ok':
                continue
            aid = d.get('agent_id') or d.get('aid')
            tot = int(d.get('cm_today_total_incl_online') or
                      d.get('cm_today_total') or 0)
            cpn = int(d.get('cm_policy_allocated_total') or 0)
            mpc = d.get('cm_mpc_new_share')
            day[aid] = (tot, cpn, mpc)
        out.append((f.split('day_')[1][:-6], day))
    return out


pol, bas = load(POL, NDAY), load(BAS, NDAY)
if not bas:
    sys.exit(f"{BAS}: 완료 일자 없음")
n = min(len(pol), len(bas))
pol, bas = pol[:n], bas[:n]

# 두 런에 모두 등장한 에이전트만 (매칭 표본)
common = set(pol[0][1]) & set(bas[0][1])
for _, d in pol + bas:
    common &= set(d)

print(f"\n{'='*70}\n  {POL}(정책) vs {BAS}(무정책) — {n}일 · 매칭 {len(common)}명\n{'='*70}")
print(f"\n  {'일자':<12}{'정책런':>11}{'baseline':>11}{'차이':>10}{'쿠폰':>10}{'일별MPC':>9}")

P = B = C = 0
for i in range(n):
    dp = sum(pol[i][1][a][0] for a in common)
    db = sum(bas[i][1][a][0] for a in common)
    dc = sum(pol[i][1][a][1] for a in common)
    P += dp; B += db; C += dc
    r = (dp - db) / dc if dc > 0 else float('nan')
    print(f"  {pol[i][0]:<12}{dp/len(common):>11,.0f}{db/len(common):>11,.0f}"
          f"{(dp-db)/len(common):>10,.0f}{dc/len(common):>10,.0f}{r:>9.3f}")

print(f"  {'─'*61}")
print(f"  {'합계(1인)':<12}{P/len(common):>11,.0f}{B/len(common):>11,.0f}"
      f"{(P-B)/len(common):>10,.0f}{C/len(common):>10,.0f}")

# 자기보고 MPC (금액가중)
w = [(m, c) for _, d in pol for (_, c, m) in d.values() if m is not None and c > 0]
self_mpc = sum(m * c for m, c in w) / sum(c for _, c in w) if w else float('nan')
real_mpc = (P - B) / C if C > 0 else float('nan')

# 에이전트별 짝지은 차이 → 실현 MPC의 신뢰구간
# 점추정만 비교하면 안 된다. 하루하루 소비 변동이 정책 효과보다 훨씬 커서,
# 표본이 작으면 실현 MPC의 구간이 [0,1]을 통째로 덮는다.
diffs = [sum(pol[i][1][a][0] for i in range(n)) - sum(bas[i][1][a][0] for i in range(n))
         for a in common]
m_ = st.mean(diffs)
s_ = st.stdev(diffs) if len(diffs) > 1 else 0.0
se = s_ / (len(diffs) ** .5) if len(diffs) > 1 else 0.0
cpa = C / len(common)                      # 1인당 쿠폰 사용액
lo, hi = (m_ - 1.96 * se) / cpa, (m_ + 1.96 * se) / cpa

print(f"\n  자기보고 MPC   {self_mpc:.3f}   (에이전트가 답한 extra_spent 기준)")
print(f"  실현    MPC   {real_mpc:.3f}   (정책런 소비 − baseline 소비) / 쿠폰 사용액")
print(f"                 95% 신뢰구간 [{lo:+.3f}, {hi:+.3f}]")

if lo <= self_mpc <= hi:
    if hi - lo > 0.30:
        tag = ("검정력 부족 — 구간이 자기보고값을 포함하나 너무 넓어 "
               "일치/불일치를 판정할 수 없다")
    else:
        tag = "일치 — 자기보고값이 실현값의 신뢰구간 안에 있다"
else:
    tag = "불일치 — 자기보고값이 실현값의 신뢰구간 밖이다"
print(f"  판정          {tag}")
print(f"\n  참고: BOK 실측 MPC 0.21 (1차, 30항) — "
      f"구간 {'포함' if lo <= 0.21 <= hi else '미포함'}")

print(f"\n  1인당 소비 증가 {m_:,.0f}원 / {n}일   표준오차 {se:,.0f}"
      + (f"   t={m_/se:.2f}" if se > 0 else ""))
if se > 0 and abs(m_ / se) < 1.96:
    need = int(len(common) * (1.96 * se / m_) ** 2) if m_ else 0
    print(f"  → 5% 유의수준 미달. 같은 효과크기라면 약 {need:,}명이 필요하다"
          f"(현재 {len(common)}명).")
print(f"{'='*70}")
