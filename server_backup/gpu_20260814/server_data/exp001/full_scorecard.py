#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOK 이슈노트 2026-13 대조 + 미시 행동 검증 통합 스코어카드.

사용:  python3 full_scorecard.py <RUN_NAME>
실측값은 **판정에만** 쓰인다(시뮬 입력으로 들어가지 않는다).
"""
import sys, json, glob, subprocess, statistics as st
from collections import defaultdict

sys.path.insert(0, '/data/exp001_repo/scripts/neo4j_load')
sys.path.insert(0, '/data/exp001')
from _common import driver_session          # noqa: E402
from item_map import to_item, to_t3         # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else 'T10'
OUT = f'/data/exp001/out_{RUN}'

# ── 실측 기준값 (BOK 2026-13 / BDC) ────────────────────────────────────────
REF_DEP_DAY = 2.73                                       # 표1 4주 76.4% → 일평균
REF_T3 = {"음식점": 46.0, "마트식료품": 21.6, "의료": 6.2,
          "미용": 5.1, "학원": 4.0, "약국": 3.9}          # 표3 1차
REF_MPC = 0.21                                            # 30항 1차
REF_MPC_Q = {1: .27, 2: .245, 3: .185, 4: .21, 5: .165}   # 31항·그림18 ◆
REF_SHARE = {"비내구재": .445, "숙박음식": .27, "병원": .09, "준내구재": .04,
             "내구재": .035, "개인미용": .035, "학원": .035, "여가": .02}  # 그림19 ▲
REF_ITEM_MPC = {"내구재": .48, "준내구재": .43, "여가": .41, "개인미용": .35,
                "숙박음식": .29, "비내구재": .13, "병원": .09, "학원": .04}  # 그림19 막대
REF_ELIG_POI = 96.2                                       # 표6 사용처 사업자 비중
REF_WE_WD = 0.735                                         # BDC 동별 주말/평일 중앙값

USED = ("i.spent_from_policy IS NOT NULL AND i.spent_from_policy <> '{}' "
        "AND i.spent_from_policy <> 'null'")


def verdict(obs, ref, tol=.25):
    if ref == 0:
        return "[OK]" if obs == 0 else "[NG]"
    d = abs(obs - ref) / ref
    return "[OK]" if d <= tol else ("[근접]" if d <= tol * 2 else "[NG]")


def band(obs, lo, hi):
    if lo <= obs <= hi:
        return "[OK]"
    w = (hi - lo) * .5
    return "[근접]" if (lo - w) <= obs <= (hi + w) else "[NG]"


def main():
    files = sorted(glob.glob(f'{OUT}/metrics/day_*.jsonl'))
    live = subprocess.run("pgrep -f run_simulation.py", shell=True,
                          capture_output=True).returncode == 0
    if live and files:
        files = files[:-1]          # 진행 중 파일 제외
    if not files:
        print(f"  {RUN}: 완료 일자 없음"); return
    days = [f.split('day_')[1][:-6] for f in files]

    # ── 지표 A·C: metrics jsonl ────────────────────────────────────────────
    g0, dep, mpc, mq = {}, defaultdict(list), [], defaultdict(list)
    n_ev, n_agent_day, amt_ev = 0, 0, []
    usedays = defaultdict(int)
    for idx, f in enumerate(files):
        for line in open(f, encoding='utf-8'):
            d = json.loads(line)
            if d.get('status') != 'ok':
                continue
            aid = d.get('agent_id') or d.get('aid')
            if idx == 0:
                w = int(d.get('cm_policy_wallet_available') or 0) or \
                    int(d.get('cm_intended_grant_today') or 0)
                if w > 0:
                    g0[aid] = ("40만" if w >= 350000 else
                               "30만" if w >= 250000 else "15만", w)
            if aid not in g0:
                continue
            tier, w = g0[aid]
            alloc = int(d.get('cm_policy_allocated_total') or 0)
            dep[tier].append(100 * alloc / w)
            n_agent_day += 1
            ne = int(d.get('n_includes') or d.get('n_events') or 0)
            n_ev += ne
            tot = int(d.get('cm_today_total') or 0)
            if ne > 0 and tot > 0:
                amt_ev.append(tot / ne)
            if alloc > 0:
                usedays[aid] += 1
                v = d.get('cm_mpc_new_share')
                if v is not None:
                    mpc.append((float(v), alloc))
                    dec = d.get('spend_decile')
                    if dec:
                        mq[min(5, max(1, (int(dec) + 1) // 2))].append((float(v), alloc))

    WT = {'40만': 400000, '30만': 300000, '15만': 150000}
    dep_all = (sum(st.mean(dep[t]) * len(dep[t]) * WT[t] for t in dep) /
               sum(len(dep[t]) * WT[t] for t in dep)) if dep else 0.0

    ok = near = ng = 0

    def score(v):
        nonlocal ok, near, ng
        ok += v == "[OK]"; near += v == "[근접]"; ng += v == "[NG]"
        return v

    print(f"\n{'='*72}\n  {RUN} — {len(files)}일 · 에이전트 {len(g0)}명"
          f"{'  (진행 중)' if live else ''}\n{'='*72}")

    print("\n[A] 소진 궤적")
    v = score(verdict(dep_all, REF_DEP_DAY))
    print(f"  전체 일간 소진   {dep_all:5.2f}%/일   (실측 {REF_DEP_DAY}) {v}")
    for t in ('40만', '30만', '15만'):
        if t in dep:
            print(f"    {t} tier      {st.mean(dep[t]):5.2f}%/일   (n={len(dep[t])})")

    print("\n[C] MPC — 쿠폰 사용액 중 신규 소비 비율")
    if mpc:
        tw = sum(w for _, w in mpc)
        m = sum(x * w for x, w in mpc) / tw
        v = score(verdict(m, REF_MPC))
        print(f"  전체 MPC         {m:.3f}      (실측 {REF_MPC}) {v}")
        vals = {}
        for q in sorted(mq):
            t2 = sum(w for _, w in mq[q])
            o = sum(x * w for x, w in mq[q]) / t2
            vals[q] = o
            v = score(verdict(o, REF_MPC_Q[q]))
            print(f"    {q}분위          {o:.3f}      (실측 {REF_MPC_Q[q]}) {v}")
        if 1 in vals and 5 in vals:
            print(f"    → 1−5분위 격차 {vals[1]-vals[5]:+.3f}  (실측 +0.105)")
    else:
        print("  측정 불가"); ng += 6

    # ── 그래프 기반 지표 ──────────────────────────────────────────────────
    # 건별로 받아 파이썬에서 계산한다. spent_from_policy는 JSON 문자열이고,
    # extra_spent는 결제액을 넘겨 적히는 경우가 있어(모델 오차) 상한을 씌워야 한다.
    with driver_session() as s:
        evs = list(s.run("""
            UNWIND $d AS dy MATCH (:Plan {day: date(dy)})-[i:INCLUDES]->()
            WHERE coalesce(i.actual_spent,0) > 0
            RETURN i.category AS l1, i.sub_category AS sub,
                   i.actual_spent AS amt, i.spent_from_policy AS sp,
                   i.extra_spent AS ex, i.would_buy_anyway AS wba,
                   i.coupon_eligible AS elig""", d=days))
        poi = s.run("""MATCH (p:POI) RETURN
                       100.0*sum(CASE WHEN p.coupon_eligible THEN 1 ELSE 0 END)
                       /count(p) AS pct""").single()['pct']
        wewd = list(s.run("""
            UNWIND $d AS dy MATCH (pl:Plan {day: date(dy)})-[i:INCLUDES]->()
            WHERE coalesce(i.actual_spent,0) > 0
            RETURN pl.day_type AS dt, sum(i.actual_spent) AS amt,
                   count(DISTINCT pl) AS np""", d=days))

    t3 = defaultdict(float); sh = defaultdict(float)
    imp = defaultdict(lambda: [0.0, 0.0])
    T = C = X = 0.0; BAD = 0; n_com = 0; amts = []
    over = 0
    for r in evs:
        amt = float(r['amt'] or 0)
        if amt <= 0:
            continue
        n_com += 1; amts.append(amt); T += amt
        cpn = 0.0
        sp = r['sp']
        if sp and sp not in ('{}', 'null'):
            try:
                cpn = float(sum(json.loads(sp).values()))
            except Exception:
                cpn = 0.0
        cpn = max(0.0, min(amt, cpn))
        if cpn <= 0:
            continue
        C += cpn
        if r['elig'] is False:
            BAD += 1
        ex = r['ex']
        if ex is None:
            ex = 0.0 if r['wba'] else amt          # 구버전 폴백(참/거짓)
        ex = float(ex)
        if ex > amt:
            over += 1
        ex = max(0.0, min(amt, ex)) * (cpn / amt)  # 정책 결제분만큼 안분
        X += ex
        k = to_t3(r['l1'], r['sub'])
        if k:
            t3[k] += cpn
        it = to_item(r['l1'], r['sub'])
        sh[it] += cpn
        imp[it][0] += cpn; imp[it][1] += ex

    print("\n[B] 업종별 사용 비중 (표3 1차)")
    for k, ref in REF_T3.items():
        o = 100 * t3.get(k, 0) / C if C else 0
        print(f"  {k:<8} {o:6.1f}%   (실측 {ref:5.1f}) {score(verdict(o, ref, .20))}")

    print("\n[D1] 품목별 사용 비중 (그림19 ▲)")
    for k, ref in sorted(REF_SHARE.items(), key=lambda z: -z[1]):
        o = sh.get(k, 0) / C if C else 0
        print(f"  {k:<8} {o:6.3f}    (실측 {ref:.3f}) {score(verdict(o, ref, .30))}")

    print("\n[D2] 품목별 MPC (그림19 막대)")
    obs = []
    for k, ref in sorted(REF_ITEM_MPC.items(), key=lambda z: -z[1]):
        c, x = imp[k]
        o = x / c if c > 0 else 0.0
        obs.append((k, o))
        tag = score(verdict(o, ref, .35)) if c > 0 else "[자료없음]"
        print(f"  {k:<8} {o:6.3f}    (실측 {ref:.2f}) {tag}")
    have = [(k, o) for k, o in obs if imp[k][0] > 0]
    rank = [k for k, _ in sorted(have, key=lambda z: -z[1])]
    refr = [k for k, _ in sorted(REF_ITEM_MPC.items(), key=lambda z: -z[1]) if k in rank]
    hit = sum(1 for a, b in zip(rank, refr) if a == b)
    print(f"  순서 일치 {hit}/{len(rank)}   우리 순서: {' > '.join(rank[:4])}")

    print("\n[E·G·F] 사용처 · 미시 지표")
    print(f"  E  사용처 준수율   {100*(1-BAD/max(1,n_com)):6.2f}%   (실측 100) "
          f"{score('[OK]' if BAD == 0 else '[NG]')}")
    print(f"  G1 POI 사용처 비중 {poi:6.1f}%   (실측 {REF_ELIG_POI}) "
          f"{score(verdict(poi, REF_ELIG_POI, .05))}")
    print(f"  G2 1인 누적 사용액 {C/max(1,len(g0)):,.0f}원 / {len(files)}일")
    if usedays:
        u = list(usedays.values())
        print(f"  G3 쿠폰 사용 일수  평균 {st.mean(u):.2f}일 / {len(files)}일 "
              f"(사용자 {len(u)}명, 미사용 {len(g0)-len(u)}명)")
    dm = {r['dt']: r for r in wewd}
    if '주말' in dm and '평일' in dm:
        we = dm['주말']['amt'] / max(1, dm['주말']['np'])
        wd = dm['평일']['amt'] / max(1, dm['평일']['np'])
        r_ = we / wd if wd else 0
        print(f"  F4 주말/평일 소비비 {r_:.3f}     (BDC {REF_WE_WD}) "
              f"{score(verdict(r_, REF_WE_WD, .15))}")
    else:
        print("  F4 주말/평일 소비비 — 기간에 주말 없음 (대조 생략)")
    if amts:
        print(f"  F1 결제 건당 금액  {st.median(amts):,.0f}원 (중앙값) — 기술통계")
    print(f"  F2 1인 1일 결제건수 {n_com/max(1,n_agent_day):.2f}건 — 기술통계")
    if over:
        print(f"  ※ extra_spent가 결제액 초과 {over}건 — 상한 적용 후 집계")

    print(f"\n{'='*72}\n  판정  [OK] {ok} · [근접] {near} · [NG] {ng}"
          f"   (대조 {ok+near+ng}개)\n{'='*72}")


if __name__ == '__main__':
    main()
