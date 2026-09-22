#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BOK 이슈노트 2026-13 대조 스코어카드 — 결제원장(events.jsonl) 기반.

Neo4j 그래프 없이 산출한다. 런이 끝나면 다음 런이 덤프를 재적재하며 그래프를 덮으므로,
검증은 내보낸 원장으로 재현 가능해야 한다(FINAL 그래프를 잃고 얻은 교훈).

실측값은 **판정에만** 쓴다 — 시뮬 입력으로 들어가지 않는다.

사용:  python scripts/analysis/scorecard_ledger.py [RUN=FINAL]
"""
import sys, json, glob, statistics as st
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from item_map import to_item, to_t3           # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
ARCH = ROOT / "output" / "exp001_archive"
RUN = sys.argv[1] if len(sys.argv) > 1 else "FINAL"

# ── 실측 기준값 (BOK 2026-13 / BDC) ────────────────────────────────────────
REF_DEP_DAY = 2.73                                        # 표1 4주 76.4% → 일평균
REF_T3 = {"음식점": 46.0, "마트식료품": 21.6, "의료": 6.2,
          "미용": 5.1, "학원": 4.0, "약국": 3.9}           # 표3 1차
REF_MPC = 0.21                                            # 30항 1차
REF_MPC_Q = {1: .27, 2: .245, 3: .185, 4: .21, 5: .165}   # 31항·그림18 ◆
REF_SHARE = {"비내구재": .445, "숙박음식": .27, "병원": .09, "준내구재": .04,
             "내구재": .035, "개인미용": .035, "학원": .035, "여가": .02}  # 그림19 ▲
REF_ITEM_MPC = {"내구재": .48, "준내구재": .43, "여가": .41, "개인미용": .35,
                "숙박음식": .29, "비내구재": .13, "병원": .09, "학원": .04}  # 그림19 막대
REF_ELIG_POI = 96.2                                       # 표6 사용처 사업자 비중
REF_WE_WD = 0.735                                         # BDC 동별 주말/평일 중앙값

TALLY = {"OK": 0, "근접": 0, "NG": 0}


def verdict(obs, ref, tol=.25):
    if ref == 0:
        v = "OK" if obs == 0 else "NG"
    else:
        d = abs(obs - ref) / ref
        v = "OK" if d <= tol else ("근접" if d <= tol * 2 else "NG")
    TALLY[v] += 1
    return f"[{v}]"


def main():
    # ── 지표 A·C : metrics jsonl ───────────────────────────────────────────
    files = sorted(glob.glob(str(ARCH / f"out_{RUN}" / "metrics" / "day_*.jsonl")))
    if not files:
        sys.exit(f"{RUN}: metrics 없음")

    g0, dep, mpc, mq = {}, defaultdict(list), [], defaultdict(list)
    n_agent_day = 0
    usedays = defaultdict(int)
    for idx, f in enumerate(files):
        for line in open(f, encoding="utf-8"):
            d = json.loads(line)
            if d.get("status") != "ok":
                continue
            aid = d.get("agent_id") or d.get("aid")
            if idx == 0:
                w = int(d.get("cm_policy_wallet_available") or 0) or \
                    int(d.get("cm_intended_grant_today") or 0)
                if w > 0:
                    g0[aid] = ("40만" if w >= 350000 else
                               "30만" if w >= 250000 else "15만", w)
            if aid not in g0:
                continue
            tier, w = g0[aid]
            alloc = int(d.get("cm_policy_allocated_total") or 0)
            dep[tier].append(100 * alloc / w)
            n_agent_day += 1
            if alloc > 0:
                usedays[aid] += 1
                v = d.get("cm_mpc_new_share")
                if v is not None:
                    mpc.append((float(v), alloc))
                    dec = d.get("spend_decile")
                    if dec:
                        mq[min(5, max(1, (int(dec) + 1) // 2))].append((float(v), alloc))

    WT = {"40만": 400000, "30만": 300000, "15만": 150000}
    dep_all = (sum(st.mean(dep[t]) * len(dep[t]) * WT[t] for t in dep) /
               sum(len(dep[t]) * WT[t] for t in dep)) if dep else 0.0

    print(f"\n{'='*74}\n  {RUN} — {len(files)}일 · 에이전트 {len(g0)}명"
          f" · 원장 기반\n{'='*74}")

    print("\n[A] 소진 궤적 (표1)")
    print(f"  전체 일간 소진     {dep_all:5.2f}%/일   (실측 {REF_DEP_DAY}) "
          f"{verdict(dep_all, REF_DEP_DAY)}")
    for t in ("40만", "30만", "15만"):
        if t in dep:
            print(f"    {t} tier       {st.mean(dep[t]):5.2f}%/일   (n={len(dep[t])})")

    print("\n[C] MPC — 쿠폰 사용액 중 신규 소비 비율 (30·31항, 그림18)")
    vals = {}
    if mpc:
        tw = sum(w for _, w in mpc)
        m = sum(x * w for x, w in mpc) / tw
        print(f"  전체 MPC          {m:.3f}      (실측 {REF_MPC}) {verdict(m, REF_MPC)}")
        for q in sorted(mq):
            t2 = sum(w for _, w in mq[q])
            o = sum(x * w for x, w in mq[q]) / t2
            vals[q] = o
            print(f"    {q}분위           {o:.3f}      (실측 {REF_MPC_Q[q]}) "
                  f"{verdict(o, REF_MPC_Q[q])}")
        if 1 in vals and 5 in vals:
            print(f"    → 1−5분위 격차  {vals[1]-vals[5]:+.3f}   (실측 +0.105)")

    # ── 결제원장 ──────────────────────────────────────────────────────────
    t3 = defaultdict(float); sh = defaultdict(float)
    imp = defaultdict(lambda: [0.0, 0.0])
    C = 0.0; BAD = 0; n_com = 0; amts = []; over = 0
    wewd = defaultdict(lambda: [0.0, set()])
    for line in open(ARCH / f"{RUN}_events.jsonl", encoding="utf-8"):
        e = json.loads(line)
        amt = float(e.get("amt") or 0)
        if amt <= 0:
            continue
        n_com += 1; amts.append(amt)
        dt = e.get("day_type") or "?"
        wewd[dt][0] += amt; wewd[dt][1].add(e.get("day"))
        cpn = 0.0
        sp = e.get("sp")
        if sp and sp not in ("{}", "null"):
            try:
                cpn = float(sum(json.loads(sp).values()))
            except Exception:
                cpn = 0.0
        cpn = max(0.0, min(amt, cpn))
        if cpn <= 0:
            continue
        C += cpn
        if e.get("elig") is False:
            BAD += 1
        ex = e.get("ex")
        if ex is None:
            ex = 0.0 if e.get("wba") else amt        # 구버전 폴백(참/거짓)
        ex = float(ex)
        if ex > amt:
            over += 1
        ex = max(0.0, min(amt, ex)) * (cpn / amt)    # 정책 결제분만큼 안분
        k = to_t3(e.get("l1"), e.get("sub"))
        if k:
            t3[k] += cpn
        it = to_item(e.get("l1"), e.get("sub"))
        sh[it] += cpn
        imp[it][0] += cpn; imp[it][1] += ex

    print("\n[B] 업종별 사용 비중 (표3 1차)")
    for k, ref in REF_T3.items():
        o = 100 * t3.get(k, 0) / C if C else 0
        print(f"  {k:<9} {o:6.1f}%    (실측 {ref:5.1f}) {verdict(o, ref, .20)}")

    print("\n[D1] 품목별 사용 비중 (그림19 ▲)")
    for k, ref in sorted(REF_SHARE.items(), key=lambda z: -z[1]):
        o = sh.get(k, 0) / C if C else 0
        print(f"  {k:<9} {o:6.3f}     (실측 {ref:.3f}) {verdict(o, ref, .30)}")

    print("\n[D2] 품목별 MPC (그림19 막대)")
    obs = []
    for k, ref in sorted(REF_ITEM_MPC.items(), key=lambda z: -z[1]):
        c, x = imp[k]
        o = x / c if c > 0 else 0.0
        obs.append((k, o, c))
        tag = verdict(o, ref, .35) if c > 0 else "[자료없음]"
        print(f"  {k:<9} {o:6.3f}     (실측 {ref:.2f}) {tag}")
    rank = [k for k, _, c in sorted([z for z in obs if z[2] > 0], key=lambda z: -z[1])]
    refr = [k for k, _ in sorted(REF_ITEM_MPC.items(), key=lambda z: -z[1]) if k in rank]
    hit = sum(1 for a, b in zip(rank, refr) if a == b)
    print(f"  순서 일치 {hit}/{len(rank)}    우리 순서: {' > '.join(rank[:4])}")

    print("\n[E·G·F] 사용처 · 미시 지표")
    ok_rate = 100 * (1 - BAD / max(1, n_com))
    TALLY["OK" if BAD == 0 else "NG"] += 1
    print(f"  E  사용처 준수율    {ok_rate:6.2f}%    (실측 100) "
          f"{'[OK]' if BAD == 0 else '[NG]'}")
    ps = json.load(open(ARCH / f"{RUN}_poi_summary.json", encoding="utf-8"))
    poi = 100 * ps["poi_eligible"] / ps["poi_total"]
    print(f"  G1 POI 사용처 비중  {poi:6.1f}%    (실측 {REF_ELIG_POI}) "
          f"{verdict(poi, REF_ELIG_POI, .05)}")
    print(f"  G2 1인 누적 사용액  {C/max(1,len(g0)):,.0f}원 / {len(files)}일")
    if usedays:
        u = list(usedays.values())
        print(f"  G3 쿠폰 사용 일수   {st.mean(u):.2f}일 / {len(files)}일 "
              f"(사용자 {len(u)}명, 미사용 {len(g0)-len(u)}명)")
    if "weekend" in wewd and "weekday" in wewd:
        we = wewd["weekend"][0] / max(1, len(wewd["weekend"][1]))
        wd = wewd["weekday"][0] / max(1, len(wewd["weekday"][1]))
        r_ = we / wd if wd else 0
        print(f"  F4 주말/평일 소비비 {r_:6.3f}     (BDC {REF_WE_WD}) "
              f"{verdict(r_, REF_WE_WD, .15)}")
    print(f"  F1 결제 건당 금액   {st.median(amts):,.0f}원 (중앙값) — 기술통계")
    print(f"  F2 1인 1일 결제건수 {n_com/max(1,n_agent_day):.2f}건 — 기술통계")
    if over:
        print(f"  ※ extra_spent가 결제액 초과 {over}건 — 상한 적용 후 집계")

    tot = sum(TALLY.values())
    print(f"\n{'='*74}\n  판정  [OK] {TALLY['OK']} · [근접] {TALLY['근접']} · "
          f"[NG] {TALLY['NG']}   (대조 {tot}개)\n{'='*74}")


if __name__ == "__main__":
    main()
