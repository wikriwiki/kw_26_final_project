#!/usr/bin/env python3
"""업종·품목 대조 — 표3(업종별 사용 비중), 그림19(품목별 MPC·사용 비중), 사용처 준수율.

사용: python3 cat_scorecard.py <날짜...>      (예: 2025-07-21 2025-07-22 ...)
주의: 실측 참조값은 판정 표시용이며 시뮬 입력이 아니다.
"""
import sys
sys.path.insert(0, "/data/exp001_repo/scripts/neo4j_load")
from _common import driver_session

DAYS = sys.argv[1:] or ["2025-07-21"]

REF_T3 = {"음식점": 46.0, "마트식료품": 21.6, "의료": 6.2, "미용": 5.1, "학원": 4.0}
REF_F19_MPC = {"내구재": .48, "준내구재": .43, "여가": .41, "개인미용": .35,
               "숙박음식": .29, "비내구재": .13, "병원": .09, "학원": .04}
REF_F19_SH = {"비내구재": .445, "숙박음식": .27, "병원": .09, "준내구재": .04,
              "내구재": .035, "개인미용": .035, "학원": .035, "여가": .02}
T3 = {"식사": "음식점", "카페": "음식점", "디저트": "음식점", "주점": "음식점",
      "마트": "마트식료품", "편의점": "마트식료품", "건강": "의료", "미용": "미용", "교육": "학원"}
ITEM = {"식사": "숙박음식", "카페": "숙박음식", "디저트": "숙박음식", "주점": "숙박음식",
        "마트": "비내구재", "편의점": "비내구재", "건강": "병원", "미용": "개인미용",
        "교육": "학원", "여가": "여가", "쇼핑": "준내구재"}
USED = ("i.spent_from_policy IS NOT NULL AND i.spent_from_policy <> '{}' "
        "AND i.spent_from_policy <> 'null'")

def vd(o, r, tol=.15):
    if not r: return "—"
    d = abs(o - r) / abs(r)
    return "[OK]" if d <= tol else ("[근접]" if d <= tol * 2 else "[NG]")

with driver_session() as s:
    print(f"═══ 업종·품목 대조 ({len(DAYS)}일) ═══\n")

    print("── A4. 사용처 준수율 (쿠폰이 사용 가능 매장에서만 결제됐는가) ──")
    r = s.run(f"""UNWIND $d AS dy
        MATCH (:Plan {{day: date(dy)}})-[i:INCLUDES]->(p:POI)
        WHERE {USED}
        RETURN count(i) AS n,
               sum(CASE WHEN p.coupon_eligible=false THEN 1 ELSE 0 END) AS bad""", d=DAYS).single()
    if r and r["n"]:
        print(f"  쿠폰 결제 {r['n']:,}건 중 비사용처 {r['bad']}건 → 준수율 "
              f"{100*(1-r['bad']/r['n']):.2f}%  {'[OK]' if r['bad']==0 else '[NG]'}\n")

    rows = list(s.run(f"""UNWIND $d AS dy
        MATCH (:Plan {{day: date(dy)}})-[i:INCLUDES]->(p:POI)
        WHERE {USED} AND coalesce(i.actual_spent,0)>0
        RETURN coalesce(i.category,'기타') AS l1, count(i) AS n,
               sum(i.actual_spent) AS amt,
               sum(CASE WHEN i.would_buy_anyway=false THEN i.actual_spent ELSE 0 END) AS new_amt,
               sum(CASE WHEN i.would_buy_anyway IS NOT NULL THEN i.actual_spent ELSE 0 END) AS flagged
        ORDER BY amt DESC""", d=DAYS))
    tot = sum(x["amt"] for x in rows) or 1

    print("── A3. 표3 업종별 사용 비중 ──")
    agg = {}
    for x in rows:
        k = T3.get(x["l1"])
        if k: agg[k] = agg.get(k, 0) + x["amt"]
    for k, ref in REF_T3.items():
        o = 100 * agg.get(k, 0) / tot
        print(f"  {k:<8} 우리 {o:>5.1f}%  실측 {ref:>5.1f}%  차 {o-ref:>+6.1f}p  {vd(o, ref, .20)}")
    print(f"  (분류 외 {100*(tot-sum(agg.values()))/tot:.1f}%)\n")

    print("── C4. 그림19 ▲ 품목별 사용 비중 ──")
    ia = {}
    for x in rows:
        k = ITEM.get(x["l1"])
        if k: ia[k] = ia.get(k, 0) + x["amt"]
    for k, ref in sorted(REF_F19_SH.items(), key=lambda z: -z[1]):
        o = ia.get(k, 0) / tot
        note = "  ※우리에 전용 카테고리 없음" if k == "내구재" else ""
        print(f"  {k:<7} 우리 {o:>5.3f}  실측 {ref:>5.3f}  {vd(o, ref, .30)}{note}")

    print("\n── C3. 그림19 막대 품목별 MPC (금액가중 신규 비중) ──")
    im = {}
    for x in rows:
        k = ITEM.get(x["l1"])
        if not k or not x["flagged"]: continue
        a, b = im.get(k, (0, 0))
        im[k] = (a + x["new_amt"], b + x["flagged"])
    if not im:
        print("  would_buy_anyway 미기록 — 측정 불가")
    else:
        for k, ref in sorted(REF_F19_MPC.items(), key=lambda z: -z[1]):
            if k not in im: print(f"  {k:<7} 우리 —      실측 {ref:>5.2f}  (해당 지출 없음)"); continue
            n, d = im[k]; o = n / d if d else 0
            print(f"  {k:<7} 우리 {o:>5.3f}  실측 {ref:>5.2f}  {vd(o, ref, .30)}")
        ours_rank = [k for k, _ in sorted(im.items(), key=lambda z: -(z[1][0]/z[1][1] if z[1][1] else 0))]
        ref_rank = [k for k, _ in sorted(REF_F19_MPC.items(), key=lambda z: -z[1]) if k in im]
        print(f"\n  순서 우리: {' > '.join(ours_rank)}")
        print(f"  순서 실측: {' > '.join(ref_rank)}")
print("\n※ 실측 수치는 판정 표시용이며 시뮬 입력이 아니다.")
