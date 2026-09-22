#!/usr/bin/env python3
"""소비 앵커(s_daily_wd) 규격 교정 — 순위보존 재배치.

배경: `scripts/bdc/generate_agents.py`가 페르소나 생성 시 명시한 분위 내 샘플링 규칙을
LLM이 지키지 않았다(측정: 1분위 중위 11,800원, 규격 하단 25,774원의 46%. 2분위 최소 1,200원으로
분위 경계 42,502원을 벗어남. 10분위 최소 38,000원으로 경계 188,330원 미달).

명시된 규격(generate_agents.py 규칙 3):
  · 1분위 : 하한은 극단 outlier이므로 무시하고 범위의 **상단 60~95%**
  · 10분위: 상한은 극단 고소비자이므로 무시하고 범위의 **하단 5~40%**
  · 2~9분위: 범위 중앙 근처

이 스크립트는 **우리 문서에 적힌 그 규격 안으로 되돌릴 뿐**이며, 분위 내 순위를 보존한다.
어떤 실측 결과에도 맞추지 않는다(폐기한 calib_anchor.py의 멱변환은 계층별 소진 갭을 목표로
지수를 골랐던 것이라 성격이 다르다).

주말치(s_daily_we)는 개인별 주말/평일 비를 보존하도록 같은 배율로 옮긴다.
"""
import sys, json
sys.path.insert(0, "/data/exp001_repo/scripts/neo4j_load")
from _common import driver_session

BOUND = json.load(open("/data/exp001_repo/output/stats/decile_boundaries.json", encoding="utf-8")) \
    if False else None   # 경계는 아래 상수로 고정(런타임 파일 의존 제거)

# decile_boundaries.json 의 weekday_spending_level.boundaries (1인·일당 평균 소비, 원)
B = {1:(682,42502), 2:(42502,56746), 3:(56746,69902), 4:(69902,81591), 5:(81591,94774),
     6:(94774,109078), 7:(109078,125633), 8:(125633,148609), 9:(148609,188330), 10:(188330,798508)}
# 규격 대역 (분위별 사용 구간)
BAND = {1:(0.60,0.95), 10:(0.05,0.40)}
DEFAULT_BAND = (0.25,0.75)

with driver_session() as s:
    rows = [(r["id"], int(r["d"]), float(r["wd"]), (float(r["we"]) if r["we"] is not None else None))
            for r in s.run("MATCH (a:Agent) WHERE a.spending_level_wd IS NOT NULL "
                           "AND a.s_daily_wd IS NOT NULL "
                           "RETURN a.id AS id, a.spending_level_wd AS d, a.s_daily_wd AS wd, "
                           "a.s_daily_we AS we")]
    print(f"  대상 {len(rows):,}명")
    by = {}
    for aid, d, wd, we in rows:
        by.setdefault(d, []).append((aid, wd, we))

    updates = []
    for d in sorted(by):
        lo, hi = B[d]
        f0, f1 = BAND.get(d, DEFAULT_BAND)
        tlo, thi = lo + f0*(hi-lo), lo + f1*(hi-lo)
        grp = sorted(by[d], key=lambda x: x[1])          # 순위 보존
        n = len(grp)
        for i, (aid, wd, we) in enumerate(grp):
            q = (i + 0.5) / n                            # 분위 내 상대 순위
            new = tlo + q * (thi - tlo)
            ratio = (new / wd) if wd > 0 else 1.0
            updates.append({"id": aid, "wd": int(round(new)),
                            "we": (int(round(we * ratio)) if we else None)})
        print(f"   {d:>2}분위 n={n:>5}  {grp[0][1]:>9,.0f}~{grp[-1][1]:>9,.0f}"
              f"  →  {tlo:>9,.0f}~{thi:>9,.0f}")

    # 원값 1회 보존 후 갱신
    s.run("MATCH (a:Agent) WHERE a.s_daily_wd_raw IS NULL AND a.s_daily_wd IS NOT NULL "
          "SET a.s_daily_wd_raw = a.s_daily_wd, a.s_daily_we_raw = a.s_daily_we")
    for i in range(0, len(updates), 1000):
        s.run("UNWIND $rows AS r MATCH (a:Agent {id: r.id}) "
              "SET a.s_daily_wd = r.wd, a.s_daily_we = coalesce(r.we, a.s_daily_we), "
              "    a.spending_daily_wd = r.wd",
              rows=updates[i:i+1000])
    r = s.run("MATCH (a:Agent) WHERE a.s_daily_wd IS NOT NULL "
              "RETURN avg(a.s_daily_wd) AS m, percentileCont(a.s_daily_wd,0.5) AS med").single()
    print(f"  교정 후 전체: 평균 {r['m']:,.0f}원  중위 {r['med']:,.0f}원")
