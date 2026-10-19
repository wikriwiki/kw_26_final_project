#!/usr/bin/env python3
"""소득 라벨 정합화 — 페르소나의 p_income_level을 지급액 배정 변수(소비분위)와 일치시킨다.

배경: 지원금 tier는 spending_level_wd(소비 10분위)로 배정하는데, 페르소나의 p_income_level은
그와 약하게만 상관된 별개 속성이었다. 그래서 40만원(최하위 소비분위) 수령자의 31.4%가 스스로를
'중~상'으로 인지해 프롬프트의 형편 서술이 어긋났다. 원값은 p_income_level_raw에 보존한다.

이것은 **내부 일관성 교정**이며 어떤 실측 결과에도 맞추지 않는다.
(폐기: 예전 calib_anchor.py의 s_daily_wd 멱변환은 계층별 소진율 갭을 줄이려고 지수를 고른
 것이어서 검증 대상에 맞춘 튜닝이었다. 재작성하지 않는다.)
"""
import sys
sys.path.insert(0, "/data/exp001_repo/scripts/neo4j_load")
from _common import driver_session

LABELS = ["하", "중하", "중", "중상", "상"]   # 소비 10분위 → 5구간

with driver_session() as s:
    r = s.run("MATCH (a:Agent) WHERE a.p_income_level_raw IS NULL "
              "SET a.p_income_level_raw = a.p_income_level RETURN count(*) AS n").single()
    print(f"  원값 보존(p_income_level_raw): {r['n']:,}")
    r = s.run("""
        MATCH (a:Agent) WHERE a.spending_level_wd IS NOT NULL
        WITH a, toInteger(a.spending_level_wd) AS d
        SET a.p_income_level = $labels[ CASE
              WHEN d <= 2 THEN 0 WHEN d <= 4 THEN 1 WHEN d <= 6 THEN 2
              WHEN d <= 8 THEN 3 ELSE 4 END ]
        RETURN count(*) AS n""", labels=LABELS).single()
    print(f"  라벨 정합화: {r['n']:,}")
    print("  결과 분포:")
    for x in s.run("MATCH (a:Agent) RETURN a.p_income_level AS lv, count(*) AS n ORDER BY n DESC"):
        print(f"    {x['lv']}: {x['n']:,}")
