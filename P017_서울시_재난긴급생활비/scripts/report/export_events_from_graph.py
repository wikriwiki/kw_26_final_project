#!/usr/bin/env python3
"""Neo4j 그래프에서 **결제 원장**을 뽑아 run 산출물 모양으로 내보낸다.

왜 필요한가
-----------
시뮬레이션 산출물 중 `metrics/day_*.jsonl` 은 **대상자 한 명당 한 줄**이라 업종별
금액이 없다. 업종·정책지급·자기부담이 들어 있는 건 결제 원장뿐이고, 그건 그래프
안에만 있다. 보고서의 업종별 분석·이중차분은 전부 이 원장 위에서 계산된다.

그래서 그래프 덤프만 있고 원장 파일이 없는 run 은 보고서를 만들 수 없다.
이 스크립트가 그 간극을 메운다 — 서버의 `export_run.py` 와 **같은 질의**를 쓴다.

읽기만 한다. 그래프에 아무것도 쓰지 않는다.

사용::

    python scripts/report/export_events_from_graph.py --out <run 디렉터리>
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neo4j import GraphDatabase

#: 서버 `export_run.py` 와 동일. 필드 이름이 달라지면 보고서가 조용히 빈다.
LEDGER = """
MATCH (pl:Plan {day: date($day)})-[i:INCLUDES]->(p:POI)
WHERE coalesce(i.actual_spent, 0) > 0
RETURN pl.day_type AS day_type, i.category AS l1, i.sub_category AS sub,
       i.actual_spent AS amt, i.spent_from_policy AS sp, i.extra_spent AS ex,
       i.would_buy_anyway AS wba, i.coupon_eligible AS elig, p.dong_code AS dong
"""
# 서버 `export_run.py` 는 `p.adm_cd` 를 읽는데 그 속성은 그래프에 없다 — 그래서
# 기존 산출물의 `dong` 은 전부 null 이고 지역 분포 절이 비어 있었다.
# 실제로 채워져 있는 건 `p.dong_code`(8자리 행정동 코드)다. 사전·사후 양쪽을
# 같은 질의로 뽑으므로 비교가 성립한다.

DAYS = "MATCH (pl:Plan) RETURN DISTINCT toString(pl.day) AS day ORDER BY day"

#: 행정동 코드(8자리)의 앞 5자리가 시군구다. 보고서의 지역 분포는 **이름**으로 읽혀야
#: 하므로 여기서 코드를 구 이름으로 바꾼다. 코드만 남기면 화면에 `11110` 이 뜬다.
SEOUL_GU = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


def gu_of(dong_code: object) -> str | None:
    """행정동 코드 → 구 이름. 모르는 코드는 지어내지 않고 None 으로 둔다."""
    code = str(dong_code or "")
    return SEOUL_GU.get(code[:5]) if len(code) >= 5 else None

POI = """
MATCH (p:POI)
RETURN count(p) AS n, sum(CASE WHEN p.coupon_eligible THEN 1 ELSE 0 END) AS elig
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, type=Path, help="run 디렉터리")
    ap.add_argument("--uri", default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    ap.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    ap.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD", "exp001pass"))
    ap.add_argument("--days", default="", help="쉼표로 구분한 일자. 비우면 그래프에 있는 전부")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    written: dict[str, int] = {}
    try:
        with driver.session() as session:
            days = (
                [d.strip() for d in args.days.split(",") if d.strip()]
                or [record["day"] for record in session.run(DAYS)]
            )
            print(f"일자 {len(days)}개: {days[0]} .. {days[-1]}" if days else "일자 없음")
            for day in days:
                path = args.out / f"events_{day}.jsonl"
                # 부분 파일이 완성본으로 오인되지 않게 다 쓴 뒤 바꿔 끼운다
                temp = path.with_suffix(".jsonl.part")
                count = 0
                with temp.open("w", encoding="utf-8", newline="\n") as fp:
                    for record in session.run(LEDGER, day=day):
                        row = dict(record)
                        row["day"] = day
                        row["gu"] = gu_of(row.get("dong"))
                        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                        count += 1
                os.replace(temp, path)
                written[day] = count
                print(f"  {day}  결제 {count:,}건  ({path.stat().st_size / 1e6:.1f} MB)")

            poi = session.run(POI).single()
            (args.out / "poi_summary.json").write_text(
                json.dumps(
                    {"poi_total": poi["n"], "poi_eligible": poi["elig"]}, ensure_ascii=False
                ),
                encoding="utf-8",
            )

        # 하루치 파일을 이어 붙여 run 전체 원장도 만든다 (보고서 엔진이 읽는 형태)
        merged = args.out / "events.jsonl"
        with merged.open("w", encoding="utf-8", newline="\n") as sink:
            for day in sorted(written):
                sink.write((args.out / f"events_{day}.jsonl").read_text(encoding="utf-8"))
        print(f"합계 {sum(written.values()):,}건 → {merged}")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
