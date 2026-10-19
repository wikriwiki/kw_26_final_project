"""P016 마트 POI 적재(2026-10-11). 실행기 2단계(복원) 직후, 정책 전 주보다 먼저 정책 있음 그래프에 한 번 넣는다.
복제(5단계)로 정책 없음 그래프에도 똑같이 들어가므로 두 갈래의 가게 구성은 같다.

입력: data/experiments/p016_marts_20261010.json (tools 밖에서 만든 목록 — /data/ab3w/audit_tools/build_marts.py)
  · tag_existing: 그래프에 이미 있는 같은 체인 가게(150m 안) — chain·mart_format 표시만 붙인다(장보기 업종 가게일 때만).
  · add_new     : 카카오맵 검색으로 찾은 서울 매장 중 그래프에 없던 것 — 새 POI(슈퍼마켓 업종)로 넣는다.
  · 짝지은 그래프 가게가 장보기 업종이 아니면(이마트 안 화장품 매장·홈플러스 안 세탁소·하나로마트 안 카페 등
    매장 안 입점 가게와 이름으로 짝지어진 경우, 10명 시험에서 23곳 확인) 표시하지 않고 새 POI로 넣는다.
    동은 짝지은 그래프 가게(같은 건물)의 동을 쓴다.
값을 지어내지 않는다: 업종코드·상생 속성·가격은 넣지 않는다(없으면 엔진의 동·업종 기준값을 쓴다).
멱등: 다시 돌려도 같은 결과(MERGE).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session  # noqa: E402

SRC = ROOT / "data" / "experiments" / "p016_marts_20261010.json"
GROCERY = ["슈퍼마켓", "식료품", "청과", "정육"]

TAG = """
UNWIND $rows AS r
MATCH (p:POI {id: r.poi_id})-[:IN_CATEGORY]->(c:Category)
WHERE c.name IN $grocery
SET p.chain = r.brand, p.mart_format = r.format, p.kakao_id = r.kakao_id
RETURN collect(DISTINCT r.poi_id) AS ids
"""

# 장보기 업종이 아니어서 표시하지 못한 짝 — 짝지은 그래프 가게의 동을 가져온다.
NOT_GROCERY_DONG = """
UNWIND $rows AS r
MATCH (p:POI {id: r.poi_id})-[:IN_DONG]->(d:Dong)
RETURN r.kakao_id AS kakao_id, d.code AS dong_code
"""

ADD = """
UNWIND $rows AS r
MATCH (d:Dong {code: r.dong_code})
MATCH (c:Category {name: '슈퍼마켓'})
MERGE (p:POI {id: r.poi_id})
ON CREATE SET p.name = r.name, p.lat = r.lat, p.lon = r.lon, p.type = 'commerce',
              p.dong_code = r.dong_code, p.source = 'kakao_map_20261010'
SET p.chain = r.brand, p.mart_format = r.format, p.kakao_id = r.kakao_id, p.address = r.address
MERGE (p)-[:IN_DONG]->(d)
MERGE (p)-[:IN_CATEGORY]->(c)
RETURN count(DISTINCT p) AS n
"""

KEYS = ("poi_id", "name", "lat", "lon", "dong_code", "brand", "format", "kakao_id", "address")


def main() -> int:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    tag = [{k: r[k] for k in ("poi_id", "brand", "format", "kakao_id")} for r in data["tag_existing"]]
    add = [{k: r.get(k) for k in KEYS} for r in data["add_new"] if r.get("dong_code")]
    with driver_session() as s:
        s.run("CREATE INDEX poi_mart_format IF NOT EXISTS FOR (p:POI) ON (p.mart_format)").consume()
        tagged = set(s.run(TAG, rows=tag, grocery=GROCERY).single()["ids"])
        miss = [r for r in data["tag_existing"] if r["poi_id"] not in tagged]
        dong = {x["kakao_id"]: x["dong_code"] for x in s.run(NOT_GROCERY_DONG, rows=miss).data()}
        moved = [dict({k: r.get(k) for k in KEYS}, poi_id=f"K_{r['kakao_id']}", dong_code=dong.get(r["kakao_id"]))
                 for r in miss]
        moved_ok = [r for r in moved if r["dong_code"]]
        n_add = s.run(ADD, rows=add + moved_ok).single()["n"]
        hyper = s.run("MATCH (p:POI {mart_format:'hypermarket'}) RETURN count(p) AS n").single()["n"]
        ssm = s.run("MATCH (p:POI {mart_format:'ssm'}) RETURN count(p) AS n").single()["n"]
    print(f"P016 마트 적재: 기존 표시 {len(tagged)}/{len(tag)} · 매장 안 입점 가게와 짝지어져 새 POI로 옮김 {len(moved_ok)}/{len(miss)} · "
          f"새 POI {n_add}/{len(add) + len(moved_ok)} · 대형마트 {hyper} · 기업형 슈퍼 {ssm}")
    for r in moved:
        print(f"  옮김: {r['name']} ({r['format']}) 동 {r['dong_code']}")
    if n_add != len(add) + len(moved_ok) or len(moved_ok) != len(miss):
        print("새 POI 일부가 들어가지 않았다(동 코드 없음?)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
