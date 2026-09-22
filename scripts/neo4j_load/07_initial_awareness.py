"""KNOWS_POI {source:'initial'} 시딩.

Day 0 인지 풀 = 각 agent당 약 70~90개 POI:
- 거주 동의 POI 상위 40개 (거리순) — 식사·카페·편의점·마트 등 일상 POI
- 직장 동의 POI 상위 30개 (거리순)
- 서울 핫스팟 랜드마크 10개 (전체 agent 공통)

KNOWS_POI 엣지(source/since/affinity)만으로 사전 인지를 표현. 별도 Memory 노드는 생성하지 않음
(initial Memory는 KNOWS_POI 엣지 속성의 중복일 뿐이고 시계열 가치도 없어서 redundant).
"""
from __future__ import annotations

import math
import os
import random
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

SEED = 42
N_HOME = 40
N_WORK = 30
N_LANDMARK = 10
# 카테고리별 최소 인지. 거리순 상위만 뽑으면 동네에 많은 업종(식당·카페·편의점)이
# 자리를 다 차지해 드물지만 존재하는 업종을 아무도 모르게 된다. 실제로 가전·가구는
# 전체 POI 의 1.4% 라 상위 40개 중 0.6개꼴로만 들어왔다(실측 1인당 0.7개).
# 사람은 자주 가지 않는 업종이라도 생활권에 뭐가 있는지는 안다 — 그 최소선을 보장한다.
N_PER_CAT = int(os.environ.get("EXP_KNOWS_PER_CAT", "3"))
DAY_ZERO = "2026-05-01"  # POC sim start

# 서울 핫스팟 랜드마크 (전체 agent 공통 인지)
LANDMARK_DONG_CODES = [
    "11680640",  # 역삼1동 (강남역·테헤란로)
    "11440550",  # 서교동 (홍대)
    "11200730",  # 성수1가1동 (성수 카페거리)
    "11140550",  # 을지로3가
    "11110530",  # 사직동 (광화문)
    "11680510",  # 신사동 (가로수길)
    "11710570",  # 잠실3동 (롯데타워)
    "11680600",  # 삼성1동 (코엑스)
    "11305610",  # 종암동 (성북 주거지)
    "11140600",  # 명동 (관광)
]


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def _add_per_category(poi_ids: set, ranked: list, k: int) -> None:
    """거리순 정렬된 풀에서 카테고리마다 가까운 k 개를 보장한다."""
    if k <= 0:
        return
    seen = defaultdict(int)
    for p in ranked:
        c = p.get("kdi")
        if not c or seen[c] >= k:
            continue
        seen[c] += 1
        poi_ids.add(p["id"])


def main():
    rng = random.Random(SEED)

    with driver_session() as s:
        # 1. 모든 commerce POI를 동별로 그룹화 + 좌표
        print("[fetch] commerce POI by dong ...")
        dong_pois = defaultdict(list)
        for r in s.run("MATCH (p:POI {type:'commerce'}) RETURN p.id AS id, p.dong_code AS d, p.lon AS lon, p.lat AS lat, p.sangsaeng_kdi AS kdi"):
            dong_pois[r["d"]].append({"id": r["id"], "lon": r["lon"], "lat": r["lat"],
                                      "kdi": r["kdi"]})
        print(f"  dongs: {len(dong_pois)}, total commerce POIs: {sum(len(v) for v in dong_pois.values())}")

        # 2. 랜드마크 POI: 각 랜드마크 동에서 인기 commerce POI 1개씩 샘플링
        landmarks = []
        for cd in LANDMARK_DONG_CODES:
            pool = dong_pois.get(cd) or []
            if pool:
                landmarks.append(rng.choice(pool)["id"])
        print(f"  landmarks selected: {len(landmarks)}")

        # 3. agent별 anchor POI 좌표 + dong
        print("[fetch] agent anchors ...")
        agents = []
        for r in s.run("""
            MATCH (a:Agent)
            OPTIONAL MATCH (a)-[:LIVES_AT]->(rp:POI)
            OPTIONAL MATCH (a)-[:WORKS_AT]->(wp:POI)
            RETURN a.id AS id,
                   rp.dong_code AS home_dong, rp.lon AS home_lon, rp.lat AS home_lat,
                   wp.dong_code AS work_dong, wp.lon AS work_lon, wp.lat AS work_lat
        """):
            agents.append(dict(r))
        print(f"  agents: {len(agents)}")

        # 4. KNOWS_POI batch 생성 (Memory 노드는 생성하지 않음)
        kp_rows = []
        for a in agents:
            poi_ids = set()
            # 거주 동
            if a["home_dong"] and a["home_lon"] is not None:
                pool = dong_pois.get(a["home_dong"]) or []
                ranked = sorted(
                    pool,
                    key=lambda p: haversine_km(p["lon"], p["lat"], a["home_lon"], a["home_lat"])
                )
                for p in ranked[:N_HOME]:
                    poi_ids.add(p["id"])
                _add_per_category(poi_ids, ranked, N_PER_CAT)
            # 직장 동
            if a["work_dong"] and a["work_lon"] is not None:
                pool = dong_pois.get(a["work_dong"]) or []
                ranked = sorted(
                    pool,
                    key=lambda p: haversine_km(p["lon"], p["lat"], a["work_lon"], a["work_lat"])
                )
                for p in ranked[:N_WORK]:
                    poi_ids.add(p["id"])
                _add_per_category(poi_ids, ranked, N_PER_CAT)
            # 랜드마크 (공통)
            for lm in landmarks:
                poi_ids.add(lm)

            for poi_id in poi_ids:
                kp_rows.append({
                    "aid": a["id"], "poi_id": poi_id,
                    "since": DAY_ZERO, "source": "initial",
                    "affinity": 0.5,
                    "visit_count": 0,
                })

        print(f"  KNOWS_POI rows: {len(kp_rows)}")

        # 5. write KNOWS_POI 엣지만
        bulk_run(s, """
            UNWIND $batch AS p
            MATCH (a:Agent {id: p.aid}), (poi:POI {id: p.poi_id})
            MERGE (a)-[kp:KNOWS_POI]->(poi)
            ON CREATE SET
              kp.since = date(p.since),
              kp.source = p.source,
              kp.visit_count = p.visit_count,
              kp.affinity = p.affinity
        """, batch=kp_rows, batch_size=5000)
        print(f"  + KNOWS_POI MERGEd")

    print("[done] initial awareness seeded.")


if __name__ == "__main__":
    main()
