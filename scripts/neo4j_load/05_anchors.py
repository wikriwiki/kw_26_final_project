"""LIVES_AT + WORKS_AT 균등 랜덤 배정.

전략:
- residence: agent.residence_dong_code_raw (8자리 NSO) 사용
- workplace: agent.workplace_dong_code_raw가 8자리면 그대로,
             아니면 workplace_dong_name 텍스트로 Dong 노드 매칭
             둘 다 실패 시 WORKS_AT 생성 안 함

POI 매칭:
- 해당 동의 POI(type=residence) 중 균등 랜덤 1개 → LIVES_AT
- 해당 동의 POI(type=workplace) 중 균등 랜덤 1개 → WORKS_AT
- 동에 POI 없으면 nearest fallback (전체 POI 중 좌표 거리 최소)
- 그것도 실패면 skip (warning 로그)
"""
from __future__ import annotations

import random
import sys
import math
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

SEED = 42


def haversine_km(lon1, lat1, lon2, lat2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def main():
    rng = random.Random(SEED)
    with driver_session() as s:
        # 1. POI 풀 가져오기 (type별로 동별 그룹)
        print("[fetch] POI pools by (type, dong_code) ...")
        res_pool = defaultdict(list)
        wk_pool = defaultdict(list)
        for r in s.run("MATCH (p:POI) WHERE p.type IN ['residence','workplace'] RETURN p.id AS id, p.type AS t, p.dong_code AS d, p.lon AS lon, p.lat AS lat"):
            entry = {"id": r["id"], "lon": r["lon"], "lat": r["lat"]}
            (res_pool if r["t"] == "residence" else wk_pool)[r["d"]].append(entry)
        print(f"  residence pool: {sum(len(v) for v in res_pool.values())} POIs across {len(res_pool)} dongs")
        print(f"  workplace pool: {sum(len(v) for v in wk_pool.values())} POIs across {len(wk_pool)} dongs")

        # 2. Dong → name 매핑 (workplace dong_code 정상화용)
        print("[fetch] Dong name lookup ...")
        dong_by_code = {}
        dong_by_name = defaultdict(list)
        dong_coord = {}  # code -> (lon, lat)
        for r in s.run("MATCH (d:Dong) RETURN d.code AS code, d.name AS name, d.lon AS lon, d.lat AS lat"):
            dong_by_code[r["code"]] = r["name"]
            dong_by_name[r["name"]].append(r["code"])
            if r["lon"] is not None:
                dong_coord[r["code"]] = (r["lon"], r["lat"])

        # 3. Agent fetch (raw dong codes + workplace dong name)
        print("[fetch] Agents ...")
        agents = []
        for r in s.run("""
            MATCH (a:Agent)
            RETURN a.id AS id,
                   a.residence_dong_code_raw AS res_cd,
                   a.workplace_dong_code_raw AS wk_cd,
                   a.workplace_dong_name AS wk_nm,
                   a.workplace_commute_min AS commute,
                   a.p_life_stage AS life_stage
        """):
            agents.append(dict(r))
        print(f"  agents: {len(agents)}")

        # 4. 균등 랜덤 매칭
        lives_at = []
        works_at = []
        stat = {
            "lives_ok": 0, "lives_fallback": 0, "lives_fail": 0,
            "works_ok": 0, "works_by_name": 0, "works_fallback": 0,
            "works_skip_no_workplace": 0, "works_skip_life_stage": 0, "works_fail": 0,
        }
        SKIP_LIFE_STAGES = {"학생", "은퇴", "주부"}

        # nearest fallback prep: 전체 residence/workplace POI 평면 리스트
        all_res = [p for ps in res_pool.values() for p in ps]
        all_wk = [p for ps in wk_pool.values() for p in ps]

        def nearest(pool, lon, lat, top=1):
            return min(pool, key=lambda p: haversine_km(p["lon"], p["lat"], lon, lat))

        for a in agents:
            # LIVES_AT
            res_cd = a["res_cd"]
            if res_cd and res_cd in res_pool and res_pool[res_cd]:
                chosen = rng.choice(res_pool[res_cd])
                lives_at.append({"aid": a["id"], "poi_id": chosen["id"]})
                stat["lives_ok"] += 1
            elif res_cd and res_cd in dong_coord and all_res:
                # fallback: 좌표 가장 가까운 residence
                lon, lat = dong_coord[res_cd]
                chosen = nearest(all_res, lon, lat)
                lives_at.append({"aid": a["id"], "poi_id": chosen["id"]})
                stat["lives_fallback"] += 1
            else:
                stat["lives_fail"] += 1

            # WORKS_AT
            life_stage = (a.get("life_stage") or "").strip()
            if any(s in life_stage for s in SKIP_LIFE_STAGES):
                stat["works_skip_life_stage"] += 1
                continue
            wk_cd = a["wk_cd"]
            wk_nm = a["wk_nm"]
            commute = a.get("commute")
            chosen_wk = None
            # try 1: dong_code 8자리 NSO 직접 매칭
            if wk_cd and len(wk_cd) == 8 and wk_cd in wk_pool and wk_pool[wk_cd]:
                chosen_wk = rng.choice(wk_pool[wk_cd])
                stat["works_ok"] += 1
            # try 2: dong_name으로 매칭
            elif wk_nm:
                cands = dong_by_name.get(wk_nm) or []
                # 첫 번째 매칭되는 코드 사용
                for cd in cands:
                    if cd in wk_pool and wk_pool[cd]:
                        chosen_wk = rng.choice(wk_pool[cd])
                        stat["works_by_name"] += 1
                        break
            # try 3: residence와 동일 (재택 또는 무직 처리)
            if chosen_wk is None and not wk_cd and not wk_nm:
                stat["works_skip_no_workplace"] += 1
                continue
            # try 4: nearest fallback
            if chosen_wk is None:
                # use residence dong's centroid (most reasonable proxy)
                if res_cd and res_cd in dong_coord and all_wk:
                    lon, lat = dong_coord[res_cd]
                    chosen_wk = nearest(all_wk, lon, lat)
                    stat["works_fallback"] += 1
                else:
                    stat["works_fail"] += 1
                    continue

            works_at.append({
                "aid": a["id"], "poi_id": chosen_wk["id"],
                "commute_min": commute or 0,
            })

        print(f"[match] {stat}")

        # 5. write
        bulk_run(s, """
            UNWIND $batch AS p
            MATCH (a:Agent {id: p.aid}), (poi:POI {id: p.poi_id})
            MERGE (a)-[:LIVES_AT]->(poi)
        """, batch=lives_at, batch_size=2000)
        print(f"  + LIVES_AT x {len(lives_at)}")

        bulk_run(s, """
            UNWIND $batch AS p
            MATCH (a:Agent {id: p.aid}), (poi:POI {id: p.poi_id})
            MERGE (a)-[w:WORKS_AT]->(poi)
            SET w.commute_min = p.commute_min
        """, batch=works_at, batch_size=2000)
        print(f"  + WORKS_AT x {len(works_at)}")

        # =========================================================
        # Cleanup: residence/workplace 중 LIVES_AT/WORKS_AT 연결 안 된 미사용 POI 삭제
        # =========================================================
        # commerce는 시뮬 중 동적 선택되므로 cleanup 대상 아님.
        before = s.run("""
            MATCH (p:POI) WHERE p.type IN ['residence','workplace']
            RETURN p.type AS t, count(p) AS c
        """).data()
        print(f"[cleanup] before: {before}")

        res_cleanup = s.run("""
            MATCH (p:POI {type:'residence'})
            WHERE NOT ()-[:LIVES_AT]->(p)
            DETACH DELETE p
            RETURN count(*) AS n
        """).single()["n"]
        print(f"  - residence unused removed: {res_cleanup}")

        wk_cleanup = s.run("""
            MATCH (p:POI {type:'workplace'})
            WHERE NOT ()-[:WORKS_AT]->(p)
            DETACH DELETE p
            RETURN count(*) AS n
        """).single()["n"]
        print(f"  - workplace unused removed: {wk_cleanup}")

        after = s.run("""
            MATCH (p:POI) WHERE p.type IN ['residence','workplace']
            RETURN p.type AS t, count(p) AS c
        """).data()
        print(f"[cleanup] after:  {after}")

    print("[done] anchors set + unused residence/workplace cleaned.")


if __name__ == "__main__":
    main()
