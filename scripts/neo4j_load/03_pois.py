"""POI 3종 (residence·workplace·commerce) 적재 + IN_DONG + IN_CATEGORY.

입력:
  data/neo4j_load/pois/residence.csv          # K-apt geocoded
  data/neo4j_load/pois/workplace.csv          # 건축물대장 geocoded
  data/neo4j_load/pois/소상공인...서울_202603.csv  # commerce raw
  data/neo4j_load/mapping/mapping_upjong_to_sub.json  # upjong_code → (cat, sub)

출력:
  (:POI {id, name, lon, lat, type, dong_code}) + [:IN_DONG] + [:IN_CATEGORY (commerce만)]

옵션:
  --residence  : residence만
  --workplace  : workplace만
  --commerce   : commerce만
  --all        : 셋 다 (기본)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session, bulk_run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POIS_DIR = PROJECT_ROOT / "data" / "neo4j_load" / "pois"
MAPPING_PATH = PROJECT_ROOT / "data" / "neo4j_load" / "mapping" / "mapping_upjong_to_sub.json"

# commerce CSV 컬럼 인덱스 (검증)
C_COL = {
    "id": 0,        # 상가아이디
    "name": 1,      # 상호명
    "upjong_l3_cd": 7,
    "dong_cd": 15,  # 행정동코드 NSO 8자리
    "lon": 37,
    "lat": 38,
}


def load_csv_pois(csv_path: Path, poi_type: str) -> list[dict]:
    """residence.csv 또는 workplace.csv 로드."""
    rows = []
    with open(csv_path, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "id": r["id"],
                    "name": r["name"] or "",
                    "lon": float(r["lon"]),
                    "lat": float(r["lat"]),
                    "type": poi_type,
                    "dong_code": r["dong_code"],
                })
            except (ValueError, KeyError):
                continue
    return rows


def load_commerce_pois() -> tuple[list[dict], list[dict]]:
    """commerce CSV + mapping → (pois, in_category_pairs)."""
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    csv_path = next(POIS_DIR.glob("소상공인*.csv"))
    pois, ic_pairs = [], []
    with open(csv_path, encoding="utf-8-sig") as f:
        r = csv.reader(f)
        next(r, None)  # header
        for row in r:
            if len(row) <= C_COL["lat"]:
                continue
            try:
                lon = float(row[C_COL["lon"]])
                lat = float(row[C_COL["lat"]])
            except ValueError:
                continue
            dong_code = row[C_COL["dong_cd"]]
            if not dong_code:
                continue
            upjong = row[C_COL["upjong_l3_cd"]]
            mp = mapping.get(upjong)
            if not mp:
                continue
            sub = mp["sub"]
            poi_id = f"C_{row[C_COL['id']]}"
            pois.append({
                "id": poi_id,
                "name": row[C_COL["name"]] or "",
                "lon": lon, "lat": lat,
                "type": "commerce",
                "dong_code": dong_code,
            })
            ic_pairs.append((poi_id, sub))
    return pois, ic_pairs


def write_pois(session, pois: list[dict], label: str, batch_size: int = 5000):
    bulk_run(session, """
        UNWIND $batch AS p
        MERGE (n:POI {id: p.id})
        SET n.name = p.name,
            n.lon = p.lon, n.lat = p.lat,
            n.type = p.type,
            n.dong_code = p.dong_code
        WITH n, p
        MATCH (d:Dong {code: p.dong_code})
        MERGE (n)-[:IN_DONG]->(d)
    """, batch=pois, batch_size=batch_size)
    print(f"  + POI({label}) x {len(pois)} + IN_DONG")


def write_in_category(session, pairs: list[tuple[str, str]], batch_size: int = 5000):
    rows = [{"poi_id": a, "cat_name": b} for a, b in pairs]
    bulk_run(session, """
        UNWIND $batch AS p
        MATCH (poi:POI {id: p.poi_id}), (c:Category {name: p.cat_name})
        MERGE (poi)-[:IN_CATEGORY]->(c)
    """, batch=rows, batch_size=batch_size)
    print(f"  + IN_CATEGORY x {len(pairs)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--residence", action="store_true")
    ap.add_argument("--workplace", action="store_true")
    ap.add_argument("--commerce", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    if not (args.residence or args.workplace or args.commerce or args.all):
        args.all = True

    with driver_session() as s:
        if args.all or args.residence:
            fp = POIS_DIR / "residence.csv"
            if fp.exists():
                pois = load_csv_pois(fp, "residence")
                print(f"[residence] {len(pois)} rows")
                write_pois(s, pois, "residence")
            else:
                print("[residence] residence.csv not found, skip")

        if args.all or args.workplace:
            fp = POIS_DIR / "workplace.csv"
            if fp.exists():
                pois = load_csv_pois(fp, "workplace")
                print(f"[workplace] {len(pois)} rows")
                write_pois(s, pois, "workplace")
            else:
                print("[workplace] workplace.csv not found, skip")

        if args.all or args.commerce:
            pois, ic = load_commerce_pois()
            print(f"[commerce] {len(pois)} POIs, {len(ic)} IN_CATEGORY pairs")
            write_pois(s, pois, "commerce")
            write_in_category(s, ic)

    print()
    print("[done] POIs loaded.")


if __name__ == "__main__":
    main()
