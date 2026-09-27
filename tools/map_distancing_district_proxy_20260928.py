"""Exploratory 2023 commercial-district proxy for a preserved DISTANCING pair.

This never changes the graph or simulator. It cannot reproduce the Seoul
Institute's 2020 card-merchant panel estimand: district boundaries, merchants,
period, counterfactual and transaction coverage differ. Requires pyshp,
shapely and pyproj. Run only on complete, externally backed-up arms.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import shapefile
from pyproj import CRS, Transformer
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

ROOT = Path(__file__).resolve().parents[1]
TARGET_TYPES = {"관광특구", "발달상권"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_paths() -> tuple[Path, Path]:
    zips = list((ROOT / "data/policy_raw_data/사회적거리두기_DISTANCING2020/환경자료").glob("*2023-10-23.zip"))
    pois = list((ROOT / "data/neo4j_load/pois").glob("소상공인*서울_202603.csv"))
    if len(zips) != 1 or len(pois) != 1:
        raise ValueError(f"Expected one official 2023 ZIP and one POI CSV; got {len(zips)}, {len(pois)}")
    return zips[0], pois[0]


def load_districts(zip_path: Path):
    with zipfile.ZipFile(zip_path) as archive:
        members = {}
        for suffix in (".shp", ".dbf", ".prj"):
            found = [name for name in archive.namelist() if name.lower().endswith(suffix)]
            if len(found) != 1:
                raise ValueError(f"ZIP has {len(found)} {suffix} entries")
            members[suffix] = found[0]
        crs = CRS.from_wkt(archive.read(members[".prj"]).decode("utf-8"))
        with archive.open(members[".shp"]) as shp, archive.open(members[".dbf"]) as dbf:
            reader = shapefile.Reader(shp=shp, dbf=dbf, encoding="utf-8")
            geometries = []
            categories = []
            codes = []
            for item in reader.iterShapeRecords():
                record = item.record.as_dict()
                category = record["TRDAR_SE_1"]
                if category not in TARGET_TYPES:
                    continue
                geometry = shape(item.shape.__geo_interface__)
                if geometry.is_empty or not geometry.is_valid:
                    raise ValueError(f"Invalid polygon {record['TRDAR_CD']}")
                geometries.append(geometry)
                categories.append(category)
                codes.append(record["TRDAR_CD"])
    return crs, geometries, categories, codes


def read_receipts(metrics_dir: Path, days: set[str]):
    files = [metrics_dir / f"day_{day}.jsonl" for day in sorted(days)]
    if not all(p.is_file() for p in files):
        raise FileNotFoundError([str(p) for p in files if not p.is_file()])
    seen_days = set()
    seen_events = set()
    receipts = []
    for path in files:
        day = path.stem.removeprefix("day_")
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                raise ValueError(f"Non-complete row in {path}: {row.get('aid')}")
            day_key = (row["aid"], day)
            if day_key in seen_days:
                raise ValueError(f"Duplicate citizen-day {day_key}")
            seen_days.add(day_key)
            for receipt in row.get("execution_receipts") or []:
                if receipt.get("kind") != "purchase_receipt" or int(receipt.get("amount") or 0) <= 0:
                    continue
                event_id = receipt["event_id"]
                if event_id in seen_events:
                    raise ValueError(f"Duplicate receipt {event_id}")
                seen_events.add(event_id)
                if receipt.get("observed_at") != day:
                    raise ValueError(f"Receipt day mismatch {event_id}")
                receipts.append(receipt)
    return receipts, seen_days, {p.as_posix(): sha256(p) for p in files}


def poi_coordinates(csv_path: Path, desired: set[str]):
    coordinates = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.reader(source)
        next(reader, None)
        for row in reader:
            if len(row) <= 38:
                continue
            pid = "C_" + row[0]
            if pid not in desired:
                continue
            if pid in coordinates:
                raise ValueError(f"Duplicate POI {pid}")
            try:
                coordinates[pid] = (float(row[37]), float(row[38]))
            except ValueError:
                continue
    return coordinates


def poi_coordinates_json(path: Path, desired: set[str]):
    """Use a separate, read-only export when a cloud-backed raw CSV is unavailable."""
    data = json.loads(path.read_text(encoding="utf-8"))
    coordinates = data.get("coordinates", data)
    return {pid: (float(coordinates[pid][0]), float(coordinates[pid][1]))
            for pid in desired if pid in coordinates}


def classify_pois(coordinates, crs, polygons, categories, codes):
    transform = Transformer.from_crs(4326, crs, always_xy=True).transform
    tree = STRtree(polygons)
    result = {}
    ambiguous = {}
    for pid, (lon, lat) in coordinates.items():
        point = Point(*transform(lon, lat))
        matched = [int(i) for i in tree.query(point) if polygons[int(i)].covers(point)]
        kinds = {categories[i] for i in matched}
        if len(kinds) > 1:
            ambiguous[pid] = [{"type": categories[i], "code": codes[i]} for i in matched]
        elif kinds:
            result[pid] = next(iter(kinds))
    return result, ambiguous


def summarize(receipts, types):
    amount = Counter()
    event_count = Counter()
    citizens = defaultdict(set)
    for receipt in receipts:
        category = types.get(receipt["poi_id"])
        if category is None:
            continue
        amount[category] += int(receipt["amount"])
        event_count[category] += 1
        citizens[category].add(receipt["agent_id"])
    return {
        category: {"spend_won": amount[category], "positive_receipts": event_count[category],
                   "citizens_with_receipts": len(citizens[category])}
        for category in sorted(TARGET_TYPES)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-metrics", type=Path, required=True)
    parser.add_argument("--off-metrics", type=Path, required=True)
    parser.add_argument("--days", nargs="+", required=True, help="Exact YYYY-MM-DD score dates")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--poi-coordinates-json", type=Path, action="append",
                        help="Read-only {poi_id: [lon, lat]} export; repeat for ON/OFF files")
    args = parser.parse_args()
    days = set(args.days)
    if len(days) != len(args.days):
        raise ValueError("Duplicate days")
    on, on_days, on_sources = read_receipts(args.on_metrics, days)
    off, off_days, off_sources = read_receipts(args.off_metrics, days)
    if on_days != off_days:
        raise ValueError(f"Different citizen-day keys: ON {len(on_days)}, OFF {len(off_days)}")
    aid_sets = [{aid for aid, observed_day in on_days if observed_day == day}
                for day in sorted(days)]
    if not aid_sets[0] or any(aids != aid_sets[0] for aids in aid_sets[1:]):
        raise ValueError("Incomplete or inconsistent citizen roster across score dates")
    zip_path, csv_path = source_paths()
    crs, polygons, categories, codes = load_districts(zip_path)
    desired = {r["poi_id"] for r in on + off}
    if args.poi_coordinates_json is None:
        coordinates = poi_coordinates(csv_path, desired)
        coordinate_sources = [csv_path]
    else:
        coordinates = {}
        coordinate_sources = args.poi_coordinates_json
        for path in coordinate_sources:
            for pid, point in poi_coordinates_json(path, desired).items():
                if pid in coordinates and coordinates[pid] != point:
                    raise ValueError(f"ON/OFF coordinate conflict for {pid}")
                coordinates[pid] = point
    matched, ambiguous = classify_pois(coordinates, crs, polygons, categories, codes)
    on_summary, off_summary = summarize(on, matched), summarize(off, matched)
    rates = {}
    for category in sorted(TARGET_TYPES):
        denominator = off_summary[category]["spend_won"]
        rates[category] = None if denominator <= 0 else 100 * (
            on_summary[category]["spend_won"] / denominator - 1)
    gap = None if any(v is None for v in rates.values()) else rates["관광특구"] - rates["발달상권"]
    on_target_won = sum(r["spend_won"] for r in on_summary.values())
    off_target_won = sum(r["spend_won"] for r in off_summary.values())
    on_total_won = sum(int(r["amount"]) for r in on)
    off_total_won = sum(int(r["amount"]) for r in off)
    coordinate_join_rate = {
        arm: (sum(r["poi_id"] in coordinates for r in receipts) / len(receipts)
              if receipts else None)
        for arm, receipts in (("on", on), ("off", off))}
    ambiguous_receipts = {
        arm: sum(r["poi_id"] in ambiguous for r in receipts)
        for arm, receipts in (("on", on), ("off", off))}
    result = {
        "status": "exploratory_geographic_proxy_not_direct_empirical_comparison",
        "reason": "2023 official district polygons and 2026 POIs; short simulated policy ON/OFF, unlike 2020 merchant-panel year-over-year card-sales changes",
        "days": sorted(days), "citizen_days_each_arm": len(on_days),
        "sources_sha256": {**on_sources, **off_sources,
                           zip_path.as_posix(): sha256(zip_path),
                           **{path.as_posix(): sha256(path) for path in coordinate_sources}},
        "polygon_epsg": crs.to_epsg(),
        "polygon_count_by_type": dict(Counter(categories)),
        "positive_receipts": {"on": len(on), "off": len(off)},
        "coordinate_join_rate_receipts": coordinate_join_rate,
        "ambiguous_receipts_excluded": ambiguous_receipts,
        "target_type_spend_capture_share": {
            "on": on_target_won / on_total_won if on_total_won else None,
            "off": off_target_won / off_total_won if off_total_won else None},
        "unique_pois_in_receipts": len(desired),
        "pois_with_coordinates": len(coordinates),
        "pois_classified_target_types": len(matched),
        "pois_ambiguous_excluded": ambiguous,
        "pois_missing_coordinates": sorted(desired - set(coordinates)),
        "on": on_summary, "off": off_summary,
        "on_off_percent_change": rates,
        "tourism_minus_developed_percentage_points": gap,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"gap_pp": gap, "rates": rates, "matched_pois": len(matched),
                      "ambiguous_pois": len(ambiguous), "output_sha256": sha256(args.out)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
