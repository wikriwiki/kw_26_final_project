"""건축물대장 표제부 25 CSV → workplace POI CSV.

Mode:
  python 03b_workplace_from_bldg.py --analyze       : 주용도 분포만 dump (filter 결정용)
  python 03b_workplace_from_bldg.py --geocode       : 필터 + Geocoding 실행
  python 03b_workplace_from_bldg.py --filter-only   : 필터만 (Geocoding skip, 미리보기용)

흐름:
1. 25 CSV 머지 (streaming, in-memory dict 누적)
2. 주용도 분포 분석 → workplace 적합 용도 선정
3. 필터링: 업무·근린·판매·교육연구·의료·문화집회만
4. 주소 추출: 도로명대지위치 우선, 없으면 대지위치
5. V-WORLD Geocoding → 좌표 + 행정동 NSO 8자리
6. workplace.csv 저장
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from geocode.vworld_client import VWorldGeocoder, load_api_key

# === 경로 ===
PROJECT_ROOT = Path(__file__).resolve().parents[2]
BLDG_DIR = PROJECT_ROOT / "data" / "neo4j_load" / "pois" / "건축물대장"
OUT_DIR = PROJECT_ROOT / "data" / "neo4j_load" / "pois"
OUT_CSV = OUT_DIR / "workplace.csv"
OUT_FAIL = OUT_DIR / "_workplace_failures.csv"
OUT_STATS = OUT_DIR / "_workplace_summary.json"
OUT_USAGE_DIST = OUT_DIR / "_bldg_usage_distribution.json"
CACHE_PATH = PROJECT_ROOT / "scripts" / "geocode" / "cache.sqlite"

# 표제부 컬럼 인덱스 (검증)
COL = {
    "site_addr": 0,        # 대지위치
    "sigungu_cd": 1,       # 시군구코드
    "bjdong_cd": 2,        # 법정동코드
    "main_bun": 4,         # 번
    "sub_bun": 5,          # 지
    "pk": 6,               # 관리건축물대장PK
    "road_addr": 11,       # 도로명대지위치
    "bld_name": 12,        # 건물명
    "main_purpose_cd": 34, # 주용도코드
    "main_purpose": 35,    # 주용도코드명
    "households": 40,      # 세대수
    "households_g": 41,    # 가구수
    "above_floors": 43,    # 지상층수
    "gross_area": 28,      # 연면적
}

# workplace 적합 주용도코드명 (분포 분석 후 확정 — 총 ~152K 예상)
WORKPLACE_USAGE = {
    # 코어 workplace
    "업무시설",
    "공공업무시설",
    "제1종근린생활시설",
    "제2종근린생활시설",
    "근린생활시설",          # 1·2종 외
    "판매시설",
    "교육연구시설",
    "교육연구및복지시설",    # 변형 표기
    "의료시설",
    # 다양성 보강
    "문화및집회시설",
    "문화 및 집회시설",      # 띄어쓰기 변형
    "운수시설",              # 역·터미널
    "노유자시설",            # 어린이집·요양원
    "공장",                   # 생산직
    "운동시설",              # 헬스장·체육관
    "관광휴게시설",          # 관광지
    "교정및군사시설",        # 공무원
    "방송통신시설",
}

# 제외 (residence·commerce 중복 또는 workplace 부적합)
EXCLUDED_USAGE_HINTS = {
    "단독주택", "공동주택", "다세대", "다중주택", "기숙사",
    "창고", "위험물", "동물",
    "묘지", "장례",
    "종교",
    "위락",  # 술집·노래방 (commerce 와 중복)
}


def merge_and_analyze() -> dict:
    """25 CSV 머지하면서 주용도 분포 + 시군구 분포 dump."""
    files = sorted(BLDG_DIR.glob("*.csv"))
    print(f"[merge] {len(files)} CSV files in {BLDG_DIR}")
    usage_counter = Counter()
    sigungu_counter = Counter()
    total = 0
    for fp in files:
        with open(fp, encoding="utf-8-sig") as f:
            r = csv.reader(f)
            header = next(r, None)
            for row in r:
                if len(row) <= COL["main_purpose"]:
                    continue
                total += 1
                usage_counter[row[COL["main_purpose"]]] += 1
                sigungu_counter[row[COL["sigungu_cd"]]] += 1
    result = {
        "total_rows": total,
        "usage_unique": len(usage_counter),
        "sigungu_unique": len(sigungu_counter),
        "usage_top30": usage_counter.most_common(30),
        "sigungu_distribution": dict(sigungu_counter),
        "workplace_baseline_match": sum(
            c for u, c in usage_counter.items() if u in WORKPLACE_USAGE
        ),
    }
    return result


def iter_filtered_records():
    """필터링된 행 yield."""
    files = sorted(BLDG_DIR.glob("*.csv"))
    pk_seen = set()
    for fp in files:
        with open(fp, encoding="utf-8-sig") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if len(row) <= COL["main_purpose"]:
                    continue
                purpose = row[COL["main_purpose"]]
                # 빠른 필터: 주용도가 WORKPLACE_USAGE에 있고, EXCLUDED 키워드 미포함
                if purpose not in WORKPLACE_USAGE:
                    continue
                if any(hint in purpose for hint in EXCLUDED_USAGE_HINTS):
                    continue
                pk = row[COL["pk"]] or ""
                if pk in pk_seen:
                    continue
                pk_seen.add(pk)
                yield {
                    "pk": pk,
                    "site_addr": (row[COL["site_addr"]] or "").strip(),
                    "road_addr": (row[COL["road_addr"]] or "").strip(),
                    "bld_name": (row[COL["bld_name"]] or "").strip(),
                    "purpose": purpose,
                    "sigungu_cd": row[COL["sigungu_cd"]] or "",
                    "bjdong_cd": row[COL["bjdong_cd"]] or "",
                    "gross_area": row[COL["gross_area"]] or "",
                    "above_floors": row[COL["above_floors"]] or "",
                }


def choose_address(rec: dict) -> tuple[str, str]:
    """도로명대지위치 우선, 없으면 대지위치(지번). 단지명/특이지명 제거 후."""
    if rec["road_addr"]:
        addr = rec["road_addr"].split("(")[0].strip()  # "...(역삼동)" 부수 정보 제거
        return addr, "road"
    if rec["site_addr"]:
        addr = rec["site_addr"]
        # "...번지" 제거
        addr = addr.replace("번지", "").strip()
        return addr, "parcel"
    return "", "road"


def run_analyze():
    print("[analyze] merging + counting 주용도 ...")
    t0 = time.time()
    result = merge_and_analyze()
    OUT_USAGE_DIST.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[analyze] elapsed {time.time()-t0:.1f}s")
    print(f"  total_rows:                  {result['total_rows']}")
    print(f"  usage_unique:                {result['usage_unique']}")
    print(f"  workplace_baseline_match:    {result['workplace_baseline_match']}")
    print()
    print("  Top 30 주용도:")
    for u, c in result["usage_top30"]:
        marker = "✅" if u in WORKPLACE_USAGE else ("❌" if any(h in u for h in EXCLUDED_USAGE_HINTS) else "  ")
        print(f"    {marker} {c:>8d}  {u}")
    print()
    print(f"  -> {OUT_USAGE_DIST}")


def run_filter_only():
    n = 0
    sigungu = Counter()
    purpose = Counter()
    addr_road = 0
    addr_jibun_only = 0
    for rec in iter_filtered_records():
        n += 1
        sigungu[rec["sigungu_cd"]] += 1
        purpose[rec["purpose"]] += 1
        if rec["road_addr"]:
            addr_road += 1
        elif rec["site_addr"]:
            addr_jibun_only += 1
    print(f"[filter] selected: {n}")
    print(f"  road_addr available:  {addr_road} ({addr_road/n*100:.1f}%)")
    print(f"  jibun-only:           {addr_jibun_only} ({addr_jibun_only/n*100:.1f}%)")
    print(f"  no address:           {n - addr_road - addr_jibun_only}")
    print(f"  by purpose: {dict(purpose.most_common())}")
    print(f"  by sigungu top10: {dict(sigungu.most_common(10))}")


def run_geocode(concurrency: int = 10):
    key = load_api_key()
    print(f"[init] V-WORLD key loaded: {key[:8]}...")
    geo = VWorldGeocoder(key, CACHE_PATH, max_retries=3)

    print("[filter] selecting workplace records ...")
    records = list(iter_filtered_records())
    print(f"[filter] selected: {len(records)}")

    print(f"[geocode] starting batch (concurrency={concurrency}) ...")
    t0 = time.time()
    results = [None] * len(records)

    def worker(i_rec):
        i, rec = i_rec
        addr, type_first = choose_address(rec)
        if not addr:
            from geocode.vworld_client import GeocodeResult
            return i, addr, type_first, GeocodeResult(
                address_in="", type_used=type_first, status="ERROR", raw_error="empty_addr"
            )
        r = geo.geocode_one(addr, type_first=type_first, fallback=True)
        return i, addr, type_first, r

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker, (i, r)) for i, r in enumerate(records)]
        done = 0
        for fut in as_completed(futures):
            i, addr, type_first, gr = fut.result()
            results[i] = (addr, type_first, gr)
            done += 1
            if done % 2000 == 0 or done == len(records):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(records) - done) / rate if rate > 0 else 0
                print(f"  progress {done}/{len(records)}  ({rate:.1f} req/s, cache hit {geo.stats['hit']}, ETA {eta/60:.1f}min)")

    # write
    out_rows = []
    fail_rows = []
    status_counter = Counter()
    sigungu_counter = Counter()
    for rec, (addr, type_first, gr) in zip(records, results):
        status_counter[gr.status] += 1
        if gr.status == "OK" and gr.lon is not None and gr.lat is not None:
            dong_code_nso = (gr.level4AC[:8] if gr.level4AC and len(gr.level4AC) >= 8 else "")
            sigungu_counter[gr.level2 or ""] += 1
            out_rows.append({
                "id": f"W_{rec['pk']}",
                "name": rec["bld_name"] or "",
                "lon": f"{gr.lon:.7f}",
                "lat": f"{gr.lat:.7f}",
                "dong_code": dong_code_nso,
                "dong_name": gr.level4A or "",
                "sigungu": gr.level2 or "",
                "main_purpose": rec["purpose"],
                "source_addr": addr,
                "addr_type": gr.type_used,
            })
        else:
            fail_rows.append({
                "id": f"W_{rec['pk']}",
                "name": rec["bld_name"] or "",
                "purpose": rec["purpose"],
                "tried_addr": addr,
                "status": gr.status,
                "error": gr.raw_error or "",
            })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "lon", "lat", "dong_code", "dong_name", "sigungu", "main_purpose", "source_addr", "addr_type"])
        w.writeheader()
        w.writerows(out_rows)

    with open(OUT_FAIL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "purpose", "tried_addr", "status", "error"])
        w.writeheader()
        w.writerows(fail_rows)

    stats = {
        "input_filtered": len(records),
        "success": len(out_rows),
        "failed": len(fail_rows),
        "status_distribution": dict(status_counter),
        "sigungu_distribution": dict(sigungu_counter),
        "geocode_client_stats": geo.stats,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"[done] success: {len(out_rows)} / failed: {len(fail_rows)}")
    print(f"[done] elapsed: {stats['elapsed_sec']}s ({stats['elapsed_sec']/60:.1f} min)")
    print(f"  -> {OUT_CSV}")
    print(f"  -> {OUT_FAIL}")
    print(f"  -> {OUT_STATS}")


def run_from_cache():
    """캐시에서 OK 결과만 추출하여 workplace.csv 생성. V-WORLD 호출 안 함.

    Day 1 미완료 처리용:
    - V-WORLD 일 한도 초과로 캐시에 ERROR/NOT_FOUND인 record는 skip
    - 캐시에 hit OK 인 record만 workplace.csv에 포함
    - 내일 ERROR 캐시 invalidate 후 재실행하면 추가 채워짐
    """
    from geocode.vworld_client import VWorldGeocoder, load_api_key
    key = load_api_key()
    geo = VWorldGeocoder(key, CACHE_PATH)

    print("[from-cache] iterating filtered records ...")
    records = list(iter_filtered_records())
    print(f"[from-cache] filtered: {len(records)}")

    out_rows = []
    fail_rows = []
    counter = Counter()
    sigungu_counter = Counter()

    for rec in records:
        addr, type_first = choose_address(rec)
        if not addr:
            counter["empty_addr"] += 1
            continue

        # cache lookup — 도로명 우선, 지번 fallback (V-WORLD 호출 없이 캐시만)
        gr = None
        for tu in [type_first, "parcel" if type_first == "road" else "road"]:
            cached = geo._cache_get(geo._cache_key(addr, tu))
            if cached and cached.status == "OK":
                gr = cached
                break

        if gr and gr.lon is not None and gr.lat is not None:
            dong_code_nso = (gr.level4AC[:8] if gr.level4AC and len(gr.level4AC) >= 8 else "")
            sigungu_counter[gr.level2 or ""] += 1
            out_rows.append({
                "id": f"W_{rec['pk']}",
                "name": rec["bld_name"] or "",
                "lon": f"{gr.lon:.7f}",
                "lat": f"{gr.lat:.7f}",
                "dong_code": dong_code_nso,
                "dong_name": gr.level4A or "",
                "sigungu": gr.level2 or "",
                "main_purpose": rec["purpose"],
                "source_addr": addr,
                "addr_type": gr.type_used,
            })
            counter["ok"] += 1
        else:
            # 캐시 miss 또는 ERROR/NOT_FOUND — 내일 V-WORLD 한도 복구 후 처리
            cached_status = "missing"
            for tu in [type_first, "parcel" if type_first == "road" else "road"]:
                c = geo._cache_get(geo._cache_key(addr, tu))
                if c:
                    cached_status = c.status
                    break
            counter[cached_status] += 1
            fail_rows.append({
                "id": f"W_{rec['pk']}",
                "name": rec["bld_name"] or "",
                "purpose": rec["purpose"],
                "tried_addr": addr,
                "status": cached_status,
                "error": "",
            })

    # write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "lon", "lat", "dong_code", "dong_name", "sigungu", "main_purpose", "source_addr", "addr_type"])
        w.writeheader()
        w.writerows(out_rows)

    with open(OUT_FAIL, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "name", "purpose", "tried_addr", "status", "error"])
        w.writeheader()
        w.writerows(fail_rows)

    stats = {
        "mode": "from-cache (no V-WORLD call)",
        "input_filtered": len(records),
        "success_from_cache": len(out_rows),
        "skipped_for_tomorrow": len(fail_rows),
        "status_counter": dict(counter),
        "sigungu_top10": dict(sigungu_counter.most_common(10)),
        "note": "ERROR/missing은 내일 V-WORLD 한도 reset 후 ERROR 캐시 invalidate + 재실행",
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"[done] OK from cache: {len(out_rows)} / skipped: {len(fail_rows)}")
    print(f"  status: {dict(counter)}")
    print(f"  -> {OUT_CSV}")
    print(f"  -> {OUT_FAIL}")
    print(f"  -> {OUT_STATS}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--analyze", action="store_true", help="주용도 분포만 dump")
    g.add_argument("--filter-only", action="store_true", help="필터 결과 카운트만")
    g.add_argument("--geocode", action="store_true", help="필터 + Geocoding 실행")
    g.add_argument("--from-cache", action="store_true", help="캐시 OK만 추출, V-WORLD 호출 없음")
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()
    if args.analyze:
        run_analyze()
    elif args.filter_only:
        run_filter_only()
    elif args.geocode:
        run_geocode(args.concurrency)
    elif getattr(args, "from_cache", False):
        run_from_cache()
