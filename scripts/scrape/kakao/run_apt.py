"""K-apt 단지 기본정보 xlsx → 카카오 리뷰/별점 풀크롤 러너.

흐름:
  1. data/neo4j_load/pois/20260508_단지_기본정보.xlsx → 서울 3,164건
  2. data/neo4j_load/pois/residence.csv → 좌표 조인 (단지코드 → lat/lon)
  3. 병렬 카카오 best_match + panel3 (5-pass cascade + 아파트 fallback)
  4. SQLite 체크포인트 (kakao_enrich.db) + JSONL 리포트

매칭률 100% 목표 → cascade에서 nomatch 발생 시 아파트 fallback
(시군구+동+이름, "아파트" 접미사, 동 centroid 근접 검색)

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.run_apt \
    --workers 16 --limit 0   # limit=0 → 전체
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openpyxl

from .client import KakaoClient, _haversine_m, _norm
from .extract import extract_reviews, extract_summary
from .store import (
    bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats, upsert_poi,
)

# === 경로 ===
ROOT = Path(__file__).resolve().parents[3]
XLSX = ROOT / "data" / "neo4j_load" / "pois" / "20260508_단지_기본정보.xlsx"
RES_CSV = ROOT / "data" / "neo4j_load" / "pois" / "residence.csv"
OUT_DIR = ROOT / "output" / "scrape" / "kakao"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# xlsx 컬럼 인덱스 (03a_residence_from_kapt.py 검증)
COL_SIDO, COL_SIGUNGU, COL_DONG_TEXT = 0, 1, 2
COL_CODE, COL_NAME = 4, 5
COL_ADDR_JIBUN, COL_ADDR_ROAD = 7, 9


def load_seoul_xlsx() -> list[dict]:
    # read_only=True는 이 파일의 dimension 메타가 깨져서 max_row=1로 잘못 보고됨
    # → read_only=False (03a_residence_from_kapt.py 패턴 따라감)
    wb = openpyxl.load_workbook(XLSX, read_only=False, data_only=True)
    ws = wb["sheet1"]
    out = []
    # row 1=공지, row 2=헤더, row 3+ 데이터
    for row in ws.iter_rows(values_only=True, min_row=3):
        if row[COL_SIDO] != "서울특별시":
            continue
        out.append({
            "code": str(row[COL_CODE]) if row[COL_CODE] else "",
            "name": str(row[COL_NAME]) if row[COL_NAME] else "",
            "sigungu": str(row[COL_SIGUNGU]) if row[COL_SIGUNGU] else "",
            "dong_text": str(row[COL_DONG_TEXT]) if row[COL_DONG_TEXT] else "",
            "addr_road": str(row[COL_ADDR_ROAD]).strip() if row[COL_ADDR_ROAD] else "",
            "addr_jibun": str(row[COL_ADDR_JIBUN]).strip() if row[COL_ADDR_JIBUN] else "",
        })
    wb.close()
    return out


def load_residence_coords() -> dict[str, dict]:
    """단지코드 → {lat, lon, dong_code, sigungu}."""
    out = {}
    with open(RES_CSV, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            # id 포맷: R_A11007001 → 단지코드 = A11007001
            code = r["id"][2:] if r["id"].startswith("R_") else r["id"]
            out[code] = {
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "dong_code": r["dong_code"],
                "sigungu_geo": r["sigungu"],
            }
    return out


def build_poi_list(limit: int = 0) -> list[dict]:
    """xlsx + residence 조인 → poi dict 리스트."""
    apts = load_seoul_xlsx()
    coords = load_residence_coords()
    pois = []
    n_with_coord = 0
    for a in apts:
        c = coords.get(a["code"])
        if c:
            n_with_coord += 1
        pois.append({
            "poi_id": f"APT_{a['code']}",
            "poi_name": a["name"],
            "poi_addr": a["addr_road"] or a["addr_jibun"] or "",
            "poi_lat": c["lat"] if c else None,
            "poi_lon": c["lon"] if c else None,
            "sigungu": a["sigungu"],
            "dong_text": a["dong_text"],
        })
    print(f"[load] Seoul apts: {len(pois)}, with coords: {n_with_coord}")
    if limit > 0:
        pois = pois[:limit]
        print(f"[load] limit applied: {len(pois)}")
    return pois


# === 아파트 전용 매칭 (카테고리 필터로 false positive 차단) ===

from rapidfuzz import fuzz

# Kakao 카카오맵 부동산 카테고리:
#   depth1=부동산, depth2=주거시설, depth3=아파트/오피스텔/주상복합 등
# 충전소·약국·경로당 같은 false positive를 차단하기 위해 사용
RESIDENTIAL_D3 = {
    "아파트", "오피스텔", "주상복합", "빌라", "연립주택", "다세대주택",
    "주택", "단독주택", "타운하우스", "도시형생활주택",
}


def is_residential(c: dict) -> bool:
    d1 = (c.get("cate_name_depth1") or "").strip()
    d2 = (c.get("cate_name_depth2") or "").strip()
    d3 = (c.get("cate_name_depth3") or "").strip()
    if d1 == "부동산":
        return True
    if d2 == "주거시설":
        return True
    if d3 in RESIDENTIAL_D3:
        return True
    return False


def _safe_dist(c: dict, lat, lng) -> float:
    if lat is None or lng is None:
        return 99999.0
    try:
        return _haversine_m(lat, lng, float(c.get("lat", 0)), float(c.get("lon", 0)))
    except (ValueError, TypeError):
        return 99999.0


def _score_and_pick(cands: list[dict], target: str, lat, lng, *,
                    min_sim: int, max_dist: float | None) -> dict | None:
    """주거 카테고리만 필터 → sim·distance 점수로 정렬 → 첫 통과 후보 반환.

    좌표가 None이면 거리 무시 (max_dist 비활성).
    """
    res = [c for c in cands if is_residential(c)]
    if not res:
        return None
    t = _norm(target)
    has_coord = lat is not None and lng is not None
    scored = []
    for c in res:
        n = _norm(c.get("name", ""))
        sim_tsr = fuzz.token_sort_ratio(t, n)
        sim_par = fuzz.partial_ratio(t, n)
        # token_sort + partial 평균 — 부분일치(컨테인먼트)에도 robust
        sim = max(sim_tsr, sim_par * 0.85)
        d = _safe_dist(c, lat, lng) if has_coord else 0.0
        scored.append((sim, d, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    for sim, d, c in scored:
        if has_coord and max_dist is not None and d > max_dist:
            continue
        if sim >= min_sim:
            return c
        if has_coord and d <= 50 and sim >= 40:
            return c
    return None


def best_match_apt(client: KakaoClient, poi: dict) -> tuple[dict | None, str]:
    """아파트 전용 매칭 — 주거 카테고리만 후보. (match, pass_label) 반환."""
    import re
    name = poi["poi_name"]
    lat, lng = poi["poi_lat"], poi["poi_lon"]

    # === Pass 1: 원본 이름 + 좌표 500m ===
    cands = client.search(query=name, lat=lat, lng=lng, radius=500)
    m = _score_and_pick(cands, name, lat, lng, min_sim=55, max_dist=500)
    if m:
        return m, "P1_orig_500m"

    # === Pass 2: "아파트" 접미사 추가/제거 ===
    name_alt = name.replace("아파트", "").strip() if "아파트" in name else f"{name} 아파트"
    cands = client.search(query=name_alt, lat=lat, lng=lng, radius=800)
    m = _score_and_pick(cands, name, lat, lng, min_sim=55, max_dist=800)
    if m:
        return m, "P2_apt_suffix"

    # === Pass 3: "(임대아파트)" 같은 괄호·꼬리표 제거 + 숫자 앞 공백 정규화 ===
    cleaned = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    cleaned = re.sub(r"([가-힣])(\d)", r"\1 \2", cleaned)
    if cleaned != name:
        cands = client.search(query=cleaned, lat=lat, lng=lng, radius=800)
        m = _score_and_pick(cands, name, lat, lng, min_sim=50, max_dist=800)
        if m:
            return m, "P3_cleaned"

    # === Pass 4: 차수("3단지") 제거 — 단일 단지로 잡힌 경우 ===
    no_danji = re.sub(r"\s*\d+단지\s*", " ", name).strip()
    no_danji = re.sub(r"\s+", " ", no_danji)
    if no_danji and no_danji != name:
        cands = client.search(query=no_danji, lat=lat, lng=lng, radius=600)
        m = _score_and_pick(cands, name, lat, lng, min_sim=45, max_dist=300)
        if m:
            return m, "P4_no_danji"

    # === Pass 5: 시군구 + 이름 ===
    if poi["sigungu"]:
        q = f"{poi['sigungu']} {name}"
        cands = client.search(query=q, lat=lat, lng=lng, radius=1500)
        m = _score_and_pick(cands, name, lat, lng, min_sim=50, max_dist=1500)
        if m:
            return m, "P5_sigungu_prefix"

    # === Pass 6: 주소 keyword + 좌표 — 같은 주소의 주거 후보 채택 ===
    if poi["poi_addr"]:
        addr_first = poi["poi_addr"].split(",")[0].strip()
        cands = client.search(query=addr_first, lat=lat, lng=lng, radius=200)
        res = [c for c in cands if is_residential(c)]
        if res and lat is not None:
            # 거리 최단 + sim 30 이상
            res.sort(key=lambda c: _safe_dist(c, lat, lng))
            top = res[0]
            d = _safe_dist(top, lat, lng)
            sim = fuzz.token_sort_ratio(_norm(name), _norm(top.get("name", "")))
            if d <= 150 and sim >= 30:
                return top, "P6_addr_nearby"

    # === Pass 7: 좌표 기반 광역 — 가까운 주거 후보 + 이름 매칭 ===
    if lat is not None and lng is not None:
        cands = client.search(query=name, lat=lat, lng=lng, radius=2000)
        m = _score_and_pick(cands, name, lat, lng, min_sim=45, max_dist=200)
        if m:
            return m, "P7_wide_radius"

    # === Pass 8: 좌표 무시 — 시군구 + 이름, sim 70+ ===
    if poi["sigungu"]:
        cands = client.search(query=f"{poi['sigungu']} {name}")
        m = _score_and_pick(cands, name, None, None, min_sim=70, max_dist=None)
        if m:
            return m, "P8_name_only"

    # === Pass 9: 좌표 무시 — 시도+동+이름 ===
    if poi["dong_text"]:
        cands = client.search(query=f"서울 {poi['dong_text']} {name}")
        m = _score_and_pick(cands, name, None, None, min_sim=65, max_dist=None)
        if m:
            return m, "P9_dong_name"

    # === Pass 10: 변형 이름 — "주상복합" prefix, 후행 숫자, "{동} " prefix 제거 ===
    name_variants = []
    if name.startswith("주상복합"):
        name_variants.append(("strip_jusang", name[4:].strip()))
    no_tail_digit = re.sub(r"\d+$", "", name).strip()
    if no_tail_digit and no_tail_digit != name:
        name_variants.append(("strip_tail_digit", no_tail_digit))
    # "{동명} {본명}" 패턴: 첫 토큰이 K-apt dong_text와 같으면 제거
    first_token = name.split()[0] if name.split() else ""
    if first_token and first_token.endswith("동") and len(name.split()) > 1:
        name_variants.append(("strip_dong_prefix", " ".join(name.split()[1:])))
    for label, v in name_variants:
        cands = client.search(query=v, lat=lat, lng=lng, radius=2000) if lat else client.search(query=v)
        m = _score_and_pick(cands, name, lat, lng, min_sim=45, max_dist=2000)
        if m:
            return m, f"P10_{label}"
        # 동일 변형으로 시군구 prefix
        if poi["sigungu"]:
            cands2 = client.search(query=f"{poi['sigungu']} {v}",
                                   lat=lat, lng=lng,
                                   radius=2000) if lat else client.search(query=f"{poi['sigungu']} {v}")
            m = _score_and_pick(cands2, name, lat, lng, min_sim=45, max_dist=2000)
            if m:
                return m, f"P10sg_{label}"

    # === Pass 11: 카테고리 무시 — 건물명 기반 보조 시설 (전기차충전소·주차장 등) ===
    # 같은 건물명을 가진 다른 facility를 building proxy로 채택. partial_ratio 사용.
    if lat is not None and lng is not None:
        cands = client.search(query=name, lat=lat, lng=lng, radius=500)
        scored = []
        for c in cands:
            n = _norm(c.get("name", ""))
            sim = max(fuzz.token_sort_ratio(_norm(name), n),
                      fuzz.partial_ratio(_norm(name), n))
            d = _safe_dist(c, lat, lng)
            scored.append((sim, d, c))
        scored.sort(key=lambda x: (-x[0], x[1]))
        if scored:
            sim, d, c = scored[0]
            if sim >= 75 and d <= 80:
                return c, "P11_proxy_facility"

    # === Pass 12: 좌표 무시 — partial_ratio 90+ (이름이 카카오 라벨에 포함) ===
    queries = [name]
    if poi["sigungu"]:
        queries.append(f"{poi['sigungu']} {name}")
    seen_pids = set()
    for q in queries:
        cands = client.search(query=q)
        for c in cands:
            pid = str(c.get("confirmid") or c.get("id"))
            if pid in seen_pids:
                continue
            seen_pids.add(pid)
            n = _norm(c.get("name", ""))
            par = fuzz.partial_ratio(_norm(name), n)
            if par >= 90:
                return c, "P12_partial"

    # === Pass 13: 매우 lenient — 변형 이름들에 대해 partial 80+ ===
    base_names = {name}
    if name.startswith("주상복합"):
        base_names.add(name[4:].strip())
    base_names.add(re.sub(r"\s*\([^)]*\)\s*", " ", name).strip())
    base_names.add(re.sub(r"\d+단지\s*$", "", name).strip())
    base_names.add(re.sub(r"\d+$", "", name).strip())
    tok = name.split()
    if tok and tok[0].endswith("동"):
        base_names.add(" ".join(tok[1:]))
    # "단지" 제거
    base_names.add(re.sub(r"\s*단지\s*$", "", name).strip())
    base_names.discard("")
    for v in base_names:
        for q in (v, f"{poi['sigungu']} {v}" if poi["sigungu"] else v):
            cands = client.search(query=q)
            for c in cands:
                n = _norm(c.get("name", ""))
                par = fuzz.partial_ratio(_norm(v), n)
                if par >= 80 and len(_norm(v)) >= 3:
                    return c, "P13_lenient"

    return None, "nomatch"


# === 병렬 워커 ===

_tls = threading.local()


def _client() -> KakaoClient:
    c = getattr(_tls, "client", None)
    if c is None:
        c = KakaoClient(mean_pace=0.6)
        _tls.client = c
    return c


def process_one(poi: dict) -> dict:
    c = _client()
    c.pacer.wait()
    best, pass_label = best_match_apt(c, poi)
    if not best:
        return {"poi": poi, "status": "nomatch", "pass": pass_label}
    kpid = str(best.get("confirmid") or best.get("id"))

    c.pacer.wait()
    panel = c.panel3(kpid)
    if not panel:
        return {"poi": poi, "status": "panel_fail", "pass": pass_label, "kpid": kpid}

    sm = extract_summary(panel)
    rv = extract_reviews(panel, limit=5)
    return {
        "poi": poi, "status": "ok", "kpid": kpid, "pass": pass_label,
        "panel": panel, "summary": sm, "reviews": rv,
        "kakao_name": best.get("name"),
    }


def writer_loop(q: queue.Queue, conn, stop_evt: threading.Event, state: dict,
                jsonl_path: Path):
    jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
    try:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if stop_evt.is_set():
                    break
                continue
            if item is None:
                break
            poi = item["poi"]
            status = item["status"]
            pass_label = item.get("pass", "?")
            state["pass_counts"][pass_label] = state["pass_counts"].get(pass_label, 0) + 1

            if status == "ok":
                mark_matched(conn, poi["poi_id"], item["kpid"])
                save_panel3(conn, item["kpid"], item["panel"])
                mark_fetched(conn, poi["poi_id"])
                bump_quota(conn, 200)
                state["ok"] += 1
                sm = item["summary"]
                jsonl_fp.write(json.dumps({
                    "poi_id": poi["poi_id"],
                    "poi_name": poi["poi_name"],
                    "kakao_pid": item["kpid"],
                    "kakao_name": item["kakao_name"],
                    "pass": pass_label,
                    "rating": sm.get("rating"),
                    "rating_count": sm.get("rating_count"),
                    "menu_count": sm.get("menu_count"),
                    "review_first_page": len(item["reviews"]),
                    "category": sm.get("category"),
                    "lat": sm.get("lat"),
                    "lon": sm.get("lon"),
                }, ensure_ascii=False) + "\n")
                jsonl_fp.flush()
                print(f"  [{pass_label[:14]:14s}] {poi['poi_name'][:25]:25s} → "
                      f"{(item['kakao_name'] or '')[:22]:22s} "
                      f"★{sm.get('rating')}({sm.get('rating_count')})")
            elif status == "nomatch":
                mark_error(conn, poi["poi_id"], "nomatch", "all passes failed")
                state["nomatch"] += 1
                print(f"  [NOMATCH      ] {poi['poi_name'][:40]}")
            elif status == "panel_fail":
                mark_error(conn, poi["poi_id"], "error", "panel3 fail")
                state["err"] += 1
                print(f"  [PANEL_FAIL   ] {poi['poi_name'][:40]} kpid={item['kpid']}")
            conn.commit()
    finally:
        jsonl_fp.close()


def main():
    # Windows console에서 한글 출력 보장
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0,
                    help="0이면 전체 (서울 3164)")
    ap.add_argument("--out-tag", default="apt_seoul")
    args = ap.parse_args()

    pois = build_poi_list(limit=args.limit)
    print(f"=== K-apt → 카카오 크롤 (workers={args.workers}, N={len(pois)}) ===\n")

    # 작성자 스레드와 메인 스레드에서 같은 conn 사용 → check_same_thread=False
    conn = open_db(check_same_thread=False)
    for p in pois:
        upsert_poi(conn, poi_id=p["poi_id"], poi_name=p["poi_name"],
                   poi_addr=p["poi_addr"],
                   poi_lat=p["poi_lat"] or 0.0,
                   poi_lon=p["poi_lon"] or 0.0)
    conn.commit()

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUT_DIR / f"{args.out_tag}_{ts}.jsonl"
    print(f"[out] jsonl: {jsonl_path}\n")

    t0 = time.time()
    q: queue.Queue = queue.Queue(maxsize=args.workers * 4)
    stop = threading.Event()
    state = {"ok": 0, "nomatch": 0, "err": 0, "pass_counts": {}}
    writer = threading.Thread(target=writer_loop,
                              args=(q, conn, stop, state, jsonl_path),
                              daemon=True)
    writer.start()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p) for p in pois]
        for f in futures:
            try:
                q.put(f.result())
            except Exception as e:
                print(f"  [WORKER_EXC] {type(e).__name__}: {e}")

    stop.set()
    q.put(None)
    writer.join(timeout=15)

    elapsed = time.time() - t0
    n = len(pois)
    print(f"\n=== 결과 N={n} ===")
    print(f"  매칭+디테일 성공: {state['ok']} ({state['ok']/n*100:.1f}%)")
    print(f"  nomatch:          {state['nomatch']} ({state['nomatch']/n*100:.1f}%)")
    print(f"  panel 에러:       {state['err']} ({state['err']/n*100:.1f}%)")
    print(f"  매칭률:           {(state['ok']+state['err'])/n*100:.1f}%  ← (nomatch 제외)")
    print(f"  소요:             {elapsed:.1f}s ({elapsed/n:.2f}s/POI 직렬환산, "
          f"{n/elapsed*60:.0f} POI/min)")
    print(f"\n  pass 분포:")
    for k, v in sorted(state["pass_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:20s} : {v}")
    print(f"\n[DB stats] {stats(conn)}")
    print(f"[jsonl]    {jsonl_path}")
    conn.close()


if __name__ == "__main__":
    main()
