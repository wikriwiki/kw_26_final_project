"""소상공인시장진흥공단 상가(상권) 정보 csv → 카카오 리뷰/별점 풀크롤 러너.

흐름:
  1. data/neo4j_load/pois/소상공인시장진흥공단_상가(상권)정보_서울_*.csv 로드
  2. 컬럼 매핑: 상가업소번호(0)/상호명(1)/지점명(2)/시군구명(14)/도로명주소(31)/경도(37)/위도(38)
  3. (선택) 시군구 필터, 업종 필터, limit
  4. 병렬 카카오 best_match + panel3
  5. SQLite (poi_id prefix=COM_) + JSONL 리포트

상가는 좌표·이름·지점명 다 있어서 매칭률 매우 높음. cascade는 run_apt와 유사하지만:
  - 주거 카테고리 필터 사용 X (가게는 다양한 카테고리)
  - 대신 "{상호} {지점}" 결합·이름 정규화 등 가게-특화 변형

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.run_commerce \
    --csv "data/neo4j_load/pois/소상공인시장진흥공단_상가(상권)정보_서울_202603.csv" \
    --workers 16 --limit 0 --gu "강남구"   # gu 비우면 서울 전체
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import queue
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rapidfuzz import fuzz

from .client import KakaoClient, _norm
from .extract import extract_reviews, extract_summary
from .run_apt import _safe_dist, _haversine_m  # 재사용
from .store import (
    bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats as db_stats, upsert_poi,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "output" / "scrape" / "kakao"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 소상공인 csv 표준 39컬럼 인덱스
COL_ID = 0          # 상가업소번호
COL_NAME = 1        # 상호명
COL_BRANCH = 2      # 지점명
COL_CAT_L = 4       # 대분류명
COL_CAT_M = 6       # 중분류명
COL_CAT_S = 8       # 소분류명
COL_SIDO = 12       # 시도명
COL_SIGUNGU = 14    # 시군구명
COL_DONG_CD = 15    # 행정동코드
COL_DONG_NM = 16    # 행정동명
COL_ADDR_JIBUN = 24 # 지번주소
COL_ADDR_ROAD = 31  # 도로명주소
COL_LON = 37        # 경도
COL_LAT = 38        # 위도


def load_csv_robust(csv_path: Path, *, sido: str = "서울특별시",
                    gu: str | None = None,
                    cat_l: str | None = None,
                    limit: int = 0) -> list[dict]:
    """소상공인 csv 로드. Drive Stream/utf-8 손상 강건. errors='replace'."""
    out = []
    with open(csv_path, "rb") as f:
        raw = f.read()
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    # null padding 잘라내기 (손상된 파일 보호)
    null_start = raw.find(b"\x00\x00\x00\x00\x00\x00\x00\x00")
    if null_start > 0:
        raw = raw[:null_start]
    text = raw.decode("utf-8", errors="replace")
    rdr = csv.reader(io.StringIO(text))
    hdr = next(rdr, None)
    if not hdr or len(hdr) < 39:
        print(f"[warn] header has {len(hdr) if hdr else 0} cols (expected 39)")
    for row in rdr:
        if len(row) < 39:
            continue
        if sido and row[COL_SIDO] != sido:
            continue
        if gu and row[COL_SIGUNGU] != gu:
            continue
        if cat_l and row[COL_CAT_L] != cat_l:
            continue
        try:
            lon = float(row[COL_LON])
            lat = float(row[COL_LAT])
        except (ValueError, TypeError):
            lon = lat = None
        name = row[COL_NAME].strip()
        branch = row[COL_BRANCH].strip()
        if not name:
            continue
        full_name = f"{name} {branch}".strip() if branch else name
        out.append({
            "poi_id": f"COM_{row[COL_ID]}",
            "poi_name": full_name,
            "poi_brand": name,           # 상호 (지점 제외)
            "poi_branch": branch,        # 지점명만
            "poi_addr": (row[COL_ADDR_ROAD] or row[COL_ADDR_JIBUN] or "").strip(),
            "poi_lat": lat,
            "poi_lon": lon,
            "sigungu": row[COL_SIGUNGU],
            "dong_text": row[COL_DONG_NM],
            "cat_l": row[COL_CAT_L],
            "cat_m": row[COL_CAT_M],
            "cat_s": row[COL_CAT_S],
        })
        if limit > 0 and len(out) >= limit:
            break
    return out


# === 상가 전용 매칭 cascade ===

def _score_commerce(cands: list[dict], target: str, lat, lng, *,
                    min_sim: int, max_dist: float | None) -> dict | None:
    """일반 카테고리 cands에 대해 sim·distance 점수로 정렬 후 첫 통과."""
    if not cands:
        return None
    t = _norm(target)
    has_coord = lat is not None and lng is not None
    scored = []
    for c in cands:
        n = _norm(c.get("name", ""))
        sim = max(fuzz.token_sort_ratio(t, n),
                  fuzz.partial_ratio(t, n) * 0.9)
        d = _safe_dist(c, lat, lng) if has_coord else 0.0
        scored.append((sim, d, c))
    scored.sort(key=lambda x: (-x[0], x[1]))
    for sim, d, c in scored:
        if has_coord and max_dist is not None and d > max_dist:
            continue
        if sim >= min_sim:
            return c
        # 매우 근접 + 약한 sim
        if has_coord and d <= 30 and sim >= 35:
            return c
    return None


def _name_cleaned_variants(name: str) -> list[str]:
    """가게 이름 정규화 변형. 좌표·이름 둘 다 통과하는 strict cascade에서만 사용.

    반환 목록은 원본 + 안전한 정규화만 (잘못 매칭될 위험 없는 것):
      - "주" / "주식회사" / "(주)" 후행·전행 제거
      - 괄호 안 제거
      - 공백 정규화
    """
    out = [name]
    out.append(re.sub(r"\s*\(?주\)?\s*$", "", name).strip())
    out.append(re.sub(r"^\(?주\)?\s*", "", name).strip())
    out.append(re.sub(r"\s*주식회사\s*", " ", name).strip())
    out.append(re.sub(r"\s*유한회사\s*", " ", name).strip())
    out.append(re.sub(r"\s*\([^)]*\)\s*", " ", name).strip())
    out.append(re.sub(r"\s+", " ", name).strip())
    seen, final = set(), []
    for v in out:
        v = re.sub(r"\s+", " ", v).strip()
        if v and len(v) >= 2 and v not in seen:
            seen.add(v)
            final.append(v)
    return final


# === SMART cascade ===
# 1) POI명으로 카카오 keyword 검색 (좌표 bias 없이) → 카카오 자체 best-match ranking 사용
# 2) 반환된 후보들의 좌표를 입력 좌표와 비교 → 임계값 내 + 이름 sim 통과 시 채택
# 둘 다 통과해야만 매칭 (false positive 차단). 통과 못하면 nomatch.

def _smart_pick(cands: list[dict], target: str, lat, lng, *,
                min_sim: int, max_dist: float,
                check_top_n: int = 10) -> dict | None:
    """카카오 검색 결과 top N 중 이름 sim + 좌표 거리 둘 다 통과하는 후보 채택.

    카카오는 query 자체로 best-match ranking을 줌. 우리는 그걸 신뢰하되
    좌표 거리로 false positive를 차단.
    """
    if lat is None or lng is None:
        return None
    if not cands:
        return None
    t = _norm(target)
    scored = []
    for c in cands[:check_top_n]:  # 카카오 ranking 상위 N만
        n = _norm(c.get("name", ""))
        sim = max(fuzz.token_sort_ratio(t, n), fuzz.partial_ratio(t, n))
        try:
            d = _haversine_m(lat, lng,
                             float(c.get("lat", 0)),
                             float(c.get("lon", 0)))
        except (ValueError, TypeError):
            continue
        scored.append((sim, d, c))
    if not scored:
        return None
    # sim 높은 순, 거리 짧은 순
    scored.sort(key=lambda x: (-x[0], x[1]))
    for sim, d, c in scored:
        if sim >= min_sim and d <= max_dist:
            return c
    return None


def best_match_commerce(client: KakaoClient, poi: dict) -> tuple[dict | None, str]:
    """SMART cascade — 좌표 hint를 시군구급(2km) 폭으로 주어 검색하고
    결과를 좌표·이름 sim 임계값으로 검증.

    핵심 원리:
      - radius=2000m hint → 카카오 결과를 입력 좌표 같은 시군구로 유도 (이름 ranking 유지)
      - 결과의 좌표를 max_dist=200m로 검증 → false positive 차단
      - 매칭 못 잡으면 nomatch

    Pass:
      P1: 원본 이름 + 2km hint → top 10, sim ≥ 60 + d ≤ 200m
      P2: 원본 이름 + 5km hint → top 10, sim ≥ 60 + d ≤ 200m (1단계가 결과 없을 때)
      P3: 브랜드만 + 2km hint → top 10, sim ≥ 60 + d ≤ 200m
      P4: 정규화 변형 + 2km hint → top 10, sim ≥ 60 + d ≤ 200m
    """
    name = poi["poi_name"]
    brand = poi["poi_brand"]
    lat, lng = poi["poi_lat"], poi["poi_lon"]

    if lat is None or lng is None:
        return None, "nomatch_no_coord"

    # P1: 원본 이름 + 2km hint
    cands = client.search(query=name, lat=lat, lng=lng, radius=2000)
    m = _smart_pick(cands, name, lat, lng, min_sim=60, max_dist=200)
    if m:
        return m, "P1_smart_2km"

    # P2: 원본 이름 + 5km hint (1단계 결과 없을 때 대비)
    cands = client.search(query=name, lat=lat, lng=lng, radius=5000)
    m = _smart_pick(cands, name, lat, lng, min_sim=60, max_dist=200)
    if m:
        return m, "P2_smart_5km"

    # P3: 브랜드만 (지점명 제외) + 2km hint
    if brand and brand != name:
        cands = client.search(query=brand, lat=lat, lng=lng, radius=2000)
        m = _smart_pick(cands, name, lat, lng, min_sim=60, max_dist=200)
        if m:
            return m, "P3_brand_2km"

    # P4: 정규화 변형 (괄호·주식회사 제거) + 2km hint
    for v in _name_cleaned_variants(name):
        if v == name:
            continue
        cands = client.search(query=v, lat=lat, lng=lng, radius=2000)
        m = _smart_pick(cands, name, lat, lng, min_sim=60, max_dist=200)
        if m:
            return m, "P4_cleaned_2km"

    return None, "nomatch"


# === 병렬 워커 ===

_tls = threading.local()


def _client():
    c = getattr(_tls, "client", None)
    if c is None:
        # 풀런용 — 워커당 quota 100k, pace는 cli에서 옵션화
        c = KakaoClient(mean_pace=_MEAN_PACE, daily_limit=100_000)
        _tls.client = c
    return c


_MEAN_PACE = 0.5  # main()에서 갱신


def process_one(poi: dict) -> dict:
    c = _client()
    c.pacer.wait()
    best, label = best_match_commerce(c, poi)
    if not best:
        return {"poi": poi, "status": "nomatch", "pass": label}
    kpid = str(best.get("confirmid") or best.get("id"))
    c.pacer.wait()
    panel = c.panel3(kpid)
    if not panel:
        return {"poi": poi, "status": "panel_fail", "pass": label, "kpid": kpid}
    sm = extract_summary(panel)
    rv = extract_reviews(panel, limit=5)
    return {
        "poi": poi, "status": "ok", "kpid": kpid, "pass": label,
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
            st = item["status"]
            label = item.get("pass", "?")
            state["pass_counts"][label] = state["pass_counts"].get(label, 0) + 1
            if st == "ok":
                mark_matched(conn, poi["poi_id"], item["kpid"])
                save_panel3(conn, item["kpid"], item["panel"])
                mark_fetched(conn, poi["poi_id"])
                bump_quota(conn, 200)
                state["ok"] += 1
                sm = item["summary"]
                jsonl_fp.write(json.dumps({
                    "poi_id": poi["poi_id"], "poi_name": poi["poi_name"],
                    "poi_brand": poi["poi_brand"],
                    "sigungu": poi["sigungu"], "dong_text": poi["dong_text"],
                    "cat_s": poi.get("cat_s"),
                    "kakao_pid": item["kpid"], "kakao_name": item["kakao_name"],
                    "pass": label, "rating": sm.get("rating"),
                    "rating_count": sm.get("rating_count"),
                    "menu_count": sm.get("menu_count"),
                    "review_first_page": len(item["reviews"]),
                    "category": sm.get("category"),
                    "lat": sm.get("lat"), "lon": sm.get("lon"),
                }, ensure_ascii=False) + "\n")
                jsonl_fp.flush()
                if state["ok"] % 50 == 0:
                    print(f"  [{label[:14]:14s}] {poi['poi_name'][:25]:25s} → "
                          f"{(item['kakao_name'] or '')[:25]:25s} "
                          f"★{sm.get('rating')}({sm.get('rating_count')})  "
                          f"({state['ok']}/{state.get('total','?')})")
            elif st == "nomatch":
                mark_error(conn, poi["poi_id"], "nomatch", "all passes failed")
                state["nomatch"] += 1
            elif st == "panel_fail":
                mark_error(conn, poi["poi_id"], "error", "panel3 fail")
                state["err"] += 1
            conn.commit()
    finally:
        jsonl_fp.close()


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "data" / "neo4j_load" / "pois" /
                                          "소상공인시장진흥공단_상가(상권)정보_서울_202603.csv"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--pace", type=float, default=0.5, help="평균 inter-arrival sec")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--gu", default="", help="시군구명 (빈값=서울 전체)")
    ap.add_argument("--cat-l", default="", help="대분류명 필터")
    ap.add_argument("--out-tag", default="commerce_seoul")
    args = ap.parse_args()
    global _MEAN_PACE
    _MEAN_PACE = args.pace

    csv_path = Path(args.csv)
    print(f"[load] {csv_path.name}")
    pois = load_csv_robust(csv_path, gu=args.gu or None,
                            cat_l=args.cat_l or None, limit=args.limit)
    print(f"[load] commerce POIs: {len(pois)}")
    if not pois:
        print("[stop] no POIs")
        return

    print(f"=== 상가 → 카카오 크롤 (workers={args.workers}, N={len(pois)}) ===\n")
    conn = open_db(check_same_thread=False)
    for p in pois:
        upsert_poi(conn, poi_id=p["poi_id"], poi_name=p["poi_name"],
                   poi_addr=p["poi_addr"],
                   poi_lat=p["poi_lat"] or 0.0,
                   poi_lon=p["poi_lon"] or 0.0)
    conn.commit()

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = args.out_tag
    if args.gu:
        tag = f"{tag}_{args.gu}"
    jsonl_path = OUT_DIR / f"{tag}_{ts}.jsonl"
    print(f"[out] jsonl: {jsonl_path}\n")

    t0 = time.time()
    q: queue.Queue = queue.Queue(maxsize=args.workers * 4)
    stop = threading.Event()
    state = {"ok": 0, "nomatch": 0, "err": 0, "pass_counts": {},
             "total": len(pois)}
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
    writer.join(timeout=20)

    elapsed = time.time() - t0
    n = len(pois)
    print(f"\n=== 결과 N={n} ===")
    print(f"  매칭 성공: {state['ok']} ({state['ok']/n*100:.1f}%)")
    print(f"  nomatch:   {state['nomatch']} ({state['nomatch']/n*100:.1f}%)")
    print(f"  panel 에러:{state['err']}")
    print(f"  소요:      {elapsed:.1f}s ({n/elapsed*60:.0f} POI/min)")
    print(f"\n  pass 분포:")
    for k, v in sorted(state["pass_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:24s} : {v}")
    print(f"\n[DB stats] {db_stats(conn)}")
    print(f"[jsonl]    {jsonl_path}")
    conn.close()


if __name__ == "__main__":
    main()
