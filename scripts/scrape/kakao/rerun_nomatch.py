"""SQLite에서 status='nomatch'인 POI만 골라 enhanced fallback으로 재시도.

run_apt.py가 끝난 후 nomatch만 다시 잡는 용도. 추가 변형:
  - "임대" 접미사 제거
  - "제N" / "제N차" 접두/중간 제거
  - "SH"/"LH"/"GH" 공기업 접두 제거
  - "안심주택"·"청년주택" 제거
  - 더블 스페이스 정규화
  - 카카오 자체 keyword로 주소 → 좌표 prefetch 후 cascade 재시도

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.rerun_nomatch --workers 8
"""
from __future__ import annotations

import argparse
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

from .client import KakaoClient, _haversine_m, _norm
from .extract import extract_reviews, extract_summary
from .run_apt import (
    OUT_DIR, _safe_dist, best_match_apt, build_poi_list,
    is_residential,
)
from .store import (
    DB_PATH, bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats as db_stats,
)


ROMAN_TO_DIGIT = {"Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
                  "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
                  "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5"}


def _normalize_roman(s: str) -> str:
    """로마 숫자(Ⅰ Ⅱ Ⅲ / I II III) → 아라비아 숫자."""
    for r, d in sorted(ROMAN_TO_DIGIT.items(), key=lambda x: -len(x[0])):
        s = s.replace(r, d)
    return s


def _strip_all(name: str) -> str:
    """name에서 자주 등장하는 잡 토큰을 모두 제거 — 가장 공격적인 정규화."""
    s = _normalize_roman(name)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"\s*제\s*\d+\s*차?\s*", " ", s)
    s = re.sub(r"\s*\d+\s*[,，]\s*\d+\s*차\s*$", " ", s)  # "1,2차"
    s = re.sub(r"\s*\d+\s*차\s*\d+\s*차\s*$", " ", s)    # "1차2차"
    s = re.sub(r"\s*임대\s*", " ", s)
    s = re.sub(r"\s*행복주택\s*", " ", s)
    s = re.sub(r"\s*(?:청년)?안심주택\s*", " ", s)
    s = re.sub(r"\s*청년주택\s*", " ", s)
    s = re.sub(r"\s*관리사무소\s*", " ", s)
    s = re.sub(r"\s*관리단\s*$", " ", s)
    s = re.sub(r"\s*\d*관리\s*$", " ", s)
    s = re.sub(r"\s*주상복합(?:빌딩|관리단)?\s*", " ", s)
    s = re.sub(r"\s*빌딩\s*", " ", s)
    s = re.sub(r"\s*초안\s*$", " ", s)
    s = re.sub(r"^주상복합\s*", " ", s)
    for pre in ("SH", "LH", "GH"):
        s = re.sub(rf"\b{pre}[-\s]*v?i?l?l?e?\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def name_variants(name: str) -> list[str]:
    """nomatch 회수용 변형 이름. 중복 제거."""
    out = [name]
    out.append(_normalize_roman(name))
    out.append(re.sub(r"\s*임대\s*$", "", name).strip())
    out.append(re.sub(r"\s*제\s*\d+\s*차?\s*", " ", name).strip())
    for pre in ("SH", "LH", "GH", "민영"):
        if name.startswith(pre):
            out.append(name[len(pre):].strip())
    out.append(re.sub(r"\s*SH[-\s]*v?i?l?l?e?\s*", " ", name).strip())
    out.append(re.sub(r"\s*(?:청년)?안심주택\s*$", "", name).strip())
    out.append(re.sub(r"\s*청년주택\s*$", "", name).strip())
    out.append(re.sub(r"\s*행복주택\s*$", "", name).strip())
    out.append(re.sub(r"\s*\d*관리\s*$", "", name).strip())
    out.append(re.sub(r"\s*관리사무소\s*$", "", name).strip())
    out.append(re.sub(r"\s*관리단\s*$", "", name).strip())
    out.append(re.sub(r"\s*초안\s*$", "", name).strip())
    out.append(re.sub(r"\s*주상복합\s*$", "", name).strip())
    out.append(re.sub(r"\s*주상복합빌딩\s*$", "", name).strip())
    out.append(re.sub(r"\s*주상복합관리단\s*$", "", name).strip())
    out.append(re.sub(r"\s*빌딩\s*$", "", name).strip())
    # "1차2차" / "1,2차" 후행
    out.append(re.sub(r"\s*\d+\s*[,，]\s*\d+\s*차\s*$", "", name).strip())
    out.append(re.sub(r"\s*\d+\s*차\s*\d+\s*차\s*$", "", name).strip())
    out.append(re.sub(r"\s*\d+\s*차\s*$", "", name).strip())
    # "아파트" suffix strip
    out.append(re.sub(r"\s*아파트\s*$", "", name).strip())
    norm = re.sub(r"\s+", " ", name).strip()
    out.append(norm)
    out.append(re.sub(r"\s*\([^)]*\)\s*", " ", name).strip())
    out.append(re.sub(r"\d+$", "", name).strip())
    out.append(re.sub(r"\d+단지\s*$", "", name).strip())
    out.append(re.sub(r"단지\s*\d+\s*[가-힣]*\s*$", "단지", name).strip())
    out.append(_strip_all(name))  # 모든 잡 토큰 제거
    tok = norm.split()
    if tok and tok[-1] in ("단지", "아파트", "임대", "주택", "관리", "초안",
                          "빌딩", "관리단", "관리사무소", "주상복합"):
        out.append(" ".join(tok[:-1]))
    if tok and tok[0].endswith("동") and len(tok) > 1:
        out.append(" ".join(tok[1:]))
    m = re.match(r"^([가-힣]{2,4}동)([가-힣A-Za-z].+)$", norm)
    if m:
        out.append(m.group(2).strip())
    # 끝 두 토큰 보조어 제거 — "상도건영제2 관리사무소" → "상도건영"
    if len(tok) >= 3:
        if tok[-1] in ("관리사무소", "주상복합", "빌딩", "관리단"):
            out.append(" ".join(tok[:-2]))
        # "이수자이 주상복합" → "이수자이"

    seen, final = set(), []
    for v in out:
        v = re.sub(r"\s+", " ", v).strip()
        if v and len(v) >= 2 and v not in seen:
            seen.add(v)
            final.append(v)
    return final


def addr_to_coords(client: KakaoClient, addr: str) -> tuple[float, float] | None:
    """주소 keyword 검색으로 카카오 좌표 추정. 첫 결과 사용."""
    if not addr:
        return None
    addr_first = addr.split(",")[0].strip()
    cands = client.search(query=addr_first)
    if not cands:
        return None
    try:
        c = cands[0]
        return float(c["lat"]), float(c["lon"])
    except (KeyError, ValueError, TypeError):
        return None


def enhanced_best_match(client: KakaoClient, poi: dict) -> tuple[dict | None, str]:
    """기존 cascade가 실패한 후 추가 시도."""
    # 1) 좌표 없으면 주소로 prefetch
    if poi["poi_lat"] is None and poi["poi_addr"]:
        coords = addr_to_coords(client, poi["poi_addr"])
        if coords:
            poi = {**poi, "poi_lat": coords[0], "poi_lon": coords[1]}

    # 2) 표준 cascade 재시도
    m, label = best_match_apt(client, poi)
    if m:
        return m, f"E1_{label}"

    # 3) 변형 이름들에 대해 비좌표 검색 + 주거 카테고리 + 변형 partial
    name = poi["poi_name"]
    variants = name_variants(name)
    lat, lng = poi["poi_lat"], poi["poi_lon"]
    for v in variants:
        queries = [v]
        if poi.get("sigungu"):
            queries.append(f"{poi['sigungu']} {v}")
        for q in queries:
            kw = {"query": q}
            if lat is not None and lng is not None:
                kw.update({"lat": lat, "lng": lng, "radius": 2000})
            cands = client.search(**kw)
            res = [c for c in cands if is_residential(c)]
            for cand_list, label_suffix, min_par in (
                (res, "res", 60),
                (cands, "any", 85),
            ):
                if not cand_list:
                    continue
                scored = []
                for c in cand_list:
                    n = _norm(c.get("name", ""))
                    par = fuzz.partial_ratio(_norm(v), n)
                    tsr = fuzz.token_sort_ratio(_norm(v), n)
                    par_orig = fuzz.partial_ratio(_norm(name), n)
                    sim = max(par, tsr, par_orig)
                    d = _safe_dist(c, lat, lng) if (lat is not None) else 0.0
                    scored.append((sim, d, c))
                scored.sort(key=lambda x: (-x[0], x[1]))
                top_sim, top_d, top_c = scored[0]
                if top_sim >= min_par:
                    if lat is None or top_d <= 3000:
                        return top_c, f"E2_var_{label_suffix}"

    # 4) 좌표가 있으면 — radius 300m + 주거 카테고리 → 가장 가까운 후보 (이름 무시)
    if lat is not None and lng is not None:
        # 가장 넓은 쿼리들로 후보 모으기
        all_cands: dict[str, dict] = {}
        for q in (name, name.replace("아파트", "").strip(), poi.get("sigungu", "") + " 아파트"):
            if not q:
                continue
            cands = client.search(query=q, lat=lat, lng=lng, radius=300)
            for c in cands:
                pid = str(c.get("confirmid") or c.get("id"))
                if pid and pid not in all_cands:
                    all_cands[pid] = c
        # 주거 카테고리만 거리순
        res_close = [c for c in all_cands.values() if is_residential(c)]
        if res_close:
            res_close.sort(key=lambda c: _safe_dist(c, lat, lng))
            top = res_close[0]
            d = _safe_dist(top, lat, lng)
            if d <= 200:
                return top, f"E4_nearest_res_{int(d)}m"

    # 5) 주소 keyword 검색 → 거리 가까운 주거
    if poi.get("poi_addr"):
        addr_first = poi["poi_addr"].split(",")[0].strip()
        kw = {"query": addr_first}
        if lat is not None and lng is not None:
            kw.update({"lat": lat, "lng": lng, "radius": 300})
        cands = client.search(**kw)
        res_addr = [c for c in cands if is_residential(c)]
        if res_addr:
            if lat is not None:
                res_addr.sort(key=lambda c: _safe_dist(c, lat, lng))
                top = res_addr[0]
                d = _safe_dist(top, lat, lng)
                if d <= 300:
                    return top, f"E5_addr_proxy_{int(d)}m"
            else:
                return res_addr[0], "E5_addr_proxy_nolat"
        # 주거 후보 없으면 어떤 후보든 채택 (건물 proxy)
        if cands and lat is not None:
            cands_sorted = sorted(cands, key=lambda c: _safe_dist(c, lat, lng))
            top = cands_sorted[0]
            d = _safe_dist(top, lat, lng)
            if d <= 150:
                return top, f"E5_any_proxy_{int(d)}m"

    # 6.5) 좌표 무관 검색 → ANY 카테고리, partial>=65 + (좌표 있으면) d<=200m
    all_cands: dict[str, dict] = {}
    # 좌표 없이 모든 변형으로 검색 (Kakao radius 필터가 너무 엄격해서 좌표 빼고 시도)
    for v in [name] + variants[:8]:
        for q in (v, f"{poi.get('sigungu','')} {v}".strip() if poi.get('sigungu') else v):
            cands = client.search(query=q)
            for c in cands:
                pid = str(c.get("confirmid") or c.get("id"))
                if pid and pid not in all_cands:
                    all_cands[pid] = c
    best = None
    best_par = 0
    best_d = 99999
    for c in all_cands.values():
        n = _norm(c.get("name", ""))
        par = max(
            fuzz.partial_ratio(_norm(name), n),
            *(fuzz.partial_ratio(_norm(v), n) for v in variants[:5]),
        )
        d = _safe_dist(c, lat, lng) if (lat is not None) else 0.0
        if par >= 65:
            # 좌표 있으면 200m 이내만, 없으면 거리 무시
            if lat is None or d <= 200:
                if par > best_par or (par == best_par and d < best_d):
                    best = c
                    best_par = par
                    best_d = d
    if best:
        return best, f"E6.5_proxy_lenient_par{best_par:.0f}_{int(best_d)}m"

    # 6) 최후 — 시군구만으로 lenient
    if poi.get("sigungu") and name:
        cands = client.search(query=f"{poi['sigungu']} {name}")
        if cands:
            for c in cands:
                n = _norm(c.get("name", ""))
                par = fuzz.partial_ratio(_norm(name), n)
                if par >= 70 and len(_norm(name)) >= 3:
                    return c, "E6_sigungu_lenient"
        # 변형으로도 시도
        for v in variants[:3]:
            cands = client.search(query=f"{poi['sigungu']} {v}")
            for c in cands:
                n = _norm(c.get("name", ""))
                par = fuzz.partial_ratio(_norm(v), n)
                if par >= 70:
                    return c, "E6_sigungu_var"

    return None, "nomatch"


_tls = threading.local()


def _client():
    c = getattr(_tls, "client", None)
    if c is None:
        c = KakaoClient(mean_pace=0.5)
        _tls.client = c
    return c


def process_one(poi: dict) -> dict:
    c = _client()
    c.pacer.wait()
    best, label = enhanced_best_match(c, poi)
    if not best:
        return {"poi": poi, "status": "nomatch", "pass": label}
    kpid = str(best.get("confirmid") or best.get("id"))
    c.pacer.wait()
    panel = c.panel3(kpid)
    if not panel:
        return {"poi": poi, "status": "panel_fail", "pass": label, "kpid": kpid}
    sm = extract_summary(panel)
    rv = extract_reviews(panel, limit=5)
    return {"poi": poi, "status": "ok", "kpid": kpid, "pass": label,
            "panel": panel, "summary": sm, "reviews": rv,
            "kakao_name": best.get("name")}


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                                   errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-tag", default="apt_rerun_nomatch")
    args = ap.parse_args()

    # SQLite에서 nomatch만 pull, full xlsx에서 join (sigungu/dong_text 필요)
    conn = open_db(check_same_thread=False)
    rows = conn.execute(
        "SELECT poi_id, poi_name, poi_addr, poi_lat, poi_lon "
        "FROM poi_status WHERE status='nomatch'"
    ).fetchall()
    print(f"[nomatch] {len(rows)} POIs to rerun")
    if not rows:
        print("[done] nothing to do")
        return

    # full xlsx에서 sigungu/dong_text 보강 (best_match_apt에서 사용)
    xlsx_pois = build_poi_list(limit=0)
    xlsx_idx = {p["poi_id"]: p for p in xlsx_pois}

    pois = []
    for poi_id, poi_name, poi_addr, lat, lon in rows:
        meta = xlsx_idx.get(poi_id, {})
        pois.append({
            "poi_id": poi_id,
            "poi_name": poi_name,
            "poi_addr": poi_addr or meta.get("poi_addr", ""),
            "poi_lat": (lat if lat and lat != 0.0 else None),
            "poi_lon": (lon if lon and lon != 0.0 else None),
            "sigungu": meta.get("sigungu", ""),
            "dong_text": meta.get("dong_text", ""),
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUT_DIR / f"{args.out_tag}_{ts}.jsonl"
    print(f"[out] jsonl: {jsonl_path}\n")

    t0 = time.time()
    state = {"ok": 0, "nomatch": 0, "err": 0, "pass_counts": {}}
    jsonl_fp = open(jsonl_path, "w", encoding="utf-8")

    def consume(item):
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
                "kakao_pid": item["kpid"], "kakao_name": item["kakao_name"],
                "pass": label, "rating": sm.get("rating"),
                "rating_count": sm.get("rating_count"),
                "menu_count": sm.get("menu_count"),
                "review_first_page": len(item["reviews"]),
                "category": sm.get("category"),
                "lat": sm.get("lat"), "lon": sm.get("lon"),
            }, ensure_ascii=False) + "\n")
            jsonl_fp.flush()
            print(f"  [{label[:14]:14s}] {poi['poi_name'][:25]:25s} → "
                  f"{(item['kakao_name'] or '')[:25]:25s} ★{sm.get('rating')}({sm.get('rating_count')})")
        elif st == "nomatch":
            mark_error(conn, poi["poi_id"], "nomatch", "enhanced fallback failed")
            state["nomatch"] += 1
            print(f"  [STILL_NOMATCH ] {poi['poi_name'][:40]}")
        elif st == "panel_fail":
            mark_error(conn, poi["poi_id"], "error", "panel3 fail (rerun)")
            state["err"] += 1
        conn.commit()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p) for p in pois]
        for f in futures:
            try:
                consume(f.result())
            except Exception as e:
                print(f"  [WORKER_EXC] {type(e).__name__}: {e}")
    jsonl_fp.close()

    elapsed = time.time() - t0
    n = len(pois)
    print(f"\n=== rerun 결과 N={n} ===")
    print(f"  복구 성공:    {state['ok']} ({state['ok']/n*100:.1f}%)")
    print(f"  여전히 nomatch: {state['nomatch']}")
    print(f"  panel 에러:    {state['err']}")
    print(f"  소요:         {elapsed:.1f}s")
    print(f"\n  pass 분포:")
    for k, v in sorted(state["pass_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:24s} : {v}")
    print(f"\n[DB stats] {db_stats(conn)}")
    print(f"[jsonl]    {jsonl_path}")
    conn.close()


if __name__ == "__main__":
    main()
