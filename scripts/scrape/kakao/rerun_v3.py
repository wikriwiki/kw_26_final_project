"""B2C nomatch 가게에 v3 cascade로 재시도 (시장·B2B 제외 후).

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.rerun_v3 --workers 24 --pace 0.4
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import KakaoClient
from .extract import extract_reviews, extract_summary
from .match_v3 import best_match_v3
from .store import (
    bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats as db_stats,
)

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "output" / "scrape" / "kakao"
CSV_PATH = Path("C:/Users/Administrator/naver_crawl/소상공인시장진흥공단_상가(상권)정보_서울_202512.csv")

_MEAN_PACE = 0.5
_tls = threading.local()


def _client():
    c = getattr(_tls, "client", None)
    if c is None:
        c = KakaoClient(mean_pace=_MEAN_PACE, daily_limit=100_000)
        _tls.client = c
    return c


def process_one(poi: dict) -> dict:
    c = _client()
    c.pacer.wait()
    best, label = best_match_v3(c, poi)
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


def writer_loop(q, conn, stop_evt, state, jsonl_path):
    jsonl_fp = open(jsonl_path, "a", encoding="utf-8")
    try:
        while True:
            try:
                item = q.get(timeout=0.5)
            except queue.Empty:
                if stop_evt.is_set(): break
                continue
            if item is None: break
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
                    "poi_brand": poi["poi_brand"], "sigungu": poi["sigungu"],
                    "dong_text": poi["dong_text"], "cat_s": poi.get("cat_s"),
                    "kakao_pid": item["kpid"], "kakao_name": item["kakao_name"],
                    "pass": label, "rating": sm.get("rating"),
                    "rating_count": sm.get("rating_count"),
                    "menu_count": sm.get("menu_count"),
                    "review_first_page": len(item["reviews"]),
                    "category": sm.get("category"),
                    "lat": sm.get("lat"), "lon": sm.get("lon"),
                }, ensure_ascii=False) + "\n")
                jsonl_fp.flush()
                if state["ok"] % 100 == 0:
                    print(f"  [{label[:14]:14s}] {poi['poi_name'][:25]:25s} → "
                          f"{(item['kakao_name'] or '')[:25]:25s} "
                          f"★{sm.get('rating')}({sm.get('rating_count')})  ({state['ok']}/{state['total']})")
            elif st == "nomatch":
                # 이미 nomatch였던 가게 — error_msg만 갱신
                mark_error(conn, poi["poi_id"], "nomatch", "v3 cascade failed")
                state["nomatch"] += 1
            else:
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
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--pace", type=float, default=0.4)
    ap.add_argument("--out-tag", default="commerce_b2c_v3")
    args = ap.parse_args()
    global _MEAN_PACE
    _MEAN_PACE = args.pace

    # CSV 메타 로드
    print("[load] csv meta ...")
    meta = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        rdr = csv.reader(f); next(rdr)
        for row in rdr:
            if len(row) < 39: continue
            pid = f'COM_{row[0]}'
            meta[pid] = {
                "brand": row[1], "branch": row[2], "sigungu": row[14],
                "dong": row[16], "cat_l": row[4], "cat_s": row[8],
                "addr": row[31] or row[24],
            }

    # DB에서 nomatch만 pull (시장은 이미 삭제됨)
    conn = open_db(check_same_thread=False)
    rows = conn.execute(
        "SELECT poi_id, poi_name, poi_addr, poi_lat, poi_lon "
        "FROM poi_status WHERE poi_id LIKE 'COM_%' AND status='nomatch'"
    ).fetchall()
    print(f"[load] B2C nomatch (시장 제외): {len(rows):,}")

    pois = []
    for poi_id, poi_name, poi_addr, lat, lon in rows:
        m = meta.get(poi_id, {})
        pois.append({
            "poi_id": poi_id, "poi_name": poi_name, "poi_addr": poi_addr,
            "poi_lat": (lat if lat and lat != 0.0 else None),
            "poi_lon": (lon if lon and lon != 0.0 else None),
            "poi_brand": m.get("brand", poi_name),
            "poi_branch": m.get("branch", ""),
            "sigungu": m.get("sigungu", ""),
            "dong_text": m.get("dong", ""),
            "cat_l": m.get("cat_l", ""),
            "cat_s": m.get("cat_s", ""),
        })

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = OUT_DIR / f"{args.out_tag}_{ts}.jsonl"
    print(f"[out] jsonl: {jsonl_path}\n")

    state = {"ok": 0, "nomatch": 0, "err": 0, "pass_counts": {}, "total": len(pois)}
    q = queue.Queue(maxsize=args.workers * 4)
    stop = threading.Event()
    writer = threading.Thread(target=writer_loop,
                              args=(q, conn, stop, state, jsonl_path), daemon=True)
    writer.start()

    t0 = time.time()
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
    print(f"\n=== rerun v3 결과 N={n} ===")
    print(f"  회수 성공:    {state['ok']:,} ({state['ok']/n*100:.1f}%)")
    print(f"  여전히 nomatch: {state['nomatch']:,}")
    print(f"  panel 에러:    {state['err']:,}")
    print(f"  소요:         {elapsed/60:.1f} min")
    print(f"\n  pass 분포:")
    for k, v in sorted(state["pass_counts"].items(), key=lambda x: -x[1]):
        print(f"    {k:30s}: {v:,}")
    print(f"\n[DB stats] {db_stats(conn)}")
    print(f"[jsonl]    {jsonl_path}")
    conn.close()


if __name__ == "__main__":
    main()
