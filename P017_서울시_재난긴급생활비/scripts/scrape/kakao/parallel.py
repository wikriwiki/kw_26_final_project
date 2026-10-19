"""병렬 크롤 러너. ThreadPoolExecutor + thread-local 세션.

각 워커가:
  - 자체 KakaoClient (자체 세션·페이서)
  - poi 받아 search + panel3 호출
  - 결과를 큐로 보냄
단일 라이터 스레드가 SQLite에 일괄 기록 (lock 경합 회피).

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.parallel --n 20 --workers 8 --gu 11680
"""
from __future__ import annotations

import argparse
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from .client import KakaoClient
from .extract import extract_reviews, extract_summary
from .store import (
    bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats, upsert_poi,
)

# Thread-local client (1 session per worker thread)
_tls = threading.local()


def _client() -> KakaoClient:
    c = getattr(_tls, "client", None)
    if c is None:
        c = KakaoClient(mean_pace=0.7)  # 각 워커가 0.7s 평균 → N워커 합산 ~N/0.7 req/s
        _tls.client = c
    return c


def process_one(poi: dict) -> dict:
    """워커 함수: poi 1개 → search + panel3 → 결과 dict."""
    c = _client()
    c.pacer.wait()
    best = c.best_match(
        poi_name=poi["poi_name"], lat=poi["poi_lat"], lng=poi["poi_lon"],
    )
    if not best:
        return {"poi": poi, "status": "nomatch"}
    kpid = str(best.get("confirmid") or best.get("id"))

    c.pacer.wait()
    panel = c.panel3(kpid)
    if not panel:
        return {"poi": poi, "status": "panel_fail", "kpid": kpid}

    sm = extract_summary(panel)
    rv = extract_reviews(panel, limit=5)
    return {"poi": poi, "status": "ok", "kpid": kpid,
            "panel": panel, "summary": sm, "reviews": rv}


def writer_loop(q: queue.Queue, conn, stop_evt: threading.Event,
                state: dict):
    """라이터 스레드 — 큐에서 받아 SQLite에 기록."""
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
        if status == "ok":
            mark_matched(conn, poi["poi_id"], item["kpid"])
            save_panel3(conn, item["kpid"], item["panel"])
            mark_fetched(conn, poi["poi_id"])
            bump_quota(conn, 200)
            state["ok"] += 1
        elif status == "nomatch":
            mark_error(conn, poi["poi_id"], "nomatch", "best_match=None")
            state["nomatch"] += 1
        elif status == "panel_fail":
            mark_error(conn, poi["poi_id"], "error", "panel3 fail")
            state["err"] += 1
        conn.commit()
        # 진행 로그
        sm = item.get("summary")
        if sm:
            print(f"  [ok] {poi['poi_name'][:25]:25s} → {(sm.get('name') or '')[:25]:25s} "
                  f"★{sm.get('rating')}({sm.get('rating_count')}) "
                  f"메뉴{sm.get('menu_count')}")
        elif status == "nomatch":
            print(f"  [nomatch] {poi['poi_name'][:40]}")


def neo4j_sample(n: int, gu_prefix: str) -> list[dict]:
    import sys
    sys.path.insert(0, "scripts/neo4j_load")
    from _common import driver_session  # type: ignore
    with driver_session() as s:
        rows = s.run("""
            MATCH (p:POI {type:'commerce'})
            WHERE p.dong_code STARTS WITH $gu
              AND p.name IS NOT NULL AND p.name <> ''
            RETURN p.id AS id, p.name AS name, p.lon AS lon, p.lat AS lat
            ORDER BY rand() LIMIT $n
        """, n=n, gu=gu_prefix).data()
    return [{"poi_id": r["id"], "poi_name": r["name"],
             "poi_addr": "", "poi_lat": r["lat"], "poi_lon": r["lon"]}
            for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gu", default="11680", help="dong_code prefix (강남구 11680)")
    args = ap.parse_args()

    conn = open_db()
    pois = neo4j_sample(args.n, args.gu)
    print(f"Neo4j 샘플: {len(pois)} POI (gu={args.gu})\n")
    for p in pois:
        upsert_poi(conn, **p)
    conn.commit()

    print(f"=== 병렬 PoC: workers={args.workers} ===\n")
    t0 = time.time()

    q: queue.Queue = queue.Queue()
    stop = threading.Event()
    state = {"ok": 0, "nomatch": 0, "err": 0}
    writer = threading.Thread(target=writer_loop,
                               args=(q, conn, stop, state), daemon=True)
    writer.start()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_one, p) for p in pois]
        for f in futures:
            try:
                q.put(f.result())
            except Exception as e:
                print(f"  [worker exc] {e}")

    stop.set()
    q.put(None)
    writer.join(timeout=10)

    elapsed = time.time() - t0
    print(f"\n[결과 {len(pois)}개, workers={args.workers}]")
    print(f"  매칭+디테일 성공: {state['ok']}")
    print(f"  nomatch: {state['nomatch']}")
    print(f"  panel 에러: {state['err']}")
    print(f"  매칭률: {state['ok']/len(pois)*100:.0f}%")
    print(f"  소요: {elapsed:.1f}s ({elapsed/len(pois):.2f}s/POI 직렬환산)")
    print(f"  실효 속도: {len(pois)/elapsed*60:.0f} POI/min")
    print(f"\n[537k 외삽]")
    print(f"  병렬 워커{args.workers}: {537489/(len(pois)/elapsed)/3600:.1f}h "
          f"= {537489/(len(pois)/elapsed)/86400:.1f}일")
    print(f"\n[DB stats] {stats(conn)}")
    conn.close()


if __name__ == "__main__":
    main()
