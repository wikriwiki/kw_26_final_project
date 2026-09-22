"""20-POI PoC. 무인증 풀크롤 — 매칭률·응답속도·차단률 측정.

사용:
  cd /g/내 드라이브/Kw/final_project
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.poc
"""
from __future__ import annotations

import sys
import time

from .client import KakaoClient
from .extract import extract_summary, extract_reviews
from .store import (
    bump_quota, mark_error, mark_fetched, mark_matched, open_db,
    save_panel3, stats, upsert_poi,
)


# 강남구 commerce CSV가 Drive Stream으로 손상됐을 때 쓰는 fallback
# (실제 상가데이터에서 추출한 강남·역삼·논현·청담 일대 가게 이름)
HARDCODED_POIS = [
    {"poi_id": "T01", "poi_name": "스타벅스 강남R점",       "poi_addr": "", "poi_lat": 37.4979, "poi_lon": 127.0276},
    {"poi_id": "T02", "poi_name": "투썸플레이스 강남역점",   "poi_addr": "", "poi_lat": 37.4985, "poi_lon": 127.0274},
    {"poi_id": "T03", "poi_name": "교보문고 강남점",        "poi_addr": "", "poi_lat": 37.5018, "poi_lon": 127.0241},
    {"poi_id": "T04", "poi_name": "맥도날드 강남점",        "poi_addr": "", "poi_lat": 37.4985, "poi_lon": 127.0290},
    {"poi_id": "T05", "poi_name": "이마트 역삼점",          "poi_addr": "", "poi_lat": 37.5005, "poi_lon": 127.0345},
    {"poi_id": "T06", "poi_name": "올리브영 강남역점",      "poi_addr": "", "poi_lat": 37.4983, "poi_lon": 127.0275},
    {"poi_id": "T07", "poi_name": "롯데시네마 강남점",      "poi_addr": "", "poi_lat": 37.5010, "poi_lon": 127.0260},
    {"poi_id": "T08", "poi_name": "삼겹사랑 강남직영점",    "poi_addr": "", "poi_lat": 37.4992, "poi_lon": 127.0287},
    {"poi_id": "T09", "poi_name": "공차 강남역점",          "poi_addr": "", "poi_lat": 37.4985, "poi_lon": 127.0271},
    {"poi_id": "T10", "poi_name": "버거킹 강남교보타워점",  "poi_addr": "", "poi_lat": 37.5020, "poi_lon": 127.0243},
    {"poi_id": "T11", "poi_name": "노브랜드 청담점",        "poi_addr": "", "poi_lat": 37.5235, "poi_lon": 127.0510},
    {"poi_id": "T12", "poi_name": "다이소 강남역점",        "poi_addr": "", "poi_lat": 37.4985, "poi_lon": 127.0280},
    {"poi_id": "T13", "poi_name": "신라명과 강남점",        "poi_addr": "", "poi_lat": 37.4970, "poi_lon": 127.0280},
    {"poi_id": "T14", "poi_name": "메가커피 역삼점",        "poi_addr": "", "poi_lat": 37.5006, "poi_lon": 127.0349},
    {"poi_id": "T15", "poi_name": "GS25 강남역삼점",        "poi_addr": "", "poi_lat": 37.5008, "poi_lon": 127.0353},
    {"poi_id": "T16", "poi_name": "본죽 논현점",            "poi_addr": "", "poi_lat": 37.5117, "poi_lon": 127.0220},
    {"poi_id": "T17", "poi_name": "맘스터치 청담점",        "poi_addr": "", "poi_lat": 37.5260, "poi_lon": 127.0488},
    {"poi_id": "T18", "poi_name": "써브웨이 강남파이낸스점","poi_addr": "", "poi_lat": 37.5009, "poi_lon": 127.0273},
    {"poi_id": "T19", "poi_name": "엔제리너스 신논현점",    "poi_addr": "", "poi_lat": 37.5044, "poi_lon": 127.0249},
    {"poi_id": "T20", "poi_name": "롯데리아 압구정점",      "poi_addr": "", "poi_lat": 37.5274, "poi_lon": 127.0286},
]


def csv_sample(n: int = 20, gu_prefix: str = "11680") -> list[dict]:
    """commerce CSV에서 N개 (Drive Stream으로 부분만 읽혀도 OK).

    바이너리로 읽어 utf-8 strict 디코딩 가능한 prefix만 사용 → 손상된
    파일에서도 머리 N MB는 살아있어 샘플 추출 가능.
    """
    import csv as _csv
    import glob
    import io
    import random
    csv_path = next(iter(glob.glob("data/neo4j_load/pois/소상공인*.csv")), None)
    if not csv_path:
        raise FileNotFoundError("commerce csv 없음")
    with open(csv_path, 'rb') as f:
        raw = f.read()
    # UTF-8 BOM strip
    if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]
    # find last successful utf-8 line — 모든 줄을 디코드 시도해서 실패 직전까지만
    text_lines = []
    for chunk in raw.split(b'\n'):
        try:
            text_lines.append(chunk.decode('utf-8'))
        except UnicodeDecodeError:
            break
    text = "\n".join(text_lines)
    rows_all = []
    r = _csv.reader(io.StringIO(text))
    next(r, None)  # header
    # 컬럼: id=0, name=1, dong_cd=15, lon=37, lat=38
    for row in r:
        if len(row) < 39: continue
        if not row[15].startswith(gu_prefix): continue
        if not row[1]: continue
        try:
            lon, lat = float(row[37]), float(row[38])
        except ValueError:
            continue
        rows_all.append({"poi_id": row[0], "poi_name": row[1],
                         "poi_addr": "", "poi_lat": lat, "poi_lon": lon})
    print(f"  CSV 부분 디코드: {len(rows_all)} commerce in gu={gu_prefix}")
    random.seed(42)
    return random.sample(rows_all, min(n, len(rows_all)))


def run_poc(*, n: int = 20):
    conn = open_db()
    client = KakaoClient(mean_pace=0.7)

    print(f"=== PoC: 무인증 풀크롤 (search + panel3) {n}개 ===\n")
    try:
        pois = csv_sample(n)
        print(f"CSV 샘플 POI: {len(pois)} (강남구 commerce)")
    except Exception as e:
        print(f"CSV 접근 실패: {e} → 하드코딩 fallback")
        pois = HARDCODED_POIS[:n]
        print(f"하드코딩 POI: {len(pois)} (강남·홍대 일대)")
    if not pois:
        print("POI 없음 — 종료")
        return

    for p in pois:
        upsert_poi(conn, **p)
    conn.commit()

    t0 = time.time()
    n_match = n_fetch = n_nomatch = n_err = 0
    rows = []

    for p in pois:
        client.pacer.wait()
        best = client.best_match(
            poi_name=p["poi_name"], lat=p["poi_lat"], lng=p["poi_lon"],
        )
        bump_quota(conn, 200)
        if not best:
            mark_error(conn, p["poi_id"], "nomatch", "best_match=None")
            print(f"  [nomatch] {p['poi_name']}")
            n_nomatch += 1
            continue
        kpid = str(best.get("confirmid") or best.get("id"))
        mark_matched(conn, p["poi_id"], kpid)
        n_match += 1

        client.pacer.wait()
        panel = client.panel3(kpid)
        if not panel:
            mark_error(conn, p["poi_id"], "error", "panel3 fail")
            n_err += 1
            continue
        save_panel3(conn, kpid, panel)
        mark_fetched(conn, p["poi_id"])
        bump_quota(conn, 200)
        n_fetch += 1
        sm = extract_summary(panel)
        rv = extract_reviews(panel, limit=5)
        rows.append({"poi": p["poi_name"], "kakao": sm["name"],
                     "rating": sm["rating"], "rcount": sm["rating_count"],
                     "menus": sm["menu_count"], "reviews": len(rv)})
        print(f"  [ok] {p['poi_name'][:25]:25s} → {(sm['name'] or '')[:25]:25s} "
              f"★{sm['rating']}({sm['rating_count']}) 메뉴{sm['menu_count']} 리뷰{len(rv)}")
        conn.commit()

    elapsed = time.time() - t0
    print(f"\n[결과 {len(pois)}개]")
    print(f"  매칭률: {n_match}/{len(pois)} ({n_match/len(pois)*100:.0f}%)")
    print(f"  디테일 성공: {n_fetch}/{n_match}")
    print(f"  nomatch: {n_nomatch}, panel 에러: {n_err}")
    print(f"  소요: {elapsed:.1f}s ({elapsed/len(pois):.2f}s/POI)")
    print(f"  ban_rate: {client.pacer.ban_rate():.1%}")

    if n_fetch:
        rated = [r for r in rows if r["rating"] is not None]
        print(f"  별점 보유: {len(rated)}/{n_fetch}")
        if rated:
            avg = sum(r["rating"] for r in rated) / len(rated)
            print(f"  매칭 가게 평균 별점: {avg:.2f}")

    # ETA 외삽
    if n_fetch:
        per_poi = elapsed / len(pois)
        print(f"\n[537k 외삽]")
        print(f"  단일 IP 단순 외연: {537489*per_poi/3600:.1f}h = {537489*per_poi/86400:.1f}일")

    print(f"\n[DB stats] {stats(conn)}")
    conn.close()


if __name__ == "__main__":
    run_poc(n=20)
