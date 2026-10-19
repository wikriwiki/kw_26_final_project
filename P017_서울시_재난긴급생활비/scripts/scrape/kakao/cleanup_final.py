"""의심 false positive 매칭 (약 + d>100m) + nomatch 가게 DB 삭제.

기준:
  - 의심 매칭: 주소 sim<70 AND 카테고리 불일치 AND 좌표 거리 > 100m (또는 거리 없음)
  - nomatch: status = 'nomatch'

후처리:
  - poi_status 에서 삭제
  - panel3_raw 에서 고아 row(아무 poi에서 안 쓰이는 kakao_pid) 정리
"""
from __future__ import annotations

import csv
import io
import json
import math
import re
import sqlite3
import sys
from collections import Counter

from rapidfuzz import fuzz

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _norm(s):
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def haversine(a, b):
    R = 6371_000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def addr_norm(addr):
    if not addr: return ""
    a = re.sub(r"서울특별시\s*", "", addr).strip()
    return a.split(",")[0].strip()


def addr_match(csv_addr, kakao_addr):
    a, b = addr_norm(csv_addr), addr_norm(kakao_addr)
    if not a or not b: return 0
    return fuzz.token_set_ratio(a, b)


CSV_TO_KAKAO_CAT = {
    "음식": ["음식점","식당","한식","양식","중식","일식","분식","카페","주점","치킨","패스트푸드","버거","베이커리","제과","디저트","도시락","찌개","국밥","초밥","고기","구이","보쌈","갈비","삼겹살","음료","호프","주류","술집"],
    "소매": ["편의점","슈퍼","마트","의류","신발","화장품","잡화","악세서리","가방","안경","보석","도서","문구","약국","꽃집","전자제품","가전","휴대폰","통신기기","정육","수산","식품","주류","취미","완구","스포츠용품","악기","패션"],
    "수리·개인": ["미용","이용","세탁","수선","수리","정비","에스테틱","피부관리","네일","마사지","타투","사진관","스튜디오"],
    "교육": ["학원","교습소","교육","스쿨","유치원","어린이집","독서실","도서관","강의실","어학원"],
    "보건의료": ["병원","의원","한의원","치과","약국","약방","정신과","산부인과","내과","외과","피부과","안과","이비인후과","정형외과","비뇨기과","신경과","성형외과","재활","검진"],
    "부동산": ["부동산","공인중개","중개"],
    "예술·스포츠": ["헬스","스포츠","체육","수영장","골프","테니스","당구","노래방","PC방","오락","볼링","필라테스","요가","복싱","유도","태권도","검도","갤러리","박물관","전시관","공연장","영화관","여가시설"],
    "숙박": ["호텔","모텔","여관","펜션","게스트하우스","리조트","민박","유스호스텔"],
}


def cat_match(csv_cat_l, panel_cat_str):
    if not csv_cat_l or not panel_cat_str: return False
    for k in CSV_TO_KAKAO_CAT.get(csv_cat_l, []):
        if len(k) >= 2 and k in panel_cat_str:
            return True
    return False


def main():
    CSV_PATH = r"C:/Users/Administrator/naver_crawl/소상공인시장진흥공단_상가(상권)정보_서울_202512.csv"
    csv_meta = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        rdr = csv.reader(f); next(rdr)
        for row in rdr:
            if len(row) < 39: continue
            csv_meta[f"COM_{row[0]}"] = {
                "addr_road": row[31], "addr_jibun": row[24],
                "cat_l": row[4],
                "lat": float(row[38]) if row[38] else None,
                "lon": float(row[37]) if row[37] else None,
            }

    db = sqlite3.connect("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")
    db.row_factory = sqlite3.Row

    print("=== 1. 의심 매칭 식별 (약 + d>100m) ===")
    rows = db.execute("""
      SELECT s.poi_id, s.kakao_pid, p.raw_json
      FROM poi_status s
      JOIN panel3_raw p ON s.kakao_pid = p.kakao_pid
      WHERE s.poi_id LIKE 'COM_%' AND s.status='fetched'
    """).fetchall()
    print(f"  검증 대상: {len(rows):,}")

    suspicious_pids = []
    for r in rows:
        pid = r["poi_id"]; meta = csv_meta.get(pid)
        if not meta: continue
        try:
            panel = json.loads(r["raw_json"])
        except json.JSONDecodeError:
            continue
        summary = panel.get("summary", {}) or {}
        paddr = (summary.get("address", {}) or {}).get("road", "")
        cat = summary.get("category", {}) or {}
        pcat = " ".join(str(cat.get(f"name{i}", "") or "") for i in (1, 2, 3, 4))
        point = summary.get("point", {}) or {}
        plat, plon = point.get("lat"), point.get("lon")

        d = None
        if all((meta["lat"], meta["lon"], plat, plon)):
            try:
                d = haversine((meta["lat"], meta["lon"]), (plat, plon))
            except (ValueError, TypeError):
                pass
        addr_in = meta["addr_road"] or meta["addr_jibun"] or ""
        asim = addr_match(addr_in, paddr)
        cm = cat_match(meta["cat_l"], pcat)

        # 의심: 주소 sim<70 AND 카테고리 불일치 AND (거리 없음 OR 거리>100m)
        if asim < 70 and not cm and (d is None or d > 100):
            suspicious_pids.append(pid)

    print(f"  의심 매칭: {len(suspicious_pids):,}")

    print()
    print("=== 2. nomatch 가게 식별 ===")
    nomatch_pids = [r[0] for r in db.execute(
        "SELECT poi_id FROM poi_status WHERE poi_id LIKE 'COM_%' AND status='nomatch'"
    )]
    print(f"  nomatch: {len(nomatch_pids):,}")

    to_delete = set(suspicious_pids) | set(nomatch_pids)
    print(f"\n=== 3. DB 삭제 (총 {len(to_delete):,}) ===")

    # 의심 매칭의 kakao_pid 수집 → 다른 POI에서 안 쓰이는 것만 panel3에서 삭제
    suspect_kpids = set()
    for r in db.execute(
        f"SELECT kakao_pid FROM poi_status WHERE poi_id IN ({','.join(['?']*len(suspicious_pids))})",
        suspicious_pids,
    ) if suspicious_pids else []:
        if r[0]: suspect_kpids.add(r[0])

    # poi_status 삭제
    batch = list(to_delete)
    deleted = 0
    for i in range(0, len(batch), 1000):
        chunk = batch[i:i+1000]
        ph = ",".join(["?"] * len(chunk))
        cur = db.execute(f"DELETE FROM poi_status WHERE poi_id IN ({ph})", chunk)
        deleted += cur.rowcount
    db.commit()
    print(f"  poi_status 삭제: {deleted:,}")

    # panel3_raw 고아 정리 — 의심 kakao_pid 중 더 이상 어떤 poi에도 안 쓰이는 것
    orphan_panel = 0
    if suspect_kpids:
        for kpid in suspect_kpids:
            n = db.execute("SELECT COUNT(*) FROM poi_status WHERE kakao_pid=?", (kpid,)).fetchone()[0]
            if n == 0:
                db.execute("DELETE FROM panel3_raw WHERE kakao_pid=?", (kpid,))
                orphan_panel += 1
        db.commit()
    print(f"  panel3_raw 고아 삭제: {orphan_panel:,}")

    # 최종 통계
    print()
    print("=== 최종 DB 상태 ===")
    for s, n in db.execute(
        "SELECT status, COUNT(*) FROM poi_status WHERE poi_id LIKE 'COM_%' GROUP BY status ORDER BY 2 DESC"
    ):
        print(f"  {s:12s}: {n:>9,}")
    n_panel, sz = db.execute("SELECT COUNT(*), SUM(bytes) FROM panel3_raw").fetchone()
    print(f"  panel3_raw       : {n_panel:>9,} rows, {(sz or 0)/1024/1024/1024:.2f} GB")
    db.close()

    print()
    print(f"=== 최종 매칭률 (B2C 시장 제외 모집단 298,525 기준) ===")
    final_matched = deleted_check = 0  # placeholder
    # 다시 쿼리
    db2 = sqlite3.connect("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")
    final_matched = db2.execute(
        "SELECT COUNT(*) FROM poi_status WHERE poi_id LIKE 'COM_%' AND status='fetched'"
    ).fetchone()[0]
    db2.close()
    print(f"  깨끗한 매칭: {final_matched:,} / 298,525 = {final_matched*100/298525:.1f}%")


if __name__ == "__main__":
    main()
