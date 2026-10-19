"""매칭 검증 — 좌표·주소·카테고리·이름 4축 전수 검증."""
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


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


def haversine(a, b):
    R = 6371_000
    p1, p2 = math.radians(a[0]), math.radians(b[0])
    dp = math.radians(b[0] - a[0])
    dl = math.radians(b[1] - a[1])
    h = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(h))


def addr_norm(addr: str) -> str:
    if not addr:
        return ""
    a = re.sub(r"서울특별시\s*", "", addr).strip()
    return a.split(",")[0].strip()


def addr_match(csv_addr: str, kakao_addr: str) -> int:
    a, b = addr_norm(csv_addr), addr_norm(kakao_addr)
    if not a or not b:
        return 0
    return fuzz.token_set_ratio(a, b)


CSV_TO_KAKAO_CAT = {
    "음식": ["음식점", "식당", "한식", "양식", "중식", "일식", "분식", "카페",
             "주점", "치킨", "패스트푸드", "버거", "베이커리", "제과", "디저트",
             "도시락", "찌개", "국밥", "초밥", "고기", "구이", "보쌈", "갈비",
             "삼겹살", "음료", "호프", "주류", "술집"],
    "소매": ["편의점", "슈퍼", "마트", "의류", "신발", "화장품", "잡화",
             "악세서리", "가방", "안경", "보석", "도서", "문구", "약국", "꽃집",
             "전자제품", "가전", "휴대폰", "통신기기", "정육", "수산", "식품",
             "주류", "취미", "완구", "스포츠용품", "악기", "패션"],
    "수리·개인": ["미용", "이용", "세탁", "수선", "수리", "정비",
                "에스테틱", "피부관리", "네일", "마사지", "타투",
                "사진관", "스튜디오"],
    "교육": ["학원", "교습소", "교육", "스쿨", "유치원", "어린이집",
             "독서실", "도서관", "강의실", "어학원"],
    "보건의료": ["병원", "의원", "한의원", "치과", "약국", "약방",
                "정신과", "산부인과", "내과", "외과", "피부과", "안과",
                "이비인후과", "정형외과", "비뇨기과", "신경과", "성형외과",
                "재활", "검진"],
    "부동산": ["부동산", "공인중개", "중개"],
    "예술·스포츠": ["헬스", "스포츠", "체육", "수영장", "골프", "테니스",
                  "당구", "노래방", "PC방", "오락", "볼링", "필라테스",
                  "요가", "복싱", "유도", "태권도", "검도",
                  "갤러리", "박물관", "전시관", "공연장", "영화관", "여가시설"],
    "숙박": ["호텔", "모텔", "여관", "펜션", "게스트하우스", "리조트",
             "민박", "유스호스텔"],
}


def cat_match_score(csv_cat_l: str, panel_cat_str: str) -> int:
    if not csv_cat_l or not panel_cat_str:
        return 0
    kws = CSV_TO_KAKAO_CAT.get(csv_cat_l, [])
    for k in kws:
        if len(k) >= 2 and k in panel_cat_str:
            return 100
    return 0


def main():
    CSV_PATH = r"C:/Users/Administrator/naver_crawl/소상공인시장진흥공단_상가(상권)정보_서울_202512.csv"
    csv_meta = {}
    with open(CSV_PATH, encoding="utf-8") as f:
        rdr = csv.reader(f)
        next(rdr)
        for row in rdr:
            if len(row) < 39:
                continue
            csv_meta[f"COM_{row[0]}"] = {
                "name": row[1], "addr_road": row[31], "addr_jibun": row[24],
                "cat_l": row[4], "cat_s": row[8],
                "lat": float(row[38]) if row[38] else None,
                "lon": float(row[37]) if row[37] else None,
            }

    db = sqlite3.connect("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT s.poi_id, s.kakao_pid, p.raw_json
        FROM poi_status s
        JOIN panel3_raw p ON s.kakao_pid = p.kakao_pid
        WHERE s.poi_id LIKE 'COM_%' AND s.status='fetched'
    """).fetchall()
    print(f"전수 검증 대상: {len(rows):,} 건\n")

    verified = 0
    dist_buckets = Counter()
    addr_sim_buckets = Counter()
    cat_match_count = 0
    strong = 0   # 주소 sim ≥80 + 카테고리 일치
    moderate = 0  # 주소 sim ≥70 OR 카테고리 일치
    weak = 0      # 주소 sim <70 AND 카테고리 불일치
    very_suspicious = 0  # weak + d>100m
    weak_examples = []

    for r in rows:
        pid = r["poi_id"]
        meta = csv_meta.get(pid)
        if not meta:
            continue
        try:
            panel = json.loads(r["raw_json"])
        except json.JSONDecodeError:
            continue
        summary = panel.get("summary", {}) or {}
        pname = summary.get("name", "")
        paddr_road = (summary.get("address", {}) or {}).get("road", "")
        cat = summary.get("category", {}) or {}
        pcat = " ".join(str(cat.get(f"name{i}", "") or "") for i in (1, 2, 3, 4))
        point = summary.get("point", {}) or {}
        plat = point.get("lat")
        plon = point.get("lon")

        d = None
        if all((meta["lat"], meta["lon"], plat, plon)):
            try:
                d = haversine((meta["lat"], meta["lon"]), (plat, plon))
            except (ValueError, TypeError):
                pass

        addr_in = meta["addr_road"] or meta["addr_jibun"] or ""
        asim = addr_match(addr_in, paddr_road)
        cm = cat_match_score(meta["cat_l"], pcat)
        nsim = max(fuzz.token_sort_ratio(_norm(meta["name"]), _norm(pname)),
                   fuzz.partial_ratio(_norm(meta["name"]), _norm(pname)))

        if d is not None:
            if d <= 30: dist_buckets["0-30m"] += 1
            elif d <= 100: dist_buckets["30-100m"] += 1
            elif d <= 200: dist_buckets["100-200m"] += 1
            elif d <= 500: dist_buckets["200-500m"] += 1
            else: dist_buckets[">500m"] += 1

        if asim >= 90: addr_sim_buckets["90-100"] += 1
        elif asim >= 70: addr_sim_buckets["70-90"] += 1
        elif asim >= 50: addr_sim_buckets["50-70"] += 1
        elif asim >= 30: addr_sim_buckets["30-50"] += 1
        else: addr_sim_buckets["<30"] += 1

        if cm >= 100:
            cat_match_count += 1

        if asim >= 80 and cm >= 100:
            strong += 1
        elif asim >= 70 or cm >= 100:
            moderate += 1
        else:
            weak += 1
            if d is None or d > 100:
                very_suspicious += 1
                if len(weak_examples) < 50:
                    weak_examples.append({
                        "pid": pid, "csv_name": meta["name"],
                        "kakao_name": pname, "csv_addr": addr_in[:60],
                        "kakao_addr": paddr_road[:60],
                        "csv_cat": meta["cat_s"], "kakao_cat": pcat[:50],
                        "d": d, "asim": asim, "nsim": nsim,
                    })

        verified += 1

    print(f"=== 매칭 신뢰도 분류 ({verified:,}건) ===\n")
    print(f"  강 (주소 sim≥80 + 카테고리 일치):           {strong:>7,} ({strong*100/verified:.1f}%)")
    print(f"  중 (주소 sim≥70 OR 카테고리 일치):          {moderate:>7,} ({moderate*100/verified:.1f}%)")
    print(f"  약 (주소·카테고리 둘 다 안 맞음):              {weak:>7,} ({weak*100/verified:.1f}%)")
    print(f"  강한 의심 (약 + 거리 >100m):              {very_suspicious:>7,} ({very_suspicious*100/verified:.1f}%)")
    print()
    print(f"카테고리 일치 합계: {cat_match_count:,} ({cat_match_count*100/verified:.1f}%)")
    print()
    print("좌표 거리 분포:")
    for k in ["0-30m", "30-100m", "100-200m", "200-500m", ">500m"]:
        n = dist_buckets.get(k, 0)
        print(f"  {k:10s}: {n:>7,} ({n*100/verified:.1f}%)")
    print()
    print("주소 token_set_ratio 분포:")
    for k in ["90-100", "70-90", "50-70", "30-50", "<30"]:
        n = addr_sim_buckets.get(k, 0)
        print(f"  {k:8s}: {n:>7,} ({n*100/verified:.1f}%)")
    print()
    print("=== 강한 의심 sample (주소·카테고리 둘 다 안 맞음 + d>100m) ===")
    for ex in weak_examples[:30]:
        print(f"\n  csv={ex['csv_name'][:25]:25s} ↔ kakao={ex['kakao_name'][:25]:25s} | d={ex['d']} addr_sim={ex['asim']:.0f}")
        print(f"    csv_addr: {ex['csv_addr']}")
        print(f"    kakao_addr: {ex['kakao_addr']}")
        print(f"    csv_cat={ex['csv_cat']:15s} | kakao_cat={ex['kakao_cat']}")

    db.close()


if __name__ == "__main__":
    main()
