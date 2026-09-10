"""크롤 결과 리포트 — SQLite 상태 요약 + JSONL 통계.

사용:
  PYTHONPATH=$PWD python -m scripts.scrape.kakao.report [<jsonl_path>]
"""
from __future__ import annotations

import io
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

DB_PATH = Path("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")
DEFAULT_OUT = Path("G:/내 드라이브/Kw/final_project/output/scrape/kakao")


def db_summary():
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute("SELECT status, COUNT(*) FROM poi_status GROUP BY status").fetchall()
    nomatch_rows = conn.execute(
        "SELECT poi_id, poi_name, error_msg FROM poi_status WHERE status='nomatch' ORDER BY poi_name"
    ).fetchall()
    panel_size = conn.execute("SELECT COUNT(*), SUM(bytes) FROM panel3_raw").fetchone()
    conn.close()
    return rows, nomatch_rows, panel_size


def jsonl_stats(path: Path):
    by_pass = Counter()
    rating_dist = Counter()
    rating_count_sum = 0
    has_rating = 0
    has_menu = 0
    has_review_first_page = 0
    cats = Counter()
    total = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            by_pass[r.get("pass", "?")] += 1
            rt = r.get("rating")
            if rt is not None and rt > 0:
                has_rating += 1
                rating_dist[round(rt, 1)] += 1
                rating_count_sum += (r.get("rating_count") or 0)
            if (r.get("menu_count") or 0) > 0:
                has_menu += 1
            if (r.get("review_first_page") or 0) > 0:
                has_review_first_page += 1
            if r.get("category"):
                cats[r["category"]] += 1
    return {
        "total": total, "by_pass": by_pass, "has_rating": has_rating,
        "rating_dist": rating_dist, "rating_count_sum": rating_count_sum,
        "has_menu": has_menu, "has_review_first_page": has_review_first_page,
        "cats": cats,
    }


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    args = sys.argv[1:]
    print(f"=== DB SUMMARY ({DB_PATH.name}) ===")
    rows, nomatch_rows, (n_panels, total_bytes) = db_summary()
    total = sum(c for _, c in rows)
    for s, c in sorted(rows, key=lambda x: -x[1]):
        pct = c / total * 100 if total else 0
        print(f"  {s:12s}: {c:>6d} ({pct:.1f}%)")
    print(f"  panel3_raw  : {n_panels} rows, {(total_bytes or 0)/1024/1024:.1f} MB")
    print()

    if nomatch_rows:
        print(f"=== NOMATCH ({len(nomatch_rows)}) ===")
        for poi_id, poi_name, _msg in nomatch_rows:
            print(f"  {poi_id} | {poi_name}")
        print()

    if args:
        path = Path(args[0])
    else:
        cands = sorted(DEFAULT_OUT.glob("apt_seoul_full_*.jsonl"))
        if not cands:
            cands = sorted(DEFAULT_OUT.glob("*.jsonl"))
        path = cands[-1] if cands else None
    if not path or not path.exists():
        print("[no jsonl]")
        return
    print(f"=== JSONL STATS ({path.name}) ===")
    st = jsonl_stats(path)
    print(f"  rows:               {st['total']}")
    print(f"  has rating >0:      {st['has_rating']} ({st['has_rating']/st['total']*100:.1f}%)")
    print(f"  total rating votes: {st['rating_count_sum']}")
    print(f"  has menu:           {st['has_menu']}")
    print(f"  has review (first): {st['has_review_first_page']}")
    print()
    print("  pass 분포:")
    for k, v in st["by_pass"].most_common():
        print(f"    {k:24s} : {v}")
    print()
    print("  카테고리 분포 (top 10):")
    for k, v in st["cats"].most_common(10):
        print(f"    {k:20s} : {v}")
    if st["rating_dist"]:
        print()
        print("  별점 분포 (top 10):")
        for k, v in sorted(st["rating_dist"].items(), key=lambda x: -x[1])[:10]:
            print(f"    ★{k:.1f}: {v}")


if __name__ == "__main__":
    main()
