"""V-WORLD 일 한도 초과 등으로 캐시된 ERROR entry 일괄 삭제.

내일 V-WORLD 한도 reset 후 사용:
  python scripts/geocode/invalidate_errors.py [--dry-run]

ERROR entry 삭제 후 03b --geocode 재실행하면 캐시 miss로 V-WORLD 재호출됨.
OK·NOT_FOUND 캐시는 보존 (NOT_FOUND는 정말 주소가 없는 경우라 재시도해도 의미 없음).
"""
import argparse
import json
import sqlite3
import sys
from pathlib import Path

CACHE_PATH = Path(__file__).resolve().parent / "cache.sqlite"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="삭제 카운트만 출력")
    ap.add_argument("--include-not-found", action="store_true",
                    help="NOT_FOUND도 함께 삭제 (주의: V-WORLD가 정말 못 찾는 주소도 재호출)")
    args = ap.parse_args()

    conn = sqlite3.connect(CACHE_PATH)

    total_before = conn.execute("SELECT COUNT(*) FROM geocode_cache").fetchone()[0]
    n_ok = conn.execute("SELECT COUNT(*) FROM geocode_cache WHERE payload LIKE '%\"status\": \"OK\"%'").fetchone()[0]
    n_nf = conn.execute("SELECT COUNT(*) FROM geocode_cache WHERE payload LIKE '%\"status\": \"NOT_FOUND\"%'").fetchone()[0]
    n_err = conn.execute("SELECT COUNT(*) FROM geocode_cache WHERE payload LIKE '%\"status\": \"ERROR\"%'").fetchone()[0]

    print(f"current cache: total={total_before:,}, OK={n_ok:,}, NOT_FOUND={n_nf:,}, ERROR={n_err:,}")

    if args.dry_run:
        print(f"[dry-run] would delete: ERROR={n_err}" + (f", NOT_FOUND={n_nf}" if args.include_not_found else ""))
        return

    deleted = 0
    cur = conn.execute("DELETE FROM geocode_cache WHERE payload LIKE '%\"status\": \"ERROR\"%'")
    deleted += cur.rowcount
    if args.include_not_found:
        cur = conn.execute("DELETE FROM geocode_cache WHERE payload LIKE '%\"status\": \"NOT_FOUND\"%'")
        deleted += cur.rowcount
    conn.commit()
    conn.execute("VACUUM")

    total_after = conn.execute("SELECT COUNT(*) FROM geocode_cache").fetchone()[0]
    print(f"deleted: {deleted:,}")
    print(f"remaining: {total_after:,}")
    conn.close()


if __name__ == "__main__":
    main()
