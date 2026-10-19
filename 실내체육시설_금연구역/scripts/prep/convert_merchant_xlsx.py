# -*- coding: utf-8 -*-
"""서울사랑상품권 유효 가맹점 xlsx → 쿠폰 사용처 백필용 CSV 변환.

입력: 서울사랑상품권 유효 가맹점('24년3월 기준).xlsx
      [가맹점명, 서울페이앱분류업종, 우편번호, 자치구명, 기본주소, 상세주소, 손목닥터가맹여부]
출력: data/coupon/seoul_love_merchants.csv (utf-8-sig)
      → scripts/neo4j_load/09_coupon_eligibility.py --merchants data/coupon 이 소비
        (컬럼 자동탐지: '가맹점명' → 상호, '자치구명' → 구)

+ 매칭 키 (자치구, 정규화 상호)의 중복률 리포트 — 같은 구 동일 상호(지점명 없는 체인)는
  어차피 같은 판정(True)이라 매칭 정확도에 무해하지만, 규모 파악용으로 출력.

사용: python scripts/prep/convert_merchant_xlsx.py [--xlsx PATH]
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_XLSX = ROOT.parent.parent.parent / "서울사랑상품권 유효 가맹점('24년3월 기준).xlsx"
OUT = ROOT / "data" / "coupon" / "seoul_love_merchants.csv"

_NORM_RE = re.compile(r"[\s()\-·.,'&]")


def norm_name(s: str | None) -> str:
    return _NORM_RE.sub("", (s or "")).lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, default=DEFAULT_XLSX)
    args = ap.parse_args()

    import openpyxl
    wb = openpyxl.load_workbook(args.xlsx, read_only=True)
    ws = wb.worksheets[0]
    rows_iter = ws.iter_rows(values_only=True)
    header = [str(c) if c else "" for c in next(rows_iter)]
    idx = {name: i for i, name in enumerate(header)}
    need = ["가맹점명", "서울페이앱분류업종", "자치구명", "기본주소"]
    missing = [c for c in need if c not in idx]
    if missing:
        raise SystemExit(f"기대 컬럼 누락: {missing} (헤더: {header})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    keys: Counter = Counter()
    cat_cnt: Counter = Counter()
    with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(need)
        for r in rows_iter:
            name = r[idx["가맹점명"]]
            gu = r[idx["자치구명"]]
            if not name or not gu:
                continue
            w.writerow([name, r[idx["서울페이앱분류업종"]] or "", gu, r[idx["기본주소"]] or ""])
            keys[(str(gu).strip(), norm_name(str(name)))] += 1
            cat_cnt[str(r[idx["서울페이앱분류업종"]] or "?")] += 1
            n += 1

    dup_keys = sum(1 for c in keys.values() if c > 1)
    dup_rows = sum(c for c in keys.values() if c > 1)
    print(f"가맹점 {n:,}건 → {OUT}")
    print(f"매칭 키(구,정규화상호): 고유 {len(keys):,} / 중복 키 {dup_keys:,} "
          f"({100 * dup_rows / max(n, 1):.1f}% 행이 중복 키 — 동일 판정이라 무해)")
    print("업종 분포 top10:", cat_cnt.most_common(10))


if __name__ == "__main__":
    main()
