# -*- coding: utf-8 -*-
"""commerce POI 전체에 민생회복 소비쿠폰 사용처 여부 백필 — p.coupon_eligible.

판정 우선순위 (실측 우선):
  ① 실측 가맹점 조인 — 서울사랑 상품권 가맹점 현황 (열린데이터광장 OA-21099,
     사업자번호·상호·주소 포함 공개 CSV)을 data/coupon/*.csv 에 두면
     (자치구, 정규화 상호) 매칭으로 coupon_eligible=True 확정.
     소비쿠폰의 상품권 지급분 사용처 = 이 가맹점망 (카드분 30억 기준과 거의 동일 집합).
  ② 룰 — scripts/sim/coupon_eligibility.is_coupon_eligible
     (상호명 브랜드 + L2 카테고리). 미매칭 POI의 fallback.
     주의: 가맹점 CSV 미매칭 ≠ 사용불가 (가맹은 신청 기반이라 누락 존재) → False 단정은 룰만.

런타임은 이 속성을 우선 사용하고, 없으면 같은 룰로 fallback 판정하므로
백필 전에도 시뮬은 동작한다 (백필은 실측 반영·조회 일관성·분석 편의용).

사용: python scripts/neo4j_load/09_coupon_eligibility.py [--dry-run] [--merchants data/coupon]
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from _common import driver_session  # noqa: E402
from coupon_eligibility import is_coupon_eligible  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_GU_RE = re.compile(r"([가-힣]+구)\b")
_NORM_RE = re.compile(r"[\s()\-·.,'&]")


def _norm_name(s: str | None) -> str:
    """상호 정규화 — 공백·괄호·기호 제거 (지점명은 유지해 체인 지점 구분)."""
    return _NORM_RE.sub("", (s or "")).lower()


def load_merchant_index(dir_or_glob: str) -> set[tuple[str, str]]:
    """상품권 가맹점 CSV들 → {(자치구명, 정규화상호)}. 컬럼은 키워드로 탐지."""
    idx: set[tuple[str, str]] = set()
    paths = sorted(glob.glob(str(Path(dir_or_glob) / "*.csv"))) or sorted(glob.glob(dir_or_glob))
    for p in paths:
        for enc in ("utf-8-sig", "cp949", "euc-kr"):
            try:
                with io.open(p, encoding=enc, newline="") as f:
                    rd = csv.DictReader(f)
                    cols = rd.fieldnames or []
                    c_name = next((c for c in cols if c and ("상호" in c or "가맹점명" in c or "점포명" in c)), None)
                    c_addr = next((c for c in cols if c and ("주소" in c or "소재지" in c)), None)
                    c_gu = next((c for c in cols if c and ("자치구" in c or "시군구" in c)), None)
                    if not c_name:
                        break
                    for row in rd:
                        name = _norm_name(row.get(c_name))
                        gu = (row.get(c_gu) or "").strip() if c_gu else ""
                        if not gu and c_addr:
                            m = _GU_RE.search(row.get(c_addr) or "")
                            gu = m.group(1) if m else ""
                        if name and gu:
                            idx.add((gu, name))
                    break
            except (UnicodeDecodeError, UnicodeError):
                continue
    return idx

FETCH = """
MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category)
OPTIONAL MATCH (p)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(g:District)
RETURN p.id AS id, p.name AS name, c.name AS sub, c.parent AS l1, g.name AS gu
"""
UPDATE = """
UNWIND $rows AS r
MATCH (p:POI {id: r.id})
SET p.coupon_eligible = r.el, p.coupon_src = r.src
"""
BATCH = 20_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--merchants", default="data/coupon",
                    help="서울사랑상품권 가맹점 CSV 디렉토리 (OA-21099 등). 없으면 룰만 사용")
    args = ap.parse_args()

    merchants = load_merchant_index(args.merchants)
    print(f"실측 가맹점 인덱스: {len(merchants):,}건" if merchants
          else f"가맹점 CSV 없음({args.merchants}) — 룰만 사용")

    reasons: Counter = Counter()
    by_sub_excluded: Counter = Counter()
    rows, total, n_matched = [], 0, 0

    with driver_session() as s:
        for rec in s.run(FETCH):
            total += 1
            # ① 실측 가맹점 조인 (자치구 + 정규화 상호)
            if merchants and rec["gu"] and (rec["gu"], _norm_name(rec["name"])) in merchants:
                rows.append({"id": rec["id"], "el": True, "src": "merchant_csv"})
                reasons["merchant_matched"] += 1
                n_matched += 1
                continue
            # ② 룰 fallback
            el, why = is_coupon_eligible(rec["name"], rec["sub"], rec["l1"])
            reasons[why] += 1
            if not el:
                by_sub_excluded[f"{rec['l1']}/{rec['sub']}"] += 1
            rows.append({"id": rec["id"], "el": el, "src": "rule"})

        print(f"commerce POI {total:,}개 판정 완료 (실측 매칭 {n_matched:,} = {100 * n_matched / max(total, 1):.1f}%)")
        print("사유 분포:", dict(reasons))
        print("룰 제외 상위 (L1/sub):", by_sub_excluded.most_common(12))
        n_ex = total - reasons.get("ok", 0) - n_matched
        print(f"제외 비율: {100 * n_ex / max(total, 1):.2f}%  (실제 쿠폰은 서울 48만 가맹점 — 대부분 사용가능이 정상)")

        if args.dry_run:
            print("[dry-run] DB 미기록")
            return
        for i in range(0, len(rows), BATCH):
            s.run(UPDATE, rows=rows[i:i + BATCH])
        print(f"→ p.coupon_eligible / p.coupon_src 기록 완료 ({len(rows):,}개, batch {BATCH:,})")


if __name__ == "__main__":
    main()
