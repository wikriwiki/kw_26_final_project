# -*- coding: utf-8 -*-
"""commerce POI 전체에 민생회복 소비쿠폰 사용처 여부 백필 — p.coupon_eligible.

룰: scripts/sim/coupon_eligibility.is_coupon_eligible (상호명 브랜드 + L2 카테고리).
런타임은 이 속성을 우선 사용하고, 없으면 같은 룰로 fallback 판정하므로
백필 전에도 시뮬은 동작한다 (백필은 조회 일관성·분석 편의용).

사용: python scripts/neo4j_load/09_coupon_eligibility.py [--dry-run]
"""
from __future__ import annotations

import argparse
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

FETCH = """
MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category)
RETURN p.id AS id, p.name AS name, c.name AS sub, c.parent AS l1
"""
UPDATE = """
UNWIND $rows AS r
MATCH (p:POI {id: r.id})
SET p.coupon_eligible = r.el
"""
BATCH = 20_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    reasons: Counter = Counter()
    by_sub_excluded: Counter = Counter()
    rows, total = [], 0

    with driver_session() as s:
        for rec in s.run(FETCH):
            total += 1
            el, why = is_coupon_eligible(rec["name"], rec["sub"], rec["l1"])
            reasons[why] += 1
            if not el:
                by_sub_excluded[f"{rec['l1']}/{rec['sub']}"] += 1
            rows.append({"id": rec["id"], "el": el})

        print(f"commerce POI {total:,}개 판정 완료")
        print("사유 분포:", dict(reasons))
        print("제외 상위 (L1/sub):", by_sub_excluded.most_common(12))
        n_ex = total - reasons.get("ok", 0)
        print(f"제외 비율: {100 * n_ex / max(total, 1):.2f}%  (실제 쿠폰은 서울 48만 가맹점 — 대부분 사용가능이 정상)")

        if args.dry_run:
            print("[dry-run] DB 미기록")
            return
        for i in range(0, len(rows), BATCH):
            s.run(UPDATE, rows=rows[i:i + BATCH])
        print(f"→ p.coupon_eligible 기록 완료 ({len(rows):,}개, batch {BATCH:,})")


if __name__ == "__main__":
    main()
