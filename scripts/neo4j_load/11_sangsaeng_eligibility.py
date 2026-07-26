# -*- coding: utf-8 -*-
"""commerce POI 전체에 상생소비지원금 적립/제외 여부 백필.

기록 속성:
  p.sangsaeng_eligible : bool  — 적립업종이면 True (실적 인정). 제외면 False.
  p.sangsaeng_arm      : str   — eligible / excluded_luxury(시계·귀금속=명품) / excluded_other
  p.sangsaeng_kdi      : str   — 적립 POI의 KDI 8분류(가전·가구/학원/이·미용/여행·레저/요식/유통/기타)
                                 제외 POI는 NULL. (D1/D2 채점용)
  p.sangsaeng_src      : str   — "rule" (상생은 실측 가맹점망 없음, 순수 룰)

판정: scripts/sim/sangsaeng_eligibility.is_sangsaeng_eligible / sangsaeng_arm
      KDI 매핑: data/sangsaeng/industry_map.json (l1_default + sub_override)

소비쿠폰(09_coupon_eligibility.py)과 달리 실측 가맹점 CSV 조인이 없다 — 상생은
네거티브 방식이라 '제외 소수'만 룰로 뺀다. 백필 결과는 시계·귀금속·유흥 상호명
극소수를 뺀 거의 전부가 True 가 정상 (docs/POLICY_BACKTEST_SANGSAENG.md §4.3).

런타임은 이 속성을 우선 사용하고, 없으면 같은 룰로 fallback 하므로 백필 전에도
시뮬은 동작한다 (백필은 조회 일관성·집계 편의·야간 State 집계용).

사용: python scripts/neo4j_load/11_sangsaeng_eligibility.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from _common import driver_session  # noqa: E402
from sangsaeng_eligibility import sangsaeng_arm, ARM_ELIGIBLE  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
_MAP_PATH = ROOT / "data" / "sangsaeng" / "industry_map.json"


def load_industry_map() -> tuple[dict, dict]:
    """(l1_default, sub_override) 반환."""
    raw = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    return raw.get("l1_default", {}), raw.get("sub_override", {})


def kdi_category(l1: str | None, sub: str | None, l1_default: dict, sub_override: dict) -> str | None:
    """적립 POI의 KDI 8분류. sub_override 가 l1_default 보다 우선. 미매핑이면 None."""
    s = (sub or "").strip()
    if s in sub_override:
        return sub_override[s]
    return l1_default.get((l1 or "").strip())


FETCH = """
MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category)
RETURN p.id AS id, p.name AS name, c.name AS sub, c.parent AS l1
"""
UPDATE = """
UNWIND $rows AS r
MATCH (p:POI {id: r.id})
SET p.sangsaeng_eligible = r.el,
    p.sangsaeng_arm = r.arm,
    p.sangsaeng_kdi = r.kdi,
    p.sangsaeng_src = 'rule'
"""
BATCH = 20_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    l1_default, sub_override = load_industry_map()
    print(f"industry_map 로드: L1 {len(l1_default)}종 / sub_override {len(sub_override)}종")

    arms: Counter = Counter()
    kdi_dist: Counter = Counter()
    excluded_by_sub: Counter = Counter()
    rows, total = [], 0

    with driver_session() as s:
        for rec in s.run(FETCH):
            total += 1
            arm, _why = sangsaeng_arm(rec["name"], rec["sub"], rec["l1"])
            el = (arm == ARM_ELIGIBLE)
            kdi = kdi_category(rec["l1"], rec["sub"], l1_default, sub_override) if el else None
            arms[arm] += 1
            if el:
                kdi_dist[kdi or "(미매핑)"] += 1
            else:
                excluded_by_sub[f"{rec['l1']}/{rec['sub']}"] += 1
            rows.append({"id": rec["id"], "el": el, "arm": arm, "kdi": kdi})

        n_el = arms.get(ARM_ELIGIBLE, 0)
        print(f"commerce POI {total:,}개 판정 완료")
        print(f"  적립 비율: {100 * n_el / max(total, 1):.2f}%  (상생=네거티브, 대부분 적립이 정상)")
        print("  arm 분포:", dict(arms))
        print("  KDI 8분류 분포:", dict(kdi_dist))
        print("  제외 상위 (L1/sub):", excluded_by_sub.most_common(10))
        n_lux = arms.get("excluded_luxury", 0)
        print(f"  ★ C1 제외 arm(시계·귀금속=명품 proxy): {n_lux:,}개")

        if args.dry_run:
            print("[dry-run] DB 미기록")
            return
        for i in range(0, len(rows), BATCH):
            s.run(UPDATE, rows=rows[i:i + BATCH])
        print(f"→ p.sangsaeng_eligible / p.sangsaeng_arm / p.sangsaeng_kdi 기록 완료 ({len(rows):,}개)")


if __name__ == "__main__":
    main()
