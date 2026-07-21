# -*- coding: utf-8 -*-
"""grant형 정책 JSON 적재 (income_grants·excluded_income·poi_restricted 포함).

배경: policy_pipeline(neo4j_writer)은 코어 필드만 SET 하고 income_grants 등
grant 확장 필드는 다루지 않는다 (P009는 수동 SET로 들어간 상태).
이 스크립트는 백테스트 정책(P010/P011)처럼 확장 필드가 있는 JSON을
한 번에 적재하는 독립 로더다.

사용: python scripts/neo4j_load/10_load_grant_policy.py data/neo4j_load/policies/P010.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

MERGE = """
MERGE (p:Policy {id: $id})
SET p.name = $name,
    p.type = $type,
    p.description = $description,
    p.announce_date = date($announce_date),
    p.effective_from = date($effective_from),
    p.effective_until = date($effective_until),
    p.benefit_rate = $benefit_rate,
    p.cap_per_agent = $cap_per_agent,
    p.income_grants = $income_grants_json,
    p.excluded_income = $excluded_income,
    p.decile_grants = $decile_grants_json,
    p.excluded_deciles = $excluded_deciles,
    p.poi_restricted = $poi_restricted,
    p.notes = $notes
WITH p
UNWIND $target_districts AS dname
OPTIONAL MATCH (d:District {name: dname})
FOREACH (_ IN CASE WHEN d IS NOT NULL THEN [1] ELSE [] END |
  MERGE (p)-[:applied_to]->(d))
RETURN p.id AS id
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    args = ap.parse_args()

    pol = json.loads(args.path.read_text(encoding="utf-8"))
    params = {
        "id": pol["id"], "name": pol["name"], "type": pol["type"],
        "description": pol.get("description", ""),
        "announce_date": pol["announce_date"],
        "effective_from": pol["effective_from"],
        "effective_until": pol["effective_until"],
        "benefit_rate": pol.get("benefit_rate"),
        "cap_per_agent": pol.get("cap_per_agent"),
        "income_grants_json": json.dumps(pol.get("income_grants") or {}, ensure_ascii=False),
        "excluded_income": pol.get("excluded_income") or [],
        # 소비 10분위(spending_level_wd 1~10) 기준 지급 — 있으면 income_grants보다 우선
        "decile_grants_json": json.dumps(pol.get("decile_grants") or {}, ensure_ascii=False),
        "excluded_deciles": [str(x) for x in (pol.get("excluded_deciles") or [])],
        "poi_restricted": bool(pol.get("poi_restricted", False)),
        "notes": pol.get("notes", ""),
        "target_districts": pol.get("target_districts") or [],
    }
    with driver_session() as s:
        r = s.run(MERGE, **params).single()
        basis = (f"decile_grants={params['decile_grants_json']}"
                 if params["decile_grants_json"] != "{}"
                 else f"income_grants={params['income_grants_json']}")
        print(f"적재 완료: {r['id']}  (poi_restricted={params['poi_restricted']}, {basis})")


if __name__ == "__main__":
    main()
