"""
backfill_category_desire_params.py
====================================
기존 Neo4j 그래프의 :Category 노드에 desire 파라미터 3종을
**한 번만** 채워넣는 backfill 스크립트.

`02_categories.py` 가 fresh 적재라면 이 스크립트는 이미 그래프가 있는데
새 필드(recovery_tau_days / desire_drop / saturation_n)만 추가하는 용도.
멱등 — 여러 번 돌려도 안전.

사용:
  python -m scripts.neo4j_load.backfill_category_desire_params
  python -m scripts.neo4j_load.backfill_category_desire_params --dry-run

검증:
  Neo4j 콘솔에서:
    MATCH (c:Category) WHERE c.recovery_tau_days IS NULL RETURN count(c)
    → 0 이어야 정상
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
# 02_categories 는 파일명이 숫자 시작이라 import 안 됨 → 같은 폴더에서 함수 복제 대신 importlib
import importlib.util

_HERE = Path(__file__).resolve().parent
_02_PATH = _HERE / "02_categories.py"
spec = importlib.util.spec_from_file_location("_cat_loader", _02_PATH)
_cat_loader = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(_cat_loader)  # type: ignore
parse_categories_yaml = _cat_loader.parse_categories_yaml
YAML_PATH = _cat_loader.YAML_PATH

from _common import driver_session  # noqa: E402


BACKFILL_CYPHER = """
UNWIND $batch AS c
MATCH (n:Category {name: c.name})
SET n.recovery_tau_days = c.recovery_tau_days,
    n.desire_drop = c.desire_drop,
    n.saturation_n = c.saturation_n
RETURN count(n) AS matched
"""

VERIFY_CYPHER = """
MATCH (c:Category)
RETURN
  count(c) AS total,
  count(c.recovery_tau_days) AS with_tau,
  count(c.desire_drop) AS with_drop,
  count(c.saturation_n) AS with_sat
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="검증만, SET 안 함")
    args = parser.parse_args()

    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    rows = parse_categories_yaml(raw)
    print(f"[parse] {len(rows)} categories from yaml")

    with driver_session() as s:
        before = s.run(VERIFY_CYPHER).single()
        print(f"[before] Category total={before['total']} "
              f"with_tau={before['with_tau']} "
              f"with_drop={before['with_drop']} "
              f"with_sat={before['with_sat']}")

        if args.dry_run:
            print("[dry-run] skipping SET")
            return 0

        # 어떤 카테고리가 yaml 에는 있지만 그래프엔 없는지 (sample)
        yaml_names = {r["name"] for r in rows}
        missing = s.run(
            "MATCH (c:Category) WHERE c.name IN $names "
            "RETURN c.name AS n",
            names=list(yaml_names),
        )
        in_graph = {r["n"] for r in missing}
        only_in_yaml = yaml_names - in_graph
        if only_in_yaml:
            print(f"[warn] {len(only_in_yaml)} categories in yaml but NOT in graph "
                  f"(will be skipped): {sorted(only_in_yaml)[:10]}")

        result = s.run(BACKFILL_CYPHER, batch=rows).single()
        print(f"[set ] matched={result['matched']}")

        after = s.run(VERIFY_CYPHER).single()
        print(f"[after ] Category total={after['total']} "
              f"with_tau={after['with_tau']} "
              f"with_drop={after['with_drop']} "
              f"with_sat={after['with_sat']}")

        gap = after["total"] - after["with_tau"]
        if gap > 0:
            print(f"[warn] {gap} Category nodes still missing desire params "
                  f"(yaml 에 없는 카테고리?)")
            return 2

    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
