"""Category 노드 적재 (categories.yaml).

현재 모델: 단일 라벨 :Category. 12 L1 + ~90 L2 sub.
L1/L2를 모두 별도 노드로 만들지 않고, **L2 노드만** 적재.
L2는 cat·sub 두 속성으로 구분 (`name` = sub, `parent` = L1).

이유:
- runtime_ontology.md §0: "단일 라벨 :Category 채택, 어휘 확정 시 L1/L2 분리 가능"
- POI는 IN_CATEGORY로 sub(L2)에 연결
- L1 그루핑은 `parent` 속성 필터 (인덱스 가능)
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "data" / "neo4j_load" / "categories" / "categories.yaml"


def main():
    data = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    rows = []
    for cat in data["categories"]:
        l1 = cat["name"]
        for sub in cat.get("sub", []):
            rows.append({
                "name": sub,
                "parent": l1,
                "open_hour": cat.get("open"),
                "close_hour": cat.get("close"),
            })
    print(f"[parse] {len(rows)} (L1, L2) pairs")

    with driver_session() as s:
        s.run("""
            UNWIND $batch AS c
            MERGE (n:Category {name: c.name})
            SET n.parent = c.parent,
                n.open_hour = c.open_hour,
                n.close_hour = c.close_hour
        """, batch=rows)
        print(f"  + Category x {len(rows)}")

    # parent 인덱스 (런타임 L1 필터용)
    with driver_session() as s:
        s.run("CREATE INDEX category_parent IF NOT EXISTS FOR (c:Category) ON (c.parent)")
        print(f"  + INDEX category_parent")

    print("[done]")


if __name__ == "__main__":
    main()
