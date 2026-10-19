"""Category 노드 적재 (categories.yaml).

현재 모델: 단일 라벨 :Category. 12 L1 + ~90 L2 sub.
L1/L2를 모두 별도 노드로 만들지 않고, **L2 노드만** 적재.
L2는 cat·sub 두 속성으로 구분 (`name` = sub, `parent` = L1).

이유:
- runtime_ontology.md §0: "단일 라벨 :Category 채택, 어휘 확정 시 L1/L2 분리 가능"
- POI는 IN_CATEGORY로 sub(L2)에 연결
- L1 그루핑은 `parent` 속성 필터 (인덱스 가능)

L2 가 dict 인 경우 (override):
    sub:
      - 한식                              # L1 값 상속
      - { name: 분식, saturation_n: 18 }  # 일부 필드만 override
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = PROJECT_ROOT / "data" / "neo4j_load" / "categories" / "categories.yaml"


# desire 파라미터 누락 시 안전 fallback (yaml 깨졌을 때만 발동)
DESIRE_DEFAULTS = {
    "recovery_tau_days": 7.0,
    "desire_drop": 0.75,
    "saturation_n": 6,
}


def parse_categories_yaml(raw: dict) -> list[dict]:
    """yaml dict → 적재용 row 리스트.

    각 row 는 한 L2 카테고리. L1 의 desire/operating 필드를 상속하되
    L2 가 dict 면 명시된 필드만 override.
    """
    rows: list[dict] = []
    for cat in raw.get("categories", []):
        l1 = cat["name"]
        l1_props = {
            "parent": l1,
            "open_hour": cat.get("open"),
            "close_hour": cat.get("close"),
            "recovery_tau_days": cat.get("recovery_tau_days",
                                         DESIRE_DEFAULTS["recovery_tau_days"]),
            "desire_drop": cat.get("desire_drop", DESIRE_DEFAULTS["desire_drop"]),
            "saturation_n": cat.get("saturation_n", DESIRE_DEFAULTS["saturation_n"]),
        }
        for sub in cat.get("sub", []):
            if isinstance(sub, str):
                row = {"name": sub, **l1_props}
            elif isinstance(sub, dict):
                if "name" not in sub:
                    raise ValueError(f"L2 sub dict missing 'name' under L1='{l1}': {sub}")
                row = {"name": sub["name"], **l1_props}
                # override allowed: desire / operating
                for k in ("recovery_tau_days", "desire_drop", "saturation_n",
                          "open_hour", "close_hour"):
                    if k in sub:
                        row[k] = sub[k]
            else:
                raise ValueError(f"L2 sub must be str or dict under L1='{l1}': {sub!r}")
            rows.append(row)
    return rows


def main():
    raw = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))
    rows = parse_categories_yaml(raw)
    print(f"[parse] {len(rows)} (L1, L2) pairs")

    with driver_session() as s:
        s.run("""
            UNWIND $batch AS c
            MERGE (n:Category {name: c.name})
            SET n.parent = c.parent,
                n.open_hour = c.open_hour,
                n.close_hour = c.close_hour,
                n.recovery_tau_days = c.recovery_tau_days,
                n.desire_drop = c.desire_drop,
                n.saturation_n = c.saturation_n
        """, batch=rows)
        print(f"  + Category x {len(rows)}  (with desire params)")

    # parent 인덱스 (런타임 L1 필터용)
    with driver_session() as s:
        s.run("CREATE INDEX category_parent IF NOT EXISTS FOR (c:Category) ON (c.parent)")
        print(f"  + INDEX category_parent")

    print("[done]")


if __name__ == "__main__":
    main()
