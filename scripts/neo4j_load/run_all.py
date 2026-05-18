"""Day 0 적재 전체 순차 실행.

순서:
  00_constraints.cypher   (cypher-shell로 별도 실행 권장)
  01_admin.py             District/Dong + ADJACENT_TO
  02_categories.py        Category
  03_pois.py              POI 3종 + IN_DONG + IN_CATEGORY
  04_agents.py            Agent
  05_anchors.py           LIVES_AT + WORKS_AT
  06_social.py            KNOWS
  07_initial_awareness.py KNOWS_POI{initial} + Memory{initial}
  99_validate.py          무결성 검증
"""
from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

ORDER = [
    ("01_admin", "01_admin"),
    ("02_categories", "02_categories"),
    ("03_pois", "03_pois"),
    ("04_agents", "04_agents"),
    ("05_anchors", "05_anchors"),
    ("06_social", "06_social"),
    ("07_initial_awareness", "07_initial_awareness"),
    ("08_initial_state", "08_initial_state"),
    ("99_validate", "99_validate"),
]


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    for module_name, label in ORDER:
        print()
        print(f"=== {label} ===")
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
        except Exception as e:
            print(f"[FAIL] {label}: {e}")
            raise
        print(f"=== {label} ok ({time.time()-t0:.1f}s) ===")
    print()
    print("ALL DONE.")


if __name__ == "__main__":
    main()
