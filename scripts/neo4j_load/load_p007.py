"""P007 정책 적재 — DEPRECATED.

이 스크립트는 하위 호환을 위해 유지됨. 새 코드는 다음을 사용하세요:

  python -m scripts.policy_pipeline.inject_json \\
      data/neo4j_load/policies/P007.json \\
      --deactivate-others-from 2026-05-01

이 thin wrapper 는 같은 결과를 만들지만 검증·Scope·캐시 무효화·summary 잡 큐
등록까지 한번에 수행한다. 하드코딩된 Cypher 는 사라졌다.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    print(
        "[load_p007] DEPRECATED — use:\n"
        "  python -m scripts.policy_pipeline.inject_json "
        "data/neo4j_load/policies/P007.json --deactivate-others-from 2026-05-01\n"
        "[load_p007] forwarding to new pipeline..."
    )
    # 프로젝트 루트가 sys.path 에 들어가도록
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from scripts.policy_pipeline.inject_json import main as inject_main

    sys.argv = [
        "inject_json",
        str(root / "data" / "neo4j_load" / "policies" / "P007.json"),
        "--deactivate-others-from", "2026-05-01",
    ]
    return inject_main()


if __name__ == "__main__":
    raise SystemExit(main())
