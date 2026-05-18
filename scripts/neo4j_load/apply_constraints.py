"""00_constraints.cypher 파일을 Python으로 적용.

cypher-shell 없이 실행 가능. 주석 라인을 안전하게 건너뛰면서
세미콜론 단위로 statement 분리하여 1개씩 실행.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import driver_session

CYPHER_PATH = Path(__file__).resolve().parent / "00_constraints.cypher"


def split_statements(text: str) -> list[str]:
    """// 주석 라인 제거 후 ';'로 분리."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//") or not stripped:
            continue
        lines.append(line)
    joined = "\n".join(lines)
    return [s.strip() for s in joined.split(";") if s.strip()]


def main():
    if not CYPHER_PATH.exists():
        raise FileNotFoundError(CYPHER_PATH)
    stmts = split_statements(CYPHER_PATH.read_text(encoding="utf-8"))
    print(f"[apply] {len(stmts)} statements from {CYPHER_PATH.name}")

    with driver_session() as s:
        for i, stmt in enumerate(stmts, 1):
            first = re.sub(r"\s+", " ", stmt)[:80]
            try:
                s.run(stmt)
                print(f"  [{i:2}/{len(stmts)}] OK  {first}")
            except Exception as e:
                print(f"  [{i:2}/{len(stmts)}] ERR {first}\n      -> {e}")

        # 결과 요약
        n_c = sum(1 for _ in s.run("SHOW CONSTRAINTS YIELD name"))
        n_i = sum(1 for _ in s.run("SHOW INDEXES YIELD name"))
        print(f"\n[done] constraints={n_c}, indexes={n_i}")


if __name__ == "__main__":
    main()
