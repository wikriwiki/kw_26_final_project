from __future__ import annotations

from src.graph.client import load_neo4j_settings, run_return_one


def main() -> None:
    settings = load_neo4j_settings()
    result = run_return_one(settings)
    print(f"Neo4j connection OK: RETURN 1 -> {result}")


if __name__ == "__main__":
    main()
