from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - fallback for partially prepared envs
    load_dotenv = None

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError
except ImportError:  # pragma: no cover - reported at runtime with a clearer message
    GraphDatabase = None
    Neo4jError = Exception


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class Neo4jSettings:
    uri: str
    user: str
    password: str
    database: str = "neo4j"


class Neo4jConfigurationError(RuntimeError):
    pass


def load_neo4j_settings(env_path: Path = DEFAULT_ENV_PATH) -> Neo4jSettings:
    if load_dotenv is not None:
        load_dotenv(env_path)
    else:
        _load_env_file_without_dependency(env_path)

    missing = [
        name
        for name in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        if not os.getenv(name)
    ]
    if missing:
        joined = ", ".join(missing)
        raise Neo4jConfigurationError(f"Missing required Neo4j env vars: {joined}")

    return Neo4jSettings(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
    )


def _load_env_file_without_dependency(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def run_return_one(settings: Neo4jSettings | None = None) -> int:
    if GraphDatabase is None:
        raise Neo4jConfigurationError(
            "neo4j package is not installed. Run: python -m pip install -r requirements.txt"
        )

    settings = settings or load_neo4j_settings()
    driver = GraphDatabase.driver(
        settings.uri,
        auth=(settings.user, settings.password),
    )
    try:
        with driver.session(database=settings.database) as session:
            result: Any = session.run("RETURN 1 AS ok").single(strict=True)
            return int(result["ok"])
    except Neo4jError:
        raise
    finally:
        driver.close()
