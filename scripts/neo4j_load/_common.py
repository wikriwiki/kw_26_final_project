"""Neo4j 적재 공용 헬퍼.

- .env 또는 환경변수에서 NEO4J_* 로드
- driver 컨텍스트 매니저
- 벌크 UNWIND 헬퍼
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import warnings
from neo4j import GraphDatabase, Driver
from neo4j.warnings import Neo4jWarning

# Suppress Neo4j 'property/rel type does not exist' WARN notifications (시뮬 노이즈)
warnings.filterwarnings("ignore", category=Neo4jWarning)


def load_env(dotenv_path: Path | None = None) -> dict:
    """Load NEO4J_* env vars from .env file."""
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[2] / "data" / "neo4j_load" / ".env"
    cfg = {}
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip().strip('"').strip("'")
    # env vars override
    for k, v in os.environ.items():
        if k.startswith("NEO4J_") or k.startswith("VWORLD_"):
            cfg[k] = v
    return cfg


def get_neo4j_config() -> dict:
    cfg = load_env()
    return {
        "uri": cfg.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": cfg.get("NEO4J_USER", "neo4j"),
        "password": cfg.get("NEO4J_PASSWORD"),
        "database": cfg.get("NEO4J_DATABASE", "neo4j"),
    }


@contextmanager
def driver_session(database: str | None = None):
    cfg = get_neo4j_config()
    if not cfg["password"]:
        raise RuntimeError(
            "NEO4J_PASSWORD not set. Add to data/neo4j_load/.env "
            "or env. Example: NEO4J_PASSWORD=your_password"
        )
    drv: Driver = GraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))
    db = database or cfg["database"]
    try:
        with drv.session(database=db) as session:
            yield session
    finally:
        drv.close()


def bulk_run(session, cypher: str, batch: list[dict], batch_size: int = 5000, **params):
    """UNWIND 패턴 벌크 실행. batch_size 단위로 청크 처리."""
    total = len(batch)
    for i in range(0, total, batch_size):
        chunk = batch[i : i + batch_size]
        session.run(cypher, batch=chunk, **params)


def chunked(it: Iterable, n: int):
    buf = []
    for x in it:
        buf.append(x)
        if len(buf) >= n:
            yield buf
            buf = []
    if buf:
        yield buf
