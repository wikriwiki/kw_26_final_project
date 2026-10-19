"""Neo4j 적재 공용 헬퍼.

- .env 또는 환경변수에서 NEO4J_* 로드
- driver 컨텍스트 매니저 (싱글톤, thread-safe — 매 호출마다 driver 재생성 안 함)
- 벌크 UNWIND 헬퍼
"""
from __future__ import annotations

import atexit
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import warnings
from neo4j import GraphDatabase, Driver
from neo4j.warnings import Neo4jWarning

# Suppress Neo4j 'property/rel type does not exist' WARN notifications (시뮬 노이즈)
warnings.filterwarnings("ignore", category=Neo4jWarning)


_env_cache: dict | None = None
_env_lock = threading.Lock()

def load_env(dotenv_path: Path | None = None) -> dict:
    """Load NEO4J_* env vars from .env file. 한 번 성공하면 캐시 — Google Drive
    같이 일시 unmount되는 경로에서 매 호출 재읽기로 실패하는 것 방지."""
    global _env_cache
    if _env_cache is not None:
        return _env_cache

    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parents[2] / "data" / "neo4j_load" / ".env"
    cfg = {}
    # .env read는 best-effort (G: 일시 unmount 등 안전망)
    try:
        if dotenv_path.exists():
            for line in dotenv_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass  # 환경변수로 fallback
    # env vars override (가장 우선)
    for k, v in os.environ.items():
        if k.startswith("NEO4J_") or k.startswith("VWORLD_"):
            cfg[k] = v
    # 첫 성공 시 캐시 (NEO4J_PASSWORD 확보됐을 때만)
    if cfg.get("NEO4J_PASSWORD"):
        with _env_lock:
            if _env_cache is None:
                _env_cache = cfg
    return cfg


def get_neo4j_config() -> dict:
    cfg = load_env()
    return {
        "uri": cfg.get("NEO4J_URI", "bolt://localhost:7687"),
        "user": cfg.get("NEO4J_USER", "neo4j"),
        "password": cfg.get("NEO4J_PASSWORD"),
        "database": cfg.get("NEO4J_DATABASE", "neo4j"),
    }


# ---------------------------------------------------------------------------
# Driver 싱글톤 (thread-safe)
# ---------------------------------------------------------------------------
# Bolt driver 는 자체 connection pool 을 가짐 — 한 프로세스에 1개만 만들고
# 모든 thread / agent 가 공유하면 connection 재사용 효과 극대화.
# 시뮬 워크로드는 workers=16~32 × agent 한 명당 5~10 session = 매분 수백~수천 회
# session 생성. driver 를 매번 만들면 인증·핸드셰이크 비용이 추가.
# ---------------------------------------------------------------------------
_driver_lock = threading.Lock()
_driver_cache: dict[str, Driver] = {}     # uri → Driver
_atexit_registered = False


def _get_or_create_driver(cfg: dict) -> Driver:
    """URI 별 싱글톤 driver. 첫 호출에만 생성, 이후 재사용."""
    global _atexit_registered
    uri = cfg["uri"]

    # double-checked locking
    drv = _driver_cache.get(uri)
    if drv is not None:
        return drv

    with _driver_lock:
        drv = _driver_cache.get(uri)
        if drv is not None:
            return drv

        if not cfg["password"]:
            raise RuntimeError(
                "NEO4J_PASSWORD not set. Add to data/neo4j_load/.env "
                "or env. Example: NEO4J_PASSWORD=your_password"
            )

        # connection pool 크기를 workers 고려해 충분히 (기본 100)
        pool_size = int(os.environ.get("NEO4J_POOL_SIZE", "100"))
        drv = GraphDatabase.driver(
            cfg["uri"], auth=(cfg["user"], cfg["password"]),
            max_connection_pool_size=pool_size,
            connection_acquisition_timeout=60.0,
            # 서버 알림 억제: avg()에 null 섞일 때의 '01G11'(null value eliminated) 등
            # 무해한 경고가 로그를 수백만 줄로 오염시켜 모니터링을 막는다. 서버가
            # 알림 자체를 계산하지 않아 미세한 성능 이점도 있음.
            notifications_min_severity="OFF",
        )
        _driver_cache[uri] = drv

        if not _atexit_registered:
            atexit.register(close_all_drivers)
            _atexit_registered = True

    return drv


def close_all_drivers() -> None:
    """프로세스 종료 시 또는 명시적 호출 — 모든 driver 정리."""
    with _driver_lock:
        for drv in _driver_cache.values():
            try:
                drv.close()
            except Exception:
                pass
        _driver_cache.clear()


@contextmanager
def driver_session(database: str | None = None):
    """세션 컨텍스트 — driver 는 싱글톤 재사용, 세션만 매번 생성/종료.

    Bolt 의 session() 은 가볍다 — connection pool 에서 conn 한 개 잠시 빌리는 정도.
    driver() 가 진짜 비싼 작업(인증·라우팅 테이블) 인데 그건 한 번만.
    """
    cfg = get_neo4j_config()
    drv = _get_or_create_driver(cfg)
    db = database or cfg["database"]
    with drv.session(database=db) as session:
        yield session


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
