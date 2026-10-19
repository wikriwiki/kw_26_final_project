"""SQLite 체크포인트 + JSONL 익스포트. 중단/재개 가능."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

DB_PATH = Path("C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS poi_status (
    poi_id      TEXT PRIMARY KEY,        -- 우리 POI ID (소상공인 id)
    poi_name    TEXT,
    poi_addr    TEXT,
    poi_lat     REAL,
    poi_lon     REAL,
    kakao_pid   TEXT,                    -- 매칭된 카카오 place_id
    status      TEXT NOT NULL,           -- pending|matched|fetched|error|nomatch|banned
    error_msg   TEXT,
    fetched_at  TEXT,
    attempts    INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_status ON poi_status(status);

CREATE TABLE IF NOT EXISTS panel3_raw (
    kakao_pid   TEXT PRIMARY KEY,
    raw_json    TEXT NOT NULL,
    fetched_at  TEXT NOT NULL,
    bytes       INTEGER
);

CREATE TABLE IF NOT EXISTS quota_log (
    date_key    TEXT PRIMARY KEY,        -- YYYY-MM-DD
    n_calls     INTEGER DEFAULT 0,
    n_ok        INTEGER DEFAULT 0,
    n_4xx       INTEGER DEFAULT 0,
    n_5xx       INTEGER DEFAULT 0
);
"""


def open_db(path: Path = DB_PATH, *, check_same_thread: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=20,
                            check_same_thread=check_same_thread)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.commit()
    return conn


def upsert_poi(conn: sqlite3.Connection, *, poi_id: str, poi_name: str,
               poi_addr: str, poi_lat: float, poi_lon: float, status: str = "pending"):
    conn.execute(
        "INSERT INTO poi_status(poi_id, poi_name, poi_addr, poi_lat, poi_lon, status) "
        "VALUES (?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(poi_id) DO NOTHING",
        (poi_id, poi_name, poi_addr, poi_lat, poi_lon, status),
    )


def mark_matched(conn: sqlite3.Connection, poi_id: str, kakao_pid: str):
    conn.execute(
        "UPDATE poi_status SET kakao_pid=?, status='matched', attempts=attempts+1 "
        "WHERE poi_id=?",
        (kakao_pid, poi_id),
    )


def mark_error(conn: sqlite3.Connection, poi_id: str, status: str, msg: str):
    conn.execute(
        "UPDATE poi_status SET status=?, error_msg=?, attempts=attempts+1 "
        "WHERE poi_id=?",
        (status, msg[:500], poi_id),
    )


def save_panel3(conn: sqlite3.Connection, kakao_pid: str, raw: dict):
    blob = json.dumps(raw, ensure_ascii=False)
    conn.execute(
        "INSERT INTO panel3_raw(kakao_pid, raw_json, fetched_at, bytes) "
        "VALUES (?, ?, datetime('now'), ?) "
        "ON CONFLICT(kakao_pid) DO UPDATE SET raw_json=excluded.raw_json, "
        "fetched_at=excluded.fetched_at, bytes=excluded.bytes",
        (kakao_pid, blob, len(blob)),
    )


def mark_fetched(conn: sqlite3.Connection, poi_id: str):
    conn.execute(
        "UPDATE poi_status SET status='fetched', fetched_at=datetime('now') "
        "WHERE poi_id=?",
        (poi_id,),
    )


def bump_quota(conn: sqlite3.Connection, status_code: int):
    import datetime as dt
    today = dt.date.today().isoformat()
    conn.execute(
        "INSERT INTO quota_log(date_key, n_calls) VALUES (?, 0) "
        "ON CONFLICT(date_key) DO NOTHING",
        (today,),
    )
    bucket = "n_ok" if 200 <= status_code < 300 else (
        "n_4xx" if 400 <= status_code < 500 else "n_5xx"
    )
    conn.execute(
        f"UPDATE quota_log SET n_calls=n_calls+1, {bucket}={bucket}+1 "
        "WHERE date_key=?",
        (today,),
    )


def stats(conn: sqlite3.Connection) -> dict:
    s = {}
    for row in conn.execute("SELECT status, COUNT(*) FROM poi_status GROUP BY status"):
        s[row[0]] = row[1]
    return s


def pending_pois(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    rows = conn.execute(
        "SELECT poi_id, poi_name, poi_addr, poi_lat, poi_lon "
        "FROM poi_status WHERE status IN ('pending', 'matched') AND attempts < 3 "
        "LIMIT ?",
        (limit,),
    ).fetchall()
    return [{"poi_id": r[0], "poi_name": r[1], "poi_addr": r[2],
             "poi_lat": r[3], "poi_lon": r[4]} for r in rows]
