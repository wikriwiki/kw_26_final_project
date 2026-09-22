"""V-WORLD Geocoder API 2.0 client.

- 도로명/지번 주소 → 좌표 (WGS84) + 행정동 코드
- SQLite 캐시 (재실행 시 캐시 hit으로 즉시 완료)
- 지수 백오프 재시도
- ThreadPoolExecutor로 batch 동시 호출

API 문서: https://www.vworld.kr/dev/v4dv_geocoderguide2_s001.do
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Iterable, Callable

import requests

VWORLD_URL = "https://api.vworld.kr/req/address"
DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF = 0.5  # seconds


@dataclass
class GeocodeResult:
    address_in: str
    type_used: str  # "road" | "parcel"
    status: str    # "OK" | "NOT_FOUND" | "ERROR"
    lon: Optional[float] = None
    lat: Optional[float] = None
    level1: Optional[str] = None   # 시도
    level2: Optional[str] = None   # 시군구
    level3: Optional[str] = None   # 읍면동 (법정동 텍스트)
    level4L: Optional[str] = None  # 리
    level4LC: Optional[str] = None # 법정동 코드 (10자리)
    level4A: Optional[str] = None  # 행정동 텍스트
    level4AC: Optional[str] = None # 행정동 코드 (10자리 MOPAS) — [:8] 하면 NSO 8자리
    refined: Optional[str] = None
    raw_error: Optional[str] = None

    def to_row(self) -> dict:
        return asdict(self)


def load_api_key(dotenv_path: Optional[Path] = None) -> str:
    """Load VWORLD_API_KEY from env or .env file."""
    key = os.environ.get("VWORLD_API_KEY")
    if key:
        return key
    if dotenv_path is None:
        # default location
        dotenv_path = Path(__file__).resolve().parents[2] / "data" / "neo4j_load" / ".env"
    if dotenv_path.exists():
        for line in dotenv_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "VWORLD_API_KEY":
                return v.strip().strip('"').strip("'")
    raise RuntimeError(f"VWORLD_API_KEY not found in env or {dotenv_path}")


class VWorldGeocoder:
    """V-WORLD Geocoder API 2.0 wrapper with SQLite cache + retry."""

    def __init__(
        self,
        api_key: str,
        cache_path: Path | str,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff: float = DEFAULT_BACKOFF,
    ):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self._lock = threading.Lock()
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_cache()
        # per-thread session
        self._tls = threading.local()
        # stats
        self.stats = {"hit": 0, "miss": 0, "not_found": 0, "error": 0, "retries": 0}

    # ---------- cache ----------
    def _init_cache(self):
        conn = sqlite3.connect(self.cache_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS geocode_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                ts INTEGER NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def _cache_key(self, address: str, type_used: str) -> str:
        return f"{type_used}|{address.strip()}"

    def _cache_get(self, key: str) -> Optional[GeocodeResult]:
        with self._lock:
            conn = sqlite3.connect(self.cache_path)
            row = conn.execute(
                "SELECT payload FROM geocode_cache WHERE cache_key=?", (key,)
            ).fetchone()
            conn.close()
        if not row:
            return None
        try:
            data = json.loads(row[0])
            return GeocodeResult(**data)
        except Exception:
            return None

    def _cache_set(self, key: str, result: GeocodeResult):
        with self._lock:
            conn = sqlite3.connect(self.cache_path)
            conn.execute(
                "INSERT OR REPLACE INTO geocode_cache (cache_key, payload, ts) VALUES (?, ?, ?)",
                (key, json.dumps(result.to_row(), ensure_ascii=False), int(time.time())),
            )
            conn.commit()
            conn.close()

    # ---------- HTTP ----------
    def _session(self) -> requests.Session:
        if not hasattr(self._tls, "session"):
            s = requests.Session()
            s.headers.update({"User-Agent": "Kw-final-project-neo4j-load/1.0"})
            self._tls.session = s
        return self._tls.session

    def _request(self, address: str, type_used: str) -> dict:
        params = {
            "service": "address",
            "request": "getCoord",
            "version": "2.0",
            "crs": "epsg:4326",
            "address": address,
            "refine": "true",
            "simple": "false",
            "format": "json",
            "type": type_used,
            "key": self.api_key,
        }
        resp = self._session().get(VWORLD_URL, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _parse(self, address: str, type_used: str, data: dict) -> GeocodeResult:
        try:
            response = data.get("response", {})
            status = response.get("status", "ERROR")
            if status != "OK":
                return GeocodeResult(
                    address_in=address, type_used=type_used,
                    status="NOT_FOUND" if status == "NOT_FOUND" else "ERROR",
                    raw_error=response.get("error", {}).get("text") or json.dumps(response)[:200],
                )
            result = response.get("result", {})
            refined = response.get("refined", {})
            point = result.get("point", {})
            struct = refined.get("structure", {})  # V-WORLD 2.0: structure는 refined 하위
            return GeocodeResult(
                address_in=address,
                type_used=type_used,
                status="OK",
                lon=float(point["x"]) if "x" in point else None,
                lat=float(point["y"]) if "y" in point else None,
                level1=struct.get("level1") or None,
                level2=struct.get("level2") or None,
                level3=struct.get("level3") or None,
                level4L=struct.get("level4L") or None,
                level4LC=struct.get("level4LC") or None,
                level4A=struct.get("level4A") or None,
                level4AC=struct.get("level4AC") or None,
                refined=refined.get("text") or None,
            )
        except Exception as e:
            return GeocodeResult(
                address_in=address, type_used=type_used,
                status="ERROR", raw_error=f"parse_err: {e}",
            )

    # ---------- public ----------
    def geocode_one(
        self,
        address: str,
        type_first: str = "road",
        fallback: bool = True,
    ) -> GeocodeResult:
        """Geocode one address. road 시도 후 실패 시 parcel fallback."""
        if not address or not address.strip():
            return GeocodeResult(address_in=address or "", type_used=type_first, status="ERROR", raw_error="empty")

        last_err = None
        # try first type with cache
        for type_used in ([type_first, "parcel" if type_first == "road" else "road"] if fallback else [type_first]):
            key = self._cache_key(address, type_used)
            cached = self._cache_get(key)
            if cached is not None:
                self.stats["hit"] += 1
                if cached.status == "OK":
                    return cached
                # cached miss → 다음 type 시도
                continue

            self.stats["miss"] += 1
            last_err = None
            for attempt in range(self.max_retries):
                try:
                    data = self._request(address, type_used)
                    parsed = self._parse(address, type_used, data)
                    self._cache_set(key, parsed)
                    if parsed.status == "OK":
                        return parsed
                    if parsed.status == "NOT_FOUND":
                        self.stats["not_found"] += 1
                        break  # try next type
                    last_err = parsed.raw_error
                    break  # ERROR (non-retryable parse)
                except requests.exceptions.RequestException as e:
                    last_err = str(e)
                    self.stats["retries"] += 1
                    time.sleep(self.backoff * (2 ** attempt))
            if last_err is not None and not isinstance(self._cache_get(key), GeocodeResult):
                # retries exhausted: cache the error
                err = GeocodeResult(
                    address_in=address, type_used=type_used,
                    status="ERROR", raw_error=last_err,
                )
                self._cache_set(key, err)

        # 둘 다 실패
        self.stats["error"] += 1
        return GeocodeResult(
            address_in=address, type_used=type_first,
            status="NOT_FOUND" if last_err is None else "ERROR",
            raw_error=last_err,
        )

    def geocode_batch(
        self,
        addresses: Iterable[str],
        concurrency: int = 10,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> list[GeocodeResult]:
        addresses = list(addresses)
        n = len(addresses)
        results: list[Optional[GeocodeResult]] = [None] * n
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            futures = {ex.submit(self.geocode_one, addr): i for i, addr in enumerate(addresses)}
            done = 0
            for fut in as_completed(futures):
                i = futures[fut]
                try:
                    results[i] = fut.result()
                except Exception as e:
                    results[i] = GeocodeResult(
                        address_in=addresses[i], type_used="road",
                        status="ERROR", raw_error=str(e),
                    )
                done += 1
                if progress and done % 50 == 0:
                    progress(done, n)
            if progress:
                progress(done, n)
        return [r for r in results if r is not None]
