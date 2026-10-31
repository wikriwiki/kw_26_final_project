"""3D 지도의 바탕 지도·라이브러리를 이 컴퓨터에서 내준다 (`SIM_MAP_CACHE_DIR`).

3D 지도는 원래 브라우저가 인터넷에서 두 가지를 받아 온다.
- 지도 라이브러리: unpkg.com 의 maplibre-gl·deck.gl (합쳐서 약 2MB)
- 바탕 지도: tiles.openfreemap.org 의 스타일·벡터 조각·글꼴·아이콘

시연장 와이파이가 느리면 첫 화면이 비고(스타일을 최대 15초 기다린다), 지도를 움직일 때마다
조각이 늦게 채워진다. 그래서 서버가 같은 주소 구조를 `/viz/map/...`·`/viz/vendor/...` 로
흉내 내고, 디스크 저장본을 먼저 준다. 저장본에 없으면 한 번 받아 저장한 뒤 준다(인터넷이 될 때).
인터넷도 안 되면 조각은 빈 응답(204)으로 넘긴다 — 지도는 그 칸만 비우고 계속 움직인다.

저장본은 `web/tools/prefetch_map.py` 가 서울 일대(확대 0~14단계)를 미리 채운다.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UPSTREAM = "https://tiles.openfreemap.org"
VENDOR = {
    # 화면 껍데기(viz.shell_html)가 쓰는 판과 같아야 한다 — scripts/sim/visualization_3d/assets.py
    "maplibre-gl.js": "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js",
    "maplibre-gl.css": "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css",
    "deck.gl.min.js": "https://unpkg.com/deck.gl@9.3.3/dist.min.js",
}
FETCH_TIMEOUT = float(os.environ.get("SIM_MAP_FETCH_TIMEOUT", "4"))
_SAFE = re.compile(r"^[A-Za-z0-9 _.@,{}\-/]+$")
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def cache_dir() -> Path | None:
    raw = os.environ.get("SIM_MAP_CACHE_DIR", "").strip()
    return Path(raw) if raw else None


def enabled() -> bool:
    directory = cache_dir()
    return directory is not None and directory.is_dir()


def vendor_ready() -> bool:
    directory = cache_dir()
    return directory is not None and all((directory / "vendor" / name).is_file() for name in VENDOR)


def _lock_for(key: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(key, threading.Lock())


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "kw-sim-console/1"})
    with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT) as response:
        body = response.read()
        if (response.headers.get("Content-Encoding") or "").lower() == "gzip":
            body = gzip.decompress(body)
        return body


def _store(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp{os.getpid()}.{threading.get_ident()}")
    temp.write_bytes(body)
    os.replace(temp, path)


def upstream_path(path: str) -> str:
    """주소 뒤쪽(`planet/2026.../14/1/2.pbf`, `fonts/Noto Sans Regular/0-255.pbf`)을 검사해 돌려준다."""
    clean = urllib.parse.unquote(path).lstrip("/")
    if not clean or ".." in clean.split("/") or not _SAFE.match(clean):
        raise ValueError("지도 주소가 아닙니다")
    return clean


def local_file(clean: str) -> Path:
    directory = cache_dir()
    assert directory is not None
    name = clean
    # 'planet'·'styles/bright' 처럼 확장자 없는 JSON 은 파일 이름에 붙여 둔다(같은 이름의 폴더와 겹치지 않게)
    if not Path(clean).suffix:
        name = clean + ".json"
    return directory / "ofm" / name


def get(clean: str, *, allow_fetch: bool = True) -> bytes | None:
    """저장본 → (없으면) 내려받아 저장. 둘 다 안 되면 None."""
    path = local_file(clean)
    if path.is_file():
        return path.read_bytes()
    if not allow_fetch:
        return None
    with _lock_for(clean):
        if path.is_file():
            return path.read_bytes()
        try:
            body = download(f"{UPSTREAM}/{urllib.parse.quote(clean)}")
        except (urllib.error.URLError, TimeoutError, OSError, ValueError):
            return None
        _store(path, body)
        return body


def rewrite(body: bytes, local_base: str) -> bytes:
    """스타일·타일 설명 JSON 안의 원래 주소를 이 서버 주소로 바꾼다."""
    text = body.decode("utf-8").replace(UPSTREAM, local_base)
    return text.encode("utf-8")


def media_type(clean: str) -> str:
    suffix = Path(clean).suffix
    if suffix == ".pbf":
        return "application/x-protobuf"
    if suffix == ".png":
        return "image/png"
    return "application/json"


def vendor_file(name: str) -> Path | None:
    directory = cache_dir()
    if directory is None or name not in VENDOR:
        return None
    path = directory / "vendor" / name
    return path if path.is_file() else None


def vendor_url(name: str) -> str:
    return f"/viz/vendor/{name}" if vendor_file(name) else VENDOR[name]


def describe() -> dict:
    directory = cache_dir()
    if directory is None:
        return {"enabled": False}
    tiles = sum(1 for _ in (directory / "ofm").rglob("*.pbf")) if (directory / "ofm").is_dir() else 0
    return {"enabled": directory.is_dir(), "vendor": vendor_ready(), "files": tiles}


def json_body(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")
