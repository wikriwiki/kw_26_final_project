"""3D 지도 서빙 — 뷰어 껍데기, 정적 파일, 데이터 팩.

예전에는 데이터까지 구운 단일 HTML(시연 표본 133MB)을 통째로 내려보냈다. 이제는
`docs/DESIGN_VIZ3D_INTEGRATION.md` 의 구조대로 나눈다.

* ``/viz/shell.html``            데이터 없는 껍데기. 정적 JS 를 ``<script src>`` 로 싣는다.
* ``/viz/static/<파일>``          ``scripts/sim/visualization_3d/static`` 원본을 그대로 읽어 보낸다.
* ``/viz/packs/<팩>/<파일>``      ``visualization_3d/pack.py`` 가 만든 팩 (manifest + gzip 조각)
* ``/api/viz/packs``             고를 수 있는 팩 목록
* ``/api/viz/packs/<팩>/residents`` 그 팩 시민의 집·직장 좌표만 (첫 화면 점 지도용)

껍데기는 ``?pack=/viz/packs/<팩>/`` 로 어떤 팩을 그릴지 받는다. 팩 파일 주소에는
해시(``?v=``)가 붙으므로 오래 캐시해도 팩을 다시 만들면 저절로 새로 받는다.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse

from scripts.sim.build_standalone_html import STATIC_JS_FILES, render_template
from scripts.sim.visualization_3d import assets as viz_assets
from scripts.sim.visualization_3d.pack import PACK_FORMAT

from .store import StoreError

REPO_ROOT = Path(__file__).resolve().parents[2]
VIZ3D_DIR = REPO_ROOT / "scripts" / "sim" / "visualization_3d"
STATIC_DIR = VIZ3D_DIR / "static"
TEMPLATE_PATH = VIZ3D_DIR / "template.html"
THEME_PATH = REPO_ROOT / "web" / "ui" / "viz-embed.css"
STATIC_FILES = frozenset((*STATIC_JS_FILES, "styles.css"))
PACK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
PACK_SUFFIXES = (".json", ".gz")
IMMUTABLE = "public, max-age=31536000, immutable"
REVALIDATE = "no-cache"
MEDIA_TYPES = {".js": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8"}


def packs_root() -> Path | None:
    """팩 폴더들의 부모. 없으면 ``None`` — 지어낸 경로로 빈 목록을 만들지 않는다."""
    configured = os.environ.get("SIM_VIZ_PACKS_DIR", "").strip()
    if configured:
        path = Path(configured).resolve()
        return path if path.is_dir() else None
    candidates = []
    data_root = os.environ.get("SIM_DATA_ROOT", "").strip()
    if data_root:
        candidates.append(Path(data_root) / "viz" / "packs")
    candidates.append(REPO_ROOT / "web" / "viz_store" / "packs")
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    return None


def list_packs() -> dict[str, Any]:
    root = packs_root()
    if root is None:
        return {"total": 0, "items": [], "unknown": ["viz_packs_root"]}
    items = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not PACK_ID.match(child.name):
            continue
        manifest_path = child / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            items.append({"pack_id": child.name, "error": "manifest 를 읽을 수 없습니다"})
            continue
        if manifest.get("format") != PACK_FORMAT:
            continue
        items.append(_pack_summary(child.name, manifest, manifest_path.stat().st_size))
    return {"total": len(items), "items": items, "unknown": []}


def _pack_summary(pack_id: str, manifest: dict[str, Any], manifest_bytes: int) -> dict[str, Any]:
    days = [frame.get("day") for frame in manifest.get("frames") or [] if frame.get("day")]
    chunks = manifest.get("chunks") or []
    shards = (manifest.get("detail") or {}).get("shards") or []
    agents_entry = manifest.get("agents") or {}
    return {
        "pack_id": pack_id,
        "label": manifest.get("label"),
        "run_id": manifest.get("run_id"),
        "agent_count": manifest.get("agent_count"),
        "frame_count": manifest.get("frame_count"),
        "first_day": days[0] if days else None,
        "last_day": days[-1] if days else None,
        "day_count": len(set(days)),
        "created_at": manifest.get("created_at"),
        # 지도를 처음 그리는 데 받는 양과, 사람을 고를 때 나눠 받는 양
        "boot_bytes": manifest_bytes + int(agents_entry.get("bytes") or 0) + sum(int(c.get("bytes") or 0) for c in chunks),
        "detail_bytes": sum(int(s.get("bytes") or 0) for s in shards),
        "shell_url": f"/viz/shell.html?pack=/viz/packs/{pack_id}/",
    }


def pack_file(pack_id: str, file_path: str) -> Path:
    if not PACK_ID.match(pack_id):
        raise StoreError(400, "잘못된 팩 이름입니다")
    root = packs_root()
    if root is None:
        raise StoreError(404, "시각화 팩 폴더가 설정되지 않았습니다")
    pack_dir = (root / pack_id).resolve()
    target = (pack_dir / file_path).resolve()
    if pack_dir not in target.parents or target.suffix not in PACK_SUFFIXES or not target.is_file():
        raise StoreError(404, "팩 파일이 없습니다")
    return target


#: 좌표를 정수로 보낼 때의 원점과 눈금. 1e-4 도는 서울에서 대략 9~11m — 점 지도에는 넘친다
_ORIGIN = (126.7, 37.4)
_SCALE = 1e-4
_residents_lock = threading.Lock()
_residents_cache: dict[str, tuple[tuple[int, int], dict[str, Any]]] = {}


def _q(value: Any, origin: float) -> int | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return round((v - origin) / _SCALE)


def residents(pack_id: str) -> dict[str, Any]:
    """팩 시민의 집·직장 좌표만 촘촘하게 돌려준다.

    대상자 파일 전체(5,000명에 약 650KB)를 첫 화면에서 받지 않게 하려는 것이다. 좌표는
    1e-4 도 단위 정수로 줄이고 `[집x, 집y, 직장x, 직장y]` 를 한 줄로 이어 붙인다.
    직장이 없는 사람은 직장 칸이 null 이다 — 집 좌표로 채우지 않는다.
    """
    manifest = json.loads(pack_file(pack_id, "manifest.json").read_text(encoding="utf-8"))
    agents_path = pack_file(pack_id, (manifest.get("agents") or {}).get("file") or "agents.json.gz")
    st = agents_path.stat()
    key = (st.st_mtime_ns, st.st_size)
    with _residents_lock:
        hit = _residents_cache.get(pack_id)
    if hit is not None and hit[0] == key:
        return hit[1]
    raw = agents_path.read_bytes()
    agents = json.loads(gzip.decompress(raw) if agents_path.suffix == ".gz" else raw)
    points: list[int | None] = []
    count = 0
    for agent in agents if isinstance(agents, list) else []:
        hx, hy = _q(agent.get("home_lon"), _ORIGIN[0]), _q(agent.get("home_lat"), _ORIGIN[1])
        if hx is None or hy is None:
            continue
        wx, wy = _q(agent.get("work_lon"), _ORIGIN[0]), _q(agent.get("work_lat"), _ORIGIN[1])
        points += [hx, hy, wx, wy] if wx is not None and wy is not None else [hx, hy, None, None]
        count += 1
    payload = {
        "pack_id": pack_id,
        "label": manifest.get("label"),
        "run_id": manifest.get("run_id"),
        "count": count,
        "origin": list(_ORIGIN),
        "scale": _SCALE,
        "stride": 4,
        "points": points,
        "unknown": [],
    }
    with _residents_lock:
        _residents_cache[pack_id] = (key, payload)
    return payload


def static_file(name: str) -> Path:
    if name not in STATIC_FILES:
        raise StoreError(404, "정적 파일이 없습니다")
    return STATIC_DIR / name


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def shell_html(theme: bool) -> str:
    styles = [
        f'<link rel="stylesheet" href="{viz_assets.MAPLIBRE_CSS_URL}">',
        f'<link rel="stylesheet" href="/viz/static/styles.css?v={_digest(STATIC_DIR / "styles.css")}">',
    ]
    if theme and THEME_PATH.is_file():
        styles.append(f'<link rel="stylesheet" href="/viz/theme.css?v={_digest(THEME_PATH)}">')
    scripts = [
        f'<script src="{viz_assets.MAPLIBRE_JS_URL}"></script>',
        f'<script src="{viz_assets.DECK_JS_URL}"></script>',
    ]
    scripts += [f'<script src="/viz/static/{name}?v={_digest(STATIC_DIR / name)}"></script>' for name in STATIC_JS_FILES]
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    return render_template(template, {"styles": "\n".join(styles), "data": "", "scripts": "\n".join(scripts)})


def _cache_header(request: Request) -> str:
    return IMMUTABLE if request.query_params.get("v") else REVALIDATE


def register(app: FastAPI) -> None:
    @app.get("/api/viz/packs")
    async def viz_packs() -> dict[str, Any]:
        return list_packs()

    @app.get("/api/viz/packs/{pack_id}/residents")
    async def viz_residents(pack_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(residents, pack_id)

    @app.get("/viz/shell.html")
    async def viz_shell(theme: str | None = None) -> HTMLResponse:
        return HTMLResponse(shell_html(theme == "console"), headers={"Cache-Control": REVALIDATE})

    @app.get("/viz/static/{name}")
    async def viz_static(name: str, request: Request) -> FileResponse:
        path = static_file(name)
        return FileResponse(path, media_type=MEDIA_TYPES[path.suffix], headers={"Cache-Control": _cache_header(request)})

    @app.get("/viz/theme.css")
    async def viz_theme(request: Request) -> FileResponse:
        if not THEME_PATH.is_file():
            raise StoreError(404, "콘솔 테마 파일이 없습니다")
        return FileResponse(THEME_PATH, media_type=MEDIA_TYPES[".css"], headers={"Cache-Control": _cache_header(request)})

    @app.get("/viz/packs/{pack_id}/{file_path:path}")
    async def viz_pack_file(pack_id: str, file_path: str, request: Request) -> FileResponse:
        path = pack_file(pack_id, file_path)
        # manifest 는 팩을 다시 만들면 바뀌므로 매번 확인한다. 조각은 해시가 주소에 붙는다.
        cache = REVALIDATE if path.name == "manifest.json" else _cache_header(request)
        media = "application/json" if path.suffix == ".json" else "application/octet-stream"
        return FileResponse(path, media_type=media, headers={"Cache-Control": cache})
