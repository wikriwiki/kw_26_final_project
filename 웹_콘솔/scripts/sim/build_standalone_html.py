"""Build the 3D simulation visualization as a single offline-friendly HTML file.

데이터는 JS 리터럴로 박지 않는다. ``visualization_3d/pack.py`` 의 팩(열 단위 바이너리
타임라인 + 대상자별 기억·일정 묶음)을 gzip·base64 로 ``<script type="application/octet-stream">``
에 담는다. 브라우저는 이 태그를 실행하지 않으므로 HTML 을 여는 데 JS 파싱 비용이 들지 않고,
뷰어가 필요한 순간에 풀어 쓴다. 예전 133MB 파일이 같은 데이터로 약 17MB 가 된다.
"""
from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

try:
    from scripts.sim.visualization_3d.assets import (
        CHART_JS_URL,
        DECK_JS_URL,
        GOOGLE_FONTS_URL,
        MAPLIBRE_CSS_URL,
        MAPLIBRE_JS_URL,
        script_tag,
        style_tag,
    )
    from scripts.sim.visualization_3d.derive import build_viz_meta
    from scripts.sim.visualization_3d.pack import build_pack, write_pack
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    from visualization_3d.assets import (  # type: ignore[no-redef]
        CHART_JS_URL,
        DECK_JS_URL,
        GOOGLE_FONTS_URL,
        MAPLIBRE_CSS_URL,
        MAPLIBRE_JS_URL,
        script_tag,
        style_tag,
    )
    from visualization_3d.derive import build_viz_meta  # type: ignore[no-redef]
    from visualization_3d.pack import build_pack, write_pack  # type: ignore[no-redef]

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VIZ_DIR = Path(
    os.environ.get(
        "VIZ_OUT_DIR",
        str(Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization"),
    )
)

VIZ3D_DIR = Path(__file__).resolve().parent / "visualization_3d"
TEMPLATE_PATH = VIZ3D_DIR / "template.html"
STATIC_DIR = VIZ3D_DIR / "static"
TEMPLATE_MARKERS = (
    "<!-- __SIM_STYLES__ -->",
    "<!-- __SIM_DATA__ -->",
    "<!-- __SIM_SCRIPTS__ -->",
)
TEMPLATE_KEYS = {
    "<!-- __SIM_STYLES__ -->": "styles",
    "<!-- __SIM_DATA__ -->": "data",
    "<!-- __SIM_SCRIPTS__ -->": "scripts",
}
JSON_INPUTS = {
    "__AGENTS__": "agents.json",
    "__TIMELINE__": "timeline.json",
    "__MEMORIES__": "memories.json",
    "__EVENTS__": "events.json",
}
STATIC_JS_FILES = (
    "pack_loader.js",
    "data_model.js",
    "map_scene.js",
    "layers.js",
    "detail.js",
    "hud.js",
    "app.js",
)


def fetch_policy_dongs_safe() -> list[dict]:
    """정책-동 적용 범위 조회. Neo4j 접속이 안 되면(오프라인 재빌드 등) 빈 리스트로
    폴백 — 정책 마커는 빠지지만 나머지 시각화 빌드는 계속 진행된다."""
    try:
        return fetch_policy_dongs()
    except Exception as exc:  # pragma: no cover - best-effort network/db call
        print(f"  WARNING: policy dong fetch failed ({exc}), skipping policy zones")
        return []


def fetch_policy_dongs() -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
    from _common import driver_session  # noqa: PLC0415

    with driver_session() as s:
        rows = s.run(
            """
            MATCH (p:Policy)-[:applied_to]->(d:Dong)
            WHERE d.lon IS NOT NULL AND d.lat IS NOT NULL
            RETURN p.id AS policy_id, p.name AS policy_name,
                   p.effective_from AS effective_from, p.effective_until AS effective_until,
                   d.code AS dong_code, d.name AS dong_name, d.lon AS lon, d.lat AS lat
            """
        )
        return [dict(r) for r in rows]


def fetch_url(url: str) -> str:
    print(f"  downloading {url} ...")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def render_template(template: str, replacements: dict[str, str]) -> str:
    for marker in TEMPLATE_MARKERS:
        if marker not in template:
            raise RuntimeError(f"missing template marker: {marker}")

    rendered = template
    for marker in TEMPLATE_MARKERS:
        rendered = rendered.replace(marker, _template_replacement(marker, replacements))
    return rendered


def build_3d_standalone(
    viz_dir: Path | None = None,
    *,
    include_detail: bool = True,
    pack_dir: Path | None = None,
) -> Path:
    """``viz_dir`` 의 JSON 4종으로 단일 HTML 을 만든다.

    ``include_detail=False`` 면 대상자별 기억·일정을 빼서 파일을 더 줄인다(지도·궤적만).
    ``pack_dir`` 를 주면 같은 팩을 폴더로도 써서 서버(``/viz/packs/<id>/``)가 바로 내보낼 수 있다.
    """
    viz_dir = Path(viz_dir) if viz_dir is not None else _resolve_viz_dir()
    payload, source = _load_json_payload(viz_dir)
    viz_meta = build_viz_meta(
        payload["__AGENTS__"],
        payload["__TIMELINE__"],
        payload["__MEMORIES__"],
        payload["__EVENTS__"],
        fetch_policy_dongs_safe(),
    )
    manifest, files = build_pack(
        payload["__AGENTS__"],
        payload["__TIMELINE__"],
        payload["__MEMORIES__"],
        payload["__EVENTS__"],
        viz_meta=viz_meta,
        pack_id=viz_dir.name,
        source=source,
    )
    if pack_dir is not None:
        write_pack(Path(pack_dir), manifest, files)
        print(f"  -> pack: {pack_dir}")

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = render_template(
        template,
        {
            "styles": _load_styles(),
            "data": _build_pack_script(manifest, files, include_detail=include_detail),
            "scripts": _load_runtime_assets(),
        },
    )

    out_path = viz_dir / "sim_standalone.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  -> {out_path}: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    return out_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="3D 시각화 단일 HTML 을 만든다")
    parser.add_argument("--viz-dir", type=Path, help="JSON 4종이 있는 폴더 (기본: VIZ_OUT_DIR)")
    parser.add_argument("--no-detail", action="store_true", help="대상자별 기억·일정을 넣지 않는다")
    parser.add_argument("--pack-dir", type=Path, help="같은 팩을 폴더로도 쓴다 (서버 서빙용)")
    args = parser.parse_args(argv)
    build_3d_standalone(args.viz_dir, include_detail=not args.no_detail, pack_dir=args.pack_dir)


def _resolve_viz_dir() -> Path:
    return Path(
        os.environ.get(
            "VIZ_OUT_DIR",
            str(Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization"),
        )
    )


def _template_replacement(marker: str, replacements: dict[str, str]) -> str:
    logical_key = TEMPLATE_KEYS[marker]
    if logical_key in replacements:
        return replacements[logical_key]
    if marker in replacements:
        return replacements[marker]
    raise RuntimeError(f"missing template replacement: {marker}")


def _load_json_payload(viz_dir: Path) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    payload: dict[str, object] = {}
    source: dict[str, dict[str, object]] = {}
    for key, filename in JSON_INPUTS.items():
        path = viz_dir / filename
        raw = path.read_bytes()
        print(f"  loading {filename} ({len(raw) / 1024:.0f} KB)")
        source[filename] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        payload[key] = json.loads(raw.decode("utf-8"))
    return payload, source


def _load_styles() -> str:
    styles = [
        style_tag(fetch_url(MAPLIBRE_CSS_URL)),
        style_tag(_fetch_optional_google_fonts()),
        style_tag(_read_static("styles.css")),
    ]
    return "\n".join(styles)


def _fetch_optional_google_fonts() -> str:
    try:
        return _neutralize_google_font_urls(fetch_url(GOOGLE_FONTS_URL))
    except Exception as exc:
        print(f"  WARNING: Google Fonts download failed ({exc}), falling back to system fonts")
        return ""


def _neutralize_google_font_urls(css: str) -> str:
    return re.sub(
        r"url\((['\"]?)https://fonts\.gstatic\.com/[^)'\"\s]+(['\"]?)\)",
        "url(about:blank)",
        css,
    )


def _load_runtime_assets() -> str:
    scripts = [
        script_tag(fetch_url(MAPLIBRE_JS_URL)),
        script_tag(fetch_url(DECK_JS_URL)),
        script_tag(fetch_url(CHART_JS_URL)),
    ]
    scripts.extend(script_tag(_read_static(filename)) for filename in STATIC_JS_FILES)
    return "\n".join(scripts)


def _build_pack_script(manifest: dict, files: dict[str, bytes], *, include_detail: bool = True) -> str:
    """팩을 HTML 에 싣는다.

    대상자 표와 화면 메타는 예전과 같은 전역(``window.__AGENTS__``, ``window.__VIZ_META__``)으로,
    타임라인·기억 묶음은 실행되지 않는 ``application/octet-stream`` 태그에 base64 로 둔다.
    """
    manifest = json.loads(json.dumps(manifest))
    if not include_detail:
        manifest["detail"] = {**manifest["detail"], "shards": [], "omitted": True}
    roster = json.loads(gzip.decompress(files[manifest["agents"]["file"]]).decode("utf-8"))

    lines = ['<script id="__SIM_DATA__">']
    lines.append(f"window.__AGENTS__ = {_safe_json(roster)};")
    lines.append(f"window.__VIZ_META__ = {_safe_json(manifest.get('meta') or {})};")
    lines.append(f"window.__SIM_PACK__ = {_safe_json(manifest)};")
    lines.append("</script>")

    embedded = [chunk["file"] for chunk in manifest["chunks"]]
    embedded += [shard["file"] for shard in manifest["detail"]["shards"]]
    for name in embedded:
        encoded = base64.b64encode(files[name]).decode("ascii")
        lines.append(f'<script type="application/octet-stream" data-pack-file="{name}">{encoded}</script>')
    return "\n".join(lines)


def _read_static(filename: str) -> str:
    return (STATIC_DIR / filename).read_text(encoding="utf-8")


def _safe_json(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


if __name__ == "__main__":
    main()
