"""Build the 3D simulation visualization as a single offline-friendly HTML file."""
from __future__ import annotations

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
    "data_model.js",
    "map_scene.js",
    "layers.js",
    "detail.js",
    "hud.js",
    "app.js",
)


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


def build_3d_standalone(viz_dir: Path | None = None) -> Path:
    viz_dir = Path(viz_dir) if viz_dir is not None else _resolve_viz_dir()
    payload = _load_json_payload(viz_dir)
    viz_meta = build_viz_meta(
        payload["__AGENTS__"],
        payload["__TIMELINE__"],
        payload["__MEMORIES__"],
        payload["__EVENTS__"],
    )

    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    html = render_template(
        template,
        {
            "styles": _load_styles(),
            "data": _build_data_script(payload, viz_meta),
            "scripts": _load_runtime_assets(),
        },
    )

    out_path = viz_dir / "sim_standalone.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"\n  -> {out_path}: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    return out_path


def main() -> None:
    build_3d_standalone()


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


def _load_json_payload(viz_dir: Path) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, filename in JSON_INPUTS.items():
        path = viz_dir / filename
        size_kb = path.stat().st_size / 1024
        print(f"  loading {filename} ({size_kb:.0f} KB)")
        payload[key] = json.loads(path.read_text(encoding="utf-8"))
    return payload


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


def _build_data_script(payload: dict[str, object], viz_meta: dict[str, object]) -> str:
    lines = ['<script id="__SIM_DATA__">']
    for key in JSON_INPUTS:
        lines.append(f"window.{key} = {_safe_json(payload[key])};")
    lines.append(f"window.__VIZ_META__ = {_safe_json(viz_meta)};")
    lines.append("</script>")
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
