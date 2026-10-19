from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HTML = ROOT / "output" / "sim" / "visualization" / "sim_standalone.html"


REQUIRED_STRINGS = [
    "window.__AGENTS__",
    "window.__TIMELINE__",
    "window.__MEMORIES__",
    "window.__EVENTS__",
    "window.__VIZ_META__",
    "maplibregl.Map",
    "deck.MapboxOverlay",
    "deck.ScatterplotLayer",
    "deck.Tile3DLayer",
    "deck.PathLayer",
    "deck.ColumnLayer",
    "deck.ArcLayer",
    "deck.HeatmapLayer",
    "deck.TripsLayer",
    "fill-extrusion-height",
    "3d-buildings",
    "tiles.openfreemap.org",
    "name:ko",
    "story-mode-btn",
    "explore-mode-btn",
    "base-style-btn",
    "base-google-btn",
    "google-key-input",
    "toggle-heatmap-btn",
    "toggle-policy-btn",
    "toggle-od-btn",
    "policy_zones",
    "detail-card",
    "agent-search",
    "selectAgent",
    "followAgent",
    "chart-cat",
    "chart-dist",
]


def main() -> None:
    if not HTML.exists():
        raise SystemExit(f"standalone visualization does not exist: {HTML}")

    html = HTML.read_text(encoding="utf-8")
    missing = [needle for needle in REQUIRED_STRINGS if needle not in html]
    if missing:
        raise SystemExit("missing required visualization strings: " + ", ".join(missing))

    data_blocks = len(re.findall(r"window\.__[A-Z_]+__\s*=", html))
    if data_blocks < 5:
        raise SystemExit(f"expected at least 5 embedded data blocks, found {data_blocks}")

    size_mb = HTML.stat().st_size / 1024 / 1024
    if size_mb < 50:
        raise SystemExit(f"standalone visualization is unexpectedly small: {size_mb:.1f} MB")

    print(f"3D visualization smoke check passed: {HTML} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
