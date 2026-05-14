"""인터랙티브 시각화를 단일 HTML 파일로 빌드 — 팀원 공유용 (offline-friendly).

- JSON 4개 임베디드
- Leaflet.css + Leaflet.js 임베디드 (CDN 차단 환경 대비)
- 타일 서버: OSM 기본 + CARTO dark 폴백
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
VIZ_DIR = Path(os.environ.get(
    "VIZ_OUT_DIR",
    str(Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization")
))
LEAFLET_CSS_URL = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
LEAFLET_JS_URL = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"


def fetch_url(url: str) -> str:
    print(f"  downloading {url} ...")
    with urllib.request.urlopen(url) as r:
        return r.read().decode("utf-8")


def main():
    index_html = (VIZ_DIR / "index.html").read_text(encoding="utf-8")

    leaflet_css = fetch_url(LEAFLET_CSS_URL)
    leaflet_js = fetch_url(LEAFLET_JS_URL)

    payload = {}
    for key, fname in [("agents", "agents.json"),
                       ("timeline", "timeline.json"),
                       ("memories", "memories.json"),
                       ("events", "events.json")]:
        p = VIZ_DIR / fname
        size_kb = p.stat().st_size / 1024
        print(f"  loading {fname} ({size_kb:.0f} KB)")
        payload[key] = json.loads(p.read_text(encoding="utf-8"))

    embedded_script = (
        '<script id="__SIM_DATA__">\n'
        f'window.__AGENTS__ = {json.dumps(payload["agents"], ensure_ascii=False)};\n'
        f'window.__TIMELINE__ = {json.dumps(payload["timeline"], ensure_ascii=False)};\n'
        f'window.__MEMORIES__ = {json.dumps(payload["memories"], ensure_ascii=False)};\n'
        f'window.__EVENTS__ = {json.dumps(payload["events"], ensure_ascii=False)};\n'
        '</script>\n'
    )

    # loadData를 임베디드 데이터로 교체
    new_load = """async function loadData() {
  try {
    AGENTS = window.__AGENTS__; TIMELINE = window.__TIMELINE__;
    MEMORIES = window.__MEMORIES__; EVENTS = window.__EVENTS__;
    if (!AGENTS || !TIMELINE) throw new Error('embedded data missing');
    AGENTS.forEach(ag => agentById[ag.id] = ag);
    document.getElementById('total-agents').textContent = AGENTS.length.toLocaleString();
    let totMem = 0;
    for (const aid in MEMORIES) totMem += (MEMORIES[aid].visited || []).length;
    document.getElementById('total-mem').textContent = totMem.toLocaleString();
    initMap();
    buildMarkers();
    setFrame(0);
    initEvents();
  } catch (e) {
    document.body.innerHTML = '<div style="color:#fff;background:#0a0a14;padding:40px;font-family:monospace;">' +
      '<h2 style="color:#e76f51">ERROR</h2><pre>' + e.message + '\\n\\n' + e.stack + '</pre></div>';
  }
}"""

    # loadData() 본체만 교체 — 다음 함수(function initMap()) 시작 직전까지만 잘라낸다.
    # 그래야 initMap·buildMarkers·setFrame·showDetail·play·initEvents 모두 보존됨.
    start = index_html.find("async function loadData()")
    end = index_html.find("function initMap()", start)
    if start < 0 or end < 0:
        raise RuntimeError("Cannot locate loadData/initMap boundary")
    standalone = (
        index_html[:start] + new_load + "\n\n" + index_html[end:]
    )
    # loadData().catch(...) 호출은 단순 호출로 교체 (마지막 줄)
    catch_start = standalone.find("loadData().catch")
    if catch_start >= 0:
        catch_end = standalone.find("});", catch_start) + 3
        standalone = standalone[:catch_start] + "loadData();" + standalone[catch_end:]

    # CDN <link>/<script> → 임베디드 + OSM 폴백 타일
    head_close = standalone.find("</head>")
    embed_block = (
        f"<style>\n{leaflet_css}\n</style>\n"
        f"<script>\n{leaflet_js}\n</script>\n"
        f"{embedded_script}"
    )
    standalone = (
        standalone.replace(
            '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>', "")
        .replace('<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>', "")
    )
    head_close = standalone.find("</head>")
    standalone = standalone[:head_close] + embed_block + standalone[head_close:]

    # 타일 서버를 OSM으로 변경 + 폴백 fallback (CARTO 차단 시 OSM 표준)
    standalone = standalone.replace(
        "L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {\n    attribution: '© OSM · CARTO', maxZoom: 19,\n  }).addTo(map);",
        """// 1차 시도: CARTO dark / 2차 폴백: OSM 표준
  const tileCarto = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    { attribution: '© OSM · CARTO', maxZoom: 19, subdomains: 'abcd' });
  const tileOSM = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    { attribution: '© OpenStreetMap', maxZoom: 19 });
  let cartoLoadFail = 0;
  tileCarto.on('tileerror', () => {
    cartoLoadFail++;
    if (cartoLoadFail >= 3) {
      console.warn('CARTO 차단 추정 → OSM 폴백');
      map.removeLayer(tileCarto); tileOSM.addTo(map);
    }
  });
  tileCarto.addTo(map);"""
    )

    out_path = VIZ_DIR / "sim_standalone.html"
    out_path.write_text(standalone, encoding="utf-8")
    print(f"\n  → {out_path}: {out_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  Leaflet 임베디드 ✓ · OSM 폴백 타일 ✓ · 에러 표시 ✓")


if __name__ == "__main__":
    main()
