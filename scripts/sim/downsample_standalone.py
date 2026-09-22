"""이미 생성된 sim_standalone(.zip/.html)의 내장 데이터를 DB 없이 다운샘플링.

sim_standalone.html 은 4개 데이터를 인라인으로 품고 있다:
  window.__AGENTS__   = [...]   # agent dict list (home_dong 포함)
  window.__TIMELINE__ = [...]   # frame list, 각 frame.agents = [{id,...}]
  window.__MEMORIES__ = {...}   # aid -> {...}
  window.__EVENTS__   = {...}   # aid -> [...]

행정동(home_dong)당 PER_DONG명만 남기고 네 데이터 모두를 그 agent 집합으로
필터링해 훨씬 작은 HTML 을 다시 쓴다. Neo4j 불필요.

사용:
  python scripts/sim/downsample_standalone.py --per-dong 5
  python scripts/sim/downsample_standalone.py --per-dong 5 --in output/sim/visualization/sim_standalone.zip
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

VIZ_DIR = Path(__file__).resolve().parents[2] / "output" / "sim" / "visualization"

PREFIXES = {
    "agents": "window.__AGENTS__ = ",
    "timeline": "window.__TIMELINE__ = ",
    "memories": "window.__MEMORIES__ = ",
    "events": "window.__EVENTS__ = ",
}


# ---------------------------------------------------------------------------
# HTML JS 패치용 — 함수 본문 통째 교체 (원본 함수는 column-0 '}' 로 닫힘)
# ---------------------------------------------------------------------------
def _patch_block(html: str, marker: str, new_text: str, label: str) -> str:
    start = html.find(marker)
    if start < 0:
        print(f"  [{label}] '{marker}' 못 찾음 — 스킵", file=sys.stderr)
        return html
    end = html.find("\n}", start)
    if end < 0:
        print(f"  [{label}] 함수 끝 못 찾음 — 스킵", file=sys.stderr)
        return html
    end += len("\n}")
    print(f"  [{label}] 함수 본문 교체")
    return html[:start] + new_text + html[end:]


# 히트맵을 __HEATFULL__(전체 격자 집계) 우선, 없으면 기존 f.agents 폴백.
_NEW_UPDATE_HEATMAP = """function updateHeatmap() {
  if (!heatLayer) return;
  const mode = document.getElementById('heat-mode').value;
  const cells = (window.__HEATFULL__ && window.__HEATFULL__[currentFrame]) || null;
  let pts;
  if (cells) {
    if (mode === 'density') pts = cells.map(c => [c[0], c[1], Math.min(1.0, c[2] * 0.15)]);
    else pts = cells.filter(c => c[3] > 0).map(c => [c[0], c[1], Math.min(1.0, c[3] / 100000)]);
  } else {
    const f = TIMELINE[currentFrame];
    if (!f) { heatLayer.setLatLngs([]); return; }
    if (mode === 'density') pts = f.agents.map(a => [a.lat, a.lon, 0.6]);
    else pts = f.agents.filter(a => (a.spent||0) > 0).map(a => [a.lat, a.lon, Math.min(1.0, (a.spent||0)/25000)]);
  }
  heatLayer.setLatLngs(pts);
}"""

# 프레임 간 위치를 requestAnimationFrame 으로 선형보간 — 순간이동 제거.
_NEW_PLAY = """function play() {
  if (isPlaying) {
    clearInterval(playerHandle); clearTimeout(playerHandle);
    if (window.__raf) cancelAnimationFrame(window.__raf);
    isPlaying = false;
    document.getElementById('play-btn').textContent = '\\u25B6'; return;
  }
  isPlaying = true;
  document.getElementById('play-btn').textContent = '\\u23F8';
  const speed = parseInt(document.getElementById('speed').value);
  const dur = Math.max(150, speed * 0.85);
  const gap = Math.max(0, speed - dur);
  function stepFrame() {
    if (!isPlaying) return;
    let next = currentFrame + 1; if (next >= TIMELINE.length) next = 0;
    const fNext = TIMELINE[next];
    const endPos = {};
    if (fNext) fNext.agents.forEach(a => { endPos[a.id] = [a.lat, a.lon]; });
    const movers = [];
    AGENTS.forEach(ag => {
      const m = markers[ag.id]; if (!m) return;
      const ll = m.getLatLng();
      const e = endPos[ag.id] || [ag.home_lat, ag.home_lon];
      if (e[0] == null || e[1] == null) return;
      if (Math.abs(ll.lat - e[0]) > 1e-6 || Math.abs(ll.lng - e[1]) > 1e-6)
        movers.push([m, ll.lat, ll.lng, e[0], e[1]]);
    });
    const t0 = performance.now();
    function anim(now) {
      if (!isPlaying) return;
      const k = Math.min(1, (now - t0) / dur);
      for (const mv of movers)
        mv[0].setLatLng([mv[1] + (mv[3]-mv[1])*k, mv[2] + (mv[4]-mv[2])*k]);
      if (k < 1) { window.__raf = requestAnimationFrame(anim); }
      else { setFrame(next); if (selected) showDetail(selected); playerHandle = setTimeout(stepFrame, gap); }
    }
    window.__raf = requestAnimationFrame(anim);
  }
  stepFrame();
}"""


def _read_html(in_path: Path) -> str:
    if in_path.suffix == ".zip":
        with zipfile.ZipFile(in_path) as z:
            name = next(n for n in z.namelist() if n.endswith(".html"))
            print(f"  [read] {in_path.name} :: {name}")
            return z.read(name).decode("utf-8")
    return in_path.read_text(encoding="utf-8")


def _parse_embedded(line: str, prefix: str):
    """'window.__X__ = <JSON>;' 한 줄에서 JSON 부분만 떼어 파싱."""
    body = line[len(prefix):]
    body = body.rstrip()
    if body.endswith(";"):
        body = body[:-1]
    return json.loads(body)


def _patch_block(html: str, start_marker: str, new_code: str, label: str) -> str:
    """start_marker로 시작하는 JS 함수 블록을 new_code로 교체.

    함수의 닫는 괄호는 컬럼0의 '\\n}' (최상위 close)로 식별. 못 찾으면 원본 유지.
    """
    start = html.find(start_marker)
    if start < 0:
        print(f"  [{label}] '{start_marker}' 못 찾음 — 스킵", file=sys.stderr)
        return html
    end = html.find("\n}", start)
    if end < 0:
        print(f"  [{label}] 닫는 괄호 못 찾음 — 스킵", file=sys.stderr)
        return html
    end += len("\n}")
    print(f"  [{label}] 함수 교체")
    return html[:start] + new_code + html[end:]


# updateHeatmap: __HEATFULL__(전체 격자 집계) 우선, 없으면 기존 표본 기반 폴백
_NEW_UPDATE_HEATMAP = """function updateHeatmap() {
  if (!heatLayer) return;
  const mode = document.getElementById('heat-mode').value;
  const HF = window.__HEATFULL__;
  let pts;
  if (HF && HF[currentFrame]) {
    const cells = HF[currentFrame];           // [lat, lon, count, sumspent]
    if (mode === 'density') {
      pts = cells.map(c => [c[0], c[1], Math.min(1.0, c[2] * 0.18)]);
    } else {
      pts = cells.filter(c => c[3] > 0).map(c => [c[0], c[1], Math.min(1.0, c[3] / 120000)]);
    }
  } else {
    const f = TIMELINE[currentFrame];
    if (!f) return;
    if (mode === 'density') {
      pts = f.agents.map(a => [a.lat, a.lon, 0.6]);
    } else {
      pts = f.agents.filter(a => (a.spent || 0) > 0)
                    .map(a => [a.lat, a.lon, Math.min(1.0, (a.spent || 0) / 25000)]);
    }
  }
  heatLayer.setLatLngs(pts);
}"""


# play: 프레임 간 위치를 requestAnimationFrame으로 보간 — 순간이동 대신 부드럽게 이동.
# 움직이는 마커(movers)만 보간해 부하 최소화. 색/필터/히트맵은 도착 시 setFrame이 처리.
_NEW_PLAY = """function play() {
  if (isPlaying) {
    clearTimeout(playerHandle); clearInterval(playerHandle);
    if (window.__raf) cancelAnimationFrame(window.__raf);
    isPlaying = false;
    document.getElementById('play-btn').textContent = '▶'; return;
  }
  isPlaying = true;
  document.getElementById('play-btn').textContent = '⏸';
  function stepFrame() {
    if (!isPlaying) return;
    const speed = parseInt(document.getElementById('speed').value);
    const dur = Math.max(150, speed * 0.85);
    let next = currentFrame + 1;
    if (next >= TIMELINE.length) next = 0;
    const fNext = TIMELINE[next];
    const endPos = {};
    if (fNext) fNext.agents.forEach(a => { endPos[a.id] = [a.lat, a.lon]; });
    const movers = [];
    AGENTS.forEach(ag => {
      const m = markers[ag.id]; if (!m) return;
      const ll = m.getLatLng();
      const e = endPos[ag.id] || [ag.home_lat, ag.home_lon];
      if (e[0] == null || e[1] == null) return;
      if (Math.abs(ll.lat - e[0]) > 1e-6 || Math.abs(ll.lng - e[1]) > 1e-6)
        movers.push([m, ll.lat, ll.lng, e[0], e[1]]);
    });
    const t0 = performance.now();
    function anim(now) {
      if (!isPlaying) return;
      const k = Math.min(1, (now - t0) / dur);
      for (const mv of movers)
        mv[0].setLatLng([mv[1] + (mv[3] - mv[1]) * k, mv[2] + (mv[4] - mv[2]) * k]);
      if (k < 1) { window.__raf = requestAnimationFrame(anim); }
      else {
        setFrame(next);
        if (selected) showDetail(selected);
        const gap = Math.max(0, parseInt(document.getElementById('speed').value) - dur);
        playerHandle = setTimeout(stepFrame, gap);
      }
    }
    window.__raf = requestAnimationFrame(anim);
  }
  stepFrame();
}"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dong", type=int, default=5, help="행정동당 남길 agent 수")
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=VIZ_DIR / "sim_standalone.zip",
                    help="입력 sim_standalone (.zip 또는 .html)")
    ap.add_argument("--out", type=Path,
                    default=VIZ_DIR / "sim_standalone_small.html",
                    help="출력 HTML 경로")
    ap.add_argument("--zip", action="store_true", help="출력 HTML 을 .zip 으로도 압축")
    ap.add_argument("--drop-detail", action="store_true",
                    help="MEMORIES+EVENTS(클릭 전용, ~65MB) 를 비워 초기 로드 렉 분리 검증")
    ap.add_argument("--canvas", action="store_true",
                    help="마커 렌더러를 Canvas로 전환(preferCanvas:true) — 재생 렌더 렉 분리 검증")
    ap.add_argument("--seoul-bounds", action="store_true",
                    help="지도를 서울 영역으로 제한(maxBounds+minZoom) — 줌아웃 시 타일 요청 억제")
    ap.add_argument("--smooth", action="store_true",
                    help="재생 시 노드 위치를 프레임 간 보간 — 순간이동 대신 부드럽게 이동")
    ap.add_argument("--heat-full", action="store_true",
                    help="히트맵을 다운샘플 전 전체 timeline(원본 5,000명) 기반으로 생성")
    args = ap.parse_args()

    html = _read_html(args.in_path)
    lines = html.split("\n")

    # 1) 4개 데이터 라인 위치 찾기 + 파싱
    data = {}
    idx = {}
    for i, line in enumerate(lines):
        for key, prefix in PREFIXES.items():
            if line.startswith(prefix):
                data[key] = _parse_embedded(line, prefix)
                idx[key] = i
    missing = [k for k in PREFIXES if k not in data]
    if missing:
        print(f"[ERROR] 내장 데이터 라인을 못 찾음: {missing}", file=sys.stderr)
        return 1

    agents = data["agents"]
    print(f"  [orig] agents={len(agents)} "
          f"frames={len(data['timeline'])} "
          f"memories={len(data['memories'])} events={len(data['events'])}")

    # 2) 행정동(home_dong)당 per_dong명 선택 — id 정렬로 결정적
    by_dong: dict[str, list] = defaultdict(list)
    for a in agents:
        by_dong[a.get("home_dong") or "?"].append(a)
    keep_ids: set[str] = set()
    kept_agents = []
    for dong, members in by_dong.items():
        members.sort(key=lambda a: str(a.get("id")))
        for a in members[: args.per_dong]:
            keep_ids.add(a["id"])
            kept_agents.append(a)
    print(f"  [keep] {len(keep_ids)} agents across {len(by_dong)} dongs "
          f"({args.per_dong}/dong)")

    # 3) 네 데이터 모두 keep_ids 로 필터
    new_agents = [a for a in agents if a["id"] in keep_ids]

    new_timeline = []
    for fr in data["timeline"]:
        fr2 = dict(fr)
        fr2["agents"] = [g for g in fr.get("agents", []) if g.get("id") in keep_ids]
        new_timeline.append(fr2)

    if args.drop_detail:
        new_memories = {}
        new_events = {}
        print("  [drop-detail] MEMORIES/EVENTS 비움 (클릭 상세 패널 비활성, 로드 렉 분리용)")
    else:
        new_memories = {aid: v for aid, v in data["memories"].items() if aid in keep_ids}
        new_events = {aid: v for aid, v in data["events"].items() if aid in keep_ids}

    # 4) 라인 교체 (compact JSON)
    repl = {
        "agents": new_agents,
        "timeline": new_timeline,
        "memories": new_memories,
        "events": new_events,
    }
    for key, prefix in PREFIXES.items():
        lines[idx[key]] = prefix + json.dumps(repl[key], ensure_ascii=False) + ";"

    # 4b) (선택) 전체(다운샘플 전) timeline 기반 히트맵 집계 — 100m 격자(소수3자리)로 누적
    if args.heat_full:
        heatfull = []
        for fr in data["timeline"]:           # 원본 5,000명 timeline
            cells: dict[tuple, list[int]] = {}
            for a in fr.get("agents", []):
                lat, lon = a.get("lat"), a.get("lon")
                if lat is None or lon is None:
                    continue
                key = (round(lat, 3), round(lon, 3))
                c = cells.setdefault(key, [0, 0])
                c[0] += 1                       # 인원수(밀집도)
                c[1] += int(a.get("spent") or 0)  # 소비합
            heatfull.append([[k[0], k[1], v[0], v[1]] for k, v in cells.items()])
        n_cells = sum(len(f) for f in heatfull)
        heat_line = "window.__HEATFULL__ = " + json.dumps(heatfull, ensure_ascii=False) + ";"
        lines.insert(idx["events"] + 1, heat_line)
        print(f"  [heat-full] 전체 timeline 격자 집계: {len(heatfull)} 프레임, {n_cells:,} 셀")

    out_html = "\n".join(lines)

    # 5) (선택) 렌더러/지도 범위 HTML 변형 — 렉 원인 분리 검증용
    if args.canvas:
        if "preferCanvas: false" in out_html:
            out_html = out_html.replace("preferCanvas: false", "preferCanvas: true")
            print("  [canvas] preferCanvas:true 로 전환 (Canvas 렌더러)")
        else:
            print("  [canvas] 'preferCanvas: false' 패턴 못 찾음 — 스킵", file=sys.stderr)
    if args.seoul_bounds:
        anchor = ".setView([37.530, 127.020], 13);"
        inject = (".setView([37.530, 127.020], 13);\n"
                  "  map.setMaxBounds([[37.41, 126.76],[37.70, 127.18]]);\n"
                  "  map.setMinZoom(10);")
        if anchor in out_html:
            out_html = out_html.replace(anchor, inject, 1)
            print("  [seoul-bounds] maxBounds=서울, minZoom=10 적용")
        else:
            print("  [seoul-bounds] setView 앵커 못 찾음 — 스킵", file=sys.stderr)

    if args.heat_full:
        out_html = _patch_block(
            out_html, "function updateHeatmap() {", _NEW_UPDATE_HEATMAP,
            "heat-full(updateHeatmap)")

    if args.smooth:
        out_html = _patch_block(
            out_html, "function play() {", _NEW_PLAY, "smooth(play)")

    args.out.write_text(out_html, encoding="utf-8")
    mb = args.out.stat().st_size / 1024 / 1024
    print(f"  → {args.out}  ({mb:.1f} MB)")

    if args.zip:
        zip_path = args.out.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(args.out, arcname=args.out.name)
        zmb = zip_path.stat().st_size / 1024 / 1024
        print(f"  → {zip_path}  ({zmb:.1f} MB, zipped)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
