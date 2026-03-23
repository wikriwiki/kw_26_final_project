"""
애니메이션 지도 — Insightful Interactive Map

에이전트의 일간 좌표 + 소비 행동 데이터를 인터랙티브 맵으로 시각화.
- 에이전트 클릭 → 프로필/행동 팝업
- 소비 밀도 히트맵 레이어
- 이벤트 배너 (카테고리 + 핫스팟)
- 실시간 통계 대시보드 (소비총액, TOP 업종, 만족도)
- 색상 모드 전환 (세그먼트/라이프스타일/소비수준)
"""
import json
from pathlib import Path
from config import OUTPUT_DIR


def generate_animation_html(
    daily_positions: list,     # [(day_idx, [{agent_id, segment, lat, lng, lifestyle, ...}])]
    weekly_stats: list = None, # [{week, total_spending, events, policy, ...}]
    output_name: str = "simulation_animation.html",
) -> Path:
    """인사이트 중심 에이전트 이동 애니메이션 HTML 생성."""
    # 프레임 데이터 정리 — 에이전트 행동 정보 포함
    frames = []
    for day_idx, agents_snap in daily_positions:
        frame = {"day": day_idx, "week": day_idx // 7, "agents": []}
        for a in agents_snap:
            agent_data = {
                "id": a["agent_id"],
                "seg": a.get("segment", "commuter"),
                "lat": round(a.get("current_lat", a.get("lat", 37.5665)), 6),
                "lng": round(a.get("current_lng", a.get("lng", 126.978)), 6),
                # 리치 데이터 (있으면)
                "ls": a.get("lifestyle", ""),
                "age": a.get("age_group", ""),
                "ind": a.get("industry", ""),
                "amt": a.get("amount", 0),
                "sat": round(a.get("satisfaction", 0), 2),
                "ts": a.get("time_slot", 0),
                "trg": a.get("triggered_by", [])[:2],
            }
            frame["agents"].append(agent_data)
        frames.append(frame)

    # 주간 이벤트 매핑
    week_events = {}
    week_stats_map = {}
    if weekly_stats:
        for ws in weekly_stats:
            w = ws["week"]
            parts = []
            if ws.get("policy") and ws["policy"] != "없음":
                parts.append(f"📋 {ws['policy']}")
            for e in ws.get("events", []):
                if isinstance(e, dict):
                    cat = e.get("category", "")
                    hl = e.get("headline", "")
                    parts.append(f"[{cat}] {hl}")
                else:
                    parts.append(str(e))
            if parts:
                week_events[w] = parts
            week_stats_map[w] = {
                "spending": ws.get("total_spending", 0),
                "actions": ws.get("actions_count", 0),
                "avg": ws.get("avg_spending_per_agent", 0),
            }

    frames_json = json.dumps(frames, ensure_ascii=False)
    events_json = json.dumps(week_events, ensure_ascii=False)
    stats_json = json.dumps(week_stats_map, ensure_ascii=False)

    html = _build_html(frames_json, events_json, stats_json, len(frames))

    out_path = OUTPUT_DIR / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[animation] Saved to {out_path.name} ({len(frames)} frames)")
    return out_path


def _build_html(frames_json, events_json, stats_json, total_frames):
    """Leaflet + JS 인터랙티브 애니메이션 HTML 생성."""
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8"/>
<title>Seoul Consumer Simulation — Interactive Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0a1a; }}
  #map {{ width: 100vw; height: 100vh; }}

  #controls {{
    position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(10,10,30,0.92); padding: 14px 24px;
    border-radius: 14px; display: flex; gap: 14px; align-items: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.5); backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
  }}
  #controls button {{
    background: #4cc9f0; color: #0a0a1a; border: none; padding: 8px 18px;
    border-radius: 8px; cursor: pointer; font-weight: 600; font-size: 13px;
    transition: all 0.2s;
  }}
  #controls button:hover {{ background: #72efdd; }}
  #controls button.active {{ background: #e76f51; color: white; }}
  #controls label {{ color: #aaa; font-size: 12px; }}
  #controls input[type=range] {{ accent-color: #4cc9f0; }}
  #controls span {{ color: white; font-size: 13px; font-weight: 500; min-width: 60px; }}
  #controls select {{
    background: #1a1a2e; color: white; border: 1px solid #4cc9f0;
    padding: 4px 8px; border-radius: 6px; font-size: 12px;
  }}

  #info {{
    position: fixed; top: 20px; left: 20px; z-index: 1000;
    background: rgba(10,10,30,0.92); padding: 16px 20px; border-radius: 12px;
    color: white; font-size: 13px; line-height: 1.7; min-width: 240px;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
  }}
  #info h3 {{ color: #4cc9f0; margin-bottom: 6px; font-size: 15px; }}
  .seg-dot {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }}
  .stat-row {{ display: flex; justify-content: space-between; margin: 2px 0; }}
  .stat-label {{ color: #888; }}
  .stat-value {{ color: #72efdd; font-weight: 600; }}
  .stat-divider {{ border-top: 1px solid rgba(255,255,255,0.1); margin: 8px 0; }}

  #dashboard {{
    position: fixed; top: 20px; right: 20px; z-index: 1000;
    background: rgba(10,10,30,0.92); padding: 16px 20px; border-radius: 12px;
    color: white; font-size: 12px; line-height: 1.6; min-width: 260px;
    max-height: 80vh; overflow-y: auto;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(10px);
  }}
  #dashboard h3 {{ color: #e76f51; margin-bottom: 8px; font-size: 14px; }}
  .industry-bar {{
    display: flex; align-items: center; gap: 6px; margin: 3px 0;
  }}
  .industry-bar .bar {{
    height: 12px; border-radius: 3px; min-width: 4px;
    transition: width 0.5s ease;
  }}
  .industry-bar .name {{ min-width: 52px; color: #aaa; font-size: 11px; }}
  .industry-bar .cnt {{ color: #72efdd; font-size: 11px; min-width: 24px; text-align: right; }}

  #event-banner {{
    position: fixed; bottom: 80px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(231,111,81,0.92); padding: 10px 20px;
    border-radius: 10px; color: white; font-size: 12px; max-width: 500px;
    display: none; text-align: center;
    box-shadow: 0 4px 16px rgba(231,111,81,0.3);
    animation: slideUp 0.4s ease;
  }}
  @keyframes slideUp {{ from {{ transform: translateX(-50%) translateY(20px); opacity: 0; }} to {{ transform: translateX(-50%); opacity: 1; }} }}

  #agent-popup {{
    position: fixed; z-index: 1100; display: none;
    background: rgba(15,15,35,0.96); padding: 14px 18px; border-radius: 10px;
    color: white; font-size: 12px; line-height: 1.7; min-width: 240px;
    border: 1px solid #4cc9f0; box-shadow: 0 8px 32px rgba(0,0,0,0.6);
  }}
  #agent-popup h4 {{ color: #4cc9f0; margin-bottom: 6px; }}
  #agent-popup .close {{ position: absolute; top: 8px; right: 12px; cursor: pointer;
    color: #888; font-size: 16px; }}
  .tag {{ display: inline-block; background: rgba(76,201,240,0.2); color: #4cc9f0;
    padding: 1px 6px; border-radius: 4px; font-size: 10px; margin: 1px 2px; }}
  .sat-bar {{ height: 6px; border-radius: 3px; background: #1a1a2e; margin: 4px 0; }}
  .sat-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
</style>
</head>
<body>
<div id="map"></div>

<div id="info">
  <h3>🏙️ Seoul Simulation</h3>
  <div id="day-display">Day 0 / Week 0</div>
  <div class="stat-divider"></div>
  <div class="stat-row"><span class="stat-label">총 소비</span><span class="stat-value" id="total-spend">-</span></div>
  <div class="stat-row"><span class="stat-label">평균 만족도</span><span class="stat-value" id="avg-sat">-</span></div>
  <div class="stat-row"><span class="stat-label">활동 에이전트</span><span class="stat-value" id="active-cnt">-</span></div>
  <div class="stat-divider"></div>
  <div><span class="seg-dot" style="background:#3388ff"></span>Commuter <span id="cnt-commuter">0</span></div>
  <div><span class="seg-dot" style="background:#33cc33"></span>Weekend <span id="cnt-weekend_visitor">0</span></div>
  <div><span class="seg-dot" style="background:#ff8800"></span>Resident <span id="cnt-resident">0</span></div>
  <div><span class="seg-dot" style="background:#cc33ff"></span>Evening <span id="cnt-evening_visitor">0</span></div>
</div>

<div id="dashboard">
  <h3>📊 실시간 업종 분포</h3>
  <div id="industry-chart"></div>
  <div class="stat-divider"></div>
  <h3 style="color:#4cc9f0;font-size:13px">🏷️ 라이프스타일 분포</h3>
  <div id="lifestyle-chart"></div>
</div>

<div id="event-banner"></div>
<div id="agent-popup"><span class="close" onclick="closePopup()">✕</span><div id="popup-content"></div></div>

<div id="controls">
  <button id="btn-play" onclick="togglePlay()">▶ Play</button>
  <button onclick="stepFrame(-1)">◀</button>
  <button onclick="stepFrame(1)">▶</button>
  <input type="range" id="progress" min="0" max="{total_frames - 1}" value="0" oninput="seekFrame(this.value)" style="width:160px"/>
  <span id="frame-label">0/{total_frames - 1}</span>
  <label>Speed</label>
  <input type="range" id="speed" min="1" max="20" value="4" style="width:80px"/>
  <label>색상</label>
  <select id="color-mode" onchange="changeColorMode()">
    <option value="segment">세그먼트</option>
    <option value="lifestyle">라이프스타일</option>
    <option value="spending">소비수준</option>
    <option value="satisfaction">만족도</option>
  </select>
  <label style="color:#aaa;font-size:11px">Heat</label>
  <select id="heat-mode" onchange="toggleHeatmap()" style="background:#1a1a2e;color:white;border:1px solid #e76f51;padding:3px 6px;border-radius:5px;font-size:11px">
    <option value="off">OFF</option>
    <option value="density">밀집도</option>
    <option value="spending">소비금액</option>
  </select>
</div>

<script>
const FRAMES = {frames_json};
const EVENTS = {events_json};
const WSTATS = {stats_json};
const TOTAL = FRAMES.length;

const SEG_COLORS = {{
  commuter: '#3388ff', weekend_visitor: '#33cc33',
  resident: '#ff8800', evening_visitor: '#cc33ff',
}};
const LS_COLORS = {{
  '카페러버': '#e07c4c', '미식가': '#d63384', '가성비추구': '#20c997',
  '건강지향': '#6fdb6f', '쇼핑중독': '#fd7e14', '문화예술': '#6f42c1',
  '집순이': '#6c757d', '야식파': '#dc3545', '': '#ffffff55',
}};
const IND_COLORS = {{
  '카페':'#8B4513','한식':'#DAA520','일식':'#FF6347','중식':'#FF4500',
  '패스트푸드':'#FFD700','치킨':'#D2691E','피자':'#B22222','주류':'#800080',
  '편의점':'#4682B4','패션잡화':'#FF69B4','슈퍼마켓':'#228B22','베이커리':'#DEB887',
  '분식':'#CD853F','문화여가':'#6A5ACD','전자제품':'#4169E1','숙박':'#2E8B57',
  '디저트':'#FF85C0','미용':'#BA55D3','교육':'#5F9EA0','주유':'#708090',
}};

// Map
const map = L.map('map', {{zoomControl: false}}).setView([37.5665, 126.978], 12);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 18,
}}).addTo(map);

// Markers
let markers = {{}};
let agentData = {{}};  // 에이전트별 최신 데이터
let currentPositions = {{}};
let targetPositions = {{}};
let heatLayer = null;

function getAgentColor(a) {{
  const mode = document.getElementById('color-mode').value;
  if (mode === 'lifestyle') return LS_COLORS[a.ls] || '#ffffff55';
  if (mode === 'spending') {{
    if (a.amt > 30000) return '#ff4444';
    if (a.amt > 15000) return '#ffaa00';
    if (a.amt > 5000) return '#44cc44';
    return '#4488ff';
  }}
  if (mode === 'satisfaction') {{
    if (a.sat > 0.7) return '#00ff88';
    if (a.sat > 0.4) return '#ffcc00';
    return '#ff4444';
  }}
  return SEG_COLORS[a.seg] || '#ffffff';
}}

function initMarkers(frame) {{
  frame.agents.forEach(a => {{
    agentData[a.id] = a;
    const color = getAgentColor(a);
    const m = L.circleMarker([a.lat, a.lng], {{
      radius: 4, color: color, fillColor: color, fillOpacity: 0.8, weight: 0,
    }}).addTo(map);
    m.on('click', () => showAgentPopup(a.id));
    markers[a.id] = m;
    currentPositions[a.id] = {{lat: a.lat, lng: a.lng}};
  }});
  updateSegCounts(frame);
}}

function showAgentPopup(agentId) {{
  const a = agentData[agentId];
  if (!a) return;
  const satPct = Math.round(a.sat * 100);
  const satColor = a.sat > 0.7 ? '#00ff88' : a.sat > 0.4 ? '#ffcc00' : '#ff4444';
  const triggers = (a.trg || []).map(t => `<span class="tag">${{t}}</span>`).join('');
  const amtFmt = a.amt ? a.amt.toLocaleString() + '원' : '소비 없음';

  document.getElementById('popup-content').innerHTML = `
    <h4>${{a.id}}</h4>
    <div class="stat-row"><span class="stat-label">세그먼트</span><span class="stat-value">${{a.seg}}</span></div>
    <div class="stat-row"><span class="stat-label">라이프스타일</span><span class="stat-value">${{a.ls || '미지정'}}</span></div>
    <div class="stat-row"><span class="stat-label">연령</span><span class="stat-value">${{a.age || '-'}}</span></div>
    <div class="stat-divider"></div>
    <div class="stat-row"><span class="stat-label">업종</span><span class="stat-value" style="color:${{IND_COLORS[a.ind]||'#fff'}}">${{a.ind || '활동없음'}}</span></div>
    <div class="stat-row"><span class="stat-label">소비</span><span class="stat-value">${{amtFmt}}</span></div>
    <div class="stat-row"><span class="stat-label">시간</span><span class="stat-value">${{a.ts ? a.ts + '시' : '-'}}</span></div>
    <div class="stat-row"><span class="stat-label">만족도</span><span class="stat-value" style="color:${{satColor}}">${{satPct}}%</span></div>
    <div class="sat-bar"><div class="sat-fill" style="width:${{satPct}}%;background:${{satColor}}"></div></div>
    ${{triggers ? '<div style="margin-top:6px">영향: ' + triggers + '</div>' : ''}}
  `;
  const popup = document.getElementById('agent-popup');
  popup.style.display = 'block';
  popup.style.left = '50%';
  popup.style.top = '40%';
  popup.style.transform = 'translate(-50%, -50%)';
}}

function closePopup() {{ document.getElementById('agent-popup').style.display = 'none'; }}

function updateSegCounts(frame) {{
  const counts = {{}};
  const lsCounts = {{}};
  let totalSpend = 0, totalSat = 0, activeCnt = 0;
  const indCounts = {{}};

  frame.agents.forEach(a => {{
    counts[a.seg] = (counts[a.seg] || 0) + 1;
    if (a.ls) lsCounts[a.ls] = (lsCounts[a.ls] || 0) + 1;
    if (a.amt > 0) {{
      totalSpend += a.amt;
      activeCnt++;
    }}
    totalSat += a.sat || 0;
    if (a.ind) indCounts[a.ind] = (indCounts[a.ind] || 0) + 1;
  }});

  for (const [seg, cnt] of Object.entries(counts)) {{
    const el = document.getElementById('cnt-' + seg);
    if (el) el.textContent = cnt;
  }}

  document.getElementById('total-spend').textContent = totalSpend.toLocaleString() + '원';
  document.getElementById('avg-sat').textContent = (totalSat / frame.agents.length * 100).toFixed(0) + '%';
  document.getElementById('active-cnt').textContent = activeCnt + '/' + frame.agents.length;

  // Industry chart
  const sorted = Object.entries(indCounts).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const maxCnt = sorted.length > 0 ? sorted[0][1] : 1;
  let indHtml = '';
  sorted.forEach(([name, cnt]) => {{
    const pct = (cnt / maxCnt * 100).toFixed(0);
    const color = IND_COLORS[name] || '#4cc9f0';
    indHtml += `<div class="industry-bar">
      <span class="name">${{name}}</span>
      <div class="bar" style="width:${{pct}}%;background:${{color}}"></div>
      <span class="cnt">${{cnt}}</span>
    </div>`;
  }});
  document.getElementById('industry-chart').innerHTML = indHtml || '<div style="color:#666">활동 없음</div>';

  // Lifestyle chart
  const lsSorted = Object.entries(lsCounts).sort((a, b) => b[1] - a[1]);
  let lsHtml = '';
  const lsMax = lsSorted.length > 0 ? lsSorted[0][1] : 1;
  lsSorted.forEach(([name, cnt]) => {{
    const pct = (cnt / lsMax * 100).toFixed(0);
    const color = LS_COLORS[name] || '#4cc9f0';
    lsHtml += `<div class="industry-bar">
      <span class="name">${{name}}</span>
      <div class="bar" style="width:${{pct}}%;background:${{color}}"></div>
      <span class="cnt">${{cnt}}</span>
    </div>`;
  }});
  document.getElementById('lifestyle-chart').innerHTML = lsHtml;
}}

// Animation state
let currentFrame = 0;
let playing = false;
let animId = null;
let interpProgress = 1.0;

function setFrame(idx) {{
  if (idx < 0 || idx >= TOTAL) return;
  currentFrame = idx;
  const frame = FRAMES[idx];

  frame.agents.forEach(a => {{
    agentData[a.id] = a;
    targetPositions[a.id] = {{lat: a.lat, lng: a.lng}};
    if (!currentPositions[a.id]) {{
      currentPositions[a.id] = {{lat: a.lat, lng: a.lng}};
    }}
    // Update colors
    if (markers[a.id]) {{
      const color = getAgentColor(a);
      markers[a.id].setStyle({{color: color, fillColor: color}});
    }}
  }});

  interpProgress = 0.0;
  updateUI(frame);
  updateSegCounts(frame);
}}

function updateUI(frame) {{
  const dayNames = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];
  document.getElementById('day-display').textContent =
    `Day ${{frame.day}} / Week ${{frame.week}} (${{dayNames[frame.day % 7]}})`;
  document.getElementById('frame-label').textContent = `${{currentFrame}}/${{TOTAL-1}}`;
  document.getElementById('progress').value = currentFrame;

  // Events
  const evts = EVENTS[String(frame.week)];
  const banner = document.getElementById('event-banner');
  if (evts && frame.day % 7 === 0) {{
    banner.innerHTML = evts.join(' &nbsp;|&nbsp; ');
    banner.style.display = 'block';
    setTimeout(() => {{ banner.style.display = 'none'; }}, 6000);
  }}
}}

function lerp(a, b, t) {{ return a + (b - a) * t; }}

function animationLoop() {{
  if (!playing && interpProgress >= 1.0) return;

  const speed = parseInt(document.getElementById('speed').value);
  const step = speed * 0.008;

  if (interpProgress < 1.0) {{
    interpProgress = Math.min(interpProgress + step, 1.0);
    let t = 1 - Math.pow(1 - interpProgress, 4);  // easeOutQuart

    for (const [id, target] of Object.entries(targetPositions)) {{
      const cur = currentPositions[id];
      if (!cur || !markers[id]) continue;
      const nlat = lerp(cur.lat, target.lat, t);
      const nlng = lerp(cur.lng, target.lng, t);
      markers[id].setLatLng([nlat, nlng]);
    }}

    // Update heatmap during animation
    if (heatLayer && interpProgress > 0.9) updateHeatmap();

  }} else if (playing) {{
    for (const [id, target] of Object.entries(targetPositions)) {{
      if (currentPositions[id]) {{
        currentPositions[id].lat = target.lat;
        currentPositions[id].lng = target.lng;
      }}
    }}
    if (currentFrame < TOTAL - 1) {{
      setFrame(currentFrame + 1);
    }} else {{
      playing = false;
      document.getElementById('btn-play').textContent = '▶ Play';
      document.getElementById('btn-play').classList.remove('active');
    }}
  }}

  animId = requestAnimationFrame(animationLoop);
}}

function togglePlay() {{
  playing = !playing;
  const btn = document.getElementById('btn-play');
  if (playing) {{
    btn.textContent = '⏸ Pause';
    btn.classList.add('active');
    if (currentFrame >= TOTAL - 1) setFrame(0);
    animationLoop();
  }} else {{
    btn.textContent = '▶ Play';
    btn.classList.remove('active');
  }}
}}

function stepFrame(delta) {{
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  document.getElementById('btn-play').classList.remove('active');
  for (const [id, target] of Object.entries(targetPositions)) {{
    if (currentPositions[id]) {{
      currentPositions[id].lat = target.lat;
      currentPositions[id].lng = target.lng;
      if (markers[id]) markers[id].setLatLng([target.lat, target.lng]);
    }}
  }}
  const next = Math.max(0, Math.min(TOTAL - 1, currentFrame + delta));
  setFrame(next);
  animationLoop();
}}

function seekFrame(val) {{
  const idx = parseInt(val);
  playing = false;
  document.getElementById('btn-play').textContent = '▶ Play';
  const frame = FRAMES[idx];
  frame.agents.forEach(a => {{
    agentData[a.id] = a;
    currentPositions[a.id] = {{lat: a.lat, lng: a.lng}};
    targetPositions[a.id] = {{lat: a.lat, lng: a.lng}};
    if (markers[a.id]) {{
      markers[a.id].setLatLng([a.lat, a.lng]);
      const color = getAgentColor(a);
      markers[a.id].setStyle({{color: color, fillColor: color}});
    }}
  }});
  currentFrame = idx;
  interpProgress = 1.0;
  updateUI(frame);
  updateSegCounts(frame);
  if (heatLayer) updateHeatmap();
}}

function changeColorMode() {{
  const frame = FRAMES[currentFrame];
  frame.agents.forEach(a => {{
    if (markers[a.id]) {{
      const color = getAgentColor(a);
      markers[a.id].setStyle({{color: color, fillColor: color}});
    }}
  }});
}}

// Heatmap
function toggleHeatmap() {{
  const mode = document.getElementById('heat-mode').value;
  if (heatLayer) {{ map.removeLayer(heatLayer); heatLayer = null; }}
  if (mode !== 'off') {{
    heatLayer = L.heatLayer([], {{
      radius: 25, blur: 15, maxZoom: 17,
      gradient: {{0.2: '#0000ff', 0.4: '#00ffff', 0.6: '#00ff00', 0.8: '#ffff00', 1.0: '#ff0000'}}
    }}).addTo(map);
    updateHeatmap();
  }}
}}

function updateHeatmap() {{
  if (!heatLayer) return;
  const mode = document.getElementById('heat-mode').value;
  const frame = FRAMES[currentFrame];
  let pts;
  if (mode === 'density') {{
    // 밀집도: 모든 에이전트 동일 intensity
    pts = frame.agents.map(a => [a.lat, a.lng, 0.6]);
  }} else {{
    // 소비금액: intensity = amt / 30000
    pts = frame.agents
      .filter(a => a.amt > 0)
      .map(a => [a.lat, a.lng, Math.min(a.amt / 30000, 1.0)]);
  }}
  heatLayer.setLatLngs(pts);
}}

// Init
if (FRAMES.length > 0) {{
  initMarkers(FRAMES[0]);
  setFrame(0);
  FRAMES[0].agents.forEach(a => {{
    if (markers[a.id]) markers[a.id].setLatLng([a.lat, a.lng]);
  }});
}}
</script>
</body>
</html>"""


# ═══════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    test_positions = []
    agents = []
    segs = ["commuter"] * 15 + ["weekend_visitor"] * 5 + ["resident"] * 5 + ["evening_visitor"] * 5
    lifestyles = ["카페러버", "미식가", "가성비추구", "건강지향", "쇼핑중독", "문화예술", "집순이", "야식파"]
    industries = ["카페", "한식", "일식", "편의점", "패스트푸드", "치킨", "주류"]

    for i in range(30):
        agents.append({
            "agent_id": f"consumer_{i:04d}",
            "segment": segs[i],
            "lifestyle": lifestyles[i % len(lifestyles)],
            "age_group": rng.choice(["20_29세", "30_39세", "40_49세"]),
            "current_lat": 37.5665 + rng.normal(0, 0.015),
            "current_lng": 126.978 + rng.normal(0, 0.015),
        })

    for day in range(14):
        snap = []
        for a in agents:
            a["current_lat"] += rng.normal(0, 0.003)
            a["current_lng"] += rng.normal(0, 0.003)
            snap.append({
                "agent_id": a["agent_id"],
                "segment": a["segment"],
                "current_lat": a["current_lat"],
                "current_lng": a["current_lng"],
                "lifestyle": a["lifestyle"],
                "age_group": a["age_group"],
                "industry": rng.choice(industries),
                "amount": int(rng.integers(5000, 50000)),
                "satisfaction": round(rng.random() * 0.5 + 0.3, 2),
                "time_slot": int(rng.choice([8, 12, 14, 19])),
                "triggered_by": ["뉴스:성수카페바이럴"] if rng.random() < 0.2 else [],
            })
        test_positions.append((day, snap))

    path = generate_animation_html(test_positions)
    print(f"\n[OK] Open in browser: {path}")

