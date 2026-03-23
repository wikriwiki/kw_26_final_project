"""
에이전트 지도 시각화 (Folium 2D)

에이전트의 현재 위치(소비/활동 위치)만 표시.
시뮬레이션 중 이벤트에 따른 이동을 시각화.
"""
import folium
from pathlib import Path

from config import OUTPUT_DIR


# 세그먼트별 색상
SEGMENT_COLORS = {
    "commuter": "#3388ff",        # 파랑
    "weekend_visitor": "#33cc33",  # 초록
    "resident": "#ff8800",        # 주황
    "evening_visitor": "#cc33ff",  # 보라
}


def create_folium_map(agents: list[dict], title: str = "Initial") -> folium.Map:
    """에이전트 현재 위치를 Folium 지도에 표시

    세그먼트별 색상 + 레이어 토글
    """
    center_lat = 37.5665
    center_lng = 126.9780

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=12,
        tiles="CartoDB dark_matter",
    )

    # 세그먼트별 레이어
    for seg_name, color in SEGMENT_COLORS.items():
        fg = folium.FeatureGroup(name=seg_name)
        seg_agents = [a for a in agents if a.get("segment") == seg_name]

        for agent in seg_agents:
            lat = agent.get("current_lat")
            lng = agent.get("current_lng")
            if lat is None or lng is None:
                continue

            folium.CircleMarker(
                location=[lat, lng],
                radius=4,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(
                    f"<b>{agent['agent_id']}</b><br>"
                    f"Segment: {agent['segment']}<br>"
                    f"Gender: {agent.get('gender', '?')}<br>"
                    f"Age: {agent.get('age_group', '?')}<br>"
                    f"Dong: {str(agent.get('adm_cd', '?'))[:8]}<br>"
                    f"Spending: {agent.get('monthly_spending', 0):,}won",
                    max_width=220,
                ),
            ).add_to(fg)

        fg.add_to(m)

    # 범례
    seg_counts = {}
    for a in agents:
        seg_counts[a["segment"]] = seg_counts.get(a["segment"], 0) + 1

    legend_html = f"""
    <div style="position:fixed; bottom:50px; left:50px; z-index:1000;
         background:rgba(0,0,0,0.85); padding:15px; border-radius:8px;
         font-family:monospace; color:white; font-size:13px; line-height:1.6;">
    <b>{title}</b><br>
    <span style="color:{SEGMENT_COLORS['commuter']}">●</span> Commuter ({seg_counts.get('commuter', 0)})<br>
    <span style="color:{SEGMENT_COLORS['weekend_visitor']}">●</span> Weekend ({seg_counts.get('weekend_visitor', 0)})<br>
    <span style="color:{SEGMENT_COLORS['resident']}">●</span> Resident ({seg_counts.get('resident', 0)})<br>
    <span style="color:{SEGMENT_COLORS['evening_visitor']}">●</span> Evening ({seg_counts.get('evening_visitor', 0)})<br>
    <b>Total: {len(agents)}</b>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    return m


def save_map(agents: list[dict], filename: str = "agent_map.html", title: str = "Initial"):
    """Folium 지도를 HTML로 저장"""
    m = create_folium_map(agents, title=title)
    path = OUTPUT_DIR / filename
    m.save(str(path))
    print(f"[map] Saved to {path}")
    return path


if __name__ == "__main__":
    from etl_transform import run_etl
    from geo_utils import build_grid_map, build_dong_to_grids, assign_coordinates_to_agents

    result = run_etl(300)
    agents = result["agents"]
    kt_od = result["kt_od"]

    grid_map = build_grid_map()
    dong_grids = build_dong_to_grids(grid_map)
    agents = assign_coordinates_to_agents(agents, dong_grids, kt_od=kt_od)

    # 현재 위치 지도 저장
    path = save_map(agents, "agent_map_initial.html", title="Initial Position")
    print(f"\nOpen in browser: {path}")
