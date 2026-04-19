"""
서울시 상권 매출(행정동 단위) 시각화 모듈
────────────────────────────────────────────────
입력  : dong_sales_ground_truth.parquet  (pipeline.py 출력)
출력  : seoul_sales_map.png   — 정적 Matplotlib 지도
        seoul_sales_map.html  — 인터랙티브 Folium Choropleth

개선 요약 (v3)
  [CRITICAL] H3 매핑 제거 -> 행정동 원본 폴리곤 자체 시각화 (Folium / Matplotlib)
  [CRITICAL] Parquet 데이터의 adstrd_cd, adstrd_nm에 맞춰 스키마 갱신
"""

from __future__ import annotations

import logging
from pathlib import Path

import folium
import geopandas as gpd
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

log = logging.getLogger(__name__)

# ── 한글 폰트 설정 (우선순위 폴백 체인) ──────────────────────────────────────
def _configure_korean_font() -> None:
    import sys
    import matplotlib.font_manager as fm

    if sys.platform == "win32":
        candidates = ["Malgun Gothic"]
    elif sys.platform == "darwin":
        candidates = ["AppleGothic", "Apple SD Gothic Neo"]
    else:
        candidates = ["NanumGothic", "NanumBarunGothic", "Nanum Gothic"]

    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in candidates if c in available), None)

    if chosen:
        matplotlib.rc("font", family=chosen)
        log.info("한글 폰트 설정: %s", chosen)
    else:
        log.warning("한글 폰트를 찾지 못했습니다.")
    matplotlib.rcParams["axes.unicode_minus"] = False


# ── 데이터 로드 및 GeoDataFrame 구성 ─────────────────────────────────────────
def _load_latest_quarter(parquet_path: Path) -> gpd.GeoDataFrame:
    """Parquet에서 가장 최근 년도/분기 데이터를 로드하고 GeoDataFrame으로 반환."""
    df = pd.read_parquet(parquet_path)

    latest_yy = df["STDR_YY_CD"].max()
    latest_qu = df.loc[df["STDR_YY_CD"] == latest_yy, "STDR_QU_CD"].max()
    log.info("시각화 기준: %d년 %d분기", latest_yy, latest_qu)

    df_latest = df[
        (df["STDR_YY_CD"] == latest_yy) & (df["STDR_QU_CD"] == latest_qu)
    ].copy()

    # 이미 parquet 안에 geometry 데이터가 WKB 혹은 WKT 형태로 탑재됨. 
    # geopandas로 불러오면서 active geometry를 선언하고 crs 설정.
    import shapely.wkb
    if type(df_latest["geometry"].iloc[0]) == bytes:
        df_latest["geometry"] = df_latest["geometry"].apply(shapely.wkb.loads)
    
    gdf_plot = gpd.GeoDataFrame(df_latest, geometry="geometry", crs="EPSG:4326")

    # 결측치나 0원 제외
    gdf_plot = gdf_plot[gdf_plot["SALES_AMT"] > 0].copy()
    # 단위를 억원으로 변경
    gdf_plot["sales_100m"] = (gdf_plot["SALES_AMT"] / 1e8).round(4)

    log.info("시각화 대상 행정동: %d개", len(gdf_plot))
    return gdf_plot, latest_yy, latest_qu


# ── 정적 지도 (Matplotlib) ────────────────────────────────────────────────────
def _save_static_map(
    gdf_plot: gpd.GeoDataFrame,
    latest_yy: int,
    latest_qu: int,
    output_path: Path,
) -> None:
    """Matplotlib 기반 정적 PNG 지도를 저장."""
    fig, ax = plt.subplots(figsize=(15, 12))
    try:
        gdf_plot.plot(
            column="sales_100m",
            cmap="YlOrRd",
            linewidth=0.5,
            edgecolor="gray",
            ax=ax,
            legend=True,
            legend_kwds={"label": "추정 매출 (억원)", "orientation": "vertical", "shrink": 0.7},
        )
        ax.set_axis_off()
        ax.set_title(
            f"서울시 행정동별 상권 추정매출\n({latest_yy}년 {latest_qu}분기)",
            fontsize=20,
            fontweight="bold",
            pad=16,
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        log.info("정적 지도 저장: %s", output_path)
    finally:
        plt.close(fig)


# ── 인터랙티브 지도 (Folium) ─────────────────────────────────────────────────
def _save_interactive_map(
    gdf_plot: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Folium Choropleth + Tooltip 통합 인터랙티브 지도를 저장."""
    center_lat = gdf_plot["lat"].mean()
    center_lng = gdf_plot["lng"].mean()

    m = folium.Map(
        location=[center_lat, center_lng],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    # 문자열화 (GeoJSON 키 매칭 용도)
    gdf_plot["adstrd_cd"] = gdf_plot["adstrd_cd"].astype(str)

    choropleth = folium.Choropleth(
        geo_data=gdf_plot.to_json(),
        name="choropleth",
        data=gdf_plot[["adstrd_cd", "sales_100m"]],
        columns=["adstrd_cd", "sales_100m"],
        key_on="feature.properties.adstrd_cd",
        fill_color="YlOrRd",
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name="행정동별 추정 매출 (억원)",
        bins=9,  # 분위수 등 여러 색상 구간 활용
        highlight=True,
    )
    choropleth.add_to(m)

    tooltip_layer = folium.GeoJson(
        gdf_plot.to_json(),
        name="tooltip",
        style_function=lambda _: {
            "fillColor": "#ffffff",
            "color": "#000000",
            "fillOpacity": 0.01,
            "weight": 0.5,
        },
        highlight_function=lambda _: {
            "fillColor": "#333333",
            "color": "#333333",
            "fillOpacity": 0.3,
            "weight": 2,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["adstrd_cd", "adstrd_nm", "sales_100m"],
            aliases=["행정동 코드", "행정동", "매출 (억원)"],
            localize=True,
            style=(
                "background-color: white;"
                "color: #333;"
                "font-family: 'Malgun Gothic', 'Nanum Gothic', sans-serif;"
                "font-size: 13px;"
                "padding: 10px;"
                "border-radius: 4px;"
                "box-shadow: 2px 2px 6px rgba(0,0,0,.15);"
            ),
        ),
        control=False,
    )
    m.add_child(tooltip_layer)
    m.keep_in_front(tooltip_layer)

    folium.LayerControl().add_to(m)
    m.save(str(output_path))
    log.info("인터랙티브 지도 저장: %s", output_path)


# ── 메인 함수 ─────────────────────────────────────────────────────────────────
def create_visualizations(
    parquet_path: str | Path = "dong_sales_ground_truth.parquet",
    output_dir: str | Path = ".",
) -> None:
    """
    Parquet Ground Truth 데이터를 읽어 정적/인터랙티브 지도를 생성합니다.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    _configure_korean_font()

    parquet_path = Path(parquet_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("데이터 로딩 중: %s", parquet_path)
    if not parquet_path.exists():
        log.error("파일이 존재하지 않습니다: %s (pipeline.py를 먼저 실행하세요)", parquet_path)
        return

    gdf_plot, latest_yy, latest_qu = _load_latest_quarter(parquet_path)

    log.info("정적 지도 생성 중...")
    _save_static_map(
        gdf_plot,
        latest_yy,
        latest_qu,
        output_dir / "seoul_sales_map.png",
    )

    log.info("인터랙티브 지도 생성 중...")
    _save_interactive_map(
        gdf_plot,
        output_dir / "seoul_sales_map.html",
    )

    log.info("모든 시각화 완료.")


if __name__ == "__main__":
    create_visualizations()
