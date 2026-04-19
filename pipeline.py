"""
서울시 행정동 기반 매출 Ground Truth 데이터 파이프라인
────────────────────────────────────────────────
입력  : 서울시 상권분석서비스(추정매출-행정동)_2024년.csv
        서울시 상권분석서비스(영역-행정동).shp
출력  : dong_sales_ground_truth.parquet

개편 이력 (v3)
  [CRITICAL] 구역을 H3가 아닌 행정동 폴리곤 자체로 1:1 매핑 
  [CRITICAL] Voronoi 및 H3 폴리필 연산 전면 제거하여 퍼포먼스 및 정확도 극대화
"""

from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

# ── 로거 설정 ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── 상수 ───────────────────────────────────────────────────────────────────────
CRS_WGS84 = "EPSG:4326"

CSV_DONG_COL = "행정동_코드"
CSV_YYQU_COL = "기준_년분기_코드"
CSV_SALES_COL = "당월_매출_금액"
SHP_DONG_COL = "ADSTRD_CD"
SHP_NM_COL = "ADSTRD_NM"

# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def _read_csv_auto(path: str | Path) -> pd.DataFrame:
    """cp949 → utf-8 → utf-8-sig 폴백 CSV 로더."""
    for enc in ("cp949", "utf-8", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc)
        except (UnicodeDecodeError, LookupError):
            continue
    raise ValueError(f"지원되는 인코딩으로 파일을 읽을 수 없습니다: {path}")

# ── 메인 파이프라인 ────────────────────────────────────────────────────────────
def process_ground_truth(
    csv_path: str | Path = "서울시 상권분석서비스(추정매출-행정동)_2024년.csv",
    shp_path: str | Path = "서울시 상권분석서비스(영역-행정동).shp",
    output_dir: str | Path = ".",
) -> gpd.GeoDataFrame:
    """
    행정동별 매출 데이터를 공간 정보와 병합하여 Ground Truth로 변환합니다.

    Parameters
    ----------
    csv_path : str | Path
        추정매출(행정동) CSV 경로
    shp_path : str | Path
        영역(행정동) Shapefile 경로
    output_dir : str | Path
        Parquet 출력 디렉토리
        
    Returns
    -------
    gpd.GeoDataFrame
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 데이터 로드 및 스키마 매핑 ───────────────────────────────────
    log.info("Step 1: 데이터 로드")
    
    df_sales = _read_csv_auto(csv_path)
    gdf_area = gpd.read_file(shp_path)

    # 문자열로 타입 통일
    df_sales[CSV_DONG_COL] = df_sales[CSV_DONG_COL].astype(str)
    gdf_area[SHP_DONG_COL] = gdf_area[SHP_DONG_COL].astype(str)

    # 연도·분기 분리 
    df_sales["STDR_YY_CD"] = df_sales[CSV_YYQU_COL] // 10
    df_sales["STDR_QU_CD"] = df_sales[CSV_YYQU_COL] % 10

    # 행정동 단위 전체 업종 매출 집계
    df_sales_grouped = (
        df_sales
        .groupby(["STDR_YY_CD", "STDR_QU_CD", CSV_DONG_COL], as_index=False)[CSV_SALES_COL]
        .sum()
        .rename(columns={CSV_SALES_COL: "SALES_AMT"})
    )
    log.info("매출 집계 완료: %d 레코드", len(df_sales_grouped))

    # ── Step 2: 공간 데이터 병합(Merge) ───────────────────────────────────────
    log.info("Step 2: 공간 데이터 병합")
    
    # SHP 원본 CRS → WGS84 변환 (시각화 호환성)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.to_crs(CRS_WGS84)

    merged_gdf = gdf_area.merge(
        df_sales_grouped,
        left_on=SHP_DONG_COL,
        right_on=CSV_DONG_COL,
        how="inner"
    )

    # 중심점 계산 (라벨링 또는 시각화 마커용 중심점)
    # CRS가 4326(WGS84)이므로 경도/위도입니다. centroid.y가 위도(lat), x가 경도(lng)입니다.
    # UserWarning(Geometry is in a geographic CRS)이 뜰 수 있지만, 단순히 시각화용 지도 중심 좌표를 
    # 구하는 것이므로 무시해도 충분합니다.
    merged_gdf["lat"] = merged_gdf.geometry.centroid.y
    merged_gdf["lng"] = merged_gdf.geometry.centroid.x

    log.info("공간 데이터 병합 완료: %d 행정동 매핑됨", merged_gdf[SHP_DONG_COL].nunique())

    # ── Step 3: 컬럼 정리 및 저장 ─────────────────────────────────────────────
    log.info("Step 3: Parquet 저장")
    
    final_gdf = merged_gdf[[
        "STDR_YY_CD", "STDR_QU_CD", SHP_DONG_COL, SHP_NM_COL, 
        "SALES_AMT", "lat", "lng", "geometry"
    ]].rename(columns={SHP_DONG_COL: "adstrd_cd", SHP_NM_COL: "adstrd_nm"})

    output_path = output_dir / "dong_sales_ground_truth.parquet"
    final_gdf.to_parquet(output_path, index=False)

    log.info("저장 완료: %s | 레코드 수: %d", output_path, len(final_gdf))
    return final_gdf


if __name__ == "__main__":
    process_ground_truth()
