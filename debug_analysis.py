"""파이프라인 결과 진단 스크립트"""
import pandas as pd
import geopandas as gpd
import numpy as np
import h3

print("=" * 60)
print("1. PARQUET 결과 분석")
print("=" * 60)
df = pd.read_parquet('ground_truth_res8_timeseries.parquet')
print(f'총 레코드 수: {len(df)}')
print(f'고유 H3 인덱스 수: {df["h3_index"].nunique()}')
print(f'년도: {sorted(df["STDR_YY_CD"].unique())}')
print(f'분기: {sorted(df["STDR_QU_CD"].unique())}')

latest_yy = df['STDR_YY_CD'].max()
latest_qu = df.loc[df['STDR_YY_CD'] == latest_yy, 'STDR_QU_CD'].max()
df_q = df[(df['STDR_YY_CD'] == latest_yy) & (df['STDR_QU_CD'] == latest_qu)]
print(f'\n{latest_yy}년 {latest_qu}분기: {len(df_q)}개 셀')
print(f'distributed_sales 통계:')
print(df_q['distributed_sales'].describe())

target = '8830e1d911fffff'
target_data = df_q[df_q['h3_index'] == target]
if len(target_data) > 0:
    val = target_data['distributed_sales'].values[0]
    total = df_q['distributed_sales'].sum()
    pct = val / total * 100
    print(f'\n*** 문제 셀 {target} ***')
    print(f'  매출: {val:,.0f}')
    print(f'  전체 대비: {pct:.2f}%')
    print(f'  좌표: ({target_data["lat"].values[0]:.4f}, {target_data["lng"].values[0]:.4f})')
else:
    print(f'\n{target} 셀이 데이터에 없습니다')

print(f'\n매출 상위 10개 셀:')
top10 = df_q.nlargest(10, 'distributed_sales')
for _, r in top10.iterrows():
    print(f'  {r["h3_index"]}: {r["distributed_sales"]/1e8:.2f}억원 ({r["lat"]:.4f}, {r["lng"]:.4f})')

print(f'\n좌표 범위: Lat {df_q["lat"].min():.4f}~{df_q["lat"].max():.4f}, Lng {df_q["lng"].min():.4f}~{df_q["lng"].max():.4f}')

print("\n" + "=" * 60)
print("2. CSV 원본 데이터 분석")
print("=" * 60)
df_csv = pd.read_csv('서울시 상권분석서비스(추정매출-상권)_2024년.csv', encoding='cp949')
print(f'총 레코드 수: {len(df_csv)}')
print(f'컬럼: {df_csv.columns.tolist()[:10]}...')
print(f'고유 상권코드 수: {df_csv["상권_코드"].nunique()}')
print(f'기준_년분기_코드: {sorted(df_csv["기준_년분기_코드"].unique())}')
print(f'\n당월_매출_금액 통계:')
print(df_csv['당월_매출_금액'].describe())

latest_code = df_csv['기준_년분기_코드'].max()
df_latest_csv = df_csv[df_csv['기준_년분기_코드'] == latest_code]
sales_by_trdar = df_latest_csv.groupby('상권_코드')['당월_매출_금액'].sum()
print(f'\n{latest_code} 기준 상권별 매출:')
print(f'  상권 수: {len(sales_by_trdar)}')
print(sales_by_trdar.describe())
print(f'\n  상위 10개 상권 (총 매출):')
for code, val in sales_by_trdar.nlargest(10).items():
    print(f'    상권코드 {code}: {val/1e8:.2f}억원')

print("\n" + "=" * 60)
print("3. SHP 파일 분석")
print("=" * 60)
gdf = gpd.read_file('서울시 상권분석서비스(영역-상권).shp')
print(f'CRS: {gdf.crs}')
print(f'총 상권 수: {len(gdf)}')
print(f'컬럼: {gdf.columns.tolist()}')
print(f'Geometry 타입 분포: {gdf.geometry.geom_type.value_counts().to_dict()}')

trdar_col = None
for col in ['TRDAR_CD', '상권_코드', 'TRDAR_CD_I']:
    if col in gdf.columns:
        trdar_col = col
        print(f'상권코드 컬럼: {col}, 고유값 수: {gdf[col].nunique()}')
        break

if gdf.crs and gdf.crs.to_epsg() != 5181:
    gdf_proj = gdf.to_crs('EPSG:5181')
else:
    gdf_proj = gdf

areas = gdf_proj.geometry.area
print(f'\n상권 면적 통계(m²):')
print(areas.describe())
print(f'면적 0인 상권: {(areas == 0).sum()}')
print(f'면적 < 100m²: {(areas < 100).sum()}')
print(f'면적 > 1,000,000m² (1km²): {(areas > 1_000_000).sum()}')

print(f'\n상위 10개 대형 상권 (m²):')
top_areas = areas.nlargest(10)
for idx in top_areas.index:
    trdar = gdf_proj.iloc[idx][trdar_col] if trdar_col else 'N/A'
    print(f'  {trdar}: {areas.iloc[idx]:,.0f} m²')

print("\n" + "=" * 60)
print("4. 매칭 분석 (CSV ↔ SHP)")
print("=" * 60)
csv_codes = set(df_csv['상권_코드'].astype(str).unique())
shp_codes = set(gdf[trdar_col].astype(str).unique())
print(f'CSV 상권코드: {len(csv_codes)}')
print(f'SHP 상권코드: {len(shp_codes)}')
print(f'교집합: {len(csv_codes & shp_codes)}')
print(f'CSV에만 존재: {len(csv_codes - shp_codes)}')
print(f'SHP에만 존재: {len(shp_codes - csv_codes)}')

print("\n" + "=" * 60)
print("5. H3 폴리필 분석 (resolution=8)")
print("=" * 60)
from shapely.ops import unary_union

# Union of all clipped polygons
gdf_4326 = gdf_proj.to_crs('EPSG:4326')
union_geom = unary_union(gdf_4326.geometry)
print(f'Union geometry type: {union_geom.geom_type}')
if union_geom.geom_type == 'MultiPolygon':
    print(f'구성 폴리곤 수: {len(union_geom.geoms)}')
    areas_4326 = [g.area for g in union_geom.geoms]
    print(f'폴리곤 면적(도²) 통계: min={min(areas_4326):.10f}, max={max(areas_4326):.10f}')

# H3 resolution 8 셀 면적 참고
print(f'\nH3 resolution 8 셀의 대략적 면적: ~0.737 km²')
print(f'서울시 전체 면적: ~605 km²')
print(f'서울 전체를 커버하는 데 필요한 res 8 셀 수: ~{605/0.737:.0f}')

# 문제 셀의 H3 정보
lat, lng = h3.h3_to_geo(target) if hasattr(h3, 'h3_to_geo') else h3.cell_to_latlng(target)
print(f'\n문제 셀 {target}의 중심: ({lat:.4f}, {lng:.4f})')
res = h3.h3_get_resolution(target) if hasattr(h3, 'h3_get_resolution') else h3.get_resolution(target)
print(f'Resolution: {res}')

# 문제 셀에 어떤 상권들이 매핑되는지 역추적
from shapely.geometry import Polygon
try:
    boundary = h3.h3_to_geo_boundary(target, geo_json=True)
except AttributeError:
    boundary = [(lng, lat) for lat, lng in h3.cell_to_boundary(target)]
h3_poly = Polygon(boundary)

# H3 셀을 투영좌표로 변환
h3_gdf_single = gpd.GeoDataFrame(geometry=[h3_poly], crs='EPSG:4326').to_crs('EPSG:5181')
print(f'문제 셀 면적 (투영): {h3_gdf_single.geometry.area.values[0]:,.0f} m²')

# 이 셀과 교차하는 상권들 찾기
intersecting = gdf_proj[gdf_proj.geometry.intersects(h3_gdf_single.geometry.values[0])]
print(f'이 셀과 교차하는 상권 수: {len(intersecting)}')
if len(intersecting) > 0:
    print(f'교차 상권 코드: {intersecting[trdar_col].tolist()[:20]}')

print("\nDone!")
