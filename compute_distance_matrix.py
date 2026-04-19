"""
서울시 행정동 간 도로 네트워크 기반 주행 거리 행렬 생성
──────────────────────────────────────────────────────
목적  : EMD(Earth Mover's Distance) ground distance matrix 생성
입력  : 서울시 상권분석서비스(영역-행정동).shp
출력  : distance_matrix.npy, distance_matrix.parquet, dong_node_mapping.csv

핵심 설계 결정
  1. representative_point()를 사용하여 폴리곤 내부 점을 보장
     (산악/하천 지역의 geometric centroid 오류 방지)
  2. SCC(최대 강결합 컴포넌트) 추출 후 노드 매핑 수행
     (고립 노드 매핑으로 인한 경로 부재 방지)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
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
SHP_PATH = "서울시 상권분석서비스(영역-행정동).shp"
NETWORK_CACHE = "seoul_drive_network.graphml"
PLACE_NAME = "Seoul, South Korea"


# ── Step 1: 행정동 중심점(대표점) 추출 ─────────────────────────────────────────
def load_dong_representative_points(shp_path: str | Path) -> gpd.GeoDataFrame:
    """
    행정동 SHP 파일에서 각 폴리곤의 대표점(representative_point)을 추출합니다.

    representative_point()는 centroid와 달리 항상 폴리곤 내부에 위치하므로,
    산악 지역이나 하천 위에 중심점이 놓이는 문제를 완화합니다.

    Returns
    -------
    gpd.GeoDataFrame
        컬럼: ADSTRD_CD, ADSTRD_NM, lat, lng, geometry(Point)
    """
    log.info("Step 1: 행정동 SHP 로드 → 대표점 추출")
    gdf = gpd.read_file(shp_path)
    log.info("  원본 CRS: %s | 행정동 수: %d", gdf.crs, len(gdf))

    # WGS84로 변환
    if gdf.crs is None or gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(CRS_WGS84)

    # representative_point: 폴리곤 내부 보장
    rep_points = gdf.geometry.representative_point()
    centroids_gdf = gpd.GeoDataFrame(
        {
            "ADSTRD_CD": gdf["ADSTRD_CD"].values,
            "ADSTRD_NM": gdf["ADSTRD_NM"].values,
            "lat": rep_points.y,
            "lng": rep_points.x,
        },
        geometry=rep_points,
        crs=CRS_WGS84,
    )

    log.info("  대표점 추출 완료: %d개", len(centroids_gdf))
    return centroids_gdf


# ── Step 2: 서울시 도로 네트워크 다운로드/로드 ──────────────────────────────────
def download_or_load_network(
    cache_path: str | Path = NETWORK_CACHE,
    place: str = PLACE_NAME,
) -> nx.MultiDiGraph:
    """
    서울시 차량 도로 네트워크를 OSM에서 다운로드하거나 캐시에서 로드합니다.

    Returns
    -------
    nx.MultiDiGraph
        OSMnx 도로 네트워크 그래프 (edge attribute 'length' 포함, 단위: 미터)
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        log.info("Step 2: 캐시 파일에서 도로 네트워크 로드 (%s)", cache_path)
        G = ox.load_graphml(cache_path)
        log.info("  노드: %d | 엣지: %d", G.number_of_nodes(), G.number_of_edges())
        return G

    log.info("Step 2: OSM에서 서울 도로 네트워크 다운로드 (수 분 소요)")
    t0 = time.time()
    G = ox.graph_from_place(place, network_type="drive")
    elapsed = time.time() - t0
    log.info(
        "  다운로드 완료 (%.1f초) | 노드: %d | 엣지: %d",
        elapsed,
        G.number_of_nodes(),
        G.number_of_edges(),
    )

    # 캐시 저장
    ox.save_graphml(G, cache_path)
    log.info("  캐시 저장: %s", cache_path)
    return G


# ── Step 3: SCC 추출 → 최근접 노드 매핑 ────────────────────────────────────────
def extract_scc_and_map_nodes(
    G: nx.MultiDiGraph,
    centroids_gdf: gpd.GeoDataFrame,
) -> tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    최대 강결합 컴포넌트(SCC)를 추출한 후, 각 행정동 대표점을
    SCC 내 최근접 노드에 매핑합니다.

    SCC 추출을 매핑 이전에 수행하여, 고립된 도로 노드에 매핑되는
    문제를 원천 차단합니다.

    Returns
    -------
    (G_scc, centroids_gdf)
        - G_scc: SCC 서브그래프
        - centroids_gdf: 'osm_node_id' 컬럼이 추가된 GeoDataFrame
    """
    log.info("Step 3: SCC 추출 및 최근접 노드 매핑")

    # 최대 강결합 컴포넌트 추출
    largest_scc_nodes = max(nx.strongly_connected_components(G), key=len)
    G_scc = G.subgraph(largest_scc_nodes).copy()
    log.info(
        "  SCC 추출: %d → %d 노드 (%.1f%% 유지)",
        G.number_of_nodes(),
        G_scc.number_of_nodes(),
        100 * G_scc.number_of_nodes() / G.number_of_nodes(),
    )

    # SCC 그래프에서 최근접 노드 매핑
    node_ids = ox.nearest_nodes(
        G_scc,
        X=centroids_gdf["lng"].values,
        Y=centroids_gdf["lat"].values,
    )
    centroids_gdf = centroids_gdf.copy()
    centroids_gdf["osm_node_id"] = node_ids

    # 매핑 통계
    n_unique = centroids_gdf["osm_node_id"].nunique()
    log.info(
        "  노드 매핑 완료: %d 행정동 → %d 고유 노드",
        len(centroids_gdf),
        n_unique,
    )
    if n_unique < len(centroids_gdf):
        log.warning(
            "  ⚠ %d개 행정동이 동일 노드에 매핑됨 (인접 행정동)",
            len(centroids_gdf) - n_unique,
        )

    return G_scc, centroids_gdf


# ── Step 4: All-pairs 최단 경로 거리 계산 ──────────────────────────────────────
def compute_all_pairs_distance(
    G: nx.MultiDiGraph,
    centroids_gdf: gpd.GeoDataFrame,
) -> tuple[np.ndarray, list[str]]:
    """
    모든 행정동 쌍 사이의 최단 주행 거리를 계산합니다.

    Parameters
    ----------
    G : nx.MultiDiGraph
        SCC 도로 네트워크
    centroids_gdf : gpd.GeoDataFrame
        osm_node_id 컬럼 포함

    Returns
    -------
    (dist_matrix, dong_codes)
        - dist_matrix: N×N numpy 배열 (거리, 미터)
        - dong_codes: 행/열 순서에 대응하는 행정동 코드 리스트
    """
    dong_codes = centroids_gdf["ADSTRD_CD"].tolist()
    node_ids = centroids_gdf["osm_node_id"].tolist()
    n = len(dong_codes)

    log.info("Step 4: All-pairs shortest path 계산 (%d × %d = %d 쌍)", n, n, n * n)
    t0 = time.time()

    # 고유 노드 ID → 인덱스 역매핑
    unique_nodes = list(set(node_ids))
    node_to_idx = {nid: i for i, nid in enumerate(unique_nodes)}

    # 고유 노드 간 최단 거리 계산 (중복 노드 재계산 방지)
    log.info("  고유 노드 수: %d", len(unique_nodes))
    unique_dist = np.full((len(unique_nodes), len(unique_nodes)), np.inf)
    np.fill_diagonal(unique_dist, 0.0)

    computed = 0
    total = len(unique_nodes)
    for src_node in unique_nodes:
        # single-source Dijkstra: 한 소스에서 다른 모든 타겟까지
        lengths = nx.single_source_dijkstra_path_length(G, src_node, weight="length")
        src_idx = node_to_idx[src_node]
        for tgt_node in unique_nodes:
            if tgt_node in lengths:
                unique_dist[src_idx, node_to_idx[tgt_node]] = lengths[tgt_node]

        computed += 1
        if computed % 50 == 0 or computed == total:
            elapsed = time.time() - t0
            log.info("  진행: %d/%d (%.1f초 경과)", computed, total, elapsed)

    # 고유 노드 거리 → 행정동 거리 행렬로 확장
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            si = node_to_idx[node_ids[i]]
            sj = node_to_idx[node_ids[j]]
            dist_matrix[i, j] = unique_dist[si, sj]

    elapsed = time.time() - t0
    log.info("  계산 완료: %.1f초", elapsed)

    # 검증
    n_inf = np.isinf(dist_matrix).sum()
    if n_inf > 0:
        log.warning("  ⚠ %d개 쌍에서 경로를 찾지 못함 (Inf)", n_inf)
    else:
        log.info("  ✓ 모든 쌍 경로 존재")

    max_km = dist_matrix[~np.isinf(dist_matrix)].max() / 1000
    mean_km = dist_matrix[~np.isinf(dist_matrix)].mean() / 1000
    log.info("  최대 거리: %.2f km | 평균 거리: %.2f km", max_km, mean_km)

    return dist_matrix, dong_codes


# ── Step 5: 결과 저장 ─────────────────────────────────────────────────────────
def save_results(
    dist_matrix: np.ndarray,
    dong_codes: list[str],
    centroids_gdf: gpd.GeoDataFrame,
    output_dir: str | Path = ".",
) -> None:
    """결과를 npy, parquet, csv로 저장합니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Step 5: 결과 저장")

    # 1) NumPy 거리 행렬
    npy_path = output_dir / "distance_matrix.npy"
    np.save(npy_path, dist_matrix)
    log.info("  저장: %s (shape=%s)", npy_path, dist_matrix.shape)

    # 2) Parquet 거리 행렬 (행정동코드를 인덱스/컬럼으로)
    df_dist = pd.DataFrame(dist_matrix, index=dong_codes, columns=dong_codes)
    df_dist.index.name = "source_dong"
    df_dist.columns.name = "target_dong"
    pq_path = output_dir / "distance_matrix.parquet"
    df_dist.to_parquet(pq_path)
    log.info("  저장: %s", pq_path)

    # 3) 행정동 → OSM 노드 매핑 CSV
    mapping_df = centroids_gdf[["ADSTRD_CD", "ADSTRD_NM", "lat", "lng", "osm_node_id"]]
    csv_path = output_dir / "dong_node_mapping.csv"
    mapping_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    log.info("  저장: %s", csv_path)


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main() -> None:
    """전체 파이프라인 실행."""
    log.info("=" * 60)
    log.info("서울시 행정동 간 도로 네트워크 주행 거리 행렬 생성 시작")
    log.info("=" * 60)
    t_total = time.time()

    # 1. 행정동 대표점
    centroids_gdf = load_dong_representative_points(SHP_PATH)

    # 2. 도로 네트워크
    G = download_or_load_network()

    # 3. SCC 추출 + 노드 매핑 (SCC 먼저 → 매핑은 SCC 내에서)
    G_scc, centroids_gdf = extract_scc_and_map_nodes(G, centroids_gdf)

    # 4. 거리 행렬 계산
    dist_matrix, dong_codes = compute_all_pairs_distance(G_scc, centroids_gdf)

    # 5. 저장
    save_results(dist_matrix, dong_codes, centroids_gdf)

    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("전체 파이프라인 완료: %.1f초", elapsed)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
