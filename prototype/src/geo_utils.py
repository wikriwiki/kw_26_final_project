"""
좌표 변환 + 격자-행정동 매핑 유틸리티

KT 데이터의 TM 좌표(EPSG:5186) → WGS84(EPSG:4326) 변환
행정동코드 → 격자 좌표 매핑 테이블 구축
거주지/소비지 분리 매핑
"""
import numpy as np
import pandas as pd
from pyproj import Transformer
from etl_loader import load_kt_time_age, load_kt_residence

# TM (EPSG:5186) → WGS84 (EPSG:4326)
_transformer = Transformer.from_crs("EPSG:5186", "EPSG:4326", always_xy=True)


def tm_to_wgs84(x: float, y: float) -> tuple[float, float]:
    """TM 좌표 → (lat, lng) 변환"""
    lng, lat = _transformer.transform(x, y)
    return lat, lng


def build_grid_map() -> pd.DataFrame:
    """KT 데이터에서 격자ID → (행정동코드, lat, lng) 매핑 테이블 구축

    KT 시간대별 + 거주지별 데이터를 합쳐서
    각 격자의 대표 좌표와 소속 행정동을 확정.
    """
    kt1 = load_kt_time_age()
    kt2 = load_kt_residence()

    # 격자별 대표 좌표 + 행정동 (두 데이터에서 합산)
    grids1 = (
        kt1.groupby("ID")
        .agg(
            x=("X_COORD", "mean"),
            y=("Y_COORD", "mean"),
            adm_cd=("ADMI_CD", "first"),
        )
        .reset_index()
    )

    grids2 = (
        kt2.groupby("ID")
        .agg(
            x=("X_COORD", "mean"),
            y=("Y_COORD", "mean"),
            adm_cd=("ADMI_CD", "first"),
        )
        .reset_index()
    )

    grids = pd.concat([grids1, grids2]).drop_duplicates(subset="ID")
    grids["adm_cd"] = grids["adm_cd"].astype(str)

    # TM → WGS84 변환
    coords = grids.apply(lambda r: tm_to_wgs84(r["x"], r["y"]), axis=1)
    grids["lat"] = [c[0] for c in coords]
    grids["lng"] = [c[1] for c in coords]

    print(f"[grid_map] {len(grids)} grid cells mapped")
    print(f"  lat range: {grids['lat'].min():.4f} ~ {grids['lat'].max():.4f}")
    print(f"  lng range: {grids['lng'].min():.4f} ~ {grids['lng'].max():.4f}")

    return grids


def build_dong_to_grids(grid_map: pd.DataFrame) -> dict:
    """행정동코드 → 해당 행정동 내 격자 리스트 매핑

    Returns: {adm_cd_8digit: [{grid_id, lat, lng}, ...]}
    """
    dong_grids = {}
    for _, row in grid_map.iterrows():
        key = str(row["adm_cd"])[:8]
        if key not in dong_grids:
            dong_grids[key] = []
        dong_grids[key].append({
            "grid_id": row["ID"],
            "lat": row["lat"],
            "lng": row["lng"],
        })
    return dong_grids


def build_residence_mapping(kt_od: pd.DataFrame, dong_grids: dict) -> dict:
    """소비지 행정동 → 거주지 행정동 후보 매핑 (유동인구 가중)

    KT OD 데이터에서 dest(소비/관측지)별로
    어떤 origin(거주지)에서 유입되는지를 가중 확률로 매핑.

    Returns: {consumption_dong_8: [(residence_dong_8, weight), ...]}
    """
    residence_map = {}

    for dest, group in kt_od.groupby("dest_adm_cd"):
        dest_8 = str(dest)[:8]
        candidates = []
        for _, row in group.iterrows():
            origin_8 = str(row["origin_adm_cd"])[:8]
            # 격자 데이터가 있는 거주지만 포함
            if origin_8 in dong_grids:
                weight = row["weekday_pop"] + row["weekend_pop"]
                candidates.append((origin_8, weight))
        if candidates:
            residence_map[dest_8] = candidates

    print(f"[residence_map] {len(residence_map)} consumption dongs with residence mapping")
    return residence_map


def _pick_grid(grids: list[dict], rng) -> dict:
    """격자 리스트에서 하나를 랜덤 선택"""
    return grids[rng.integers(0, len(grids))]


def assign_coordinates_to_agents(
    agents: list[dict],
    dong_grids: dict,
    kt_od: pd.DataFrame = None,
    seed: int = 42,
) -> list[dict]:
    """에이전트에 거주지(home) + 소비지(work) 좌표를 분리 할당

    1. work_lat/lng: B079 adm_cd(소비 행정동)의 격자에서 할당
    2. home_lat/lng: KT OD 데이터로 거주지 행정동 추정 → 해당 격자 할당
    3. current_lat/lng: 세그먼트에 따라 초기 위치 설정
       - commuter/evening_visitor → 소비지(직장/방문지)
       - resident/weekend_visitor → 거주지(집)
    """
    rng = np.random.default_rng(seed)

    # 전체 격자의 평균 좌표 (fallback)
    all_lats = []
    all_lngs = []
    for grids in dong_grids.values():
        for g in grids:
            all_lats.append(g["lat"])
            all_lngs.append(g["lng"])

    fallback_lat = np.mean(all_lats) if all_lats else 37.5665
    fallback_lng = np.mean(all_lngs) if all_lngs else 126.9780

    # 거주지 매핑 구축
    residence_map = {}
    if kt_od is not None and len(kt_od) > 0:
        residence_map = build_residence_mapping(kt_od, dong_grids)

    work_matched = 0
    home_matched = 0

    for agent in agents:
        consumption_dong_8 = str(agent["adm_cd"])[:8]

        # ── 1) 소비지(work) 좌표 ──
        work_grids = dong_grids.get(consumption_dong_8, [])
        if work_grids:
            chosen = _pick_grid(work_grids, rng)
            agent["work_lat"] = chosen["lat"] + rng.normal(0, 0.001)
            agent["work_lng"] = chosen["lng"] + rng.normal(0, 0.001)
            agent["work_grid"] = chosen["grid_id"]
            work_matched += 1
        else:
            agent["work_lat"] = fallback_lat + rng.normal(0, 0.02)
            agent["work_lng"] = fallback_lng + rng.normal(0, 0.02)
            agent["work_grid"] = "unknown"

        # ── 2) 거주지(home) 좌표 ──
        candidates = residence_map.get(consumption_dong_8, [])

        if candidates:
            # KT OD 기반: 유동인구 가중 랜덤으로 거주지 행정동 선택
            dongs = [c[0] for c in candidates]
            weights = np.array([c[1] for c in candidates], dtype=float)
            total_w = weights.sum()
            if total_w > 0:
                weights /= total_w
            else:
                weights = np.ones(len(dongs)) / len(dongs)

            chosen_idx = rng.choice(len(dongs), p=weights)
            chosen_dong = dongs[chosen_idx]

            home_grids = dong_grids.get(chosen_dong, [])
            if home_grids:
                chosen = _pick_grid(home_grids, rng)
                agent["home_lat"] = chosen["lat"] + rng.normal(0, 0.001)
                agent["home_lng"] = chosen["lng"] + rng.normal(0, 0.001)
                agent["home_grid"] = chosen["grid_id"]
                agent["home_adm_cd"] = chosen_dong
                home_matched += 1
            else:
                # 격자 없으면 소비지 근처로 fallback
                agent["home_lat"] = agent["work_lat"] + rng.normal(0, 0.005)
                agent["home_lng"] = agent["work_lng"] + rng.normal(0, 0.005)
                agent["home_grid"] = "estimated"
                agent["home_adm_cd"] = chosen_dong
        else:
            # KT OD 매핑 없음 → 세그먼트별 추정
            if agent["segment"] == "resident":
                # 거주민: 집 ≈ 소비지 (같은 동네)
                agent["home_lat"] = agent["work_lat"] + rng.normal(0, 0.002)
                agent["home_lng"] = agent["work_lng"] + rng.normal(0, 0.002)
            else:
                # 출퇴근/방문자: 소비지에서 약간 떨어진 곳
                agent["home_lat"] = agent["work_lat"] + rng.normal(0, 0.01)
                agent["home_lng"] = agent["work_lng"] + rng.normal(0, 0.01)
            agent["home_grid"] = "estimated"
            agent["home_adm_cd"] = consumption_dong_8

        # ── 3) 현재 위치: 세그먼트별 초기 위치 ──
        if agent["segment"] in ("commuter", "evening_visitor"):
            # 직장인/저녁방문: 소비지에서 시작
            agent["current_lat"] = agent["work_lat"]
            agent["current_lng"] = agent["work_lng"]
        else:
            # 거주민/주말방문: 집에서 시작
            agent["current_lat"] = agent["home_lat"]
            agent["current_lng"] = agent["home_lng"]

    print(f"[coordinates] work matched: {work_matched}/{len(agents)}")
    print(f"[coordinates] home matched (KT OD): {home_matched}/{len(agents)}")
    return agents


if __name__ == "__main__":
    from etl_transform import build_kt_od

    grid_map = build_grid_map()
    dong_grids = build_dong_to_grids(grid_map)
    kt_od = build_kt_od()

    print(f"\n{len(dong_grids)} dongs with grid cells")
    print(f"Sample: {list(dong_grids.items())[:2]}")

    # 좌표 변환 테스트
    lat, lng = tm_to_wgs84(199637.3, 553456.6)
    print(f"\nTest: TM(199637, 553457) -> lat={lat:.6f}, lng={lng:.6f}")
