"""
ETL Step 2: 데이터 변환 → 에이전트 프로필 + 환경 데이터 생성

1. 업종 코드 통합 (ss코드 ↔ sb코드 → 대분류 기준 통합)
2. 소비자 세그먼트 프로토타입 생성 (B079 행정동 × 성별 × 연령대)
3. 소비 시간 패턴 추출 (B079 시간대별)
4. 거주지→소비지 OD 매트릭스 (B079 유입지별 + KT 거주지별)
5. 지역 환경 상태 (B069 상권지수 → 4분류 파생)
6. 에이전트 프로필 생성 (세그먼트 × 개인차)
"""
import numpy as np
import pandas as pd
from etl_loader import (
    load_b079_gender_age, load_b079_inflow, load_b079_time,
    load_b063_block_gender_age,
    load_industry_code_ss, load_industry_code_sb,
    load_kt_time_age, load_kt_residence,
    load_district_index,
)


# ═══════════════════════════════════════════
# 1. 업종 코드 통합
# ═══════════════════════════════════════════

def build_industry_map() -> dict:
    """ss코드/sb코드 → 대분류 매핑 딕셔너리 생성"""
    ss = load_industry_code_ss()
    sb = load_industry_code_sb()

    industry_map = {}

    # ss코드 → 대분류
    for _, row in ss.iterrows():
        industry_map[row["upjong_cd"].upper()] = {
            "class1": row["class1"],
            "class2": row["class2"],
            "class3": row["class3"],
        }

    # sb코드 → 대분류 (sb코드는 class1 기준으로 ss코드 대분류와 매핑)
    for _, row in sb.iterrows():
        industry_map[row["sb_upjong_cd"].upper()] = {
            "class1": row["class1"],
            "class2": row["class2"],
            "class3": row["class3"],
        }

    return industry_map


# ═══════════════════════════════════════════
# 2. 소비자 세그먼트 프로토타입 생성
# ═══════════════════════════════════════════

def build_consumer_segments() -> pd.DataFrame:
    """B079 성별×연령대×행정동 → 소비 세그먼트 프로토타입

    각 (행정동, 성별, 연령대) 조합이 하나의 세그먼트.
    세그먼트별로: 총 소비금액, 소비건수, 주요 업종 Top3을 추출.
    """
    df = load_b079_gender_age()

    # 행정동 × 성별 × 연령대별 집계
    # card_amount/card_count는 해당 그룹의 전체 합계(집단 단위)
    # → 1건당 평균 소비금액을 계산하여 개인 소비 추정에 사용
    seg = (
        df.groupby(["adm_cd", "gender", "age_group"])
        .agg(
            total_amount=("card_amount", "sum"),
            total_count=("card_count", "sum"),
        )
        .reset_index()
    )
    # 1건당 평균 소비금액 (개인 소비 추정의 기반)
    seg["avg_per_transaction"] = seg["total_amount"] / seg["total_count"].clip(lower=1)

    # 세그먼트별 주요 업종 Top3
    top_industries = (
        df.groupby(["adm_cd", "gender", "age_group", "industry_major"])
        ["card_amount"].sum()
        .reset_index()
        .sort_values("card_amount", ascending=False)
        .groupby(["adm_cd", "gender", "age_group"])
        .head(3)
        .groupby(["adm_cd", "gender", "age_group"])
        ["industry_major"].apply(list)
        .reset_index()
        .rename(columns={"industry_major": "top_industries"})
    )

    seg = seg.merge(top_industries, on=["adm_cd", "gender", "age_group"], how="left")

    print(f"[segments] {len(seg)} consumer segment prototypes created")
    return seg


# ═══════════════════════════════════════════
# 3. 소비 시간 패턴 추출
# ═══════════════════════════════════════════

def build_time_patterns() -> pd.DataFrame:
    """B079 시간대별 → 행정동별 피크 시간대, 시간대 분포

    각 행정동의 소비가 집중되는 시간대를 파악.
    """
    df = load_b079_time()

    # 행정동별 시간대 분포
    time_dist = (
        df.groupby(["adm_cd", "time_slot"])
        ["card_amount"].sum()
        .reset_index()
    )

    # 행정동별 피크 시간대
    peak = (
        time_dist
        .sort_values("card_amount", ascending=False)
        .groupby("adm_cd")
        .first()
        .reset_index()
        .rename(columns={"time_slot": "peak_time", "card_amount": "peak_amount"})
    )

    # 행정동별 시간대 비율 (피벗)
    total_by_adm = time_dist.groupby("adm_cd")["card_amount"].sum().reset_index()
    total_by_adm.columns = ["adm_cd", "total_amount"]
    time_dist = time_dist.merge(total_by_adm, on="adm_cd")
    time_dist["time_ratio"] = time_dist["card_amount"] / time_dist["total_amount"]

    print(f"[time_patterns] {len(peak)} districts with peak time info")
    return peak, time_dist


# ═══════════════════════════════════════════
# 4. 거주지→소비지 OD 매트릭스
# ═══════════════════════════════════════════

def build_od_matrix() -> pd.DataFrame:
    """B079 유입지별 → 유입지(시도/시군구) × 소비지(행정동) OD

    어디서 와서 어디서 소비하는지의 OD(Origin-Destination) 매트릭스.
    """
    df = load_b079_inflow()

    # 유입지 라벨 생성 (시도 + 시군구)
    df["origin"] = df["inflow_sido"] + " " + df["inflow_sgg"]
    df["origin"] = df["origin"].str.strip()

    od = (
        df.groupby(["origin", "adm_cd"])
        .agg(
            total_amount=("card_amount", "sum"),
            total_count=("card_count", "sum"),
        )
        .reset_index()
    )

    print(f"[od_matrix] {len(od)} origin-destination pairs")
    return od


def build_kt_od() -> pd.DataFrame:
    """KT 거주지별 유동인구 → 거주지코드 × 행정동 × 평일/주말 유동인구

    B079 OD와 교차 검증 가능.
    """
    df = load_kt_residence()

    kt_od = (
        df.groupby(["INFLOW_ADMIN_CD", "ADMI_CD"])
        .agg(
            weekday_pop=("WKDY_FLPOP_CNT", "sum"),
            weekend_pop=("WKND_FLPOP_CNT", "sum"),
        )
        .reset_index()
        .rename(columns={
            "INFLOW_ADMIN_CD": "origin_adm_cd",
            "ADMI_CD": "dest_adm_cd",
        })
    )
    kt_od["origin_adm_cd"] = kt_od["origin_adm_cd"].astype(str)
    kt_od["dest_adm_cd"] = kt_od["dest_adm_cd"].astype(str)

    # 주중/주말 비율
    kt_od["weekday_ratio"] = kt_od["weekday_pop"] / (kt_od["weekday_pop"] + kt_od["weekend_pop"])
    kt_od["weekday_ratio"] = kt_od["weekday_ratio"].fillna(0.5)

    print(f"[kt_od] {len(kt_od)} KT origin-destination pairs")
    return kt_od


# ═══════════════════════════════════════════
# 5. 지역 환경 상태 (B069 → 4분류 파생)
# ═══════════════════════════════════════════

def build_district_profiles() -> pd.DataFrame:
    """B069 상권지수 → 행정동별 환경 프로필 + 4분류 파생

    5개 지수(매출/인프라/점포/인구/집객)에서 종합 점수 산출 후,
    중앙값 기준으로 LL/LH/HL/HH 4분류로 파생.

    - 축1: 기존 상권 안정성 (STORE + SALES 평균) → High/Low
    - 축2: 성장/유입 동력 (POPULATION + DEPOSIT 평균) → High/Low
    """
    df = load_district_index()
    df["ADSTRD_CD"] = df["ADSTRD_CD"].astype(str)

    # 가장 최근 데이터만 사용
    latest_date = df["DATE"].max()
    df = df[df["DATE"] == latest_date].copy()

    # 축1: 기존 상권 안정성 (점포 + 매출)
    df["stability"] = (df["STORE"] + df["SALES"]) / 2

    # 축2: 성장/유입 동력 (인구 + 집객)
    df["growth"] = (df["POPULATION"] + df["DEPOSIT"]) / 2

    # 중앙값 기준 4분류
    stability_med = df["stability"].median()
    growth_med = df["growth"].median()

    def classify(row):
        s = "H" if row["stability"] >= stability_med else "L"
        g = "H" if row["growth"] >= growth_med else "L"
        return s + g  # HH, HL, LH, LL

    df["district_type"] = df.apply(classify, axis=1)

    result = df[["ADSTRD_CD", "SALES", "INFRASTRUCTURE", "STORE",
                 "POPULATION", "DEPOSIT", "stability", "growth", "district_type"]].copy()
    result = result.rename(columns={"ADSTRD_CD": "adm_cd"})

    type_counts = result["district_type"].value_counts()
    print(f"[district_profiles] {len(result)} districts | types: {type_counts.to_dict()}")

    return result


# ═══════════════════════════════════════════
# 6. 페르소나 정의 (라이프스타일, 관심사, 일과표)
# ═══════════════════════════════════════════

# 라이프스타일 유형별 정의
LIFESTYLES = {
    "카페러버": {
        "industries_bonus": ["카페", "베이커리"],
        "interests": ["디저트", "카페투어", "인스타"],
    },
    "미식가": {
        "industries_bonus": ["한식", "일식", "중식", "주류"],
        "interests": ["맛집탐방", "미슐랭", "와인"],
    },
    "가성비추구": {
        "industries_bonus": ["패스트푸드", "분식", "편의점"],
        "interests": ["세일", "쿠폰", "가성비"],
    },
    "건강지향": {
        "industries_bonus": ["슈퍼마켓", "한식", "의료"],
        "interests": ["운동", "건강식", "요가"],
    },
    "쇼핑중독": {
        "industries_bonus": ["패션잡화", "전자제품"],
        "interests": ["패션", "신상", "브랜드"],
    },
    "문화예술": {
        "industries_bonus": ["문화여가", "카페", "숙박"],
        "interests": ["전시", "공연", "K-POP"],
    },
    "집순이": {
        "industries_bonus": ["편의점", "슈퍼마켓"],
        "interests": ["넷플릭스", "배달", "집콕"],
    },
    "야식파": {
        "industries_bonus": ["치킨", "주류", "피자"],
        "interests": ["야식", "술모임", "치맥"],
    },
}

# 연령/성별 → 라이프스타일 분포 (확률)
LIFESTYLE_DISTRIBUTIONS = {
    ("20_29세", "남"): {"카페러버": 0.15, "미식가": 0.10, "가성비추구": 0.25, "문화예술": 0.15, "야식파": 0.20, "쇼핑중독": 0.05, "집순이": 0.10},
    ("20_29세", "여"): {"카페러버": 0.30, "미식가": 0.10, "쇼핑중독": 0.20, "문화예술": 0.15, "가성비추구": 0.10, "건강지향": 0.10, "집순이": 0.05},
    ("30_39세", "남"): {"미식가": 0.20, "카페러버": 0.10, "야식파": 0.15, "가성비추구": 0.15, "건강지향": 0.15, "문화예술": 0.10, "집순이": 0.15},
    ("30_39세", "여"): {"카페러버": 0.25, "미식가": 0.15, "쇼핑중독": 0.15, "건강지향": 0.20, "문화예술": 0.15, "가성비추구": 0.05, "집순이": 0.05},
    ("40_49세", "남"): {"미식가": 0.25, "건강지향": 0.20, "야식파": 0.15, "가성비추구": 0.15, "집순이": 0.10, "문화예술": 0.10, "카페러버": 0.05},
    ("40_49세", "여"): {"건강지향": 0.25, "카페러버": 0.15, "미식가": 0.15, "쇼핑중독": 0.15, "문화예술": 0.15, "가성비추구": 0.10, "집순이": 0.05},
    ("50_59세", "남"): {"건강지향": 0.30, "미식가": 0.20, "야식파": 0.10, "집순이": 0.15, "가성비추구": 0.15, "문화예술": 0.05, "카페러버": 0.05},
    ("50_59세", "여"): {"건강지향": 0.30, "미식가": 0.15, "카페러버": 0.15, "문화예술": 0.10, "쇼핑중독": 0.10, "가성비추구": 0.10, "집순이": 0.10},
    ("60_69세", "남"): {"건강지향": 0.35, "집순이": 0.20, "미식가": 0.15, "가성비추구": 0.15, "야식파": 0.05, "문화예술": 0.05, "카페러버": 0.05},
    ("60_69세", "여"): {"건강지향": 0.35, "집순이": 0.15, "카페러버": 0.15, "미식가": 0.10, "가성비추구": 0.15, "문화예술": 0.05, "쇼핑중독": 0.05},
}

# 세그먼트별 일과표 템플릿
SCHEDULE_TEMPLATES = {
    "commuter": {
        "wake_up": (6, 8), "commute_start": (7, 9), "lunch_time": (11, 13),
        "commute_end": (17, 19), "dinner_time": (18, 20), "bed_time": (22, 24),
    },
    "resident": {
        "wake_up": (7, 10), "commute_start": (9, 11), "lunch_time": (11, 13),
        "commute_end": (15, 17), "dinner_time": (17, 19), "bed_time": (21, 23),
    },
    "weekend_visitor": {
        "wake_up": (8, 11), "commute_start": (10, 13), "lunch_time": (12, 14),
        "commute_end": (18, 21), "dinner_time": (18, 21), "bed_time": (23, 25),
    },
    "evening_visitor": {
        "wake_up": (8, 10), "commute_start": (9, 11), "lunch_time": (12, 14),
        "commute_end": (17, 19), "dinner_time": (19, 22), "bed_time": (23, 26),
    },
}

# 추가 관심사 풀 (라이프스타일 + 연령별로 조합)
EXTRA_INTERESTS = [
    "여행", "캠핑", "반려동물", "독서", "게임", "자기계발",
    "사진", "드라마", "SNS", "음악", "스포츠", "요리",
    "인테리어", "부동산", "재테크", "육아",
]


def _choose_lifestyle(age_group, gender, rng):
    """연령/성별 기반 라이프스타일 랜덤 선택."""
    key = (age_group, gender)
    dist = LIFESTYLE_DISTRIBUTIONS.get(key)
    if not dist:
        # 기본값: 균등 분포
        names = list(LIFESTYLES.keys())
        return rng.choice(names)
    names = list(dist.keys())
    probs = np.array([dist[n] for n in names])
    probs /= probs.sum()
    return names[rng.choice(len(names), p=probs)]


def _generate_schedule(segment_type, rng):
    """세그먼트별 일과표 생성 (시간대에 랜덤 변동 추가)."""
    template = SCHEDULE_TEMPLATES.get(segment_type, SCHEDULE_TEMPLATES["commuter"])
    schedule = {}
    for key, (lo, hi) in template.items():
        schedule[key] = int(rng.integers(lo, hi + 1)) % 24
    return schedule


def _generate_interests(lifestyle, age_group, rng):
    """라이프스타일 + 연령 기반 관심사 리스트 생성."""
    base = list(LIFESTYLES.get(lifestyle, {}).get("interests", []))
    # 추가 관심사 1~3개
    n_extra = rng.integers(1, 4)
    extra = rng.choice(EXTRA_INTERESTS, size=min(n_extra, len(EXTRA_INTERESTS)), replace=False)
    all_interests = list(set(base + list(extra)))
    return all_interests[:6]  # 최대 6개


def _generate_persona_desc(agent_partial, lifestyle, rng):
    """에이전트 프로필 기반 1줄 페르소나 설명 생성."""
    seg_kr = {
        "commuter": "직장인", "resident": "동네 주민",
        "weekend_visitor": "주말 나들이파", "evening_visitor": "저녁 외출파",
    }
    seg = seg_kr.get(agent_partial["segment"], "소비자")
    age = agent_partial["age_group"]
    gender = agent_partial["gender"]
    ls = lifestyle

    # 다양한 설명 템플릿
    templates = [
        f"{age} {gender}, {seg}. {ls} 성향으로 퇴근 후 동네를 즐기는 편.",
        f"서울 사는 {age} {gender} {seg}. 주로 {ls} 스타일의 소비를 즐김.",
        f"{age} {gender}. 평소 {ls} 성향이 강하고, {seg}으로 바쁜 일상을 보냄.",
        f"{seg}인 {age} {gender}. {ls} 라이프스타일로 일상의 소확행을 찾는 중.",
        f"{age} {gender} {seg}. {ls} 취향으로 가끔 새로운 곳도 도전해봄.",
    ]
    return templates[rng.integers(0, len(templates))]


# ═══════════════════════════════════════════
# 7. 에이전트 프로필 생성
# ═══════════════════════════════════════════

def generate_agent_profiles(
    segments: pd.DataFrame,
    time_patterns: pd.DataFrame,
    od_matrix: pd.DataFrame,
    district_profiles: pd.DataFrame,
    total_agents: int = 300,
    seed: int = 42,
) -> list[dict]:
    """세그먼트 프로토타입에서 개별 에이전트 프로필을 샘플링

    1. 세그먼트별 소비금액 비율로 에이전트 수 배분
    2. 세그먼트 내에서 정규분포로 개인차 부여
    3. 행동 성향 파라미터 (가격민감도, 충성도 등) 부여
    4. 라이프스타일, 일과표, 관심사, SNS 활동도 부여
    """
    rng = np.random.default_rng(seed)

    if len(segments) == 0:
        print("[WARN] No segments available, generating minimal profiles")
        return []

    # 세그먼트별 에이전트 수 배분 (소비금액 비율)
    total = segments["total_amount"].sum()
    if total == 0:
        segments["agent_share"] = 1 / len(segments)
    else:
        segments["agent_share"] = segments["total_amount"] / total

    segments["n_agents"] = (segments["agent_share"] * total_agents).astype(int)
    # 부족분 보정
    deficit = total_agents - segments["n_agents"].sum()
    if deficit > 0:
        top_idx = segments.nlargest(deficit, "total_amount").index
        segments.loc[top_idx, "n_agents"] += 1

    # 시간대 패턴 매핑 (peak_time) — 행정동 앞 10자리 기준
    peak_map = {}
    if "peak_time" in time_patterns.columns:
        for _, row in time_patterns.iterrows():
            peak_map[str(row["adm_cd"])] = int(row["peak_time"])

    # 지역 타입 매핑 (B079: 10자리, B069: 8자리 → 앞 8자리로 매칭)
    district_map = district_profiles.set_index("adm_cd")["district_type"].to_dict() if len(district_profiles) > 0 else {}

    # OD에서 대안 지역 추출
    od_alternatives = {}
    if len(od_matrix) > 0:
        for adm_cd, group in od_matrix.groupby("adm_cd"):
            alternatives = group.nlargest(3, "total_amount")["origin"].tolist()
            od_alternatives[adm_cd] = alternatives

    # KT OD에서 주중/주말 비율 매핑 (행정동 8자리 기준)
    weekday_ratio_map = {}
    try:
        kt_od_data = build_kt_od()
        for _, row in kt_od_data.iterrows():
            key = str(row["dest_adm_cd"])[:8]
            if key not in weekday_ratio_map:
                weekday_ratio_map[key] = []
            weekday_ratio_map[key].append(row["weekday_ratio"])
        weekday_ratio_map = {k: np.mean(v) for k, v in weekday_ratio_map.items()}
    except Exception:
        pass

    # 에이전트 생성
    agents = []
    agent_id = 0

    for _, seg in segments.iterrows():
        n = int(seg["n_agents"])
        if n == 0:
            continue

        for _ in range(n):
            # ── 1) 세그먼트 분류 (먼저 결정) ──
            peak = peak_map.get(seg["adm_cd"], 12)
            wd_ratio = weekday_ratio_map.get(str(seg["adm_cd"])[:8], 0.6)
            age_str = str(seg["age_group"])

            if "20" in age_str and wd_ratio < 0.5:
                segment_type = "weekend_visitor"
            elif ("20" in age_str or "30" in age_str) and 11 <= peak <= 14:
                segment_type = "commuter"
            elif "60" in age_str or "70" in age_str:
                segment_type = "resident"
            elif peak >= 18:
                segment_type = "evening_visitor"
            elif wd_ratio > 0.7:
                segment_type = "commuter"
            elif wd_ratio < 0.4:
                segment_type = "weekend_visitor"
            else:
                roll = rng.random()
                if roll < 0.4:
                    segment_type = "commuter"
                elif roll < 0.65:
                    segment_type = "weekend_visitor"
                elif roll < 0.85:
                    segment_type = "resident"
                else:
                    segment_type = "evening_visitor"

            # ── 2) 개인 월 소비금액 추정 ──
            avg_txn = seg.get("avg_per_transaction", 15000)
            avg_txn = min(avg_txn, 50000)

            if segment_type == "commuter":
                monthly_visits = rng.integers(15, 30)
            elif segment_type == "weekend_visitor":
                monthly_visits = rng.integers(4, 12)
            elif segment_type == "resident":
                monthly_visits = rng.integers(10, 25)
            else:
                monthly_visits = rng.integers(8, 16)

            monthly_spending = max(
                30000,
                int(avg_txn * monthly_visits * rng.normal(1.0, 0.15))
            )

            # ── 3) 행동 성향 파라미터 ──
            age = seg["age_group"]
            if "20" in str(age):
                price_sens, loyalty, trend_sens, social_inf = 0.7, 0.3, 0.8, 0.7
            elif "30" in str(age):
                price_sens, loyalty, trend_sens, social_inf = 0.6, 0.5, 0.5, 0.5
            elif "40" in str(age):
                price_sens, loyalty, trend_sens, social_inf = 0.5, 0.7, 0.3, 0.3
            elif "50" in str(age):
                price_sens, loyalty, trend_sens, social_inf = 0.4, 0.8, 0.2, 0.2
            else:
                price_sens, loyalty, trend_sens, social_inf = 0.5, 0.6, 0.3, 0.3

            price_sens = np.clip(price_sens + rng.normal(0, 0.1), 0, 1)
            loyalty = np.clip(loyalty + rng.normal(0, 0.1), 0, 1)
            trend_sens = np.clip(trend_sens + rng.normal(0, 0.1), 0, 1)
            social_inf = np.clip(social_inf + rng.normal(0, 0.1), 0, 1)

            # ── 4) 페르소나 (라이프스타일, 일과표, 관심사) ──
            lifestyle = _choose_lifestyle(str(age), seg["gender"], rng)
            daily_schedule = _generate_schedule(segment_type, rng)
            interests = _generate_interests(lifestyle, str(age), rng)

            # SNS 활동도 (연령 기반)
            if "20" in str(age):
                sns_activity = round(float(np.clip(rng.normal(0.75, 0.12), 0.3, 1.0)), 2)
            elif "30" in str(age):
                sns_activity = round(float(np.clip(rng.normal(0.55, 0.15), 0.1, 0.9)), 2)
            elif "40" in str(age):
                sns_activity = round(float(np.clip(rng.normal(0.35, 0.12), 0.05, 0.7)), 2)
            else:
                sns_activity = round(float(np.clip(rng.normal(0.2, 0.1), 0.0, 0.5)), 2)

            agent = {
                "agent_id": f"consumer_{agent_id:04d}",
                "segment": segment_type,
                "gender": seg["gender"],
                "age_group": seg["age_group"],
                "adm_cd": seg["adm_cd"],
                "district_type": district_map.get(str(seg["adm_cd"])[:8], "unknown"),
                "monthly_spending": monthly_spending,
                "top_industries": seg.get("top_industries", []),
                "peak_time": peak,
                "alternative_areas": od_alternatives.get(seg["adm_cd"], []),
                "price_sensitivity": round(float(price_sens), 2),
                "loyalty": round(float(loyalty), 2),
                "trend_sensitivity": round(float(trend_sens), 2),
                "social_influence_weight": round(float(social_inf), 2),
                "online_preference": round(float(rng.uniform(0.1, 0.4)), 2),
                # ── 새 페르소나 필드 ──
                "lifestyle": lifestyle,
                "daily_schedule": daily_schedule,
                "interests": interests,
                "sns_activity": sns_activity,
                "persona_description": "",  # 아래에서 일괄 생성
            }

            # persona_description 생성 (agent 완성 후)
            agent["persona_description"] = _generate_persona_desc(agent, lifestyle, rng)

            agents.append(agent)
            agent_id += 1

    print(f"[agents] {len(agents)} consumer agents generated")

    # 세그먼트 분포
    seg_counts = pd.Series([a["segment"] for a in agents]).value_counts()
    print(f"[agents] segment distribution: {seg_counts.to_dict()}")

    # 라이프스타일 분포
    ls_counts = pd.Series([a["lifestyle"] for a in agents]).value_counts()
    print(f"[agents] lifestyle distribution: {ls_counts.to_dict()}")

    return agents


# ═══════════════════════════════════════════
# 전체 ETL 실행
# ═══════════════════════════════════════════

def run_etl(total_agents: int = 300) -> dict:
    """전체 ETL 파이프라인 실행 → 시뮬레이션에 필요한 모든 데이터 반환"""

    print("=" * 60)
    print("ETL Pipeline Start")
    print("=" * 60)

    # 1. 업종 코드 통합
    print("\n[Step 1] Industry code mapping...")
    industry_map = build_industry_map()
    print(f"  -> {len(industry_map)} industry codes mapped")

    # 2. 소비자 세그먼트 생성
    print("\n[Step 2] Consumer segments...")
    segments = build_consumer_segments()

    # 3. 시간 패턴
    print("\n[Step 3] Time patterns...")
    peak_times, time_dist = build_time_patterns()

    # 4. OD 매트릭스
    print("\n[Step 4] OD matrix...")
    od = build_od_matrix()
    kt_od = build_kt_od()

    # 5. 지역 환경 프로필
    print("\n[Step 5] District profiles...")
    districts = build_district_profiles()

    # 6. 에이전트 프로필 생성
    print("\n[Step 6] Agent profiles...")
    agents = generate_agent_profiles(
        segments=segments,
        time_patterns=peak_times,
        od_matrix=od,
        district_profiles=districts,
        total_agents=total_agents,
    )

    print("\n" + "=" * 60)
    print(f"ETL Complete: {len(agents)} agents, {len(districts)} districts")
    print("=" * 60)

    return {
        "agents": agents,
        "segments": segments,
        "time_patterns": {"peak": peak_times, "distribution": time_dist},
        "od_matrix": od,
        "kt_od": kt_od,
        "district_profiles": districts,
        "industry_map": industry_map,
    }


if __name__ == "__main__":
    result = run_etl(total_agents=300)

    # 샘플 에이전트 출력
    import json
    if result["agents"]:
        import sys, io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        print("\n--- Sample Agent ---")
        print(json.dumps(result["agents"][0], ensure_ascii=False, indent=2))
        print(f"\n--- Agent 1 ---")
        print(json.dumps(result["agents"][min(1, len(result["agents"])-1)], ensure_ascii=False, indent=2))
