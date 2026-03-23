"""
합성 프로토타입 데이터 생성

풀 데이터 없이 시뮬레이션을 돌리기 위한 합성 데이터.
서울 전체 ~420개 행정동, ~2100개 KT 격자를 커버.
실행 후 config.py에서 USE_SYNTHETIC = True로 전환.
"""
import numpy as np
import pandas as pd
from pathlib import Path

SYNTHETIC_DIR = Path(__file__).parent.parent / "data" / "synthetic"
SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42
rng = np.random.default_rng(SEED)

# ═══════════════════════════════════════════
# 서울 25개 구 정의
# (구코드5자리, 구이름, 행정동수, TM중심X, TM중심Y)
# ═══════════════════════════════════════════

GU_DEFS = [
    ("11110", "종로구",   17, 198500, 552500),
    ("11140", "중구",     15, 199500, 551500),
    ("11170", "용산구",   16, 198500, 550000),
    ("11200", "성동구",   17, 201000, 551500),
    ("11215", "광진구",   15, 203000, 551500),
    ("11230", "동대문구", 14, 201000, 553000),
    ("11260", "중랑구",   16, 203500, 553500),
    ("11290", "성북구",   20, 200000, 554000),
    ("11305", "강북구",   13, 199500, 556000),
    ("11320", "도봉구",   14, 200500, 558000),
    ("11350", "노원구",   19, 202500, 557500),
    ("11380", "은평구",   16, 195000, 554500),
    ("11410", "서대문구", 14, 196500, 553000),
    ("11440", "마포구",   16, 195500, 551500),
    ("11470", "양천구",   18, 191500, 549000),
    ("11500", "강서구",   20, 189500, 550500),
    ("11530", "구로구",   15, 192000, 547500),
    ("11545", "금천구",   10, 193500, 546000),
    ("11560", "영등포구", 18, 194500, 549000),
    ("11590", "동작구",   15, 196000, 548000),
    ("11620", "관악구",   21, 196500, 546500),
    ("11650", "서초구",   18, 199500, 547000),
    ("11680", "강남구",   22, 202000, 548000),
    ("11710", "송파구",   20, 205000, 549000),
    ("11740", "강동구",   18, 206500, 550500),
]

GENDERS = ["남", "여"]
AGE_GROUPS = ["10_19세", "20_29세", "30_39세", "40_49세", "50_59세", "60_69세"]
INDUSTRIES = [
    "한식", "중식", "일식", "카페", "편의점", "패스트푸드",
    "패션잡화", "의료", "주유", "슈퍼마켓", "미용", "문화여가",
    "주류", "베이커리", "치킨", "피자", "분식", "전자제품", "숙박", "교육",
]

# 업종별 건당 평균 소비금액 (원)
INDUSTRY_AVG_SPEND = {
    "한식": 12000, "중식": 15000, "일식": 22000, "카페": 6000,
    "편의점": 5000, "패스트푸드": 8000, "패션잡화": 35000, "의료": 25000,
    "주유": 60000, "슈퍼마켓": 18000, "미용": 20000, "문화여가": 15000,
    "주류": 30000, "베이커리": 7000, "치킨": 20000, "피자": 25000,
    "분식": 7000, "전자제품": 50000, "숙박": 80000, "교육": 30000,
}

INFLOW_ORIGINS = [
    ("서울특별시", "종로구"), ("서울특별시", "중구"), ("서울특별시", "강남구"),
    ("서울특별시", "송파구"), ("서울특별시", "마포구"), ("서울특별시", "영등포구"),
    ("서울특별시", "관악구"), ("서울특별시", "노원구"), ("서울특별시", "강서구"),
    ("경기도", "성남시"), ("경기도", "고양시"), ("경기도", "수원시"),
    ("경기도", "부천시"), ("경기도", "안양시"), ("경기도", "용인시"),
    ("인천광역시", "남동구"), ("인천광역시", "부평구"),
]


# ═══════════════════════════════════════════
# 행정동 코드 생성
# ═══════════════════════════════════════════

def generate_dong_codes():
    """서울 25개구 × N개동 = ~420개 행정동 코드 생성"""
    dongs = []
    for gu_code, gu_name, n_dongs, cx, cy in GU_DEFS:
        for i in range(n_dongs):
            dong_3 = f"{(i + 1) * 25:03d}"       # 025, 050, 075, ...
            code_8 = f"{gu_code}{dong_3}"          # 11110025
            code_10 = f"{code_8}00"                # 1111002500
            # TM 좌표 (구 중심 ± 랜덤)
            x = cx + rng.normal(0, 800)
            y = cy + rng.normal(0, 800)
            dongs.append({
                "code_10": code_10,
                "code_8": code_8,
                "gu_name": gu_name,
                "tm_x": x,
                "tm_y": y,
            })
    return dongs


# ═══════════════════════════════════════════
# B079: 카드소비 (행정동별) 생성
# ═══════════════════════════════════════════

def generate_b079_gender_age(dongs):
    """B079 성별연령대별 카드소비 — 9컬럼"""
    rows = []
    dates = ["20230101", "20230201", "20230301"]

    for dong in dongs:
        for date in dates:
            # 각 동에 5~8개 주요 업종
            n_industries = rng.integers(5, 9)
            selected = rng.choice(INDUSTRIES, size=n_industries, replace=False)

            for gender in GENDERS:
                for age in AGE_GROUPS:
                    for industry in selected:
                        avg = INDUSTRY_AVG_SPEND[industry]
                        count = int(rng.integers(5, 200))
                        amount = int(count * avg * rng.normal(1.0, 0.2))
                        amount = max(amount, 1000)
                        rows.append([
                            date, dong["code_10"], "내국인",
                            gender, age, industry,
                            amount, count, 0.0,
                        ])

    df = pd.DataFrame(rows, columns=[
        "기준일자", "행정동코드", "내외국인", "성별", "연령대",
        "업종대분류", "카드이용금액합계", "카드이용건수합계", "비율",
    ])
    path = SYNTHETIC_DIR / "7.서울시 내국인 성별 연령대별(행정동별).csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B079 gender_age] {len(df)} rows -> {path.name}")
    return df


def generate_b079_time(dongs):
    """B079 시간대별 카드소비 — 6컬럼"""
    rows = []
    dates = ["20230101", "20230201", "20230301"]

    # 시간대별 소비 가중치 (12시, 18시 피크)
    time_weights = {
        h: (3.0 if h in (12, 13) else
            2.5 if h in (18, 19, 20) else
            1.5 if h in (11, 14, 17, 21) else
            0.5 if h in (1, 2, 3, 4, 5) else 1.0)
        for h in range(1, 25)
    }

    for dong in dongs:
        for date in dates:
            top_industries = rng.choice(INDUSTRIES, size=3, replace=False)
            for time_slot in range(1, 25):
                for industry in top_industries:
                    w = time_weights[time_slot]
                    count = int(rng.poisson(20 * w))
                    if count == 0:
                        continue
                    avg = INDUSTRY_AVG_SPEND[industry]
                    amount = int(count * avg * rng.normal(1.0, 0.15))
                    rows.append([date, time_slot, dong["code_10"], industry, amount, count])

    df = pd.DataFrame(rows, columns=[
        "기준일자", "시간대", "행정동코드", "업종대분류",
        "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "2.서울시민의 일별 시간대별(행정동).csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B079 time] {len(df)} rows -> {path.name}")
    return df


def generate_b079_inflow(dongs):
    """B079 유입지별 카드소비 — 7컬럼"""
    rows = []
    dates = ["20230101", "20230201"]

    for dong in dongs:
        n_origins = rng.integers(2, 6)
        origins = [INFLOW_ORIGINS[i] for i in rng.choice(len(INFLOW_ORIGINS), size=n_origins, replace=False)]

        for date in dates:
            top_ind = rng.choice(INDUSTRIES, size=3, replace=False)
            for sido, sgg in origins:
                for industry in top_ind:
                    count = int(rng.integers(3, 80))
                    amount = int(count * INDUSTRY_AVG_SPEND[industry] * rng.normal(1.0, 0.2))
                    rows.append([date, dong["code_10"], sido, sgg, industry, amount, count])

    df = pd.DataFrame(rows, columns=[
        "기준일자", "행정동코드", "유입시도", "유입시군구",
        "업종대분류", "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B079 inflow] {len(df)} rows -> {path.name}")
    return df


# ═══════════════════════════════════════════
# KT 유동인구 생성
# ═══════════════════════════════════════════

def generate_kt_grids(dongs):
    """각 행정동에 3~7개 격자 생성 → 격자 리스트 반환"""
    grids = []
    grid_id = 1000

    for dong in dongs:
        n_grids = rng.integers(3, 8)
        for _ in range(n_grids):
            grids.append({
                "grid_id": grid_id,
                "adm_cd_8": dong["code_8"],
                "x": dong["tm_x"] + rng.normal(0, 300),
                "y": dong["tm_y"] + rng.normal(0, 300),
            })
            grid_id += 1

    return grids


def generate_kt_time_age(grids, dongs):
    """KT 시간대별 유동인구 — 38컬럼 (Korean(English) 컬럼명)"""
    rows = []
    age_cols_m = [f"M{a:02d}" for a in range(0, 80, 10)]  # M00~M70 (8)
    age_cols_f = [f"F{a:02d}" for a in range(0, 80, 10)]   # F00~F70 (8)

    # 상업 밀집 행정동 (강남, 중구, 종로 등)은 유동인구 높게
    commercial_gus = {"11110", "11140", "11680", "11650", "11560", "11440"}

    for grid in grids:
        gu_code = grid["adm_cd_8"][:5]
        is_commercial = gu_code in commercial_gus
        base_pop = 500 if is_commercial else 200

        for ym in ["202301", "202302", "202303"]:
            for dow in ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]:
                for ts in range(1, 25):
                    # 시간대별 배율
                    if ts in (12, 13):
                        mult = 2.5
                    elif ts in (18, 19, 20):
                        mult = 2.0
                    elif ts in (8, 9, 10):
                        mult = 1.8
                    elif ts in (1, 2, 3, 4, 5):
                        mult = 0.2
                    else:
                        mult = 1.0

                    # 주말 보정
                    if dow in ("토요일", "일요일"):
                        mult *= 0.7 if not is_commercial else 1.2

                    pop_scale = base_pop * mult

                    row = {
                        "grid_id": grid["grid_id"],
                        "x": grid["x"],
                        "y": grid["y"],
                        "dow": dow,
                        "ts": ts,
                    }
                    total = 0
                    for col in age_cols_m + age_cols_f:
                        val = max(0, int(rng.poisson(pop_scale / 16)))
                        row[col] = val
                        total += val
                    row["total"] = total
                    row["adm_cd"] = grid["adm_cd_8"]
                    row["etl_ym"] = ym

                    rows.append(row)

    # 전체 조합 → 너무 크면 샘플링 (420동 × 5격자 × 3월 × 7요일 × 24시간 ≈ 2.1M)
    # 프로토타입이므로 일부만 생성: 월 1개, 요일 2개, 시간대 6개
    # 위에서 전부 생성하면 너무 크므로 샘플 전략 변경
    pass  # 아래에서 축소 버전 생성


def generate_kt_time_age_compact(grids):
    """KT 시간대별 유동인구 — 축소 버전 (프로토타입용)

    격자별 대표 3행만 생성 (시간대 12, 18, 22)
    목적: grid_map 구축 (좌표+행정동 매핑)에 필요한 최소 데이터
    """
    rows = []
    age_cols_m = [f"남성{a}대(M{a:02d})" for a in range(0, 80, 10)]
    age_cols_f = [f"여성{a}대(F{a:02d})" for a in range(0, 80, 10)]

    commercial_gus = {"11110", "11140", "11680", "11650", "11560", "11440"}

    for grid in grids:
        gu_code = grid["adm_cd_8"][:5]
        is_commercial = gu_code in commercial_gus
        base_pop = 500 if is_commercial else 200

        for ts in [12, 18, 22]:
            mult = {12: 2.5, 18: 2.0, 22: 0.8}[ts]
            pop_scale = base_pop * mult

            row = [grid["grid_id"], grid["x"], grid["y"], "월요일", ts]
            total = 0
            for _ in range(16):  # 8 male + 8 female age groups
                val = max(0, int(rng.poisson(pop_scale / 16)))
                row.append(val)
                total += val
            row.append(total)
            row.append(grid["adm_cd_8"])
            row.append("202303")
            rows.append(row)

    cols = (
        ["격자아이디(ID)", "X좌표(X_COORD)", "Y좌표(Y_COORD)", "요일(DAY)", "시간대(TIME)"]
        + age_cols_m + age_cols_f
        + ["합계(TOTAL)", "행정동코드(ADMI_CD)", "기준월(ETL_YM)"]
    )
    df = pd.DataFrame(rows, columns=cols)
    path = SYNTHETIC_DIR / "KT 월별 시간대별 성연령대별 유동인구.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[KT time_age] {len(df)} rows -> {path.name}")
    return df


def generate_kt_residence(grids, dongs):
    """KT 거주지별 유동인구

    각 관측 격자에 대해 3~5개 거주지 행정동에서 유입되는 인구 생성.
    이 데이터가 거주지↔소비지 OD 매핑의 핵심.
    """
    rows = []
    dong_codes = [d["code_8"] for d in dongs]

    for grid in grids:
        dest_dong = grid["adm_cd_8"]
        dest_gu = dest_dong[:5]

        # 거주지 후보: 같은 구 + 인접 구 + 외부
        same_gu_dongs = [d for d in dong_codes if d[:5] == dest_gu and d != dest_dong]
        other_dongs = [d for d in dong_codes if d[:5] != dest_gu]

        # 3~5개 거주지 선택 (같은 구 비율 높게)
        n_origins = rng.integers(3, 6)
        origins = []
        for _ in range(n_origins):
            if rng.random() < 0.4 and same_gu_dongs:
                origins.append(rng.choice(same_gu_dongs))
            else:
                origins.append(rng.choice(other_dongs))

        for origin_dong in origins:
            weekday_pop = int(rng.poisson(150) * rng.normal(1.0, 0.3))
            weekend_pop = int(rng.poisson(80) * rng.normal(1.0, 0.3))
            weekday_pop = max(weekday_pop, 1)
            weekend_pop = max(weekend_pop, 1)

            rows.append([
                grid["grid_id"], grid["x"], grid["y"],
                "M", "30대", origin_dong, dest_dong,
                weekday_pop, weekend_pop, "202303",
            ])

    cols = [
        "격자아이디(ID)", "X좌표(X_COORD)", "Y좌표(Y_COORD)",
        "성별(GENDER)", "연령대(AGE_GROUP)",
        "유입행정동코드(INFLOW_ADMIN_CD)", "행정동코드(ADMI_CD)",
        "평일유동인구(WKDY_FLPOP_CNT)", "주말유동인구(WKND_FLPOP_CNT)",
        "기준월(ETL_YM)",
    ]
    df = pd.DataFrame(rows, columns=cols)
    path = SYNTHETIC_DIR / "KT 월별 성연령대별 거주지별 유동인구.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[KT residence] {len(df)} rows -> {path.name}")
    return df


# ═══════════════════════════════════════════
# B069: 상권발달지수 생성
# ═══════════════════════════════════════════

def generate_district_index(dongs):
    """B069 상권발달 개별지수 — 행정동별 5개 지수"""
    rows = []
    for dong in dongs:
        gu_code = dong["code_8"][:5]
        # 구별 기본 특성 (상업지구는 매출/점포 높게)
        commercial_gus = {"11110", "11140", "11680", "11650", "11560"}
        residential_gus = {"11320", "11350", "11305", "11260", "11740"}

        if gu_code in commercial_gus:
            base = {"sales": 60, "infra": 50, "store": 70, "pop": 65, "deposit": 55}
        elif gu_code in residential_gus:
            base = {"sales": 25, "infra": 35, "store": 20, "pop": 40, "deposit": 30}
        else:
            base = {"sales": 40, "infra": 40, "store": 40, "pop": 45, "deposit": 40}

        rows.append([
            201907,
            dong["code_8"],
            round(base["sales"] + rng.normal(0, 12), 2),
            round(base["infra"] + rng.normal(0, 10), 2),
            round(base["store"] + rng.normal(0, 15), 2),
            round(base["pop"] + rng.normal(0, 12), 2),
            round(base["deposit"] + rng.normal(0, 10), 2),
        ])

    cols = [
        "기준년월(DATE)", "행정동코드(ADSTRD_CD)",
        "매출지수(SALES)", "인프라지수(INFRASTRUCTURE)", "점포지수(STORE)",
        "인구지수(POPULATION)", "집객지수(DEPOSIT)",
    ]
    df = pd.DataFrame(rows, columns=cols)
    path = SYNTHETIC_DIR / "행정동별 상권발달 개별지수.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B069 district] {len(df)} rows -> {path.name}")
    return df


# ═══════════════════════════════════════════
# B063: 카드소비패턴 (블록/집계구) — 최소 생성
# ═══════════════════════════════════════════

def generate_b063_block_gender_age():
    """블록별 성별연령대별 카드소비패턴"""
    ss_codes = [f"SS{i:03d}" for i in range(1, 76)]
    rows = []
    for _ in range(5000):
        rows.append([
            rng.choice(ss_codes),
            rng.choice(["201906", "201907", "201908"]),
            f"BLK{rng.integers(10000, 99999)}",
            rng.choice(["M", "F"]),
            rng.choice(["20대", "30대", "40대", "50대", "60대"]),
            int(rng.integers(10000, 500000)),
            int(rng.integers(1, 50)),
        ])
    df = pd.DataFrame(rows, columns=[
        "업종코드", "년월", "블록코드", "성별", "연령대",
        "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "블록별 성별연령대별 카드소비패턴.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B063 block_gender_age] {len(df)} rows -> {path.name}")


def generate_b063_census_gender_age():
    """집계구별 성별연령대별 카드소비패턴"""
    sb_codes = [f"SB{i:03d}" for i in range(1, 64)]
    rows = []
    for _ in range(5000):
        rows.append([
            f"CSU{rng.integers(10000, 99999)}",
            rng.choice(sb_codes),
            rng.choice(["201906", "201907"]),
            rng.choice(["20190601", "20190615"]),
            "내국인",
            rng.choice(["M", "F"]),
            rng.choice(["30대", "40대", "50대"]),
            int(rng.integers(5000, 300000)),
            int(rng.integers(1, 40)),
        ])
    df = pd.DataFrame(rows, columns=[
        "집계구코드", "신한업종코드", "년월", "일별",
        "내외국인", "성별", "연령대",
        "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "내국인(집계구) 성별연령대별.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B063 census_gender_age] {len(df)} rows -> {path.name}")


def generate_b063_census_inflow():
    """집계구별 유입지별 카드소비패턴"""
    sb_codes = [f"SB{i:03d}" for i in range(1, 64)]
    rows = []
    for _ in range(5000):
        sido, sgg = INFLOW_ORIGINS[rng.integers(0, len(INFLOW_ORIGINS))]
        rows.append([
            f"CSU{rng.integers(10000, 99999)}",
            rng.choice(sb_codes),
            "201907", "20190701",
            sido, sgg,
            int(rng.integers(5000, 200000)),
            int(rng.integers(1, 30)),
        ])
    df = pd.DataFrame(rows, columns=[
        "집계구코드", "신한업종코드", "년월", "일별",
        "유입시도", "유입시군구",
        "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "내국인(집계구) 유입지별.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B063 census_inflow] {len(df)} rows -> {path.name}")


def generate_b063_block_time():
    """블록별 일자별시간대별 카드소비패턴"""
    sb_codes = [f"SB{i:03d}" for i in range(1, 64)]
    rows = []
    for _ in range(5000):
        rows.append([
            f"BLK{rng.integers(10000, 99999)}",
            rng.choice(sb_codes),
            "201907", "20190715",
            rng.choice(["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]),
            int(rng.integers(1, 25)),
            int(rng.integers(5000, 200000)),
            int(rng.integers(1, 30)),
        ])
    df = pd.DataFrame(rows, columns=[
        "블록코드", "신한업종코드", "년월", "일별",
        "요일", "시간대",
        "카드이용금액합계", "카드이용건수합계",
    ])
    path = SYNTHETIC_DIR / "내국인(블록) 일자별시간대별.csv"
    df.to_csv(path, index=False, encoding="cp949")
    print(f"[B063 block_time] {len(df)} rows -> {path.name}")


# ═══════════════════════════════════════════
# 매핑 테이블 복사 (실제 데이터 그대로 사용)
# ═══════════════════════════════════════════

def copy_reference_tables():
    """업종코드, 임대시세 등 참조 테이블은 원본 데이터 그대로 사용"""
    import shutil
    data_dir = Path(__file__).parent.parent / "data"
    ref_files = [
        "카드소비 업종코드.csv",
        "신한카드 내국인 63업종 코드.csv",
        "월세임대 예측시세.csv",
        "전세임대 예측시세.csv",
    ]
    for f in ref_files:
        src = data_dir / f
        dst = SYNTHETIC_DIR / f
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            print(f"[copy] {f}")
        elif not src.exists():
            print(f"[skip] {f} (원본 없음)")


# ═══════════════════════════════════════════
# 실행
# ═══════════════════════════════════════════

def main():
    print("=" * 60)
    print("합성 프로토타입 데이터 생성")
    print("=" * 60)

    # 1. 행정동 코드 생성
    dongs = generate_dong_codes()
    print(f"\n[dongs] {len(dongs)} administrative dongs generated")

    # 2. KT 격자 생성
    grids = generate_kt_grids(dongs)
    print(f"[grids] {len(grids)} KT grid cells generated")

    # 3. B079 카드소비
    print("\n--- B079 카드소비 ---")
    generate_b079_gender_age(dongs)
    generate_b079_time(dongs)
    generate_b079_inflow(dongs)

    # 4. KT 유동인구
    print("\n--- KT 유동인구 ---")
    generate_kt_time_age_compact(grids)
    generate_kt_residence(grids, dongs)

    # 5. B069 상권지수
    print("\n--- B069 상권발달지수 ---")
    generate_district_index(dongs)

    # 6. B063 카드소비패턴
    print("\n--- B063 카드소비패턴 ---")
    generate_b063_block_gender_age()
    generate_b063_census_gender_age()
    generate_b063_census_inflow()
    generate_b063_block_time()

    # 7. 참조 테이블 복사
    print("\n--- 참조 테이블 ---")
    copy_reference_tables()

    print("\n" + "=" * 60)
    print(f"완료! 합성 데이터: {SYNTHETIC_DIR}")
    print("config.py에서 USE_SYNTHETIC = True로 전환하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
