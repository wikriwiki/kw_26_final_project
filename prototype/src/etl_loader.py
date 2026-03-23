"""
ETL Step 1: 원천 데이터 로딩 + 컬럼명 정규화

각 CSV의 한글(깨짐 가능) 컬럼명을 영문 표준명으로 변환하여
이후 파이프라인에서 일관되게 사용할 수 있도록 한다.
"""
import pandas as pd
from config import DATA_FILES, ENCODING


# ═══════════════════════════════════════════
# 컬럼명 매핑 (위치 기반 — 인코딩 깨짐 대응)
# ═══════════════════════════════════════════

def _rename_by_position(df: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """컬럼 수가 맞으면 위치 기반으로 이름 변경"""
    if len(df.columns) >= len(names):
        df = df.iloc[:, :len(names)]
        df.columns = names
    return df


# ── B079: 카드소비 (행정동별) ──

def load_b079_gender_age() -> pd.DataFrame:
    """7. 내국인 성별 연령대별(행정동별)
    → 행정동 × 성별 × 연령대 × 업종대분류별 카드소비
    """
    df = pd.read_csv(DATA_FILES["b079_gender_age"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "date",           # 기준일자 (20230101)
        "adm_cd",         # 행정동코드 (1114055000)
        "national",       # 내외국인구분 (내국인)
        "gender",         # 성별 (남/여)
        "age_group",      # 연령대 (30_39세)
        "industry_major",  # 업종대분류
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
        "ratio",          # 비율 (unnamed 컬럼)
    ])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["adm_cd"] = df["adm_cd"].astype(str)
    return df


def load_b079_inflow() -> pd.DataFrame:
    """8. 내국인 개인카드 기준 유입지별(행정동별)
    → 소비 행정동 × 유입 시도/시군구 × 업종대분류별 카드소비
    """
    df = pd.read_csv(DATA_FILES["b079_inflow"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "date",           # 기준일자
        "adm_cd",         # 소비지 행정동코드
        "inflow_sido",    # 유입지 시도
        "inflow_sgg",     # 유입지 시군구
        "industry_major",  # 업종대분류
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["adm_cd"] = df["adm_cd"].astype(str)
    df["inflow_sgg"] = df["inflow_sgg"].fillna("")
    return df


def load_b079_time() -> pd.DataFrame:
    """2. 일별 시간대별(행정동)
    → 행정동 × 시간대 × 업종대분류별 카드소비
    """
    df = pd.read_csv(DATA_FILES["b079_time"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "date",           # 기준일자
        "time_slot",      # 시간대 (1~24)
        "adm_cd",         # 행정동코드
        "industry_major",  # 업종대분류
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["adm_cd"] = df["adm_cd"].astype(str)
    df["time_slot"] = df["time_slot"].astype(int)
    return df


# ── B063: 카드소비패턴 (블록/집계구) ──

def load_b063_block_gender_age() -> pd.DataFrame:
    """블록별 성별연령대별 카드소비패턴"""
    df = pd.read_csv(DATA_FILES["b063_block_gender_age"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "upjong_cd",      # 업종코드 (SS013)
        "ym",             # 년월 (201906)
        "block_cd",       # 블록코드
        "gender",         # 성별 (M/F)
        "age_group",      # 연령대 (50대)
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["upjong_cd"] = df["upjong_cd"].str.upper()
    return df


def load_b063_census_gender_age() -> pd.DataFrame:
    """집계구별 성별연령대별 카드소비패턴"""
    df = pd.read_csv(DATA_FILES["b063_census_gender_age"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "census_cd",      # 집계구코드
        "sb_upjong_cd",   # 신한업종코드 (SB008)
        "ym",             # 년월
        "ymd",            # 일별
        "national",       # 내외국인구분
        "gender",         # 성별 (M/F or NaN)
        "age_group",      # 연령대 (30대)
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["sb_upjong_cd"] = df["sb_upjong_cd"].str.upper()
    return df


def load_b063_census_inflow() -> pd.DataFrame:
    """집계구별 유입지별 카드소비패턴"""
    df = pd.read_csv(DATA_FILES["b063_census_inflow"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "census_cd",      # 집계구코드
        "sb_upjong_cd",   # 신한업종코드
        "ym",             # 년월
        "ymd",            # 일별
        "inflow_sido",    # 유입지 시도
        "inflow_sgg",     # 유입지 시군구
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["sb_upjong_cd"] = df["sb_upjong_cd"].str.upper()
    return df


def load_b063_block_time() -> pd.DataFrame:
    """블록별 일자별시간대별 카드소비패턴"""
    df = pd.read_csv(DATA_FILES["b063_block_time"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "block_cd",       # 블록코드
        "sb_upjong_cd",   # 신한업종코드
        "ym",             # 년월
        "ymd",            # 일별
        "day_of_week",    # 요일 (월요일, 금요일 등)
        "time_slot",      # 시간대
        "card_amount",    # 카드이용금액합계
        "card_count",     # 카드이용건수합계
    ])
    df["sb_upjong_cd"] = df["sb_upjong_cd"].str.upper()
    return df


# ── 업종 코드 매핑 테이블 ──

def load_industry_code_ss() -> pd.DataFrame:
    """카드소비 업종코드 (ss001~ss075, 75개)"""
    df = pd.read_csv(DATA_FILES["industry_code_ss"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "upjong_cd",      # 업종코드 (ss001)
        "class1",         # 대분류 (음식/식품)
        "class2",         # 중분류 (한식)
        "class3",         # 소분류 (한식)
    ])
    df["upjong_cd"] = df["upjong_cd"].str.strip().str.lower()
    return df


def load_industry_code_sb() -> pd.DataFrame:
    """신한카드 63업종 코드 (sb001~sb063)"""
    df = pd.read_csv(DATA_FILES["industry_code_sb"], encoding=ENCODING)
    df = _rename_by_position(df, [
        "class1",         # 대분류
        "class2",         # 중분류
        "class3",         # 신한업종분류
        "sb_upjong_cd",   # 신한업종코드 (sb001)
    ])
    df["sb_upjong_cd"] = df["sb_upjong_cd"].str.strip().str.lower()
    return df


# ── B009: KT 유동인구 ──

def load_kt_time_age() -> pd.DataFrame:
    """KT 월별 시간대별 성연령대별 유동인구
    38컬럼: ID, X, Y, 요일, 시간대, M00~M70(15), F00~F70(15), TOTAL, ADMI_CD, ETL_YM
    """
    df = pd.read_csv(DATA_FILES["kt_time_age"], encoding=ENCODING)
    # 영문 alias가 괄호 안에 있으므로 추출
    new_cols = []
    for col in df.columns:
        if "(" in col and ")" in col:
            eng = col.split("(")[-1].rstrip(")")
            new_cols.append(eng)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


def load_kt_residence() -> pd.DataFrame:
    """KT 월별 성연령대별 거주지별 유동인구
    → 격자 × 성별 × 연령대 × 거주지코드 × 평일/주말 유동인구
    """
    df = pd.read_csv(DATA_FILES["kt_residence"], encoding=ENCODING)
    new_cols = []
    for col in df.columns:
        if "(" in col and ")" in col:
            eng = col.split("(")[-1].rstrip(")")
            new_cols.append(eng)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


# ── B069: 상권발달지수 ──

def load_district_index() -> pd.DataFrame:
    """행정동별 상권발달 개별지수
    → 행정동 × 5개 지수 (매출/인프라/점포/인구/집객)
    """
    df = pd.read_csv(DATA_FILES["district_index"], encoding=ENCODING)
    new_cols = []
    for col in df.columns:
        if "(" in col and ")" in col:
            eng = col.split("(")[-1].rstrip(")")
            new_cols.append(eng)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


# ── B068: 임대시세 ──

def load_rent_monthly() -> pd.DataFrame:
    """월세임대 예측시세"""
    df = pd.read_csv(DATA_FILES["rent_monthly"], encoding=ENCODING)
    new_cols = []
    for col in df.columns:
        if "(" in col and ")" in col:
            eng = col.split("(")[-1].rstrip(")")
            new_cols.append(eng)
        else:
            new_cols.append(col)
    df.columns = new_cols
    return df


# ═══════════════════════════════════════════
# 로딩 테스트
# ═══════════════════════════════════════════

if __name__ == "__main__":
    loaders = {
        "b079_gender_age": load_b079_gender_age,
        "b079_inflow": load_b079_inflow,
        "b079_time": load_b079_time,
        "b063_block_gender_age": load_b063_block_gender_age,
        "b063_census_gender_age": load_b063_census_gender_age,
        "b063_census_inflow": load_b063_census_inflow,
        "b063_block_time": load_b063_block_time,
        "industry_code_ss": load_industry_code_ss,
        "industry_code_sb": load_industry_code_sb,
        "kt_time_age": load_kt_time_age,
        "kt_residence": load_kt_residence,
        "district_index": load_district_index,
        "rent_monthly": load_rent_monthly,
    }

    for name, loader in loaders.items():
        try:
            df = loader()
            print(f"[OK] {name:30s} | {df.shape[0]:>5d} rows | cols: {list(df.columns)}")
        except Exception as e:
            print(f"[ERR] {name:30s} | ERROR: {e}")
