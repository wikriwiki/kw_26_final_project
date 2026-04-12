"""
01_data_preprocess_spark.py
====================================
서울시 빅데이터 캠퍼스 VDI 환경 최적화 PySpark 전처리 파이프라인.
Feature Parity 100%: Original(preprocess_join.py) 전체 데이터 소스 커버리지 복원.

데이터 소스:
  [B078] 수도권 생활이동 (PURPOSE_250M) — Wide Format 동적 합산
  [B009] KT 유동인구 (wlk_*.txt 시간대별 + 거주지별)
  [B063] 카드소비 (블록 파이프구분 + 집계구 CSV)
  [B069] 상권발달 개별지수
  [B079] 카드결제 (성별연령대별 + 유입지별)
  [TEL]  통신사 29종 지표 (telecom_29.csv)
  [WP]   직장인구 (상권분석서비스)
  [TS]   일별 시간대별 소비 (서울시민)

출력: output/parquet/ 디렉토리에 각 데이터소스별 Parquet 저장.
"""

import sys
from pathlib import Path
from functools import reduce
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType

# ─────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("data/raw")
MAPPING_DIR = Path("data/mapping")
PARQUET_DIR = Path("output/parquet")

# 서울시 빅데이터 캠퍼스 원본 파일 인코딩 (cp949 or euc-kr)
KR_ENCODING = "cp949"

# 텔레콤 29종 컬럼 키워드 → 약어 매핑 (공백 제거 후 매칭)
_TEL_METRIC_MAP = {
    "야간상주지변경횟수평균":       "tel_night_move",
    "주간상주지변경횟수평균":       "tel_day_move",
    "평균출근소요시간평균":         "tel_commute_time",
    "평균근무시간평균":             "tel_work_time",
    "소액결재사용횟수평균":         "tel_micropay_cnt",
    "소액결재사용금액평균":         "tel_micropay_amt",
    "SNS사용횟수":                 "tel_sns_cnt",
    "평균통화량":                   "tel_call_amt",
    "평균문자량":                   "tel_text_amt",
    "평균통화대상자수":             "tel_call_users",
    "평균문자대상자수":             "tel_text_users",
    "데이터사용량":                 "tel_data_usage",
    "평일총이동횟수":               "tel_wd_move_cnt",
    "휴일총이동횟수평균":           "tel_we_move_cnt",
    "집추정위치평일총체류시간":     "tel_home_wd_time",
    "집추정위치휴일총체류시간":     "tel_home_we_time",
    "평일총이동거리합계":           "tel_wd_move_dist",
    "휴일총이동거리합계":           "tel_we_move_dist",
    "지하철이동일수합계":           "tel_subway_days",
    "게임서비스사용일수":           "tel_game_days",
    "금융서비스사용일수":           "tel_finance_days",
    "쇼핑서비스사용일수":           "tel_shopping_days",
    "동영상/방송서비스사용일수":    "tel_video_days",
    "유튜브사용일수":               "tel_youtube_days",
    "넷플릭스사용일수":             "tel_netflix_days",
    "배달서비스사용일수":           "tel_delivery_days",
    "배달_브랜드서비스사용일수":    "tel_delivery_brand_days",
    "배달_식재료서비스사용일수":    "tel_delivery_grocery_days",
    "최근3개월내요금연체비율":      "tel_overdue_ratio",
}


# ─────────────────────────────────────────────────────────────────────
# Spark 세션 초기화
# ─────────────────────────────────────────────────────────────────────
def init_spark():
    """빅데이터 캠퍼스 단일 VM(16GB RAM) 최적 구성"""
    return SparkSession.builder \
        .appName("Agent_Persona_ETL") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "14g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()


# ─────────────────────────────────────────────────────────────────────
# 성별 / 연령대 정규화 UDF
# ─────────────────────────────────────────────────────────────────────
@F.udf(StringType())
def clean_gender_udf(g):
    if g is None:
        return "U"
    g = str(g).strip()
    if g in ("남", "M", "1", "male"):
        return "M"
    if g in ("여", "F", "2", "female"):
        return "F"
    return "U"


@F.udf(StringType())
def clean_age_udf(s):
    if s is None:
        return "U"
    s = str(s).strip()
    if "_" in s:
        s = s.split("_")[0]
    if "이상" in s:
        return "70대이상"
    if "대" in s:
        if "70" in s:
            return "70대이상"
        return s.replace("세", "")
    if "미만" in s:
        return "20세미만"
    s = s.replace("세", "")
    if s.isdigit():
        a = int(s)
        if len(s) == 4:
            a = int(s[:2])
        if a < 20:  return "20세미만"
        if a < 30:  return "20대"
        if a < 40:  return "30대"
        if a < 50:  return "40대"
        if a < 60:  return "50대"
        if a < 70:  return "60대"
        return "70대이상"
    return "U"


# ─────────────────────────────────────────────────────────────────────
# 헬퍼 함수
# ─────────────────────────────────────────────────────────────────────
def file_exists(pattern):
    """파일 또는 글로브 패턴 존재 여부 확인"""
    p = Path(pattern)
    if "*" in str(pattern):
        return len(list(p.parent.glob(p.name))) > 0
    return p.exists()


def read_csv(spark, path, header=True, sep=",", encoding=None):
    """CSV 읽기 (인코딩 + 구분자 지정)"""
    reader = spark.read.option("header", str(header).lower())
    if sep != ",":
        reader = reader.option("sep", sep)
    if encoding:
        reader = reader.option("encoding", encoding)
    return reader.csv(str(path))


def find_tel_columns(spark_columns):
    """Spark 컬럼명 리스트에서 텔레콤 29종 지표 매칭.
    Returns: {short_name: original_column_name}
    """
    result = {}
    for col_name in spark_columns:
        clean = col_name.replace(" ", "")
        if any(q in clean for q in ["25%", "50%", "75%", "4분위수", "미추정"]):
            continue
        for pattern, short in _TEL_METRIC_MAP.items():
            if pattern in clean and short not in result:
                result[short] = col_name
    return result


def save_parquet(df, name):
    """Parquet 저장 헬퍼"""
    if df is None:
        return
    out = str(PARQUET_DIR / name)
    df.write.mode("overwrite").parquet(out)
    print(f"  → {name} saved.")


# ─────────────────────────────────────────────────────────────────────
# 매핑 테이블 로드
# ─────────────────────────────────────────────────────────────────────
def load_mapping(spark):
    """code_mapping_mopas_nso.csv → (stat7_cd, adm8_cd) 매핑 DataFrame"""
    path = MAPPING_DIR / "code_mapping_mopas_nso.csv"
    if not path.exists():
        print(f"  ⚠ Mapping file not found: {path}")
        return None
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns
    # cols[0]=행안부_8자리, cols[1]=행안부_10자리, cols[2]=통계청_7자리
    return df.select(
        F.trim(F.col(cols[2])).alias("stat7_cd"),
        F.substring(F.trim(F.col(cols[0])), 1, 8).alias("adm8_cd"),
        F.trim(F.col(cols[3])).alias("gu") if len(cols) > 3 else F.lit(""),
        F.trim(F.col(cols[4])).alias("dong") if len(cols) > 4 else F.lit(""),
    ).filter(
        (F.col("stat7_cd") != "") & (F.col("adm8_cd") != "")
    ).dropDuplicates(["stat7_cd"])


# ─────────────────────────────────────────────────────────────────────
# [TEL] 통신사 29종 지표 — telecom_29.csv
# ─────────────────────────────────────────────────────────────────────
def process_telecom_29(spark, mapping_df):
    """
    telecom_29.csv 구조 (positional):
      [0] 통계청7자리코드, [1] 자치구, [2] 행정동, [3] 성별, [4] 연령대,
      [5] 총인구수, [6+] 29종 지표 (각 지표마다 평균/25%/50%/75%/4분위수 등 여러 열)

    처리: 인구수 가중평균으로 (adm8, gender, age_grp) 단위 집계 → Parquet 저장.
    """
    path = DATA_DIR / "telecom_29.csv"
    if not path.exists():
        print("  ⚠ telecom_29.csv not found — skipping.")
        return None

    print("Loading [TEL] telecom_29.csv ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 10:
        print("  ⚠ telecom_29 column count too low — skipping.")
        return None

    # 고정 위치 컬럼
    stat7_col, gu_col, dong_col = cols[0], cols[1], cols[2]
    gender_col, age_col, pop_col = cols[3], cols[4], cols[5]

    # 29종 지표 동적 탐색
    tel_cols = find_tel_columns(cols)
    matched = sorted(tel_cols.keys())
    print(f"  Matched {len(matched)}/{len(_TEL_METRIC_MAP)} telecom metrics.")

    # 기본 컬럼 선택 및 타입 변환
    select_exprs = [
        F.trim(F.col(stat7_col)).alias("stat7"),
        F.trim(F.col(gu_col)).alias("gu_raw"),
        F.trim(F.col(dong_col)).alias("dong_raw"),
        clean_gender_udf(F.col(gender_col)).alias("gender"),
        clean_age_udf(F.col(age_col)).alias("age_grp"),
        F.coalesce(F.col(pop_col).cast(DoubleType()), F.lit(0.0)).alias("pop"),
    ]
    for short, orig in tel_cols.items():
        select_exprs.append(
            F.coalesce(F.col(orig).cast(DoubleType()), F.lit(0.0)).alias(short)
        )

    df_sel = df.select(*select_exprs)

    # stat7 → adm8 매핑 조인
    if mapping_df is None:
        print("  ⚠ No mapping table — cannot resolve adm8.")
        return None

    df_joined = df_sel.join(
        mapping_df.select("stat7_cd", "adm8_cd"),
        df_sel["stat7"] == mapping_df["stat7_cd"],
        "inner"
    ).withColumnRenamed("adm8_cd", "adm8").drop("stat7_cd", "stat7")

    # (adm8, gender, age_grp) 단위 인구수 가중평균 집계
    agg_exprs = [
        F.first("gu_raw").alias("gu"),
        F.first("dong_raw").alias("dong"),
        F.sum("pop").alias("tel_pop"),
    ]
    for m in matched:
        agg_exprs.append(
            F.sum(F.col(m) * F.col("pop")).alias(f"{m}_wprod")
        )

    df_agg = df_joined.groupBy("adm8", "gender", "age_grp").agg(*agg_exprs)

    # 가중평균 계산 (음수 → 0 클램핑)
    for m in matched:
        df_agg = df_agg.withColumn(
            m,
            F.when(F.col("tel_pop") > 0,
                   F.greatest(F.lit(0.0), F.col(f"{m}_wprod") / F.col("tel_pop")))
             .otherwise(F.lit(0.0))
        ).drop(f"{m}_wprod")

    cnt = df_agg.count()
    print(f"  telecom_29: {cnt} unique (adm8, gender, age_grp) groups.")
    return df_agg


# ─────────────────────────────────────────────────────────────────────
# [B078] 수도권 생활이동 — Wide Format 동적 합산
# ─────────────────────────────────────────────────────────────────────
def process_b078_mobility(spark):
    """PURPOSE_250M_*.CSV: 15~48번 컬럼(성별/연령별 인구수) 병렬 합산"""
    pattern = str(DATA_DIR / "PURPOSE_250M_*.CSV")
    if not file_exists(pattern):
        pattern = str(DATA_DIR / "PURPOSE_250M_*.csv")
        if not file_exists(pattern):
            print("  ⚠ B078 files not found.")
            return None

    print("Loading [B078] Mobility Data ...")
    df = read_csv(spark, pattern)

    if len(df.columns) < 49:
        print(f"  ⚠ B078 column count ({len(df.columns)}) < 49 — skipping.")
        return None

    pop_columns = df.columns[15:49]
    sum_expr = reduce(add, [
        F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)) for c in pop_columns
    ])

    return df.withColumn("TOTAL_POP", sum_expr).select(
        F.col("O_CELL_ID").alias("origin_cell_id"),
        F.col("MOVE_PURPOSE").alias("purpose"),
        F.col("TOTAL_POP").alias("population")
    )


# ─────────────────────────────────────────────────────────────────────
# [B009-wlk] KT 유동인구 시간대별 — Wide Format 합산
# ─────────────────────────────────────────────────────────────────────
def process_b009_wlk(spark):
    """wlk_*.txt: _c5~_c34(30개 시간대별 인구수) 합산, _c36=행정동코드"""
    pattern = str(DATA_DIR / "wlk_*.txt")
    if not file_exists(pattern):
        print("  ⚠ B009_wlk files not found.")
        return None

    print("Loading [B009-wlk] Walk Data (Time-series) ...")
    df = read_csv(spark, pattern, header=False)

    # _c36(행정동코드) 참조 필요 → 최소 37개 컬럼 필요 (인덱스 0~36)
    if len(df.columns) < 37:
        print(f"  ⚠ B009_wlk column count ({len(df.columns)}) < 37 — skipping.")
        return None

    pop_columns = df.columns[5:35]
    sum_expr = reduce(add, [
        F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)) for c in pop_columns
    ])

    return df.withColumn("TOTAL_POP", sum_expr).select(
        F.col("_c3").alias("day_of_week"),
        F.col("_c4").alias("time_slot"),
        F.col("_c36").alias("adm_cd"),
        F.col("TOTAL_POP").alias("population")
    )


# ─────────────────────────────────────────────────────────────────────
# [B009-res] KT 유동인구 거주지별 — 직장 유동 + 성별/연령대
# ─────────────────────────────────────────────────────────────────────
def process_b009_residence(spark):
    """
    KT 월별 성연령대별 거주지별 유동인구.csv
    [0]셀id [1]x좌표 [2]y좌표 [3]성별 [4]연령대
    [5]주중보행인구수 [6]주말보행인구수 [7]거주지코드 [8]행정동코드 [9]기준년월
    """
    path = DATA_DIR / "KT 월별 성연령대별 거주지별 유동인구.csv"
    if not path.exists():
        print("  ⚠ B009_residence file not found.")
        return None

    print("Loading [B009-res] Residence-based Floating Pop ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 9:
        print("  ⚠ B009_residence column count too low — skipping.")
        return None

    return df.select(
        clean_gender_udf(F.col(cols[3])).alias("gender"),
        clean_age_udf(F.col(cols[4])).alias("age_grp"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("weekday_pop"),
        F.coalesce(F.col(cols[6]).cast(DoubleType()), F.lit(0.0)).alias("weekend_pop"),
        F.substring(F.trim(F.col(cols[7])), 1, 8).alias("residence_cd"),
        F.substring(F.trim(F.col(cols[8])), 1, 8).alias("observed_cd"),
    ).filter(F.col("residence_cd") != "")


# ─────────────────────────────────────────────────────────────────────
# [B079-card] 카드 결제 — 성별/연령대별 행정동별
# ─────────────────────────────────────────────────────────────────────
def process_b079_card(spark):
    """
    7.서울시 내국인 성별 연령대별(행정동별).csv
    [0]기준일자 [1]가맹점행정동코드(10자리) [2]개인법인구분
    [3]성별 [4]연령대 [5]업종대분류 [6]카드이용금액계 [7]카드이용건수계
    업종 컬럼 보존 → Phase 2에서 업종비율 산출 가능.
    """
    path = DATA_DIR / "7.서울시 내국인 성별 연령대별(행정동별).csv"
    if not path.exists():
        print("  ⚠ B079_card file not found.")
        return None

    print("Loading [B079-card] Card Payment ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 7:
        print("  ⚠ B079_card column count too low — skipping.")
        return None

    return df.select(
        F.substring(F.trim(F.col(cols[1])), 1, 8).alias("adm8"),
        clean_gender_udf(F.col(cols[3])).alias("gender"),
        clean_age_udf(F.col(cols[4])).alias("age_grp"),
        F.trim(F.col(cols[5])).alias("industry"),
        F.coalesce(F.col(cols[6]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
        F.coalesce(F.col(cols[7]).cast(DoubleType()), F.lit(0.0)).alias("count")
        if len(cols) > 7 else F.lit(0.0).alias("count"),
    ).filter(
        (F.col("gender") != "U") & (F.col("age_grp") != "U")
    )


# ─────────────────────────────────────────────────────────────────────
# [B063-block] 블록별 카드소비 — 파이프(|) 구분자
# ─────────────────────────────────────────────────────────────────────
def process_b063_block(spark):
    """블록_성별연령대별_*.txt (파이프 구분자)"""
    pattern = str(DATA_DIR / "블록_성별연령대별_*.txt")
    if not file_exists(pattern):
        # 대체 파일명 시도
        alt = DATA_DIR / "블록별 성별연령대별 카드소비패턴.csv"
        if alt.exists():
            print("Loading [B063-block] (CSV fallback) ...")
            df = read_csv(spark, str(alt), encoding=KR_ENCODING)
        else:
            print("  ⚠ B063_block files not found.")
            return None
    else:
        print("Loading [B063-block] Consumption (pipe-separated) ...")
        df = read_csv(spark, pattern, header=False, sep="|")

    cols = df.columns
    if len(cols) < 6:
        print("  ⚠ B063_block column count too low — skipping.")
        return None

    return df.select(
        F.col(cols[0]).alias("industry_cd"),
        F.col(cols[2]).alias("block_cd"),
        clean_gender_udf(F.col(cols[3])).alias("gender"),
        clean_age_udf(F.col(cols[4])).alias("age_grp"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
        F.coalesce(F.col(cols[6]).cast(DoubleType()), F.lit(0.0)).alias("count")
        if len(cols) > 6 else F.lit(0.0).alias("count"),
    )


# ─────────────────────────────────────────────────────────────────────
# [B063-census] 집계구 성별연령대별 — 매핑 조인 필요
# ─────────────────────────────────────────────────────────────────────
def process_b063_census(spark, mapping_df):
    """
    내국인(집계구) 성별연령대별.csv
    [0]가맹점집계구코드(13자리) [1]내국인업종코드 [2]기준연월
    [3]일별(YYYYMMDD) [4]개인법인구분 [5]성별 [6]연령대 [7]카드이용금액계 [8]카드이용건수
    집계구13자리의 앞 7자리 → stat7 → adm8 매핑.
    날짜 기반 평일/주말 구분 추가 (Phase 2 consumption_detail용).
    """
    path = DATA_DIR / "내국인(집계구) 성별연령대별.csv"
    if not path.exists():
        print("  ⚠ B063_census file not found.")
        return None

    print("Loading [B063-census] Census Consumption ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 8:
        print("  ⚠ B063_census column count too low — skipping.")
        return None

    df_sel = df.select(
        F.substring(F.trim(F.col(cols[0])), 1, 7).alias("stat7"),
        F.trim(F.col(cols[1])).alias("industry_cd"),
        F.trim(F.col(cols[3])).alias("date_str"),
        clean_gender_udf(F.col(cols[5])).alias("gender"),
        clean_age_udf(F.col(cols[6])).alias("age_grp"),
        F.coalesce(F.col(cols[7]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
        F.coalesce(F.col(cols[8]).cast(DoubleType()), F.lit(0.0)).alias("count")
        if len(cols) > 8 else F.lit(0.0).alias("count"),
    ).filter(
        (F.col("gender") != "U") & (F.col("age_grp") != "U") & (F.col("amount") > 0)
    )

    # stat7 → adm8 매핑 조인
    if mapping_df is None:
        print("  ⚠ No mapping — cannot resolve adm8 for census data.")
        return None

    df_mapped = df_sel.join(
        mapping_df.select("stat7_cd", "adm8_cd"),
        df_sel["stat7"] == mapping_df["stat7_cd"],
        "inner"
    ).withColumnRenamed("adm8_cd", "adm8").drop("stat7_cd", "stat7")

    # 평일/주말 플래그 (Spark dayofweek: 1=Sun, 7=Sat)
    df_mapped = df_mapped.withColumn(
        "date_parsed",
        F.to_date(F.substring("date_str", 1, 8), "yyyyMMdd")
    ).withColumn(
        "is_weekend",
        F.when(F.dayofweek("date_parsed").isin(1, 7), F.lit(True))
         .otherwise(F.lit(False))
    ).drop("date_parsed")

    return df_mapped


# ─────────────────────────────────────────────────────────────────────
# [B069] 상권발달 개별지수
# ─────────────────────────────────────────────────────────────────────
def process_b069_commercial(spark):
    """
    행정동별 상권발달 개별지수.csv
    [0]일자 [1]행정동코드(8자리) [2]매출지수 [3]인프라지수
    [4]가맹점지수 [5]인구지수 [6]금융지수
    """
    path = DATA_DIR / "행정동별 상권발달 개별지수.csv"
    if not path.exists():
        print("  ⚠ B069_commercial file not found.")
        return None

    print("Loading [B069] Commercial Index ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 7:
        print("  ⚠ B069 column count too low — skipping.")
        return None

    return df.select(
        F.substring(F.trim(F.col(cols[1])), 1, 8).alias("adm8"),
        F.coalesce(F.col(cols[2]).cast(DoubleType()), F.lit(0.0)).alias("b069_sales"),
        F.coalesce(F.col(cols[3]).cast(DoubleType()), F.lit(0.0)).alias("b069_infra"),
        F.coalesce(F.col(cols[4]).cast(DoubleType()), F.lit(0.0)).alias("b069_store"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("b069_pop"),
        F.coalesce(F.col(cols[6]).cast(DoubleType()), F.lit(0.0)).alias("b069_finance"),
    ).dropDuplicates(["adm8"])


# ─────────────────────────────────────────────────────────────────────
# [B079-inflow] 유입지별 (행정동별)
# ─────────────────────────────────────────────────────────────────────
def process_b079_inflow(spark):
    """
    8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv
    [0]기준일자 [1]가맹점행정동코드(10자리) [2]고객주소광역시도
    [3]고객주소시군구 [4]업종대분류 [5]카드이용금액계 [6]카드이용건수계
    """
    path = DATA_DIR / "8.서울시 내국인의 개인카드 기준 유입지별(행정동별).csv"
    if not path.exists():
        print("  ⚠ B079_inflow file not found.")
        return None

    print("Loading [B079-inflow] Card Inflow ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 6:
        return None

    return df.select(
        F.substring(F.trim(F.col(cols[1])), 1, 8).alias("adm8"),
        F.trim(F.col(cols[2])).alias("region"),
        F.when(F.col(cols[2]).contains("서울"), F.lit(True))
         .otherwise(F.lit(False)).alias("is_seoul"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
    )


# ─────────────────────────────────────────────────────────────────────
# [B063-inflow] 유입지별 (집계구) — 매핑 필요
# ─────────────────────────────────────────────────────────────────────
def process_b063_inflow(spark, mapping_df):
    """
    내국인(집계구) 유입지별.csv
    [0]가맹점집계구코드(13자리) [1]내국인업종코드 [2]기준연월 [3]일별
    [4]고객주소광역시(SIDO) [5]고객주소시군구 [6]카드이용금액계 [7]카드이용건수
    """
    path = DATA_DIR / "내국인(집계구) 유입지별.csv"
    if not path.exists():
        print("  ⚠ B063_inflow file not found.")
        return None

    print("Loading [B063-inflow] Census Inflow ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 7 or mapping_df is None:
        return None

    df_sel = df.select(
        F.substring(F.trim(F.col(cols[0])), 1, 7).alias("stat7"),
        F.trim(F.col(cols[4])).alias("region"),
        F.when(F.col(cols[4]).contains("서울"), F.lit(True))
         .otherwise(F.lit(False)).alias("is_seoul"),
        F.coalesce(F.col(cols[6]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
    )

    return df_sel.join(
        mapping_df.select("stat7_cd", "adm8_cd"),
        df_sel["stat7"] == mapping_df["stat7_cd"],
        "inner"
    ).withColumnRenamed("adm8_cd", "adm8").drop("stat7_cd", "stat7")


# ─────────────────────────────────────────────────────────────────────
# [WP] 직장인구 — 서울시 상권분석서비스
# ─────────────────────────────────────────────────────────────────────
def process_workplace_pop(spark):
    """
    서울시 상권분석서비스(직장인구-행정동).csv
    [0]기준_년분기_코드 [1]행정동_코드 [2]행정동_코드_명 [3]총_직장_인구_수
    [4]남성_직장_인구_수 [5]여성_직장_인구_수
    [6-11]연령대별(10대~60이상)_직장_인구
    [12-17]남성연령대별, [18-23]여성연령대별
    최신 분기만 사용.
    """
    path = DATA_DIR / "서울시 상권분석서비스(직장인구-행정동).csv"
    if not path.exists():
        print("  ⚠ Workplace population file not found.")
        return None

    print("Loading [WP] Workplace Population ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 24:
        print(f"  ⚠ Workplace pop column count ({len(cols)}) < 24 — skipping.")
        return None

    # 최신 분기 필터
    latest_q = df.select(F.max(F.trim(F.col(cols[0])))).collect()[0][0]
    print(f"  Using latest quarter: {latest_q}")

    df_latest = df.filter(F.trim(F.col(cols[0])) == latest_q)

    # 성별×연령대 컬럼 구성
    ga_map = {
        12: "M_20세미만", 13: "M_20대", 14: "M_30대",
        15: "M_40대",     16: "M_50대", 17: "M_60대",
        18: "F_20세미만", 19: "F_20대", 20: "F_30대",
        21: "F_40대",     22: "F_50대", 23: "F_60대",
    }

    select_exprs = [
        F.substring(F.trim(F.col(cols[1])), 1, 8).alias("dong_code"),
        F.trim(F.col(cols[2])).alias("dong_name"),
        F.coalesce(F.col(cols[3]).cast(DoubleType()), F.lit(0.0)).alias("total_pop"),
        F.coalesce(F.col(cols[4]).cast(DoubleType()), F.lit(0.0)).alias("male_pop"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("female_pop"),
    ]
    for idx, name in ga_map.items():
        if idx < len(cols):
            select_exprs.append(
                F.coalesce(F.col(cols[idx]).cast(DoubleType()), F.lit(0.0)).alias(name)
            )

    return df_latest.select(*select_exprs)


# ─────────────────────────────────────────────────────────────────────
# [TS] 일별 시간대별 소비 — 동별 소비 패턴
# ─────────────────────────────────────────────────────────────────────
def process_time_sales(spark):
    """
    2.서울시민의 일별 시간대별(행정동).csv
    [0]기준일자 [1]시간대 [2]고객행정동코드 [3]업종대분류
    [4]카드이용금액계 [5]카드이용건수계
    """
    path = DATA_DIR / "2.서울시민의 일별 시간대별(행정동).csv"
    if not path.exists():
        print("  ⚠ Time_sales file not found.")
        return None

    print("Loading [TS] Time-based Sales ...")
    df = read_csv(spark, str(path), encoding=KR_ENCODING)
    cols = df.columns

    if len(cols) < 5:
        return None

    return df.select(
        F.trim(F.col(cols[0])).alias("date_str"),
        F.trim(F.col(cols[1])).alias("time_slot"),
        F.substring(F.trim(F.col(cols[2])), 1, 8).alias("dong_cd"),
        F.trim(F.col(cols[3])).alias("industry"),
        F.coalesce(F.col(cols[4]).cast(DoubleType()), F.lit(0.0)).alias("amount"),
        F.coalesce(F.col(cols[5]).cast(DoubleType()), F.lit(0.0)).alias("count")
        if len(cols) > 5 else F.lit(0.0).alias("count"),
    ).filter(F.col("amount") > 0)


# ─────────────────────────────────────────────────────────────────────
# 메인 파이프라인
# ─────────────────────────────────────────────────────────────────────
def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")

    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  [Phase 1] Data Preprocessing — PySpark ETL")
    print("=" * 60)

    # 매핑 테이블 (다른 데이터의 조인에 공용)
    mapping_df = load_mapping(spark)
    save_parquet(
        mapping_df.select("stat7_cd", "adm8_cd") if mapping_df else None,
        "mapping_stat7_adm8.parquet"
    )

    # 각 데이터 소스 처리 및 저장
    sources = [
        ("telecom_29_agg.parquet",   lambda: process_telecom_29(spark, mapping_df)),
        ("b078_mobility.parquet",    lambda: process_b078_mobility(spark)),
        ("b009_wlk.parquet",         lambda: process_b009_wlk(spark)),
        ("b009_residence.parquet",   lambda: process_b009_residence(spark)),
        ("b079_card.parquet",        lambda: process_b079_card(spark)),
        ("b063_block.parquet",       lambda: process_b063_block(spark)),
        ("b063_census.parquet",      lambda: process_b063_census(spark, mapping_df)),
        ("b069_commercial.parquet",  lambda: process_b069_commercial(spark)),
        ("b079_inflow.parquet",      lambda: process_b079_inflow(spark)),
        ("b063_inflow.parquet",      lambda: process_b063_inflow(spark, mapping_df)),
        ("workplace_pop.parquet",    lambda: process_workplace_pop(spark)),
        ("time_sales.parquet",       lambda: process_time_sales(spark)),
    ]

    loaded = 0
    for name, loader in sources:
        try:
            df = loader()
            save_parquet(df, name)
            if df is not None:
                loaded += 1
        except Exception as e:
            print(f"  ✗ Error processing {name}: {e}")

    print(f"\n{'=' * 60}")
    print(f"  [Phase 1 Complete] {loaded}/{len(sources)} data sources → Parquet")
    print(f"  Output: {PARQUET_DIR}/")
    print(f"{'=' * 60}")
    spark.stop()


if __name__ == "__main__":
    main()