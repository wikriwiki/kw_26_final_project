"""
01_data_preprocess_spark.py
====================================
서울시 빅데이터 캠퍼스 환경에 맞춘 PySpark 기반 데이터 전처리 파이프라인.
Wide Format 병렬 합산 및 이기종 스키마 정규화를 수행한 후 Parquet 포맷으로 저장합니다.
"""

import sys
from pathlib import Path
from functools import reduce
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

# 디렉토리 설정
DATA_DIR = Path("data/raw")
MAPPING_DIR = Path("data/mapping")
PARQUET_OUT_DIR = Path("output/parquet")

def init_spark():
    """Spark 세션 초기화 (빅데이터 캠퍼스 단일 VM 리소스 최대 활용)"""
    return SparkSession.builder \
        .appName("Agent_Persona_ETL") \
        .master("local[*]") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

def process_b078_mobility(spark):
    """
    [B078] 수도권 생활이동 데이터 (PURPOSE_250M_XXXXXX.CSV)
    - Wide Format 해결: 15번~48번 컬럼(성별/연령별 인구수) 병렬 합산
    """
    file_path = str(DATA_DIR / "PURPOSE_250M_*.CSV")
    print(f"Loading B078 Mobility Data...")
    
    df = spark.read.option("header", "true").csv(file_path)
    if len(df.columns) < 49:
        print("  Warning: B078 schema does not match expected wide format.")
        return None

    # 인구수 컬럼 추출 (15~48)
    pop_columns = df.columns[15:49]
    sum_expr = reduce(add, [F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)) for c in pop_columns])

    # O_CELL_ID 기준으로 출발 격자의 이동 목적별 총 인구수 산출
    df_processed = df.withColumn("TOTAL_POP", sum_expr) \
        .select(
            F.col("O_CELL_ID").alias("origin_cell_id"),
            F.col("MOVE_PURPOSE").alias("purpose"),
            F.col("TOTAL_POP").alias("population")
        )
    return df_processed

def process_b009_wlk(spark):
    """
    [B009] 서울시 50m간격 월별 KT 유동인구 (wlk_자치구_YYYYMM.txt)
    - Wide Format 해결: 4번 컬럼(시간대), 5~34번 컬럼(인구수) 병렬 합산
    """
    file_path = str(DATA_DIR / "wlk_*.txt")
    print(f"Loading B009 Walk Data (Time-series)...")
    
    df = spark.read.option("header", "false").csv(file_path)
    if len(df.columns) < 35:
        return None

    pop_columns = df.columns[5:35]
    sum_expr = reduce(add, [F.coalesce(F.col(c).cast(DoubleType()), F.lit(0.0)) for c in pop_columns])

    # _c3(요일), _c4(시간대), _c36(행정동) 추출 및 인구수 합산
    df_processed = df.withColumn("TOTAL_POP", sum_expr) \
        .select(
            F.col("_c3").alias("day_of_week"),
            F.col("_c4").alias("time_slot"),
            F.col("_c36").alias("adm_cd"),
            F.col("TOTAL_POP").alias("population")
        )
    return df_processed

def process_b063_consumption(spark):
    """
    [B063] 서울시민의 업종별 카드소비 패턴 데이터
    - 구분자 문제 해결: 파이프(|) 인식 처리
    """
    file_path = str(DATA_DIR / "블록_성별연령대별_*.txt")
    print(f"Loading B063 Consumption Pattern Data...")
    
    df = spark.read.option("header", "false").option("sep", "|").csv(file_path)
    
    df_processed = df.select(
        F.col("_c0").alias("industry_cd"),
        F.col("_c2").alias("block_cd"),
        F.col("_c3").alias("gender"),
        F.col("_c4").alias("age_grp"),
        F.col("_c5").cast(DoubleType()).alias("amount"),
        F.col("_c6").cast(DoubleType()).alias("count")
    )
    return df_processed

def load_and_save_mappings(spark):
    """법정동/행정동 및 업종 코드 매핑 데이터 로드"""
    print("Loading Mapping Data...")
    map_df = spark.read.option("header", "true").csv(str(MAPPING_DIR / "code_mapping_mopas_nso.csv"))
    
    # 통계청 7자리 -> 행안부 8자리 매핑 추출
    map_processed = map_df.select(
        F.trim(F.col("통계청_7자리")).alias("stat7_cd"),
        F.substring(F.trim(F.col("행안부_8자리")), 1, 8).alias("adm8_cd")
    ).filter(F.col("stat7_cd") != "")
    
    return map_processed

def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")
    
    PARQUET_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. B078 처리 및 저장
    df_b078 = process_b078_mobility(spark)
    if df_b078:
        df_b078.write.mode("overwrite").parquet(str(PARQUET_OUT_DIR / "b078_mobility.parquet"))
        print("  -> B078 Parquet saved.")

    # 2. B009 처리 및 저장
    df_b009_wlk = process_b009_wlk(spark)
    if df_b009_wlk:
        df_b009_wlk.write.mode("overwrite").parquet(str(PARQUET_OUT_DIR / "b009_wlk.parquet"))
        print("  -> B009_wlk Parquet saved.")
        
    # 3. B063 처리 및 저장
    df_b063 = process_b063_consumption(spark)
    if df_b063:
        df_b063.write.mode("overwrite").parquet(str(PARQUET_OUT_DIR / "b063_consumption.parquet"))
        print("  -> B063 Parquet saved.")

    # 4. 매핑 데이터 처리 및 저장
    df_mapping = load_and_save_mappings(spark)
    if df_mapping:
        df_mapping.write.mode("overwrite").parquet(str(PARQUET_OUT_DIR / "mapping_stat7_adm8.parquet"))
        print("  -> Mapping Parquet saved.")

    print("\n[ETL Phase 1 Complete] Data successfully standardized and saved to Parquet format.")
    spark.stop()

if __name__ == "__main__":
    main()