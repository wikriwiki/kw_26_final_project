"""
02_statistical_profiling_spark.py
====================================
PySpark 기반 통계 집계 및 프로파일링 파이프라인.
1단계에서 정제된 Parquet 데이터를 로드하여 분산 연산으로 통계 지표(10분위수, 업종비율 등)를 산출하고,
LLM 에이전트 생성용 JSON 파일로 축소(Reduce)하여 반환합니다.
"""

import json
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

PARQUET_DIR = Path("output/parquet")
STATS_OUT_DIR = Path("output/stats")

def init_spark():
    """Spark 세션 초기화"""
    return SparkSession.builder \
        .appName("Agent_Persona_Profiling") \
        .master("local[*]") \
        .config("spark.driver.memory", "16g") \
        .getOrCreate()

def calculate_consumption_deciles_and_ratio(spark):
    """
    B063 데이터를 바탕으로 성별x연령대별 소비 수준 10분위수 및 업종별 소비 비율 산출
    """
    print("Calculating Consumption Profiles...")
    df_b063 = spark.read.parquet(str(PARQUET_DIR / "b063_consumption.parquet"))
    
    # 1. (성별, 연령대) 그룹별 총 소비액 집계
    df_demo_total = df_b063.groupBy("gender", "age_grp") \
        .agg(F.sum("amount").alias("total_amount"))
        
    # 2. Spark Window 함수를 이용한 10분위수(Decile) 분산 계산
    # total_amount를 기준으로 오름차순 정렬하여 1~10분위 부여
    window_spec = Window.orderBy("total_amount")
    df_decile = df_demo_total.withColumn("spending_level", F.ntile(10).over(window_spec))
    
    # 3. (성별, 연령대, 업종) 그룹별 소비 비율(Ratio) 산출
    df_industry = df_b063.groupBy("gender", "age_grp", "industry_cd") \
        .agg(F.sum("amount").alias("ind_amount"))
        
    df_ratio = df_industry.join(df_demo_total, on=["gender", "age_grp"]) \
        .withColumn("industry_ratio", F.round(F.col("ind_amount") / F.col("total_amount"), 4))
        
    return df_decile, df_ratio

def calculate_temporal_activity(spark):
    """
    B009_wlk 데이터를 바탕으로 요일별, 시간대별 유동인구 활동 비율 산출
    """
    print("Calculating Temporal Activity Distributions...")
    df_b009 = spark.read.parquet(str(PARQUET_DIR / "b009_wlk.parquet"))
    
    # 시간대별 총 유동인구 합산
    df_time = df_b009.groupBy("time_slot") \
        .agg(F.sum("population").alias("total_pop"))
        
    # 전체 인구 대비 해당 시간대 활동 비율 계산
    total_pop_sum = df_time.select(F.sum("total_pop")).collect()[0][0]
    
    df_time_ratio = df_time.withColumn("activity_ratio", F.round(F.col("total_pop") / F.lit(total_pop_sum), 4)) \
        .orderBy("time_slot")
        
    return df_time_ratio

def save_to_json(dataframe, filename, key_cols, val_cols):
    """
    Spark DataFrame을 마스터 노드로 수집(Collect)하여 JSON으로 저장.
    주의: 이 함수는 데이터가 충분히 축소(Aggregated)된 이후에만 호출해야 합니다.
    """
    STATS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Spark 연산 결과를 Python 메모리로 수집
    rows = dataframe.collect()
    
    result_dict = {}
    for row in rows:
        # 다중 키 처리 (예: "M_20대")
        key = "_".join([str(row[c]) for c in key_cols])
        
        if len(val_cols) == 1:
            result_dict[key] = row[val_cols[0]]
        else:
            if key not in result_dict:
                result_dict[key] = {}
            # 중첩 딕셔너리 구조 (예: 업종별 비율)
            # 여기서는 업종코드를 키로, 비율을 값으로 저장하는 로직을 가정
            sub_key = str(row[val_cols[0]])
            sub_val = row[val_cols[1]]
            result_dict[key][sub_key] = sub_val

    out_path = STATS_OUT_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
        
    print(f"  -> Saved {filename} ({len(result_dict)} keys)")

def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")
    
    # 1. 소비 수준 10분위 및 업종 비율 산출
    df_decile, df_ratio = calculate_consumption_deciles_and_ratio(spark)
    
    # 2. 시간대별 활동 비율 산출
    df_time_ratio = calculate_temporal_activity(spark)
    
    # 3. JSON Export (마스터 노드로 수집)
    print("\nExporting aggregated results to JSON...")
    
    # spending_level 저장
    save_to_json(df_decile, "agent_spending_deciles.json", key_cols=["gender", "age_grp"], val_cols=["spending_level"])
    
    # industry_ratio 저장 (업종코드 -> 비율)
    save_to_json(df_ratio, "agent_industry_ratios.json", key_cols=["gender", "age_grp"], val_cols=["industry_cd", "industry_ratio"])
    
    # 시간대별 비율 저장
    save_to_json(df_time_ratio, "global_temporal_activity.json", key_cols=["time_slot"], val_cols=["activity_ratio"])
    
    print("\n[Profiling Phase 2 Complete] Statistical profiles generated successfully.")
    spark.stop()

if __name__ == "__main__":
    main()