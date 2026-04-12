"""
02_statistical_profiling_spark.py
====================================
PySpark 기반 통계 집계 및 프로파일링 파이프라인 (Feature Parity 100%).
Phase 1에서 정제된 Parquet 데이터를 로드하여 분산 연산으로 통계 지표를 산출하고,
Original(analyze_stats.py)과 동일한 9종 JSON을 생성합니다.

출력 (output/stats/):
  1. agent_profiles.json        — per (adm8, gender, age) 프로파일 + 10분위
  2. dong_context.json           — 행정동별 상권/유입 지표
  3. workplace_flow.json         — 거주지→직장 이동 확률분포
  4. workplace_population.json   — 행정동별 직장인구
  5. consumption_detail.json     — 평일/주말 업종별 소비비율
  6. dong_consumption.json       — 행정동별 업종 소비패턴
  7. global_distributions.json   — 전체 소비/이동/시간 패턴
  8. agent_allocation.json       — 그룹별 에이전트 할당
  9. aggregate_stats.json        — (gender, age) 별 mean/std/min/max
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# ─────────────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────────────
PARQUET_DIR   = Path("output/parquet")
STATS_OUT_DIR = Path("output/stats")
TARGET_AGENTS = 15000


def init_spark():
    return SparkSession.builder \
        .appName("Agent_Persona_Profiling") \
        .master("local[*]") \
        .config("spark.driver.memory", "14g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()


# ─────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────
def save_json(data, filename):
    """JSON 저장 + 크기 출력"""
    STATS_OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = STATS_OUT_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    size_kb = path.stat().st_size / 1024
    print(f"    → {filename} ({size_kb:.1f} KB)")


def safe_read_parquet(spark, name):
    """Parquet 안전 읽기 (없으면 None 반환)"""
    path = PARQUET_DIR / name
    if not path.exists():
        print(f"    ⚠ Parquet not found: {name}")
        return None
    return spark.read.parquet(str(path))


def safe_round(v, digits=4):
    if v is None:
        return 0.0
    return round(float(v), digits)


# ─────────────────────────────────────────────────────────────────────
# 1. Agent Profiles — per (adm8, gender, age) 통합 프로파일
# ─────────────────────────────────────────────────────────────────────
def build_agent_profiles(spark):
    """
    telecom_29_agg + b079_card + b009_residence를 조인하여
    프로파일을 구축하고, 소비/이동 10분위를 부여합니다.
    """
    print("[1] Building agent_profiles.json ...")

    df_tel = safe_read_parquet(spark, "telecom_29_agg.parquet")
    if df_tel is None:
        return {}, {}

    tel_metric_cols = [c for c in df_tel.columns
                       if c.startswith("tel_") and c not in ("tel_pop",)]

    # — B079 카드: (adm8, gender, age_grp) 별 총 소비액 + per-capita —
    df_card = safe_read_parquet(spark, "b079_card.parquet")
    card_agg = None
    if df_card is not None:
        card_agg = df_card.groupBy("adm8", "gender", "age_grp") \
            .agg(F.sum("amount").alias("b079_card_amt"))

    # — B009 거주지: (adm8=observed_cd, gender, age_grp) 별 유동인구 합 —
    df_res = safe_read_parquet(spark, "b009_residence.parquet")
    flow_agg = None
    if df_res is not None:
        flow_agg = df_res.groupBy(
            F.col("observed_cd").alias("adm8_obs"),
            "gender", "age_grp"
        ).agg(
            F.sum("weekday_pop").alias("b009_weekday_flow"),
            F.sum("weekend_pop").alias("b009_weekend_flow"),
        )

    # — B063 집계구: (adm8, gender, age_grp) 별 총 소비액 —
    df_census = safe_read_parquet(spark, "b063_census.parquet")
    census_agg = None
    if df_census is not None:
        census_agg = df_census.groupBy("adm8", "gender", "age_grp") \
            .agg(F.sum("amount").alias("b063_sb_amt"))

    # — Left Join —
    df_profile = df_tel
    if card_agg is not None:
        df_profile = df_profile.join(
            card_agg,
            on=["adm8", "gender", "age_grp"],
            how="left"
        )
    else:
        df_profile = df_profile.withColumn("b079_card_amt", F.lit(None).cast("double"))

    if flow_agg is not None:
        df_profile = df_profile.join(
            flow_agg,
            (df_profile["adm8"] == flow_agg["adm8_obs"]) &
            (df_profile["gender"] == flow_agg["gender"]) &
            (df_profile["age_grp"] == flow_agg["age_grp"]),
            "left"
        ).drop("adm8_obs")
        # 중복 컬럼 처리 (join 후 gender, age_grp 중복 가능)
        # 이미 on= 방식이 아니라 조건 방식이므로 flow_agg의 컬럼은 별도
    else:
        df_profile = df_profile \
            .withColumn("b009_weekday_flow", F.lit(None).cast("double")) \
            .withColumn("b009_weekend_flow", F.lit(None).cast("double"))

    if census_agg is not None:
        df_profile = df_profile.join(
            census_agg,
            on=["adm8", "gender", "age_grp"],
            how="left"
        )
    else:
        df_profile = df_profile.withColumn("b063_sb_amt", F.lit(None).cast("double"))

    # — Per-capita 계산 (10분위 기준값) —
    df_profile = df_profile \
        .withColumn("card_per_capita",
                    F.when((F.col("tel_pop") > 0) & F.col("b079_card_amt").isNotNull(),
                           F.col("b079_card_amt") / F.col("tel_pop"))
                     .otherwise(F.lit(0.0))) \
        .withColumn("flow_per_capita",
                    F.when(F.col("tel_pop") > 0,
                           (F.coalesce(F.col("b009_weekday_flow"), F.lit(0.0)) +
                            F.coalesce(F.col("b009_weekend_flow"), F.lit(0.0)))
                           / F.col("tel_pop"))
                     .otherwise(F.lit(0.0)))

    # — 10분위 부여 (전체 기준 ntile) —
    w_spend = Window.orderBy("card_per_capita")
    w_mobil = Window.orderBy("flow_per_capita")

    df_profile = df_profile \
        .withColumn("spending_level",
                    F.when(F.col("card_per_capita") > 0, F.ntile(10).over(w_spend))
                     .otherwise(F.lit(1))) \
        .withColumn("mobility_level",
                    F.when(F.col("flow_per_capita") > 0, F.ntile(10).over(w_mobil))
                     .otherwise(F.lit(1)))

    # — 업종별 소비비율 (B079 card 원본에서 산출) —
    industry_ratios = {}
    if df_card is not None:
        df_ind = df_card.groupBy("adm8", "gender", "age_grp", "industry") \
            .agg(F.sum("amount").alias("ind_amt"))
        df_ind_total = df_card.groupBy("adm8", "gender", "age_grp") \
            .agg(F.sum("amount").alias("total_amt"))
        df_ind_ratio = df_ind.join(df_ind_total, on=["adm8", "gender", "age_grp"]) \
            .withColumn("ratio", F.round(F.col("ind_amt") / F.col("total_amt"), 4)) \
            .filter(F.col("total_amt") > 0)

        for row in df_ind_ratio.select(
            F.concat_ws("_", "adm8", "gender", "age_grp").alias("key"),
            "industry", "ratio"
        ).collect():
            k = row["key"]
            if k not in industry_ratios:
                industry_ratios[k] = {}
            industry_ratios[k][row["industry"]] = row["ratio"]

    # — Collect & Build final dict —
    rows = df_profile.collect()
    profiles = {}
    pop_weights = {}

    for r in rows:
        key = f"{r['adm8']}_{r['gender']}_{r['age_grp']}"
        pop = r["tel_pop"] or 0.0

        profile = {
            "location": {
                "adm_cd_8": r["adm8"],
                "gu": r["gu"] or "",
                "dong": r["dong"] or "",
            },
            "demographics": {
                "gender": r["gender"],
                "age_grp": r["age_grp"],
                "population": safe_round(pop, 1),
            },
            "telecom": {
                col: safe_round(r[col]) for col in tel_metric_cols
                if r[col] is not None
            },
            "consumption": {
                "spending_level": int(r["spending_level"] or 1),
            },
            "mobility": {
                "mobility_level": int(r["mobility_level"] or 1),
            },
        }

        # 업종비율 추가
        if key in industry_ratios:
            # 상위 10개만 보존
            sorted_ind = dict(sorted(
                industry_ratios[key].items(), key=lambda x: -x[1]
            )[:10])
            profile["consumption"]["industry_ratio"] = sorted_ind

        # 주말/평일 이동비
        wd = r["b009_weekday_flow"] or 0
        we = r["b009_weekend_flow"] or 0
        if wd > 0:
            profile["mobility"]["weekend_weekday_ratio"] = safe_round(we / wd)

        profiles[key] = profile
        pop_weights[key] = pop

    print(f"    {len(profiles)} unique profiles")
    return profiles, pop_weights


# ─────────────────────────────────────────────────────────────────────
# 2. Dong Context — 행정동별 상권 + 유입 지표
# ─────────────────────────────────────────────────────────────────────
def build_dong_context(spark):
    print("[2] Building dong_context.json ...")

    result = {}

    # B069 상권지수
    df_069 = safe_read_parquet(spark, "b069_commercial.parquet")
    if df_069 is not None:
        for r in df_069.collect():
            adm8 = r["adm8"]
            result[adm8] = {
                "b069_sales": safe_round(r["b069_sales"]),
                "b069_infra": safe_round(r["b069_infra"]),
                "b069_store": safe_round(r["b069_store"]),
                "b069_pop":   safe_round(r["b069_pop"]),
                "b069_finance": safe_round(r["b069_finance"]),
            }

    # B079 유입 비율
    df_079in = safe_read_parquet(spark, "b079_inflow.parquet")
    if df_079in is not None:
        df_079_agg = df_079in.groupBy("adm8", "is_seoul") \
            .agg(F.sum("amount").alias("amt"))
        for r in df_079_agg.collect():
            adm8 = r["adm8"]
            if adm8 not in result:
                result[adm8] = {}
            key = "b079_inflow_seoul" if r["is_seoul"] else "b079_inflow_other"
            result[adm8][key] = safe_round(r["amt"], 2)

        # 비율 계산
        for adm8, ctx in result.items():
            s = ctx.get("b079_inflow_seoul", 0)
            o = ctx.get("b079_inflow_other", 0)
            t = s + o
            if t > 0:
                ctx["b079_seoul_inflow_ratio"] = safe_round(s / t)
            ctx.pop("b079_inflow_seoul", None)
            ctx.pop("b079_inflow_other", None)

    # B063 유입 비율
    df_063in = safe_read_parquet(spark, "b063_inflow.parquet")
    if df_063in is not None:
        df_063_agg = df_063in.groupBy("adm8", "is_seoul") \
            .agg(F.sum("amount").alias("amt"))
        in063 = defaultdict(lambda: {"seoul": 0, "other": 0})
        for r in df_063_agg.collect():
            k = "seoul" if r["is_seoul"] else "other"
            in063[r["adm8"]][k] += r["amt"]
        for adm8, vals in in063.items():
            if adm8 not in result:
                result[adm8] = {}
            t = vals["seoul"] + vals["other"]
            if t > 0:
                result[adm8]["b063_seoul_inflow_ratio"] = safe_round(vals["seoul"] / t)

    print(f"    {len(result)} dongs with context")
    return result


# ─────────────────────────────────────────────────────────────────────
# 3. Workplace Flow — 거주지→직장 이동 확률분포
# ─────────────────────────────────────────────────────────────────────
def build_workplace_flow(spark, workplace_pop):
    print("[3] Building workplace_flow.json ...")

    df_res = safe_read_parquet(spark, "b009_residence.parquet")
    if df_res is None:
        return {}

    # 거주지 → 관측지(직장 추정) : 주중 보행인구수로 가중
    df_flow = df_res.filter(
        (F.col("residence_cd") != "") &
        (F.col("observed_cd") != "") &
        (F.col("weekday_pop") > 0)
    ).groupBy("residence_cd", "observed_cd") \
     .agg(F.sum("weekday_pop").alias("flow_pop"))

    # 거주지별 총 유동인구
    df_total = df_flow.groupBy("residence_cd") \
        .agg(F.sum("flow_pop").alias("total_pop"))

    df_prob = df_flow.join(df_total, on="residence_cd") \
        .withColumn("probability", F.round(F.col("flow_pop") / F.col("total_pop"), 4))

    # Top 10 per residence dong + 기타
    w = Window.partitionBy("residence_cd").orderBy(F.desc("flow_pop"))
    df_ranked = df_prob.withColumn("rank", F.row_number().over(w))

    df_top10 = df_ranked.filter(F.col("rank") <= 10)
    df_others = df_ranked.filter(F.col("rank") > 10) \
        .groupBy("residence_cd") \
        .agg(F.sum("probability").alias("other_prob"))

    # Collect
    flow_dict = defaultdict(list)
    for r in df_top10.collect():
        flow_dict[r["residence_cd"]].append({
            "dong": r["observed_cd"],
            "probability": float(r["probability"])
        })

    for r in df_others.collect():
        if r["other_prob"] > 0.001:
            flow_dict[r["residence_cd"]].append({
                "dong": "기타",
                "probability": safe_round(r["other_prob"])
            })

    # Fallback: workplace_pop에서 KT 데이터 없는 동 보완
    if workplace_pop:
        all_dongs = set(flow_dict.keys())
        wp_dongs = set(workplace_pop.keys())
        missing = wp_dongs - all_dongs

        if missing:
            # 전체 직장인구 기반 글로벌 확률분포 (top 20)
            sorted_wp = sorted(
                [(d, info.get("total", 0)) for d, info in workplace_pop.items()],
                key=lambda x: -x[1]
            )
            total_wp = sum(v for _, v in sorted_wp)
            if total_wp > 0:
                fallback = []
                top_wp = sorted_wp[:20]
                top_sum = sum(v for _, v in top_wp)
                for dong_code, cnt in top_wp:
                    fallback.append({
                        "dong": dong_code,
                        "probability": safe_round(cnt / total_wp),
                    })
                other_wp = safe_round((total_wp - top_sum) / total_wp)
                if other_wp > 0.001:
                    fallback.append({"dong": "기타", "probability": other_wp})

                for d in missing:
                    flow_dict[d] = fallback

                print(f"    {len(missing)} dongs filled with workplace_pop fallback")

    result = dict(flow_dict)
    print(f"    {len(result)} dongs with workplace flow")
    return result


# ─────────────────────────────────────────────────────────────────────
# 4. Workplace Population — 행정동별 직장인구
# ─────────────────────────────────────────────────────────────────────
def build_workplace_population(spark):
    print("[4] Building workplace_population.json ...")
    df = safe_read_parquet(spark, "workplace_pop.parquet")
    if df is None:
        return {}

    ga_cols = [c for c in df.columns if c.startswith("M_") or c.startswith("F_")]

    result = {}
    for r in df.collect():
        by_ga = {}
        for col in ga_cols:
            v = r[col]
            if v is not None and v > 0:
                by_ga[col] = int(v)

        result[r["dong_code"]] = {
            "dong_name": r["dong_name"] or "",
            "total": int(r["total_pop"] or 0),
            "male": int(r["male_pop"] or 0),
            "female": int(r["female_pop"] or 0),
            "by_gender_age": by_ga,
        }

    print(f"    {len(result)} dongs with workplace population")
    return result


# ─────────────────────────────────────────────────────────────────────
# 5. Consumption Detail — 평일/주말 업종별 소비비율
# ─────────────────────────────────────────────────────────────────────
def build_consumption_detail(spark):
    """B063 집계구 데이터에서 (adm8, gender, age, 평일/주말, 업종) 소비비율 산출"""
    print("[5] Building consumption_detail.json ...")

    df = safe_read_parquet(spark, "b063_census.parquet")
    if df is None:
        return {}

    # 그룹별 집계
    df_agg = df.groupBy("adm8", "gender", "age_grp", "is_weekend", "industry_cd") \
        .agg(F.sum("amount").alias("ind_amt"))

    # 그룹 총합 (daytype별)
    df_total = df.groupBy("adm8", "gender", "age_grp", "is_weekend") \
        .agg(F.sum("amount").alias("total_amt"))

    df_ratio = df_agg.join(df_total, on=["adm8", "gender", "age_grp", "is_weekend"]) \
        .withColumn("ratio", F.round(F.col("ind_amt") / F.col("total_amt"), 4)) \
        .filter(F.col("total_amt") > 0)

    # Collect & Structure
    result = defaultdict(lambda: {
        "weekday_ratio": {}, "weekend_ratio": {},
    })

    for r in df_ratio.select(
        F.concat_ws("_", "adm8", "gender", "age_grp").alias("key"),
        "is_weekend", "industry_cd", "ratio"
    ).collect():
        dtype = "weekend_ratio" if r["is_weekend"] else "weekday_ratio"
        result[r["key"]][dtype][r["industry_cd"]] = float(r["ratio"])

    # 주말/평일 소비비 계산
    df_daytype_totals = df.groupBy("adm8", "gender", "age_grp", "is_weekend") \
        .agg(F.sum("amount").alias("dt_total"))

    # 고유 날짜 수로 일평균 정규화
    df_dates = df.select("date_str", "is_weekend").distinct()
    n_wd = df_dates.filter(~F.col("is_weekend")).count() or 1
    n_we = df_dates.filter(F.col("is_weekend")).count() or 1

    wd_totals = {}
    we_totals = {}
    for r in df_daytype_totals.select(
        F.concat_ws("_", "adm8", "gender", "age_grp").alias("key"),
        "is_weekend", "dt_total"
    ).collect():
        if r["is_weekend"]:
            we_totals[r["key"]] = r["dt_total"]
        else:
            wd_totals[r["key"]] = r["dt_total"]

    for key in result:
        wd = wd_totals.get(key, 0)
        we = we_totals.get(key, 0)
        if wd > 0 and we > 0:
            wd_avg = wd / n_wd
            we_avg = we / n_we
            result[key]["weekend_weekday_spending_ratio"] = safe_round(we_avg / wd_avg)

        # 비율 정렬
        for dtype in ["weekday_ratio", "weekend_ratio"]:
            result[key][dtype] = dict(
                sorted(result[key][dtype].items(), key=lambda x: -x[1])
            )

    # 소비 10분위
    spending_totals = {}
    for key in result:
        spending_totals[key] = wd_totals.get(key, 0) + we_totals.get(key, 0)

    if spending_totals:
        sorted_items = sorted(spending_totals.items(), key=lambda x: x[1])
        n = len(sorted_items)
        for i, (key, val) in enumerate(sorted_items):
            decile = min(10, int(i * 10 / n) + 1)
            result[key]["detail_spending_level"] = decile

    result["_meta"] = {
        "n_weekday_dates": n_wd,
        "n_weekend_dates": n_we,
        "description": "업종별 소비비중 (행정동×성별×나이대×평일or주말)",
        "unit": "비율 (합계=1.0)",
    }

    actual = len(result) - 1  # exclude _meta
    print(f"    {actual} groups with consumption detail")
    return dict(result)


# ─────────────────────────────────────────────────────────────────────
# 6. Dong Consumption — 행정동별 업종 소비패턴
# ─────────────────────────────────────────────────────────────────────
def build_dong_consumption(spark):
    print("[6] Building dong_consumption.json ...")

    df = safe_read_parquet(spark, "time_sales.parquet")
    if df is None:
        return {}

    # 행정동 × 업종 합계
    df_total = df.groupBy("dong_cd", "industry") \
        .agg(F.sum("amount").alias("ind_amt"))
    df_dong_total = df.groupBy("dong_cd") \
        .agg(F.sum("amount").alias("dong_amt"))
    df_ratio = df_total.join(df_dong_total, on="dong_cd") \
        .withColumn("ratio", F.round(F.col("ind_amt") / F.col("dong_amt"), 4)) \
        .filter(F.col("dong_amt") > 0)

    result = defaultdict(dict)
    for r in df_ratio.collect():
        result[r["dong_cd"]][r["industry"]] = float(r["ratio"])

    # 비율 정렬
    final = {}
    for dong, industries in result.items():
        final[dong] = {
            "industry_ratio": dict(sorted(industries.items(), key=lambda x: -x[1]))
        }

    # 평일/주말 비율 (날짜 기반)
    unique_dates = df.select("date_str").distinct().collect()
    if len(unique_dates) > 1:
        import datetime as dt
        wd_dates, we_dates = set(), set()
        for r in unique_dates:
            d = r["date_str"]
            try:
                parsed = dt.datetime.strptime(str(d).strip()[:8], "%Y%m%d")
                if parsed.weekday() >= 5:
                    we_dates.add(d)
                else:
                    wd_dates.add(d)
            except ValueError:
                wd_dates.add(d)

        if wd_dates and we_dates:
            df_wd = df.filter(F.col("date_str").isin(list(wd_dates)))
            df_we = df.filter(F.col("date_str").isin(list(we_dates)))

            n_wd, n_we = len(wd_dates), len(we_dates)

            wd_sums = {r["dong_cd"]: r["s"] for r in
                       df_wd.groupBy("dong_cd").agg(F.sum("amount").alias("s")).collect()}
            we_sums = {r["dong_cd"]: r["s"] for r in
                       df_we.groupBy("dong_cd").agg(F.sum("amount").alias("s")).collect()}

            for dong in final:
                wd_t = wd_sums.get(dong, 0)
                we_t = we_sums.get(dong, 0)
                if wd_t > 0 and we_t > 0:
                    wd_avg = wd_t / n_wd
                    we_avg = we_t / n_we
                    final[dong]["weekend_to_weekday"] = safe_round(we_avg / wd_avg)

    print(f"    {len(final)} dongs with consumption patterns")
    return final


# ─────────────────────────────────────────────────────────────────────
# 7. Global Distributions — 전체 소비/이동/시간 패턴
# ─────────────────────────────────────────────────────────────────────
def build_global_distributions(spark, profiles):
    print("[7] Building global_distributions.json ...")
    distributions = {}

    # 7-a: 시간대별 활동 (B009_wlk)
    df_wlk = safe_read_parquet(spark, "b009_wlk.parquet")
    if df_wlk is not None:
        df_time = df_wlk.groupBy("time_slot") \
            .agg(F.sum("population").alias("total_pop"))
        total_pop = df_time.select(F.sum("total_pop")).collect()[0][0] or 1.0
        temporal = {}
        for r in df_time.orderBy("time_slot").collect():
            temporal[r["time_slot"]] = safe_round(r["total_pop"] / total_pop)
        distributions["hourly_consumption"] = temporal
        print(f"    Temporal: {len(temporal)} time slots")

    # 7-b: 이동 목적 (B078)
    df_mob = safe_read_parquet(spark, "b078_mobility.parquet")
    if df_mob is not None:
        df_purpose = df_mob.groupBy("purpose") \
            .agg(F.sum("population").alias("pop"))
        total_p = df_purpose.select(F.sum("pop")).collect()[0][0] or 1.0
        purpose_dist = {}
        for r in df_purpose.orderBy(F.desc("pop")).collect():
            purpose_dist[r["purpose"]] = safe_round(r["pop"] / total_p)
        if purpose_dist:
            distributions["movement_purpose"] = purpose_dist
            print(f"    Movement purposes: {len(purpose_dist)} categories")

    # 7-c: 성별×연령대별 업종 소비비율 (B079_card에서)
    df_card = safe_read_parquet(spark, "b079_card.parquet")
    if df_card is not None:
        df_ga_ind = df_card.groupBy("gender", "age_grp", "industry") \
            .agg(F.sum("amount").alias("ind_amt"))
        df_ga_total = df_card.groupBy("gender", "age_grp") \
            .agg(F.sum("amount").alias("total_amt"))
        df_ind_ratio = df_ga_ind.join(df_ga_total, on=["gender", "age_grp"]) \
            .withColumn("ratio", F.round(F.col("ind_amt") / F.col("total_amt"), 4)) \
            .filter(F.col("total_amt") > 0)

        industry_dist = defaultdict(dict)
        for r in df_ind_ratio.select(
            F.concat_ws("_", "gender", "age_grp").alias("key"),
            "industry", "ratio"
        ).collect():
            industry_dist[r["key"]][r["industry"]] = float(r["ratio"])

        # 정렬
        for k in industry_dist:
            industry_dist[k] = dict(
                sorted(industry_dist[k].items(), key=lambda x: -x[1])
            )
        distributions["industry_spending_ratio"] = dict(industry_dist)
        print(f"    Industry spending: {len(industry_dist)} demo groups")

    return distributions


# ─────────────────────────────────────────────────────────────────────
# 8. Agent Allocation — 인구 비례 배분
# ─────────────────────────────────────────────────────────────────────
def build_agent_allocation(pop_weights, target=TARGET_AGENTS):
    print(f"[8] Building agent_allocation.json (target={target}) ...")

    total_pop = sum(pop_weights.values())
    if total_pop <= 0:
        return {}

    allocation = {}
    assigned = 0

    # 인구 비례 할당 (최소 1명)
    raw_alloc = {k: (pop / total_pop) * target for k, pop in pop_weights.items()}
    for key, raw in sorted(raw_alloc.items(), key=lambda x: -x[1]):
        n = max(1, round(raw))
        allocation[key] = n
        assigned += n

    # 목표 정확히 맞추기
    diff = assigned - target
    if diff != 0:
        sorted_keys = sorted(allocation.keys(), key=lambda k: -allocation[k])
        for key in sorted_keys:
            if diff == 0:
                break
            if diff > 0 and allocation[key] > 1:
                allocation[key] -= 1
                diff -= 1
            elif diff < 0:
                allocation[key] += 1
                diff += 1

    print(f"    Allocated {sum(allocation.values())} agents across {len(allocation)} groups")
    print(f"    Range: {min(allocation.values())}-{max(allocation.values())} per group")
    return allocation


# ─────────────────────────────────────────────────────────────────────
# 9. Aggregate Stats — (gender, age) 별 mean/std/min/max
# ─────────────────────────────────────────────────────────────────────
def build_aggregate_stats(spark):
    """Spark 내장 함수로 텔레콤 29종 지표의 그룹별 통계 산출"""
    print("[9] Building aggregate_stats.json ...")

    df = safe_read_parquet(spark, "telecom_29_agg.parquet")
    if df is None:
        return {}

    tel_cols = [c for c in df.columns if c.startswith("tel_") and c != "tel_pop"]

    # Spark groupBy + 통계 함수
    agg_exprs = []
    for col in tel_cols:
        agg_exprs.extend([
            F.avg(col).alias(f"{col}_mean"),
            F.stddev(col).alias(f"{col}_std"),
            F.min(col).alias(f"{col}_min"),
            F.max(col).alias(f"{col}_max"),
            F.count(F.when(F.col(col).isNotNull(), True)).alias(f"{col}_count"),
        ])

    df_stats = df.groupBy("gender", "age_grp").agg(*agg_exprs)

    result = {}
    for r in df_stats.collect():
        key = f"{r['gender']}_{r['age_grp']}"
        metrics = {}
        for col in tel_cols:
            metrics[col] = {
                "mean": safe_round(r[f"{col}_mean"]),
                "std":  safe_round(r[f"{col}_std"]),
                "min":  safe_round(r[f"{col}_min"]),
                "max":  safe_round(r[f"{col}_max"]),
                "count": int(r[f"{col}_count"] or 0),
            }
        result[key] = metrics

    print(f"    {len(result)} demographic groups with stats")
    return result


# ─────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────
def main():
    spark = init_spark()
    spark.sparkContext.setLogLevel("ERROR")

    print("=" * 60)
    print("  [Phase 2] Statistical Profiling — PySpark")
    print("=" * 60)

    # 1. Agent Profiles + Population Weights
    profiles, pop_weights = build_agent_profiles(spark)

    # 2. Dong Context
    dong_context = build_dong_context(spark)

    # 4. Workplace Population (3번보다 먼저: fallback 제공)
    workplace_pop = build_workplace_population(spark)

    # 3. Workplace Flow (workplace_pop fallback 포함)
    workplace_flow = build_workplace_flow(spark, workplace_pop)

    # 5. Consumption Detail
    consumption_detail = build_consumption_detail(spark)

    # 6. Dong Consumption
    dong_consumption = build_dong_consumption(spark)

    # 7. Global Distributions
    global_dist = build_global_distributions(spark, profiles)

    # 8. Agent Allocation
    allocation = build_agent_allocation(pop_weights)

    # 9. Aggregate Stats (Spark mean/std)
    agg_stats = build_aggregate_stats(spark)

    # ── JSON 저장 ────────────────────────────────────────────────────
    print("\n[Save] Writing 9 JSON outputs ...")
    save_json(profiles, "agent_profiles.json")
    save_json(dong_context, "dong_context.json")
    save_json(workplace_flow, "workplace_flow.json")
    save_json(workplace_pop, "workplace_population.json")
    save_json(consumption_detail, "consumption_detail.json")
    save_json(dong_consumption, "dong_consumption.json")
    save_json(global_dist, "global_distributions.json")
    save_json(allocation, "agent_allocation.json")
    save_json(agg_stats, "aggregate_stats.json")

    # ── 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  [Phase 2 Complete] 9 JSON files generated")
    print(f"{'=' * 60}")
    print(f"  Profiles:     {len(profiles)} unique (adm8, gender, age)")
    print(f"  Dongs:        {len(dong_context)} with context")
    print(f"  Workplace:    {len(workplace_flow)} dongs with flow")
    print(f"  WorkplacePop: {len(workplace_pop)} dongs with pop")
    print(f"  ConsumpDetail:{len(consumption_detail) - 1} groups")
    print(f"  DongConsump:  {len(dong_consumption)} dongs")
    print(f"  Allocation:   {sum(allocation.values())} agents → {len(allocation)} groups")
    print(f"  Output:       {STATS_OUT_DIR}/")
    print()

    spark.stop()


if __name__ == "__main__":
    main()