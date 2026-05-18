"""
analyze_stats.py
======================
Reads joined & unjoined datasets from `output/synthetic/` (or `output/original/`)
and raw data from `data/raw/`, then produces comprehensive statistical profiles
for LLM-based agent generation.

Output (JSON) → output/stats/
  1. agent_profiles.json       – Per (adm8, gender, age) detailed statistics
  2. dong_context.json         – Per-dong commercial/infrastructure context
  3. workplace_flow.json       – Residence→workplace probability distributions
  4. global_distributions.json – Temporal, consumption, movement patterns
  5. agent_allocation.json     – How many agents to generate per combo (for 3000 target)
  7. workplace_population.json – Per-dong workplace population by gender×age

Usage:
  python analyze_stats.py                  # default: synthetic pipeline
  python analyze_stats.py --source original
"""

import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TARGET_AGENTS = 15000

def parse_args():
    source = "synthetic"
    for arg in sys.argv[1:]:
        if arg in ("--source", "-s"):
            continue
        if arg in ("original", "synthetic"):
            source = arg
    return source

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None

def csv_read(path, enc=None):
    if not path.exists():
        print(f"  ⚠ Not found: {path}")
        return [], []
    encodings = [enc] if enc else ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                reader = csv.reader(f)
                try:
                    h = [c.strip().strip('"').strip("\ufeff") for c in next(reader)]
                except StopIteration:
                    return [], []
                rows = [row for row in reader if row]
            return h, rows
        except UnicodeDecodeError:
            continue
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            h = [c.strip().strip('"').strip("\ufeff") for c in next(reader)]
        except StopIteration:
            return [], []
        rows = [row for row in reader if row]
    return h, rows

def compute_stats(values):
    """Compute mean, std, min, max for a list of floats."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}
    n = len(values)
    m = sum(values) / n
    var = sum((v - m) ** 2 for v in values) / n
    return {
        "mean": round(m, 4),
        "std": round(math.sqrt(var), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "count": n,
    }

def compute_decile(values_dict):
    """Assign 10-quantile ranks (1=lowest 10%, 10=highest 10%).
    Returns (assignments, boundaries):
      - assignments: {key: decile_rank(1~10)}
      - boundaries: {"1": [min, max], "2": [min, max], ...}
    """
    if not values_dict:
        return {}, {}
    sorted_items = sorted(values_dict.items(), key=lambda x: x[1])
    n = len(sorted_items)
    assignments = {}
    decile_values = defaultdict(list)
    for i, (key, val) in enumerate(sorted_items):
        decile = min(10, int(i * 10 / n) + 1)
        assignments[key] = decile
        decile_values[decile].append(val)

    boundaries = {}
    for d in range(1, 11):
        vals = decile_values.get(d, [])
        if vals:
            boundaries[str(d)] = [round(min(vals), 2), round(max(vals), 2)]

    return assignments, boundaries

def clean_gender(g):
    g = str(g).strip()
    if g in ("남", "M", "1", "male"):
        return "M"
    if g in ("여", "F", "2", "female"):
        return "F"
    return "U"

def clean_age(s):
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
        if a < 20:
            return "20세미만"
        if a < 30:
            return "20대"
        if a < 40:
            return "30대"
        if a < 50:
            return "40대"
        if a < 60:
            return "50대"
        if a < 70:
            return "60대"
        return "70대이상"
    return "U"

# ---------------------------------------------------------------------------
# 1. Agent Profiles – Per (adm8, gender, age) statistics
# ---------------------------------------------------------------------------
def build_agent_profiles(out_dir):
    print("[1] Building agent profiles …")
    h, rows = csv_read(out_dir / "joined_persona_base.csv")
    if not h:
        return {}, {}

    profiles = {}
    pop_weights = {}

    # Identify column groups
    tel_cols = [c for c in h if c.startswith("tel_") and c != "tel_pop"]
    join_cols = ["b079_card_amt", "b009_weekday_flow", "b009_weekend_flow", "b063_sb_amt"]

    for row in rows:
        if len(row) < len(h):
            continue
        d = dict(zip(h, row))
        key = f"{d['adm_cd_8']}_{d['gender']}_{d['age_grp']}"
        pop = safe_float(d.get("tel_pop")) or 0.0

        profile = {
            "location": {
                "adm_cd_8": d["adm_cd_8"],
                "gu": d.get("gu", ""),
                "dong": d.get("dong", ""),
            },
            "demographics": {
                "gender": d["gender"],
                "age_grp": d["age_grp"],
                "population": pop,
            },
            "telecom": {},
            "consumption": {},
            "mobility": {},
        }

        # Telecom metrics
        for col in tel_cols:
            v = safe_float(d.get(col))
            if v is not None:
                profile["telecom"][col] = round(v, 4)

        # Joined consumption/mobility
        for col in join_cols:
            v = safe_float(d.get(col))
            if v is not None:
                if "flow" in col:
                    profile["mobility"][col] = round(v, 4)
                else:
                    profile["consumption"][col] = round(v, 2)

        # Compute per-capita values (group total ÷ population)
        if pop > 0:
            for col in list(profile["consumption"].keys()):
                total_val = profile["consumption"][col]
                profile["consumption"][col + "_per_capita"] = round(total_val / pop, 2)
            for col in list(profile["mobility"].keys()):
                total_val = profile["mobility"][col]
                profile["mobility"][col + "_per_capita"] = round(total_val / pop, 4)

        profiles[key] = profile
        pop_weights[key] = pop

    print(f"    {len(profiles)} unique profiles loaded")
    return profiles, pop_weights

# ---------------------------------------------------------------------------
# 1-b. Convert agent profiles to export format (응용집계)
# ---------------------------------------------------------------------------
def convert_profiles_to_export(profiles, raw_dir):
    """Convert raw agent profiles to 응용집계 format:
    - consumption: spending_level (10분위) + industry_ratio (비율)
    - mobility: mobility_level (10분위) + weekend_weekday_ratio (비율)
    """
    print("[1b] Converting profiles to export format (응용집계) …")

    # --- Step 1: Compute 10분위 for spending & mobility ---
    spending_vals = {}
    mobility_vals = {}
    mobility_raw = {}  # store weekday/weekend for ratio calc
    for key, p in profiles.items():
        pc = p["consumption"].get("b079_card_amt_per_capita")
        if pc is not None and pc > 0:
            spending_vals[key] = pc
        wd = p["mobility"].get("b009_weekday_flow_per_capita", 0)
        we = p["mobility"].get("b009_weekend_flow_per_capita", 0)
        if wd + we > 0:
            mobility_vals[key] = wd + we
            mobility_raw[key] = (wd, we)

    spending_deciles, spending_boundaries = compute_decile(spending_vals)
    mobility_deciles, mobility_boundaries = compute_decile(mobility_vals)

    # --- Step 2: Build industry_ratio per (adm8, gender, age) from B079-1 ---
    industry_ratios = {}
    h_ind, r_ind = csv_read(raw_dir / "7.서울시 내국인 성별 연령대별(행정동별).csv")
    if h_ind:
        ind_totals = defaultdict(lambda: defaultdict(float))
        for r in r_ind:
            if len(r) < 7:
                continue
            adm8 = r[1].strip()[:8] if len(r[1].strip()) >= 8 else r[1].strip()
            gen = clean_gender(r[3])
            age = clean_age(r[4])
            ind = r[5].strip()
            amt = safe_float(r[6]) or 0.0
            if gen != "U" and age != "U" and amt > 0:
                k = f"{adm8}_{gen}_{age}"
                ind_totals[k][ind] += amt

        for k, industries in ind_totals.items():
            total = sum(industries.values())
            if total > 0:
                industry_ratios[k] = {
                    ind: round(v / total, 4)
                    for ind, v in sorted(industries.items(), key=lambda x: -x[1])
                }

    print(f"    Spending deciles: {len(spending_deciles)}, Mobility deciles: {len(mobility_deciles)}")
    print(f"    Industry ratios: {len(industry_ratios)} groups")

    # --- Step 3: Replace raw values in profiles ---
    for key, p in profiles.items():
        new_consumption = {}
        if key in spending_deciles:
            new_consumption["spending_level"] = spending_deciles[key]
        if key in industry_ratios:
            new_consumption["industry_ratio"] = industry_ratios[key]
        p["consumption"] = new_consumption

        new_mobility = {}
        if key in mobility_deciles:
            new_mobility["mobility_level"] = mobility_deciles[key]
        if key in mobility_raw:
            wd, we = mobility_raw[key]
            if wd > 0:
                new_mobility["weekend_weekday_ratio"] = round(we / wd, 4)
        p["mobility"] = new_mobility

    # --- Step 4: Print decile boundaries to console (JSON 미포함, 콘솔 출력만) ---
    print("\n    *** 경계값 (콘솔 출력만, JSON 미포함) ***")
    print("    [spending_level 경계값 (1인당 일 소비액, 원)]")
    for d in range(1, 11):
        if str(d) in spending_boundaries:
            lo, hi = spending_boundaries[str(d)]
            print(f"      {d:2d}분위: {lo:>12,.0f} ~ {hi:>12,.0f}")
    print("    [mobility_level 경계값 (1인당 보행유동량)]")
    for d in range(1, 11):
        if str(d) in mobility_boundaries:
            lo, hi = mobility_boundaries[str(d)]
            print(f"      {d:2d}분위: {lo:>12,.2f} ~ {hi:>12,.2f}")
    print("    ***********************************\n")

    return profiles

# ---------------------------------------------------------------------------
# 2. Dong Context
# ---------------------------------------------------------------------------
def build_dong_context(out_dir):
    print("[2] Building dong context …")
    h, rows = csv_read(out_dir / "joined_dong_context.csv")
    if not h:
        return {}

    context = {}
    for row in rows:
        d = dict(zip(h, row))
        adm8 = d["adm_cd_8"]
        ctx = {}

        # b069 fields — keep as-is (already exportable data)
        for col in h:
            if col.startswith("b069_"):
                v = safe_float(d.get(col))
                if v is not None:
                    ctx[col] = round(v, 4)

        # b079_2 inflow — convert to 서울유입비율 (응용집계)
        seoul_079 = safe_float(d.get("b079_2_inflow_seoul")) or 0
        other_079 = safe_float(d.get("b079_2_inflow_other")) or 0
        total_079 = seoul_079 + other_079
        if total_079 > 0:
            ctx["b079_seoul_inflow_ratio"] = round(seoul_079 / total_079, 4)

        # b063 inflow — convert to 서울유입비율 (응용집계)
        seoul_063 = safe_float(d.get("b063_inflow_seoul")) or 0
        other_063 = safe_float(d.get("b063_inflow_other")) or 0
        total_063 = seoul_063 + other_063
        if total_063 > 0:
            ctx["b063_seoul_inflow_ratio"] = round(seoul_063 / total_063, 4)

        context[adm8] = ctx

    print(f"    {len(context)} dongs with context data")
    return context

# ---------------------------------------------------------------------------
# 3. Workplace Flow – Residence→Workplace probability from KT data
# ---------------------------------------------------------------------------
def build_workplace_flow(raw_dir, workplace_pop=None):
    print("[3] Building workplace flow probabilities …")
    # KT 월별 성연령대별 거주지별 유동인구.csv
    # col7=거주지코드, col8=행정동코드(관측지=주간활동지)
    # col5=주중보행인구수 → 주중에 관측 = 직장 추정
    h, rows = csv_read(raw_dir / "KT 월별 성연령대별 거주지별 유동인구.csv")
    if not h:
        return {}

    # Aggregate: residence_dong → {observed_dong: weekday_count}
    flow = defaultdict(lambda: defaultdict(float))
    all_residence_dongs = set()
    for r in rows:
        if len(r) < 9:
            continue
        residence = r[7].strip()[:8] if r[7].strip() else ""
        observed = r[8].strip()[:8]
        wd_pop = safe_float(r[5]) or 0.0
        if residence:
            all_residence_dongs.add(residence)
        if residence and observed and wd_pop > 0:
            flow[residence][observed] += wd_pop

    # Convert to probability distributions (top 10 per dong)
    workplace_flow = {}
    for res_dong, destinations in flow.items():
        total = sum(destinations.values())
        if total <= 0:
            continue
        sorted_dests = sorted(destinations.items(), key=lambda x: -x[1])
        top = sorted_dests[:10]
        # Include "기타" for remaining probability
        top_total = sum(v for _, v in top)
        other_pct = round((total - top_total) / total, 4) if total > top_total else 0

        probs = []
        for dong, cnt in top:
            probs.append({"dong": dong, "probability": round(cnt / total, 4)})
        if other_pct > 0.001:
            probs.append({"dong": "기타", "probability": other_pct})

        workplace_flow[res_dong] = probs

    kt_count = len(workplace_flow)
    print(f"    {kt_count} residence dongs with KT flow data")

    # Fallback: fill missing dongs using workplace_population distribution
    if workplace_pop:
        # Build a global workplace probability distribution from workplace_pop
        global_wp = {}
        for dong_code, info in workplace_pop.items():
            t = info.get("total", 0)
            if t > 0:
                global_wp[dong_code] = t
        total_wp = sum(global_wp.values())

        if total_wp > 0:
            # Sort by total descending, take top 20 as fallback destinations
            sorted_wp = sorted(global_wp.items(), key=lambda x: -x[1])
            top_wp = sorted_wp[:20]
            top_wp_total = sum(v for _, v in top_wp)
            other_wp = round((total_wp - top_wp_total) / total_wp, 4)

            fallback_probs = []
            for dong_code, cnt in top_wp:
                fallback_probs.append({
                    "dong": dong_code,
                    "probability": round(cnt / total_wp, 4)
                })
            if other_wp > 0.001:
                fallback_probs.append({"dong": "기타", "probability": other_wp})

            # Collect all known dongs (from KT + workplace_pop keys)
            all_dongs = all_residence_dongs | set(workplace_pop.keys())
            fallback_count = 0
            for dong in all_dongs:
                if dong not in workplace_flow:
                    workplace_flow[dong] = fallback_probs
                    fallback_count += 1

            print(f"    {fallback_count} dongs filled with workplace_pop fallback")

    print(f"    {len(workplace_flow)} total dongs with workplace flow")
    return workplace_flow


# ---------------------------------------------------------------------------
# 3-b. Workplace Population – Per-dong workers by gender×age
# ---------------------------------------------------------------------------
def build_workplace_population(raw_dir):
    """Read 서울시 상권분석서비스(직장인구-행정동).csv and extract per-dong
    workplace population by gender × age group.
    Uses only the latest quarter."""
    print("[3b] Building workplace population …")
    csv_path = raw_dir / "서울시 상권분석서비스(직장인구-행정동).csv"
    h, rows = csv_read(csv_path)
    if not h:
        return {}

    # Find the latest quarter
    quarters = set()
    for r in rows:
        if r:
            quarters.add(r[0].strip())
    if not quarters:
        return {}
    latest_q = max(quarters)
    print(f"    Using latest quarter: {latest_q}")

    # Column mapping (based on analyzed CSV structure):
    # [0]  기준_년분기_코드
    # [1]  행정동_코드
    # [2]  행정동_코드_명
    # [3]  총_직장_인구_수
    # [4]  남성_직장_인구_수
    # [5]  여성_직장_인구_수
    # [6]  연령대_10_직장_인구_수
    # [7]  연령대_20_직장_인구_수
    # [8]  연령대_30_직장_인구_수
    # [9]  연령대_40_직장_인구_수
    # [10] 연령대_50_직장_인구_수
    # [11] 연령대_60_이상_직장_인구_수
    # [12-17] 남성연령대_10~60이상_직장_인구_수
    # [18-23] 여성연령대_10~60이상_직장_인구_수

    # gender×age column mapping → our age group names
    male_age_cols = [
        (12, "M_20세미만"),  # 남성연령대_10
        (13, "M_20대"),
        (14, "M_30대"),
        (15, "M_40대"),
        (16, "M_50대"),
        (17, "M_60대"),  # 60이상 → map to 60대 (closest)
    ]
    female_age_cols = [
        (18, "F_20세미만"),  # 여성연령대_10
        (19, "F_20대"),
        (20, "F_30대"),
        (21, "F_40대"),
        (22, "F_50대"),
        (23, "F_60대"),  # 60이상 → map to 60대
    ]
    all_ga_cols = male_age_cols + female_age_cols

    result = {}
    for r in rows:
        if len(r) < 24:
            continue
        if r[0].strip() != latest_q:
            continue

        dong_code = r[1].strip()[:8]  # Normalize to 8 digits
        dong_name = r[2].strip()
        total_pop = safe_float(r[3]) or 0

        by_gender_age = {}
        for col_idx, ga_key in all_ga_cols:
            v = safe_float(r[col_idx]) or 0
            if v > 0:
                by_gender_age[ga_key] = int(v)

        result[dong_code] = {
            "dong_name": dong_name,
            "total": int(total_pop),
            "male": int(safe_float(r[4]) or 0),
            "female": int(safe_float(r[5]) or 0),
            "by_gender_age": by_gender_age,
        }

    print(f"    {len(result)} dongs with workplace population data")
    return result


# ---------------------------------------------------------------------------
# 3-c. Consumption Detail – (adm8, gender, age, weekday/weekend, industry) avg
# ---------------------------------------------------------------------------
def build_consumption_detail(source_dir, mapping_dir):
    """Read 내국인(집계구) 성별연령대별.csv and compute daily average spending
    per (행정동, 성별, 나이대, 평일/주말, 업종).

    Uses TS_YMD to derive weekday/weekend, maps 집계구→행정동 via stat7→adm8,
    and translates SB codes to industry names.

    Output structure per key:
      "11110530_M_20대": {
        "weekday": { "한식": 1234.5, "커피전문점": 567.8, ... },
        "weekend": { "한식": 890.1, ... },
        "weekday_total": 5000.0,
        "weekend_total": 3000.0
      }
    """
    import datetime

    print("[3c-detail] Building consumption detail (adm8×gender×age×daytype×industry) …")

    # Load stat7→adm8 mapping
    h_map, r_map = csv_read(mapping_dir / "code_mapping_mopas_nso.csv")
    stat7_to_adm8 = {}
    for r in r_map:
        if len(r) > 2 and r[2].strip() and r[0].strip():
            stat7_to_adm8[r[2].strip()] = r[0].strip()[:8]

    # Load SB code→name mapping
    sb_to_name = {}
    h_sb, r_sb = csv_read(mapping_dir / "신한카드 내국인 63업종 코드.csv")
    if h_sb:
        for r in r_sb:
            if len(r) >= 4:
                code = r[3].strip().upper()
                name = r[2].strip()
                sb_to_name[code] = name

    # Read 내국인(집계구) 성별연령대별.csv
    csv_path = source_dir / "내국인(집계구) 성별연령대별.csv"
    h, rows = csv_read(csv_path)
    if not h:
        print("    ⚠ File not found or empty")
        return {}

    # Collect unique dates for normalization
    weekday_dates = set()
    weekend_dates = set()

    # Aggregate: (adm8, gender, age, daytype, industry_name) → total amt
    agg = defaultdict(float)
    skipped = 0

    for r in rows:
        if len(r) < 8:
            continue
        cen = r[0].strip()
        a8 = stat7_to_adm8.get(cen[:7])
        if not a8:
            skipped += 1
            continue

        sb_code = r[1].strip().upper()
        date_str = r[3].strip()
        gen = clean_gender(r[5])
        age = clean_age(r[6])
        amt = safe_float(r[7]) or 0.0

        if gen == "U" or age == "U" or amt <= 0:
            continue

        # Derive weekday/weekend
        try:
            dt = datetime.datetime.strptime(date_str[:8], "%Y%m%d")
            if dt.weekday() >= 5:
                daytype = "weekend"
                weekend_dates.add(date_str[:8])
            else:
                daytype = "weekday"
                weekday_dates.add(date_str[:8])
        except ValueError:
            daytype = "weekday"
            weekday_dates.add(date_str[:8])

        # Map SB code to name
        ind_name = sb_to_name.get(sb_code, sb_code)

        agg[(a8, gen, age, daytype, ind_name)] += amt

    n_wd = max(1, len(weekday_dates))
    n_we = max(1, len(weekend_dates))

    print(f"    Rows processed: {len(rows)}, skipped (no mapping): {skipped}")
    print(f"    Unique weekday dates: {n_wd}, weekend dates: {n_we}")
    print(f"    SB codes mapped: {len(sb_to_name)}")

    # Build result grouped by (adm8_gender_age) — 응용집계: 비율 + 범주
    # First pass: compute totals per (key, daytype)
    group_daytype_totals = defaultdict(lambda: {"weekday": 0.0, "weekend": 0.0})
    group_industry = defaultdict(lambda: {"weekday": defaultdict(float), "weekend": defaultdict(float)})

    for (a8, gen, age, daytype, ind_name), total_amt in agg.items():
        key = f"{a8}_{gen}_{age}"
        group_daytype_totals[key][daytype] += total_amt
        group_industry[key][daytype][ind_name] += total_amt

    # Second pass: convert to ratios
    result = {}
    spending_for_decile = {}  # key → total spending (for 10분위)

    for key in group_industry:
        entry = {}

        for daytype in ["weekday", "weekend"]:
            dt_total = group_daytype_totals[key][daytype]
            if dt_total > 0:
                ratios = {
                    ind: round(amt / dt_total, 4)
                    for ind, amt in group_industry[key][daytype].items()
                }
                entry[f"{daytype}_ratio"] = dict(
                    sorted(ratios.items(), key=lambda x: -x[1])
                )
            else:
                entry[f"{daytype}_ratio"] = {}

        # Weekend/weekday spending ratio
        wd_total = group_daytype_totals[key]["weekday"]
        we_total = group_daytype_totals[key]["weekend"]
        if wd_total > 0 and we_total > 0 and n_wd > 0 and n_we > 0:
            wd_avg = wd_total / n_wd
            we_avg = we_total / n_we
            entry["weekend_weekday_spending_ratio"] = round(we_avg / wd_avg, 4)

        result[key] = entry
        spending_for_decile[key] = wd_total + we_total

    # Compute detail_spending_level (10분위)
    deciles, detail_boundaries = compute_decile(spending_for_decile)
    for key, decile in deciles.items():
        result[key]["detail_spending_level"] = decile

    # 콘솔 출력 (JSON 미포함, 콘솔 출력만)
    print("    [detail_spending_level 경계값 (집계구 기반 총 소비액, 원)]")
    for d in range(1, 11):
        if str(d) in detail_boundaries:
            lo, hi = detail_boundaries[str(d)]
            print(f"      {d:2d}분위: {lo:>12,.0f} ~ {hi:>12,.0f}")

    # Add metadata
    result["_meta"] = {
        "n_weekday_dates": n_wd,
        "n_weekend_dates": n_we,
        "description": "업종별 소비비중 (행정동×성별×나이대×평일or주말) — 응용집계",
        "unit": "비율 (합계=1.0)",
    }

    print(f"    {len(result) - 1} unique (adm8, gender, age) groups with consumption detail")
    return result


# ---------------------------------------------------------------------------
# 3-d. Dong Consumption – Per-dong industry spending, weekday vs weekend
# ---------------------------------------------------------------------------
def build_dong_consumption(raw_dir):
    """Read 2.서울시민의 일별 시간대별(행정동).csv and compute per-dong
    industry spending breakdown with weekday/weekend split.

    Output structure per dong:
      { "industry_ratio": { "한식": 0.35, ... },
        "weekday_avg": 12345.0,
        "weekend_avg": 9876.0,
        "weekend_to_weekday": 0.8,
        "industry_weekday": { "한식": 45000, ... },
        "industry_weekend": { "한식": 32000, ... } }
    """
    import datetime

    print("[3c] Building dong consumption patterns …")
    csv_path = raw_dir / "2.서울시민의 일별 시간대별(행정동).csv"
    h, rows = csv_read(csv_path)
    if not h:
        return {}

    # Cols: [0]기준일자 [1]시간대 [2]고객행정동코드 [3]업종대분류
    #        [4]카드이용금액계 [5]카드이용건수계

    # Check how many unique dates exist
    unique_dates = set()
    for r in rows:
        if r:
            unique_dates.add(r[0].strip())

    has_multi_dates = len(unique_dates) > 1
    print(f"    Unique dates: {len(unique_dates)} ({'multi-day' if has_multi_dates else 'single-day snapshot'})")

    # Aggregate: dong → industry → weekday/weekend amounts
    dong_wd = defaultdict(lambda: defaultdict(float))  # weekday
    dong_we = defaultdict(lambda: defaultdict(float))  # weekend
    dong_total = defaultdict(lambda: defaultdict(float))
    weekday_days = set()
    weekend_days = set()

    for r in rows:
        if len(r) < 5:
            continue
        date_str = r[0].strip().replace(".0", "")
        dong = r[2].strip()[:8]
        industry = r[3].strip()
        amt = safe_float(r[4]) or 0.0
        if not dong or not industry or amt <= 0:
            continue

        dong_total[dong][industry] += amt

        if has_multi_dates:
            try:
                dt = datetime.datetime.strptime(date_str[:8], "%Y%m%d")
                if dt.weekday() >= 5:  # Saturday=5, Sunday=6
                    dong_we[dong][industry] += amt
                    weekend_days.add(date_str)
                else:
                    dong_wd[dong][industry] += amt
                    weekday_days.add(date_str)
            except ValueError:
                dong_wd[dong][industry] += amt

    n_wd = max(1, len(weekday_days))
    n_we = max(1, len(weekend_days))

    # Build result — 응용집계: 비율만 출력 (원금액 제거)
    result = {}
    for dong, industries in dong_total.items():
        total = sum(industries.values())
        if total <= 0:
            continue

        entry = {
            "industry_ratio": {
                k: round(v / total, 4)
                for k, v in sorted(industries.items(), key=lambda x: -x[1])
            },
        }

        if has_multi_dates and dong in dong_wd and dong in dong_we:
            wd_total = sum(dong_wd[dong].values())
            we_total = sum(dong_we[dong].values())
            wd_avg = wd_total / n_wd
            we_avg = we_total / n_we

            # 주말/평일 소비비 (비율만, 원금액 제거)
            entry["weekend_to_weekday"] = round(we_avg / max(1, wd_avg), 4)

            # 업종별 평일/주말 비중 (비율, 합계=1.0)
            wd_ind_total = sum(dong_wd[dong].values())
            we_ind_total = sum(dong_we[dong].values())
            if wd_ind_total > 0:
                entry["industry_weekday_ratio"] = {
                    k: round(v / wd_ind_total, 4)
                    for k, v in sorted(dong_wd[dong].items(), key=lambda x: -x[1])
                }
            if we_ind_total > 0:
                entry["industry_weekend_ratio"] = {
                    k: round(v / we_ind_total, 4)
                    for k, v in sorted(dong_we[dong].items(), key=lambda x: -x[1])
                }

        result[dong] = entry

    print(f"    {len(result)} dongs with consumption patterns")
    if has_multi_dates:
        print(f"    Weekday days: {n_wd}, Weekend days: {n_we}")
    return result

# ---------------------------------------------------------------------------
# 4. Global Distributions (from unjoined/reference data)
# ---------------------------------------------------------------------------
def build_global_distributions(out_dir, raw_dir):
    print("[4] Building global distributions …")
    distributions = {}

    # 4-a: Temporal activity (시간대별 소비)
    # unjoined_time_sales.csv = 2.서울시민의 일별 시간대별(행정동)
    # Cols: 0=기준년월, 1=업종대분류, 2=행정동코드, 3~26=시간대별(00시~23시) 매출비율
    h_ts, r_ts = csv_read(out_dir / "unjoined_time_sales.csv")
    if h_ts:
        hourly_totals = defaultdict(float)
        for r in r_ts:
            for i in range(3, min(len(r), len(h_ts))):
                v = safe_float(r[i])
                if v is not None:
                    hourly_totals[h_ts[i]] += v
        total = sum(hourly_totals.values())
        if total > 0:
            distributions["hourly_consumption"] = {
                k: round(v / total, 4) for k, v in sorted(hourly_totals.items())
            }
        print(f"    Temporal: {len(hourly_totals)} time slots")

    # 4-b: Temporal activity by dong (KT 시간대별)
    # unjoined_temporal_activity.csv = KT 월별 시간대별 성연령대별 유동인구
    h_ta, r_ta = csv_read(out_dir / "unjoined_temporal_activity.csv")
    if h_ta:
        # Extract time-of-day distribution across gender/age
        time_dist = defaultdict(lambda: defaultdict(float))
        for r in r_ta:
            if len(r) < 5:
                continue
            # Identify time slot column
            for i, col in enumerate(h_ta):
                if "시간" in col or "time" in col.lower():
                    v = safe_float(r[i])
                    if v is not None:
                        gen = clean_gender(r[3]) if len(r) > 3 else "U"
                        age = clean_age(r[4]) if len(r) > 4 else "U"
                        time_dist[f"{gen}_{age}"][col] += v
        if time_dist:
            # 응용집계: 시간대별 활동비중 (비율, 합계=1.0)
            temporal_ratios = {}
            for demo_key, cols in time_dist.items():
                total = sum(cols.values())
                if total > 0:
                    temporal_ratios[demo_key] = {
                        k: round(v / total, 4) for k, v in cols.items()
                    }
            distributions["temporal_activity_by_demo"] = temporal_ratios
            print(f"    Activity by demo: {len(temporal_ratios)} groups (ratio)")

    # 4-c: Movement purpose (생활이동 목적)
    # unjoined_mobility_purpose.csv = PURPOSE_250M_202403.csv
    h_mp, r_mp = csv_read(out_dir / "unjoined_mobility_purpose.csv")
    if h_mp:
        purpose_counts = defaultdict(float)
        purpose_col = None
        count_col = None
        for i, c in enumerate(h_mp):
            if "목적" in c or "purpose" in c.lower():
                purpose_col = i
            if "인구" in c or "수" in c or "count" in c.lower():
                count_col = i
        if purpose_col is not None:
            for r in r_mp:
                if len(r) > purpose_col:
                    purpose = r[purpose_col].strip()
                    cnt = safe_float(r[count_col]) if count_col and len(r) > count_col else 1.0
                    purpose_counts[purpose] += (cnt or 1.0)
            total = sum(purpose_counts.values())
            if total > 0:
                distributions["movement_purpose"] = {
                    k: round(v / total, 4) for k, v in sorted(purpose_counts.items(), key=lambda x: -x[1])
                }
                print(f"    Movement purposes: {len(purpose_counts)} categories")

    # 4-d: Block consumption patterns (블록별 카드소비)
    # unjoined_block_consumption.csv = 블록별 성별연령대별 카드소비패턴
    h_bc, r_bc = csv_read(out_dir / "unjoined_block_consumption.csv")
    if h_bc:
        consumption_by_demo = defaultdict(lambda: defaultdict(list))
        for r in r_bc:
            if len(r) < 6:
                continue
            gen = clean_gender(r[3]) if len(r) > 3 else "U"
            age = clean_age(r[4]) if len(r) > 4 else "U"
            # Aggregate numeric columns
            for i in range(5, len(r)):
                v = safe_float(r[i])
                if v is not None:
                    consumption_by_demo[f"{gen}_{age}"][h_bc[i] if i < len(h_bc) else f"col{i}"].append(v)
        
        # 응용집계: 소비수준 10분위 범주 + 건당단가 범주
        block_deciles = {}
        for demo_key, cols in consumption_by_demo.items():
            demo_result = {}
            for col, vals in cols.items():
                if not vals:
                    continue
                # Assign each block a decile, then report distribution
                block_vals = {f"{demo_key}_b{i}": v for i, v in enumerate(vals)}
                deciles, _ = compute_decile(block_vals)
                # Count how many blocks fall in each decile
                dist = defaultdict(int)
                for d in deciles.values():
                    dist[d] += 1
                demo_result[col] = {
                    "decile_distribution": {str(k): v for k, v in sorted(dist.items())},
                    "count": len(vals),
                }
            block_deciles[demo_key] = demo_result
        if block_deciles:
            distributions["block_consumption_by_demo"] = block_deciles
            print(f"    Block consumption: {len(block_deciles)} demo groups (decile)")

    # 4-e: Industry spending breakdown (from raw B079-1)
    h_ind, r_ind = csv_read(raw_dir / "7.서울시 내국인 성별 연령대별(행정동별).csv")
    if h_ind:
        industry_spend = defaultdict(lambda: defaultdict(float))
        for r in r_ind:
            if len(r) < 7:
                continue
            gen = clean_gender(r[3])
            age = clean_age(r[4])
            ind = r[5].strip()
            amt = safe_float(r[6]) or 0.0
            industry_spend[f"{gen}_{age}"][ind] += amt

        # Convert to proportions per demo group
        industry_dist = {}
        for demo_key, industries in industry_spend.items():
            total = sum(industries.values())
            if total > 0:
                industry_dist[demo_key] = {
                    k: round(v / total, 4)
                    for k, v in sorted(industries.items(), key=lambda x: -x[1])
                }
        if industry_dist:
            distributions["industry_spending_ratio"] = industry_dist
            print(f"    Industry spending: {len(industry_dist)} demo groups")

    # 4-f: Weekday vs Weekend spending ratio (from block daily data)
    h_ww, r_ww = csv_read(raw_dir / "내국인(블록) 일자별시간대별.csv")
    if h_ww:
        weekend_names = {"토요일", "일요일", "토", "일", "Saturday", "Sunday"}
        weekday_amt = 0.0
        weekend_amt = 0.0
        weekday_cnt = 0
        weekend_cnt = 0

        # Find column indices
        day_col = None
        amt_col = None
        for i, c in enumerate(h_ww):
            if "요일" in c or "DAW" in c.upper():
                day_col = i
            if "금액" in c or "AMT" in c.upper():
                amt_col = i

        if day_col is not None and amt_col is not None:
            for r in r_ww:
                if len(r) <= max(day_col, amt_col):
                    continue
                day_name = r[day_col].strip()
                amt = safe_float(r[amt_col]) or 0.0
                if day_name in weekend_names:
                    weekend_amt += amt
                    weekend_cnt += 1
                elif day_name:
                    weekday_amt += amt
                    weekday_cnt += 1

            if weekday_cnt > 0 and weekend_cnt > 0:
                wd_avg = weekday_amt / weekday_cnt
                we_avg = weekend_amt / weekend_cnt
                total_avg = (weekday_amt + weekend_amt) / (weekday_cnt + weekend_cnt)
                # 응용집계: 비율만 출력 (원금액 제거)
                distributions["weekday_weekend_spending"] = {
                    "weekday_ratio": round(wd_avg / total_avg, 4),
                    "weekend_ratio": round(we_avg / total_avg, 4),
                    "weekend_to_weekday": round(we_avg / wd_avg, 4),
                    "_note": "weekend_to_weekday < 1 means weekend spending is lower"
                }
                print(f"    Weekday/Weekend: ratio={we_avg/wd_avg:.3f}")

    return distributions

# ---------------------------------------------------------------------------
# 5. Agent Allocation – proportional to population
# ---------------------------------------------------------------------------
def compute_allocation(pop_weights, target=TARGET_AGENTS):
    print(f"[5] Computing agent allocation for {target} agents …")
    total_pop = sum(pop_weights.values())
    if total_pop <= 0:
        return {}

    allocation = {}
    assigned = 0
    # Proportional allocation, minimum 1 for non-zero populations
    raw_alloc = {}
    for key, pop in pop_weights.items():
        raw = (pop / total_pop) * target
        raw_alloc[key] = raw

    # Round and ensure minimum 1 for non-zero, adjust to hit target
    for key, raw in sorted(raw_alloc.items(), key=lambda x: -x[1]):
        n = max(1, round(raw))
        allocation[key] = n
        assigned += n

    # Adjust to exactly match target
    diff = assigned - target
    if diff != 0:
        # Adjust the largest allocations
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

    print(f"    Allocated {sum(allocation.values())} agents across {len(allocation)} combos")
    print(f"    Range: {min(allocation.values())}-{max(allocation.values())} agents per combo")
    return allocation

# ---------------------------------------------------------------------------
# 6. Aggregate Stats  –  (gender, age) 그룹별 요약 통계
# ---------------------------------------------------------------------------
def build_aggregate_stats(profiles):
    """Build summary statistics per (gender, age) group for LLM context."""
    print("[6] Building aggregate statistics per (gender, age) …")
    group_data = defaultdict(lambda: defaultdict(list))

    for key, profile in profiles.items():
        gen = profile["demographics"]["gender"]
        age = profile["demographics"]["age_grp"]
        group_key = f"{gen}_{age}"

        # telecom 메트릭만 포함 (이미 반출 가능 데이터 기반)
        # consumption/mobility는 응용집계 변환 후 범주이므로 mean/std 불필요
        for metric, val in profile["telecom"].items():
            group_data[group_key][metric].append(val)

    agg_stats = {}
    for group_key, metrics in group_data.items():
        agg_stats[group_key] = {
            metric: compute_stats(vals) for metric, vals in metrics.items()
        }

    print(f"    {len(agg_stats)} demographic groups summarized")
    return agg_stats

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    source = parse_args()
    out_dir = Path(f"output/{source}")
    raw_dir = Path("data/raw")
    stats_dir = Path("output/stats")
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"  Statistical Analysis (source: {source})")
    print("=" * 55)

    # 1. Agent profiles
    profiles, pop_weights = build_agent_profiles(out_dir)

    # 2. Dong context
    dong_context = build_dong_context(out_dir)

    # 3b. Workplace population (new data)
    workplace_pop = build_workplace_population(raw_dir)

    # 3. Workplace flow (enhanced with workplace_pop fallback)
    workplace_flow = build_workplace_flow(raw_dir, workplace_pop)

    # 3c. Consumption detail (adm8 × gender × age × weekday/weekend × industry)
    # Source CSV: data/raw/ (original) or data/synthetic/ (synthetic)
    source_data_dir = raw_dir if source == "original" else Path("data/synthetic")
    consumption_detail = build_consumption_detail(source_data_dir, Path("data/mapping"))

    # 3d. Dong consumption patterns
    dong_consumption = build_dong_consumption(raw_dir)

    # 4. Global distributions
    global_dist = build_global_distributions(out_dir, raw_dir)

    # 5. Agent allocation
    allocation = compute_allocation(pop_weights)

    # 6. Aggregate stats by (gender, age) — telecom 메트릭만 (반출 가능 데이터)
    agg_stats = build_aggregate_stats(profiles)

    # 7. Convert agent profiles to export format (응용집계)
    #    반드시 aggregate_stats 이후에 호출 (raw값 필요)
    profiles = convert_profiles_to_export(profiles, raw_dir)

    # --- Save outputs ---
    print("\n[Save] Writing JSON outputs …")

    def save_json(data, filename):
        path = stats_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        size_kb = path.stat().st_size / 1024
        print(f"    → {filename} ({size_kb:.1f} KB)")

    save_json(profiles, "agent_profiles.json")
    save_json(dong_context, "dong_context.json")
    save_json(workplace_flow, "workplace_flow.json")
    save_json(global_dist, "global_distributions.json")
    save_json(allocation, "agent_allocation.json")
    save_json(agg_stats, "aggregate_stats.json")
    save_json(workplace_pop, "workplace_population.json")
    save_json(consumption_detail, "consumption_detail.json")
    save_json(dong_consumption, "dong_consumption.json")

    # --- Summary ---
    print(f"\n{'='*55}")
    print(f"  [OK] Analysis Complete")
    print(f"{'='*55}")
    print(f"  Profiles:     {len(profiles)} unique (adm8, gender, age)")
    print(f"  Dongs:        {len(dong_context)} with context")
    print(f"  Workplace:    {len(workplace_flow)} dongs with flow data")
    print(f"  WorkplacePop: {len(workplace_pop)} dongs with population data")
    print(f"  ConsumpDetail:{len(consumption_detail) - 1} (adm8,gender,age) consumption details")
    print(f"  DongConsump:  {len(dong_consumption)} dongs with consumption patterns")
    print(f"  Allocation:   {sum(allocation.values())} agents → {len(allocation)} combos")
    print(f"  Output:       {stats_dir}/")
    print()


if __name__ == "__main__":
    main()
