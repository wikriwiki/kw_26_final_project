"""
validate_vs_raw.py
======================
생성된 에이전트의 분포가 원본 raw 데이터의 평균/분산과 일치하는지 검증

비교 항목:
  1. 텔레콤 지표 (배달일수, 출근시간, 이동거리, 재택시간 등) — raw telecom vs 에이전트
  2. 성별/연령대별 인구 분포 — raw 인구 vs 에이전트
  3. 자치구별 분포
  4. 이동활발도 (평일/휴일 이동거리) — raw vs 에이전트
  5. 소비 업종 비율 — raw 카드소비 vs 에이전트

사용법:
  python validate_vs_raw.py
  python validate_vs_raw.py --agents output/agents/agents_final.json
"""

import json
import csv
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter

STATS_DIR = Path(__file__).parent / "output" / "stats"
DATA_DIR = Path(__file__).parent / "data"


def csv_read(path, enc=None):
    """CSV 읽기 (인코딩 자동 감지)"""
    encodings = [enc] if enc else ["utf-8-sig", "cp949", "euc-kr", "utf-8"]
    for encoding in encodings:
        try:
            with open(path, "r", encoding=encoding, errors="strict") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)
                return header, rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    return [], []


def safe_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def normalize_age(raw_age):
    """raw 연령대를 표준 형식으로 변환"""
    s = str(raw_age).strip().replace(" ", "")
    if "미만" in s or s in ("10대", "10", "15"):
        return "20세미만"
    for decade in ["20", "30", "40", "50", "60"]:
        if s.startswith(decade):
            return f"{decade}대"
    if s.startswith("70") or s.startswith("80") or "이상" in s:
        return "70대이상"
    # telecom format: 20, 25, 30, etc.
    try:
        a = int(s[:2]) if len(s) >= 2 else int(s)
        if a < 20: return "20세미만"
        if a < 30: return "20대"
        if a < 40: return "30대"
        if a < 50: return "40대"
        if a < 60: return "50대"
        if a < 70: return "60대"
        return "70대이상"
    except ValueError:
        return None


def normalize_gender(raw_gender):
    """raw 성별을 M/F로 변환"""
    s = str(raw_gender).strip()
    if s in ("1", "M", "남", "남성"):
        return "M"
    if s in ("2", "F", "여", "여성"):
        return "F"
    return None


# ---------------------------------------------------------------------------
# Load raw telecom data
# ---------------------------------------------------------------------------
def load_raw_telecom():
    """telecom_29.csv에서 (adm8, gender, age) 별 주요 지표 로드"""
    path = DATA_DIR / "raw" / "telecom_29.csv"
    if not path.exists():
        print(f"  telecom_29.csv not found at {path}")
        return {}

    h, rows = csv_read(path)
    if not h:
        return {}

    # Map column names to our metric names
    col_map = {}
    for i, c in enumerate(h):
        if "출근 소요시간" in c and "평균" in c and "4분위" not in c:
            col_map["commute_min"] = i
        elif "배달 서비스" in c and "사용일수" in c and "4분위" not in c and "미추정" not in c:
            col_map["delivery_days"] = i
        elif "쇼핑 서비스" in c and "사용일수" in c and "4분위" not in c and "미추정" not in c:
            col_map["shopping_days"] = i
        elif "평일 총 이동 거리" in c and "합계" in c and "4분위" not in c and "미추정" not in c:
            col_map["weekday_move_dist"] = i
        elif "휴일 총 이동 거리" in c and "합계" in c and "4분위" not in c and "미추정" not in c:
            col_map["weekend_move_dist"] = i
        elif "집 추정 위치 평일" in c and "체류시간" in c and "4분위" not in c and "미추정" not in c:
            col_map["home_weekday_sec"] = i
        elif "지하철이동일수" in c and "합계" in c and "4분위" not in c and "미추정" not in c:
            col_map["subway_days"] = i
        elif "총인구수" in c:
            col_map["population"] = i

    # Find column indices for adm_cd, gender, age
    adm_idx = next((i for i, c in enumerate(h) if "행정동코드" in c), 0)
    gender_idx = next((i for i, c in enumerate(h) if "성별" in c), 3)
    age_idx = next((i for i, c in enumerate(h) if "연령대" in c), 4)

    data = {}
    for row in rows:
        if len(row) <= max(col_map.values(), default=0):
            continue
        adm8 = str(row[adm_idx]).strip()
        gender = normalize_gender(row[gender_idx])
        age_grp = normalize_age(row[age_idx])
        if not gender or not age_grp:
            continue

        key = f"{adm8}_{gender}_{age_grp}"
        pop = safe_float(row[col_map["population"]]) if "population" in col_map else 0
        entry = {"population": pop or 0}
        for metric, idx in col_map.items():
            if metric != "population":
                entry[metric] = safe_float(row[idx])
        data[key] = entry

    return data


# ---------------------------------------------------------------------------
# Aggregate agents by (gender, age)
# ---------------------------------------------------------------------------
def aggregate_agents(agents):
    """에이전트를 (gender, age_group) 별로 집계"""
    by_demo = defaultdict(list)
    for a in agents:
        p = a.get("personal", {})
        g = p.get("gender", "")
        ag = p.get("age_group", "")
        if g and ag:
            by_demo[f"{g}_{ag}"].append(a)
    return by_demo


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0


def std(values):
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return 0
    m = mean(values)
    return (sum((v - m) ** 2 for v in values) / (len(values) - 1)) ** 0.5


# ---------------------------------------------------------------------------
# Main validation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="에이전트 vs Raw 데이터 검증")
    parser.add_argument("--agents", default="output/agents/agents_final.json")
    parser.add_argument("--stats-dir", type=Path, default=STATS_DIR)
    args = parser.parse_args()

    agent_path = Path(args.agents)
    if not agent_path.is_absolute():
        agent_path = Path(__file__).parent / agent_path

    if not agent_path.exists():
        print(f"File not found: {agent_path}")
        sys.exit(1)

    print(f"Loading agents: {agent_path}")
    with open(agent_path, "r", encoding="utf-8") as f:
        agents = json.load(f)
    print(f"  {len(agents):,} agents loaded")

    # Load stats
    profiles = {}
    agg_stats = {}
    allocation = {}
    for name, var in [("agent_profiles", profiles), ("aggregate_stats", agg_stats),
                      ("agent_allocation", allocation)]:
        p = args.stats_dir / f"{name}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                var.update(json.load(f))

    print("\n" + "=" * 70)
    print("  Agent vs Raw Data Validation Report")
    print("=" * 70)

    # =====================================================================
    # 1. Telecom metrics: raw group-level mean vs agent values
    # =====================================================================
    print("\n[1] Telecom Metrics: Raw Data Mean vs Agent Mean (by gender x age)")

    raw_tel = load_raw_telecom()
    if not raw_tel:
        print("  Skipped: telecom raw data not available")
    else:
        # Aggregate raw by (gender, age)
        raw_by_demo = defaultdict(lambda: defaultdict(list))
        raw_pop_by_demo = defaultdict(float)
        for key, entry in raw_tel.items():
            parts = key.rsplit("_", 2)
            if len(parts) != 3:
                continue
            demo_key = f"{parts[1]}_{parts[2]}"
            pop = entry.get("population", 0)
            raw_pop_by_demo[demo_key] += pop
            for metric in ["commute_min", "delivery_days", "shopping_days",
                           "weekday_move_dist", "weekend_move_dist", "home_weekday_sec",
                           "subway_days"]:
                v = entry.get(metric)
                if v is not None and pop > 0:
                    # weighted by population
                    raw_by_demo[demo_key][metric].append((v, pop))

        # Compute raw weighted means
        raw_means = {}
        for demo_key, metrics in raw_by_demo.items():
            raw_means[demo_key] = {}
            for metric, val_pop_list in metrics.items():
                total_w = sum(p for _, p in val_pop_list)
                if total_w > 0:
                    wmean = sum(v * p for v, p in val_pop_list) / total_w
                    raw_means[demo_key][metric] = wmean

        # Aggregate agents by (gender, age)
        agent_by_demo = aggregate_agents(agents)

        # Agent metric extraction
        def get_agent_metric(agent, metric):
            b = agent.get("behavior", {})
            w = agent.get("workplace", {})
            if metric == "commute_min":
                return safe_float(w.get("commute_min"))
            elif metric == "delivery_days":
                return safe_float(b.get("delivery_days"))
            elif metric == "shopping_days":
                return safe_float(b.get("shopping_days"))
            elif metric == "weekday_move_dist":
                v = safe_float(b.get("weekday_move_km"))
                return v * 1000 if v is not None else None  # km -> m
            elif metric == "weekend_move_dist":
                v = safe_float(b.get("weekend_move_km"))
                return v * 1000 if v is not None else None
            elif metric == "home_weekday_sec":
                v = safe_float(b.get("home_hours_weekday"))
                return v * 3600 if v is not None else None  # hours -> sec
            elif metric == "subway_days":
                return safe_float(b.get("subway_days"))
            return None

        metrics_to_check = [
            ("commute_min", "출근시간(분)"),
            ("delivery_days", "배달일수/월"),
            ("weekday_move_dist", "평일이동거리(m)"),
            ("weekend_move_dist", "휴일이동거리(m)"),
            ("home_weekday_sec", "평일재택(초)"),
        ]

        print(f"\n  {'성별_연령':>10} | {'지표':>14} | {'Raw평균':>10} | {'Agent평균':>10} | {'차이%':>7}")
        print(f"  {'-'*10}-+-{'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*7}")

        total_checks = 0
        within_30pct = 0

        for demo_key in sorted(raw_means.keys()):
            rm = raw_means[demo_key]
            agent_list = agent_by_demo.get(demo_key, [])
            if not agent_list:
                continue

            for metric, label in metrics_to_check:
                raw_val = rm.get(metric)
                if raw_val is None or raw_val == 0:
                    continue

                agent_vals = [get_agent_metric(a, metric) for a in agent_list]
                agent_vals = [v for v in agent_vals if v is not None]
                if not agent_vals:
                    continue

                agent_mean = mean(agent_vals)
                pct_diff = (agent_mean - raw_val) / abs(raw_val) * 100

                total_checks += 1
                if abs(pct_diff) <= 30:
                    within_30pct += 1

                flag = "" if abs(pct_diff) <= 30 else " <--"
                print(f"  {demo_key:>10} | {label:>14} | {raw_val:>10.1f} | {agent_mean:>10.1f} | {pct_diff:>+6.1f}%{flag}")

        if total_checks:
            print(f"\n  30% 이내 일치율: {within_30pct}/{total_checks} ({within_30pct/total_checks*100:.1f}%)")

    # =====================================================================
    # 2. Population distribution: raw vs agent
    # =====================================================================
    print("\n[2] Population Distribution: Raw vs Agent (gender x age)")

    # Raw population from profiles
    raw_demo_pop = defaultdict(float)
    for gk, profile in profiles.items():
        parts = gk.rsplit("_", 2)
        if len(parts) != 3:
            continue
        demo_key = f"{parts[1]}_{parts[2]}"
        pop = profile.get("demographics", {}).get("population", 0)
        raw_demo_pop[demo_key] += pop

    total_raw_pop = sum(raw_demo_pop.values())

    # Agent distribution
    agent_demo_cnt = Counter()
    for a in agents:
        p = a.get("personal", {})
        g = p.get("gender", "")
        ag = p.get("age_group", "")
        if g and ag:
            agent_demo_cnt[f"{g}_{ag}"] += 1

    total_agents = len(agents)

    print(f"\n  {'성별_연령':>10} | {'Raw인구%':>8} | {'Agent%':>8} | {'차이':>6}")
    print(f"  {'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

    for demo_key in sorted(set(list(raw_demo_pop.keys()) + list(agent_demo_cnt.keys()))):
        raw_pct = raw_demo_pop.get(demo_key, 0) / max(total_raw_pop, 1) * 100
        agent_pct = agent_demo_cnt.get(demo_key, 0) / max(total_agents, 1) * 100
        diff = agent_pct - raw_pct
        print(f"  {demo_key:>10} | {raw_pct:>7.2f}% | {agent_pct:>7.2f}% | {diff:>+5.2f}%")

    # =====================================================================
    # 3. Gu distribution
    # =====================================================================
    print("\n[3] Gu Distribution: Raw vs Agent (top 10)")

    raw_gu_pop = defaultdict(float)
    for gk, profile in profiles.items():
        gu = profile.get("location", {}).get("gu")
        pop = profile.get("demographics", {}).get("population", 0)
        if gu:
            raw_gu_pop[gu] += pop

    agent_gu_cnt = Counter(a.get("residence", {}).get("gu") for a in agents)

    print(f"\n  {'자치구':>8} | {'Raw%':>7} | {'Agent%':>7} | {'차이':>6}")
    print(f"  {'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*6}")

    for gu, _ in sorted(raw_gu_pop.items(), key=lambda x: -x[1])[:10]:
        raw_pct = raw_gu_pop[gu] / max(total_raw_pop, 1) * 100
        agent_pct = agent_gu_cnt.get(gu, 0) / max(total_agents, 1) * 100
        diff = agent_pct - raw_pct
        print(f"  {gu:>8} | {raw_pct:>6.2f}% | {agent_pct:>6.2f}% | {diff:>+5.2f}%")

    # =====================================================================
    # 4. Spending level vs aggregate stats comparison
    # =====================================================================
    print("\n[4] Aggregate Stats (mean/std) vs Agent distribution")

    if agg_stats:
        for demo_key in sorted(agg_stats.keys())[:6]:  # Show a few demo groups
            agg = agg_stats[demo_key]
            agent_list = aggregate_agents(agents).get(demo_key, [])
            if not agent_list or len(agent_list) < 5:
                continue

            print(f"\n  --- {demo_key} ({len(agent_list)} agents) ---")
            for metric, label, extractor in [
                ("tel_commute_time", "출근시간", lambda a: safe_float(a.get("workplace", {}).get("commute_min"))),
                ("tel_delivery_days", "배달일수", lambda a: safe_float(a.get("behavior", {}).get("delivery_days"))),
            ]:
                stat = agg.get(metric)
                if not stat:
                    continue

                vals = [extractor(a) for a in agent_list]
                vals = [v for v in vals if v is not None]
                if not vals:
                    continue

                agent_m = mean(vals)
                agent_s = std(vals)
                raw_m = stat["mean"]
                raw_s = stat["std"]
                pct = (agent_m - raw_m) / max(abs(raw_m), 0.01) * 100

                print(f"    {label:>8}: raw mean={raw_m:.1f} std={raw_s:.1f} | "
                      f"agent mean={agent_m:.1f} std={agent_s:.1f} | diff={pct:+.1f}%")

    # =====================================================================
    # 5. Summary Score
    # =====================================================================
    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  Total agents:     {len(agents):,}")
    target = sum(allocation.values()) if allocation else 0
    print(f"  Target:           {target:,}")
    print(f"  Coverage:         {len(agents)/max(target,1)*100:.1f}%")
    print(f"  Unique dongs:     {len(set(a.get('residence',{}).get('dong_code') for a in agents))}")
    print(f"  Unique jobs:      {len(set(a.get('personal',{}).get('job') for a in agents))}")
    print(f"  Gender (M/F):     {sum(1 for a in agents if a.get('personal',{}).get('gender')=='M')}"
          f" / {sum(1 for a in agents if a.get('personal',{}).get('gender')=='F')}")


if __name__ == "__main__":
    main()
