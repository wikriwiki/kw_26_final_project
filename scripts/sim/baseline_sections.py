"""4일 baseline 무정책 시뮬용 section 함수.

generate_baseline_report.py가 import해서 사용.
"""
import sys
from datetime import date, timedelta
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def section2_baseline_daily(start: date, days: int, out_dir: Path) -> tuple[dict, dict]:
    """일별 안정성·분포 — Day 1~N의 commerce 매출·이벤트·만족도.

    baseline 정책 미주입 시뮬의 day-by-day consistency 진단.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    day_strs = [(start + timedelta(days=i)).isoformat() for i in range(days)]

    daily = []
    with driver_session() as s:
        for d in day_strs:
            r = s.run("""
                MATCH (:Plan {day: date($d)})-[i:INCLUDES]->(p:POI {type:'commerce'})
                RETURN count(i) AS n,
                       sum(coalesce(i.actual_spent, 0)) AS total_spent,
                       avg(i.actual_spent) AS avg_spent,
                       avg(i.actual_satisfaction) AS avg_sat,
                       sum(CASE WHEN i.actual_satisfaction IS NOT NULL THEN 1 ELSE 0 END) AS sat_n
            """, d=d).single()
            if r and r['n'] and r['n'] > 0:
                daily.append({
                    "day": d,
                    "n_commerce": r['n'],
                    "total_spent": int(r['total_spent'] or 0),
                    "avg_spent": round(float(r['avg_spent']) if r['avg_spent'] else 0, 0),
                    "avg_sat": round(float(r['avg_sat']) if r['avg_sat'] else 0, 3),
                })

    # 차트 — 4-panel: 거래수·총소비·1인당 평균·평균 만족도
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    xs = [d['day'][5:] for d in daily]
    axes[0,0].bar(xs, [d['n_commerce'] for d in daily], color="#4361ee")
    axes[0,0].set_title("일별 commerce 거래 수")
    axes[0,0].set_ylabel("건")
    axes[0,1].bar(xs, [d['total_spent']/1e6 for d in daily], color="#3a0ca3")
    axes[0,1].set_title("일별 총 commerce 소비 (백만원)")
    axes[1,0].plot(xs, [d['avg_spent'] for d in daily], 'o-', color="#f72585", linewidth=2)
    axes[1,0].set_title("거래 1건당 평균 소비액 (원)")
    axes[1,0].set_ylabel("원")
    axes[1,1].plot(xs, [d['avg_sat'] for d in daily], 'o-', color="#ffb703", linewidth=2)
    axes[1,1].set_title("거래 평균 만족도")
    axes[1,1].set_ylim(0.4, 0.9)
    axes[1,1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    fname = "fig2_baseline_daily.png"
    plt.savefig(out_dir / fname, dpi=140, bbox_inches="tight")
    plt.close()

    # variation 분석 — 안정성 지표
    spent_avgs = [d['avg_spent'] for d in daily]
    sat_avgs = [d['avg_sat'] for d in daily]
    stability = {
        "spent_avg_min": int(min(spent_avgs)) if spent_avgs else 0,
        "spent_avg_max": int(max(spent_avgs)) if spent_avgs else 0,
        "spent_variation_pct": round((max(spent_avgs) - min(spent_avgs)) / min(spent_avgs) * 100, 2) if spent_avgs and min(spent_avgs) > 0 else 0,
        "sat_min": min(sat_avgs) if sat_avgs else 0,
        "sat_max": max(sat_avgs) if sat_avgs else 0,
        "sat_range": round(max(sat_avgs) - min(sat_avgs), 3) if sat_avgs else 0,
    }
    return {"daily": daily, "stability": stability}, {"daily": fname}


def section3_baseline_distribution(start: date, days: int, out_dir: Path) -> tuple[dict, str]:
    """자치구·소득별 분포 — 4일 누적 commerce 소비.

    인구·소비 분포가 자치구·소득군별로 어떻게 잡혔는지.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    day_strs = [(start + timedelta(days=i)).isoformat() for i in range(days)]

    with driver_session() as s:
        # 자치구별 4일 누적 소비
        gu_rows = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI {type:'commerce'})
            WHERE toString(p.day) IN $days
            RETURN a.residence_gu AS gu, sum(coalesce(i.actual_spent, 0)) AS total,
                   count(i) AS n, count(DISTINCT a) AS n_agent
            ORDER BY total DESC
        """, days=day_strs).data()
        # 소득별 4일 일평균 소비
        income_rows = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI {type:'commerce'})
            WHERE toString(p.day) IN $days
            RETURN a.p_income_level AS inc, sum(coalesce(i.actual_spent, 0)) AS total,
                   count(DISTINCT a) AS n_agent
        """, days=day_strs).data()
        # 연령별
        age_rows = s.run("""
            MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI {type:'commerce'})
            WHERE toString(p.day) IN $days
            RETURN a.p_age_group AS age, sum(coalesce(i.actual_spent, 0)) AS total,
                   count(DISTINCT a) AS n_agent
        """, days=day_strs).data()

    # 차트
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    # 자치구
    gus = [r['gu'] for r in gu_rows[:25]]
    totals = [r['total']/1e6 for r in gu_rows[:25]]
    axes[0].barh(gus[::-1], totals[::-1], color="#4361ee")
    axes[0].set_title("자치구별 4일 누적 commerce (백만원)")
    axes[0].set_xlabel("백만원")
    # 소득별 (1인당 일평균)
    INC_ORDER = ['상','중상','중','중하','하']
    inc_map = {r['inc']: r for r in income_rows if r['inc']}
    inc_per_capita = []
    for k in INC_ORDER:
        r = inc_map.get(k)
        if r and r['n_agent'] > 0:
            inc_per_capita.append(r['total'] / r['n_agent'] / days)
        else:
            inc_per_capita.append(0)
    axes[1].bar(INC_ORDER, inc_per_capita, color="#3a0ca3")
    axes[1].set_title("소득군별 1인당 일평균 commerce (원)")
    axes[1].set_ylabel("원/일/명")
    # 연령별 (1인당 일평균)
    AGE_ORDER = ['10대','20대','30대','40대','50대','60대','70대이상']
    age_map = {r['age']: r for r in age_rows if r['age']}
    age_per_capita = []
    for k in AGE_ORDER:
        r = age_map.get(k)
        if r and r['n_agent'] > 0:
            age_per_capita.append(r['total'] / r['n_agent'] / days)
        else:
            age_per_capita.append(0)
    axes[2].bar(AGE_ORDER, age_per_capita, color="#f72585")
    axes[2].set_title("연령대별 1인당 일평균 commerce (원)")
    axes[2].set_ylabel("원/일/명")
    plt.setp(axes[2].xaxis.get_majorticklabels(), rotation=30, ha='right')
    plt.tight_layout()
    fname = "fig3_baseline_distribution.png"
    plt.savefig(out_dir / fname, dpi=140, bbox_inches="tight")
    plt.close()

    return {
        "by_gu": [{"gu": r['gu'], "total": int(r['total']), "n_agent": r['n_agent']} for r in gu_rows[:10]],
        "by_income": [{"inc": k, "per_capita_daily": int(v)} for k, v in zip(INC_ORDER, inc_per_capita)],
        "by_age": [{"age": k, "per_capita_daily": int(v)} for k, v in zip(AGE_ORDER, age_per_capita)],
    }, fname


def find_satisfaction_sample(label: str, days_strs: list[str]) -> str | None:
    """baseline 인터뷰 라벨링 — 정책 무관 만족도 기반.

    positive: 4일 평균 만족도 상위 + 거래 ≥ 6건 + 외출 다양 (단골 아닌 POI 방문 多)
    negative: 4일 평균 만족도 하위 + 거래 ≥ 6건
    neutral : 4일 평균 만족도 0.5 ± 0.05 + 거래 ≥ 6건
    """
    if label not in ("positive", "negative", "neutral"):
        return None
    with driver_session() as s:
        if label == "positive":
            rows = s.run("""
                MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI)
                WHERE toString(p.day) IN $days AND i.actual_satisfaction IS NOT NULL
                WITH a, avg(i.actual_satisfaction) AS sat, count(i) AS visits
                WHERE sat >= 0.75 AND visits >= 8
                RETURN a.id AS id ORDER BY sat DESC, visits DESC LIMIT 30
            """, days=days_strs).data()
        elif label == "negative":
            rows = s.run("""
                MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI)
                WHERE toString(p.day) IN $days AND i.actual_satisfaction IS NOT NULL
                WITH a, avg(i.actual_satisfaction) AS sat, count(i) AS visits
                WHERE sat <= 0.45 AND visits >= 6
                RETURN a.id AS id ORDER BY sat ASC, visits DESC LIMIT 30
            """, days=days_strs).data()
        else:  # neutral
            rows = s.run("""
                MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI)
                WHERE toString(p.day) IN $days AND i.actual_satisfaction IS NOT NULL
                WITH a, avg(i.actual_satisfaction) AS sat, count(i) AS visits
                WHERE sat >= 0.55 AND sat <= 0.65 AND visits >= 6
                RETURN a.id AS id ORDER BY visits DESC LIMIT 30
            """, days=days_strs).data()
    if not rows:
        return None
    import random
    random.seed(hash(label))
    return random.choice(rows)['id']
