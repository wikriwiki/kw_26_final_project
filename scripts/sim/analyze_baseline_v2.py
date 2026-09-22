"""3일 v2 baseline 시뮬 (메모리 제로 + 무정책 + 새 SYSTEM_S2) 분석·차트."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
from _common import driver_session  # noqa: E402

# 한글 폰트
import matplotlib.font_manager as fm
for candidate in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(candidate in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False

DAYS = ["2026-05-01", "2026-05-02", "2026-05-03"]
DAY_LABELS = ["Day 0", "Day 1", "Day 2"]
SIM_DIR = Path("C:/Users/Administrator/sim_output/metrics")
OUT_DIR = Path("G:/내 드라이브/Kw/final_project/output/sim/report/REPORT_3D_BASELINE_V2.d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(day: str) -> list[dict]:
    """resume 중복 제거하고 마지막 status=ok 한 번씩만."""
    f = SIM_DIR / f"day_{day}.jsonl"
    rows = []
    seen = {}
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                j = json.loads(line)
                aid = j.get("aid")
                if j.get("status") == "ok":
                    seen[aid] = j  # 마지막 ok overwrite
            except Exception:
                pass
    return list(seen.values())


def section_summary():
    """Day별 기본 통계."""
    print("=" * 70)
    print("Section 1: Day별 기본 통계")
    print("=" * 70)
    stats = []
    for d, lbl in zip(DAYS, DAY_LABELS):
        rows = load_jsonl(d)
        n = len(rows)
        n_events = [r.get("n_events", 0) for r in rows]
        balance = [r.get("balance", 0) for r in rows]
        fatigue = [r.get("fatigue", 0) for r in rows]
        tok_in = [r.get("tokens_in", 0) for r in rows]
        tok_out = [r.get("tokens_out", 0) for r in rows]
        s1 = [r.get("s1_attempts", 1) for r in rows]
        s2 = [r.get("s2_attempts", 1) for r in rows]
        zero_ev = sum(1 for x in n_events if x == 0)
        stats.append(dict(
            day=lbl, n=n,
            avg_events=mean(n_events), zero_events=zero_ev,
            avg_balance=mean(balance), avg_fatigue=mean(fatigue),
            avg_tok_in=mean(tok_in), avg_tok_out=mean(tok_out),
            avg_s1=mean(s1), avg_s2=mean(s2),
        ))
    return stats


def section_factor_dist():
    """Day별 pick_factor 분포."""
    print("=" * 70)
    print("Section 2: pick_factor 분포")
    print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            rows = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.pick_factor IS NOT NULL
                RETURN r.pick_factor AS factor, r.pick_reason AS reason
            ''').data()
            n = len(rows)
            counter = Counter(r["factor"] for r in rows)
            n_review = sum(
                1 for r in rows
                if any(k in (r["reason"] or "").lower()
                       for k in ["리뷰", "review", "평점", "별점", "★", "후기"])
            )
            out[d] = dict(n=n, counter=counter, review_mentions=n_review)
    return out


def section_visits():
    """Day별 외출 이벤트, 카테고리 분포."""
    print("=" * 70)
    print("Section 3: 외출 이벤트·카테고리")
    print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            n_visits = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                RETURN count(r) AS n
            ''').single()["n"]
            outed_agents = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[:INCLUDES]->()
                RETURN count(DISTINCT p) AS n
            ''').single()["n"]
            # 카테고리
            cats = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.category IS NOT NULL
                RETURN r.category AS cat, count(*) AS n
                ORDER BY n DESC
            ''').data()
            # intent 분포
            intents = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.intent IS NOT NULL
                RETURN r.intent AS intent, count(*) AS n
                ORDER BY n DESC LIMIT 10
            ''').data()
            # 평균 외출 시간(anchor)
            anchors = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.anchor IS NOT NULL
                RETURN r.anchor AS anchor, count(*) AS n
                ORDER BY n DESC LIMIT 8
            ''').data()
            out[d] = dict(
                n_visits=n_visits, outed_agents=outed_agents,
                categories=cats, intents=intents, anchors=anchors
            )
    return out


def section_conversation():
    """Day별 Conversation 분류·통계."""
    print("=" * 70)
    print("Section 4: 사회적 상호작용 (Conversation)")
    print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            convs = s.run(f'''
                MATCH (c:Conversation {{day: date("{d}")}})
                RETURN c.topic_type AS topic, c.threshold_used AS thr,
                       c.ambient_threshold_applied AS ambient,
                       c.urgency_score AS urg, c.exposure_score AS exp,
                       c.relationship_score AS rel, c.interaction_score AS itx
            ''').data()
            n = len(convs)
            if n == 0:
                out[d] = dict(n=0)
                continue
            topic_dist = Counter(c["topic"] for c in convs)
            ambient_n = sum(1 for c in convs if c["ambient"])
            avg_thr = mean(c["thr"] for c in convs if c["thr"] is not None)
            avg_urg = mean(c["urg"] for c in convs if c["urg"] is not None) if any(c["urg"] for c in convs) else 0
            avg_exp = mean(c["exp"] for c in convs if c["exp"] is not None) if any(c["exp"] for c in convs) else 0
            avg_rel = mean(c["rel"] for c in convs if c["rel"] is not None) if any(c["rel"] for c in convs) else 0
            avg_itx = mean(c["itx"] for c in convs if c["itx"] is not None) if any(c["itx"] for c in convs) else 0
            out[d] = dict(
                n=n, topic_dist=topic_dist, ambient_n=ambient_n,
                avg_thr=avg_thr, avg_urg=avg_urg, avg_exp=avg_exp,
                avg_rel=avg_rel, avg_itx=avg_itx
            )
    return out


def section_spending():
    """Day별 소비 분석."""
    print("=" * 70)
    print("Section 5: 소비 패턴")
    print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            r = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.actual_spent IS NOT NULL AND r.actual_spent > 0
                RETURN sum(r.actual_spent) AS total_spent,
                       avg(r.actual_spent) AS avg_spent,
                       percentileCont(r.actual_spent, 0.5) AS p50,
                       percentileCont(r.actual_spent, 0.9) AS p90,
                       count(*) AS n
            ''').single()
            out[d] = dict(
                total_spent=int(r["total_spent"] or 0),
                avg_spent=int(r["avg_spent"] or 0),
                p50=int(r["p50"] or 0),
                p90=int(r["p90"] or 0),
                n=r["n"]
            )
    return out


def chart_factor_evolution(factor_dist):
    """Day별 factor 분포 변화 누적 막대 차트."""
    factors_order = ["distance", "satisfaction", "known", "appointment", "rumor", "random", "review"]
    colors = ["#5B8FF9", "#5AD8A6", "#5D7092", "#F6BD16", "#E8684A", "#9270CA", "#FF6B6B"]
    data = {f: [] for f in factors_order}
    for d in DAYS:
        n = factor_dist[d]["n"]
        for f in factors_order:
            v = factor_dist[d]["counter"].get(f, 0)
            data[f].append(v * 100 / max(n, 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(3)
    for f, c in zip(factors_order, colors):
        ax.bar(DAY_LABELS, data[f], bottom=bottom, label=f, color=c, edgecolor="white")
        bottom += np.array(data[f])
    ax.set_ylabel("비율 (%)")
    ax.set_title("Day별 의사결정 factor 분포 진화", pad=12, fontsize=13, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_factor_evolution.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → fig1_factor_evolution.png")


def chart_review_usage(factor_dist):
    """리뷰·별점 언급 추이."""
    review_pct = [
        factor_dist[d]["review_mentions"] * 100 / max(factor_dist[d]["n"], 1)
        for d in DAYS
    ]
    factor_review = [
        factor_dist[d]["counter"].get("review", 0) * 100 / max(factor_dist[d]["n"], 1)
        for d in DAYS
    ]
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, review_pct, w, label="reasoning에 리뷰·별점 언급", color="#5B8FF9")
    ax.bar(x + w/2, factor_review, w, label="factor=review (명시적)", color="#F6BD16")
    ax.set_xticks(x)
    ax.set_xticklabels(DAY_LABELS)
    ax.set_ylabel("비율 (%)")
    ax.set_title("Day별 별점·리뷰 활용 추이", pad=12, fontsize=13, fontweight="bold")
    for i, (a, b) in enumerate(zip(review_pct, factor_review)):
        ax.text(i - w/2, a + 0.1, f"{a:.1f}%", ha="center", fontsize=9)
        ax.text(i + w/2, b + 0.1, f"{b:.1f}%", ha="center", fontsize=9)
    ax.legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_review_usage.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → fig2_review_usage.png")


def chart_state_trend(summary):
    """Day별 잔액·피로 추이."""
    balance = [s["avg_balance"] / 1000 for s in summary]
    fatigue = [s["avg_fatigue"] for s in summary]
    events = [s["avg_events"] for s in summary]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(DAY_LABELS, balance, "o-", color="#5B8FF9", linewidth=2, markersize=10, label="평균 잔액 (천원)")
    ax1.set_ylabel("평균 잔액 (천원)", color="#5B8FF9")
    ax1.tick_params(axis="y", labelcolor="#5B8FF9")
    ax2 = ax1.twinx()
    ax2.plot(DAY_LABELS, fatigue, "s-", color="#E8684A", linewidth=2, markersize=10, label="평균 피로")
    ax2.plot(DAY_LABELS, events, "^-", color="#5AD8A6", linewidth=2, markersize=10, label="평균 외출 이벤트")
    ax2.set_ylabel("피로 / 외출 이벤트")
    ax2.tick_params(axis="y", labelcolor="#666")
    ax1.set_title("Day별 잔액·피로·외출 추이", pad=12, fontsize=13, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_state_trend.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → fig3_state_trend.png")


def chart_category_top(visits):
    """Day 0 vs Day 2 상위 카테고리 분포."""
    cats_d0 = visits[DAYS[0]]["categories"][:10]
    cats_d2 = visits[DAYS[2]]["categories"][:10]
    # union 카테고리
    union = []
    seen = set()
    for c in cats_d0 + cats_d2:
        if c["cat"] not in seen:
            union.append(c["cat"])
            seen.add(c["cat"])
    union = union[:8]
    d0_map = {c["cat"]: c["n"] for c in cats_d0}
    d2_map = {c["cat"]: c["n"] for c in cats_d2}
    d0_vals = [d0_map.get(c, 0) for c in union]
    d2_vals = [d2_map.get(c, 0) for c in union]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(union))
    w = 0.35
    ax.bar(x - w/2, d0_vals, w, label="Day 0", color="#5B8FF9")
    ax.bar(x + w/2, d2_vals, w, label="Day 2", color="#F6BD16")
    ax.set_xticks(x)
    ax.set_xticklabels(union, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("외출 이벤트 수")
    ax.set_title("카테고리 분포 (Day 0 vs Day 2)", pad=12, fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_category_dist.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → fig4_category_dist.png")


def chart_conversation_trend(conv):
    """Day별 Conversation 분류 추이."""
    topics = ["약속", "이슈", "추천", "기타"]
    topic_keys = ["promise", "issue", "recommend", "casual"]  # 추정 — sim에서 사용한 key 확인
    # topic_type 키 확인 후 정확한 매핑 필요. 일단 raw counter 사용.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    n_total = [conv[d]["n"] if conv[d].get("n") else 0 for d in DAYS]
    ax.bar(DAY_LABELS, n_total, color=["#5B8FF9", "#5AD8A6", "#F6BD16"], edgecolor="white")
    for i, v in enumerate(n_total):
        ax.text(i, v + 5, f"{v}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Conversation pair 수")
    ax.set_title("Day별 사회적 상호작용 수", pad=12, fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_conv_trend.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  → fig5_conv_trend.png")


def main():
    print(f"\n출력 디렉토리: {OUT_DIR}\n")
    summary = section_summary()
    factor_dist = section_factor_dist()
    visits = section_visits()
    conv = section_conversation()
    spending = section_spending()

    print("\n=== 차트 생성 ===")
    chart_factor_evolution(factor_dist)
    chart_review_usage(factor_dist)
    chart_state_trend(summary)
    chart_category_top(visits)
    chart_conversation_trend(conv)

    # JSON dump
    result = dict(
        summary=summary,
        factor_dist={d: dict(n=fd["n"],
                              counter=dict(fd["counter"]),
                              review_mentions=fd["review_mentions"])
                     for d, fd in factor_dist.items()},
        visits={d: dict(n_visits=v["n_visits"],
                         outed_agents=v["outed_agents"],
                         categories=v["categories"][:10],
                         intents=v["intents"],
                         anchors=v["anchors"])
                for d, v in visits.items()},
        conversation={d: ({k: (dict(v) if isinstance(v, Counter) else v)
                          for k, v in c.items()})
                      for d, c in conv.items()},
        spending=spending,
    )
    with open(OUT_DIR / "data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n→ data.json 저장: {OUT_DIR / 'data.json'}")

    return result


if __name__ == "__main__":
    main()
