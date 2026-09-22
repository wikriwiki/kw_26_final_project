"""v4 시뮬 (Midm-2.0 + 모든 fix) 분석·차트."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
from _common import driver_session  # noqa: E402

for candidate in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(candidate in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = candidate
        break
plt.rcParams["axes.unicode_minus"] = False

DAYS = ["2026-05-01", "2026-05-02", "2026-05-03"]
DAY_LABELS = ["Day 0 (금)", "Day 1 (토)", "Day 2 (일)"]
SIM_DIR = Path("C:/Users/Administrator/sim_output/metrics")
OUT_DIR = Path("G:/내 드라이브/Kw/final_project/output/sim/report/REPORT_3D_V4_MIDM.d")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_jsonl(day):
    seen = {}
    with open(SIM_DIR / f"day_{day}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                j = json.loads(line)
                if j.get("status") == "ok":
                    seen[j["aid"]] = j
            except: pass
    return list(seen.values())


def pct(vs, p):
    s = sorted(vs); return s[min(int(len(s)*p), len(s)-1)] if s else 0


def section_summary():
    print("=" * 70); print("Section 1: Day별 기본 통계"); print("=" * 70)
    out = []
    for d, lbl in zip(DAYS, DAY_LABELS):
        rows = load_jsonl(d)
        n = len(rows)
        n_events = [r.get("n_events", 0) for r in rows]
        balance = [r.get("balance", 0) for r in rows]
        fatigue = [r.get("fatigue", 0) for r in rows]
        tok_in = [r.get("tokens_in", 0) for r in rows]
        rl = [r.get("review_lookup_count", 0) for r in rows]
        s1 = [r.get("s1_attempts", 1) for r in rows]
        s2 = [r.get("s2_attempts", 1) for r in rows]
        el = [r.get("elapsed") for r in rows if r.get("elapsed")]
        t_s1 = [r.get("timing_t_s1") for r in rows if r.get("timing_t_s1")]
        t_s2 = [r.get("timing_t_s2") for r in rows if r.get("timing_t_s2")]
        nz_rl = [x for x in rl if x > 0]
        out.append(dict(
            day=lbl, n=n,
            avg_events=mean(n_events), avg_balance=mean(balance), avg_fatigue=mean(fatigue),
            avg_tok=mean(tok_in), avg_s1=mean(s1), avg_s2=mean(s2),
            avg_elapsed=mean(el) if el else 0,
            avg_t_s1=mean(t_s1) if t_s1 else 0,
            avg_t_s2=mean(t_s2) if t_s2 else 0,
            review_lookup_pct=len(nz_rl)*100/max(n,1),
            review_lookup_avg_poi=mean(nz_rl) if nz_rl else 0,
        ))
        print(f"{lbl}: n={n:,}, 외출 {mean(n_events):.2f}, 잔액 {int(mean(balance)):,}, fatigue {mean(fatigue):.2f}")
        print(f"  Stage1·2 시도: {mean(s1):.2f}/{mean(s2):.2f} | timing S1 {mean(t_s1):.1f}s + S2 {mean(t_s2):.1f}s | 총 {mean(el):.1f}s")
        print(f"  review_lookup: {len(nz_rl)*100/max(n,1):.1f}% 발동, 발동 시 평균 {mean(nz_rl) if nz_rl else 0:.1f} POI")
    return out


def section_factor():
    print("\n" + "=" * 70); print("Section 2: pick_factor + 리뷰 언급"); print("=" * 70)
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
            n_review = sum(1 for r in rows if any(k in (r["reason"] or "").lower()
                            for k in ["리뷰","review","평점","별점","★","후기"]))
            out[d] = dict(n=n, counter=counter, review_mentions=n_review)
            print(f"Day {d}: n={n:,} / 리뷰 언급 {n_review} ({n_review*100/max(n,1):.1f}%)")
            for k, v in counter.most_common(7):
                print(f"  {k}: {v:,} ({v*100/max(n,1):.1f}%)")
    return out


def section_dong():
    print("\n" + "=" * 70); print("Section 3: 거주/직장/그외 외출 dong 분포"); print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            r = s.run(f'''
                MATCH (a:Agent)-[:HAS_PLAN]->(p:Plan {{day: date("{d}")}})-[rel:INCLUDES]->(poi:POI)
                WHERE poi.dong_code IS NOT NULL AND a.residence_dong_code_raw IS NOT NULL
                WITH a.residence_dong_code_raw AS home, a.workplace_dong_code_raw AS work, poi.dong_code AS visit
                RETURN count(*) AS total,
                       sum(CASE WHEN visit = home THEN 1 ELSE 0 END) AS in_home,
                       sum(CASE WHEN visit = work AND visit <> home THEN 1 ELSE 0 END) AS in_work,
                       sum(CASE WHEN visit <> home AND (work IS NULL OR visit <> work) THEN 1 ELSE 0 END) AS in_other
            ''').single()
            out[d] = dict(total=r['total'], home=r['in_home'], work=r['in_work'], other=r['in_other'])
            t=r['total']; h=r['in_home']; w=r['in_work']; o=r['in_other']
            print(f"Day {d}: 총 {t:,}건 | 거주 {h*100/t:.1f}% / 직장 {w*100/t:.1f}% / 그외 {o*100/t:.1f}%")
    return out


def section_conversation():
    print("\n" + "=" * 70); print("Section 4: Night Phase Conversation"); print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            convs = s.run(f'''
                MATCH (c:Conversation {{day: date("{d}")}})
                RETURN c.intent AS intent, c.topic_type AS topic,
                       c.ambient_threshold_applied AS ambient,
                       c.relationship_score AS rel, c.exposure_score AS exp,
                       c.urgency_score AS urg, c.interaction_score AS itx
            ''').data()
            n = len(convs)
            if n == 0:
                out[d] = dict(n=0); continue
            intent_dist = Counter(c["intent"] for c in convs)
            topic_dist = Counter(c["topic"] for c in convs)
            ambient_n = sum(1 for c in convs if c["ambient"])
            avg_rel = mean(c["rel"] for c in convs if c["rel"] is not None)
            avg_exp = mean(c["exp"] for c in convs if c["exp"] is not None)
            avg_urg = mean(c["urg"] for c in convs if c["urg"] is not None)
            avg_itx = mean(c["itx"] for c in convs if c["itx"] is not None)
            out[d] = dict(n=n, intent_dist=intent_dist, topic_dist=topic_dist,
                          ambient_n=ambient_n, avg_rel=avg_rel, avg_exp=avg_exp,
                          avg_urg=avg_urg, avg_itx=avg_itx)
            print(f"Day {d}: Conversation {n:,}")
            print(f"  intent: {dict(intent_dist.most_common())}")
            print(f"  topic: {dict(topic_dist.most_common())}")
            print(f"  ambient: {ambient_n}/{n} ({ambient_n*100/n:.1f}%)")
            print(f"  rel/exp/urg/itx 평균: {avg_rel:.3f} / {avg_exp:.3f} / {avg_urg:.3f} / {avg_itx:.3f}")
    return out


def section_spending():
    print("\n" + "=" * 70); print("Section 5: 소비"); print("=" * 70)
    out = {}
    with driver_session() as s:
        for d in DAYS:
            r = s.run(f'''
                MATCH (p:Plan {{day: date("{d}")}})-[r:INCLUDES]->()
                WHERE r.actual_spent IS NOT NULL AND r.actual_spent > 0
                RETURN sum(r.actual_spent) AS total, avg(r.actual_spent) AS avg,
                       percentileCont(r.actual_spent, 0.5) AS p50,
                       percentileCont(r.actual_spent, 0.9) AS p90, count(*) AS n
            ''').single()
            out[d] = dict(total=int(r['total'] or 0), avg=int(r['avg'] or 0),
                          p50=int(r['p50'] or 0), p90=int(r['p90'] or 0), n=r['n'])
            print(f"Day {d}: 총 {r['total']/1e6:.1f}M원 / n={r['n']:,} / avg {int(r['avg'] or 0):,}원 / p50 {int(r['p50'] or 0):,} / p90 {int(r['p90'] or 0):,}")
    return out


def section_knows_poi():
    """KNOWS_POI 누적 (workplace + KNOWS 시드 + actual_satisfaction strict 효과)."""
    print("\n" + "=" * 70); print("Section 6: KNOWS_POI 누적 검증"); print("=" * 70)
    with driver_session() as s:
        r = s.run('''
            MATCH ()-[r:KNOWS_POI]->()
            RETURN count(r) AS total,
                   sum(CASE WHEN r.visit_count > 0 THEN 1 ELSE 0 END) AS visited,
                   sum(CASE WHEN r.avg_satisfaction IS NOT NULL THEN 1 ELSE 0 END) AS with_sat,
                   avg(CASE WHEN r.avg_satisfaction IS NOT NULL THEN r.avg_satisfaction END) AS avg_sat
        ''').single()
        print(f"KNOWS_POI 총: {r['total']:,}")
        print(f"  visit_count > 0: {r['visited']:,} ({r['visited']*100/r['total']:.1f}%) ← v3에서 4건만이었던 거 검증")
        print(f"  avg_satisfaction 채움: {r['with_sat']:,}")
        if r['avg_sat']: print(f"  avg_satisfaction 평균: {r['avg_sat']:.3f}")
        return dict(total=r['total'], visited=r['visited'], with_sat=r['with_sat'],
                    avg_sat=float(r['avg_sat']) if r['avg_sat'] else None)


def section_timing():
    """단계별 timing 평균 — 병목 위치."""
    print("\n" + "=" * 70); print("Section 7: 단계별 병목 (timing)"); print("=" * 70)
    out = {}
    for d in DAYS:
        rows = load_jsonl(d)
        ts = {k: [r.get(f"timing_t_{k}") for r in rows if r.get(f"timing_t_{k}") is not None]
              for k in ["dawn", "s1", "s2", "write_plan", "night_finalize"]}
        el = [r.get("elapsed") for r in rows if r.get("elapsed")]
        out[d] = {k: (mean(v) if v else 0) for k, v in ts.items()}
        out[d]["elapsed"] = mean(el) if el else 0
        avg_el = mean(el) if el else 1
        print(f"Day {d}: elapsed {avg_el:.1f}s")
        for k, v in ts.items():
            if v: print(f"  t_{k}: {mean(v):.2f}s ({mean(v)*100/avg_el:.1f}%)")
    return out


# ─────────── 차트 ───────────

def chart_factor_evolution(factor_dist):
    factors = ["distance", "satisfaction", "known", "appointment", "rumor", "random", "review"]
    colors = ["#5B8FF9","#5AD8A6","#5D7092","#F6BD16","#E8684A","#9270CA","#FF6B6B"]
    data = {f: [] for f in factors}
    for d in DAYS:
        n = factor_dist[d]["n"]
        for f in factors:
            data[f].append(factor_dist[d]["counter"].get(f, 0)*100/max(n,1))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(3)
    for f, c in zip(factors, colors):
        ax.bar(DAY_LABELS, data[f], bottom=bottom, label=f, color=c, edgecolor="white")
        bottom += np.array(data[f])
    ax.set_ylabel("비율 (%)"); ax.set_title("Day별 의사결정 factor 진화 (v4)", pad=12, fontweight="bold")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.set_ylim(0, 105); plt.tight_layout()
    fig.savefig(OUT_DIR / "fig1_factor_evolution.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → fig1_factor_evolution.png")


def chart_dong_distribution(dong):
    fig, ax = plt.subplots(figsize=(9, 5))
    home = [dong[d]['home']*100/dong[d]['total'] for d in DAYS]
    work = [dong[d]['work']*100/dong[d]['total'] for d in DAYS]
    other = [dong[d]['other']*100/dong[d]['total'] for d in DAYS]
    x = np.arange(3); w = 0.27
    ax.bar(x-w, home, w, label="거주 행정동", color="#5B8FF9")
    ax.bar(x, work, w, label="직장 행정동", color="#F6BD16")
    ax.bar(x+w, other, w, label="그 외 (광역·타동)", color="#5AD8A6")
    for i, vals in enumerate(zip(home, work, other)):
        for j, v in enumerate(vals):
            ax.text(i + (j-1)*w, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(DAY_LABELS); ax.set_ylabel("비율 (%)")
    ax.set_title("Day별 외출 dong 분포 (v4 — workplace 81% 충원 효과)", pad=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / "fig2_dong_distribution.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → fig2_dong_distribution.png")


def chart_conversation(conv):
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ["약속", "추천", "기타", "이슈"]
    colors = ["#F6BD16", "#5AD8A6", "#5D7092", "#E8684A"]
    data = {c: [] for c in cats}
    for d in DAYS:
        info = conv[d]
        if info.get('n', 0) == 0:
            for c in cats: data[c].append(0)
            continue
        intent_dist = info.get('intent_dist', Counter())
        for c in cats: data[c].append(intent_dist.get(c, 0))
    x = np.arange(3); w = 0.2
    for i, (c, col) in enumerate(zip(cats, colors)):
        ax.bar(x + (i-1.5)*w, data[c], w, label=c, color=col)
    ax.set_xticks(x); ax.set_xticklabels(DAY_LABELS); ax.set_ylabel("Conversation 수")
    ax.set_title("Day별 Night Phase intent 분류 (v4 — KNOWS 시드 효과)", pad=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / "fig3_conversation_intent.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → fig3_conversation_intent.png")


def chart_timing(timing):
    fig, ax = plt.subplots(figsize=(9, 5))
    stages = ["dawn", "s1", "s2", "write_plan", "night_finalize"]
    colors = ["#5D7092", "#5B8FF9", "#F6BD16", "#5AD8A6", "#E8684A"]
    bottom = np.zeros(3)
    for stg, c in zip(stages, colors):
        vals = [timing[d][stg] for d in DAYS]
        ax.bar(DAY_LABELS, vals, bottom=bottom, label=f"t_{stg}", color=c)
        bottom += np.array(vals)
    ax.set_ylabel("시간 (sec)"); ax.set_title("Day별 단계별 timing", pad=12, fontweight="bold")
    ax.legend(); plt.tight_layout()
    fig.savefig(OUT_DIR / "fig4_timing_stages.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → fig4_timing_stages.png")


def chart_spending(spending):
    fig, ax = plt.subplots(figsize=(8, 5))
    totals = [spending[d]['total']/1e6 for d in DAYS]
    avgs = [spending[d]['avg']/1000 for d in DAYS]
    ax2 = ax.twinx()
    ax.bar(DAY_LABELS, totals, color="#5B8FF9", alpha=0.7, label="총 소비 (백만원)")
    ax2.plot(DAY_LABELS, avgs, "o-", color="#E8684A", linewidth=2, markersize=10, label="평균 거래액 (천원)")
    ax.set_ylabel("총 소비 (백만원)", color="#5B8FF9")
    ax2.set_ylabel("평균 거래액 (천원)", color="#E8684A")
    ax.set_title("Day별 소비 패턴", pad=12, fontweight="bold")
    for i, v in enumerate(totals):
        ax.text(i, v + 5, f"{v:.0f}M", ha="center", fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "fig5_spending.png", dpi=120, bbox_inches="tight"); plt.close()
    print(f"  → fig5_spending.png")


def main():
    print(f"\n=== 출력 디렉토리: {OUT_DIR} ===\n")
    summary = section_summary()
    factor = section_factor()
    dong = section_dong()
    conv = section_conversation()
    spending = section_spending()
    knows_poi = section_knows_poi()
    timing = section_timing()

    print("\n=== 차트 ===")
    chart_factor_evolution(factor)
    chart_dong_distribution(dong)
    chart_conversation(conv)
    chart_timing(timing)
    chart_spending(spending)

    result = dict(
        summary=summary,
        factor={d: dict(n=fd["n"], counter=dict(fd["counter"]),
                        review_mentions=fd["review_mentions"]) for d, fd in factor.items()},
        dong=dong,
        conversation={d: ({k: (dict(v) if isinstance(v, Counter) else v)
                          for k, v in c.items()}) for d, c in conv.items()},
        spending=spending,
        knows_poi=knows_poi,
        timing=timing,
    )
    with open(OUT_DIR / "data.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n→ data.json")


if __name__ == "__main__":
    main()
