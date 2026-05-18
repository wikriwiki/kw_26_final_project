"""풀런 종료 후 최종 보고서 자동 생성.

섹션:
  1. 시뮬 조건 요약
  2. 정책 시행 전 vs 후 매출 추이 (DID + 시간축)
  3. 간접 영향 지역 (spillover) — 강남 인접 자치구
  4. 소비자 행동·심리 분석
     4-1. 방문 목적·이동 패턴 (trigger 분포)
     4-2. 단골 vs 신규
     4-3. 만족도·피드백 (trigger·카테고리별)
  5. 1대1 인터뷰 (positive/negative/neutral 각 1명, 5~6 질문)

산출:
  - <out_md>           : 단일 markdown
  - <out_md>.d/*.png   : 차트 (md에서 참조)

CLI:
  python scripts/sim/generate_final_report.py \\
      --start 2026-05-01 --days 7 --policy-from 2026-05-02 \\
      --out docs/FINAL_REPORT_7D.md
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session  # noqa: E402


# ═══════════════════════════════════════════════════════════════
# matplotlib 한글 폰트
# ═══════════════════════════════════════════════════════════════
def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")  # GUI 없이
    import matplotlib.pyplot as plt
    # Windows: Malgun Gothic, Mac: AppleGothic, Linux: NanumGothic
    for f in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
        try:
            plt.rcParams["font.family"] = f
            plt.rcParams["axes.unicode_minus"] = False
            break
        except Exception:
            continue
    return plt


# ═══════════════════════════════════════════════════════════════
# 섹션 1 — 시뮬 조건 요약
# ═══════════════════════════════════════════════════════════════
def section1_conditions(start: date, days: int, policy_from: str | None) -> dict:
    out = {
        "기간": f"{start.isoformat()} ~ {(start + timedelta(days=days-1)).isoformat()}",
        "일수": days,
        "정책_시행일": policy_from or "처음부터",
    }
    with driver_session() as s:
        for k, q in [
            ("Agent 수", "MATCH (a:Agent) RETURN count(a) AS n"),
            ("Plan 수", "MATCH (p:Plan) RETURN count(p) AS n"),
            ("INCLUDES 엣지", "MATCH ()-[i:INCLUDES]->() RETURN count(i) AS n"),
            ("State 수", "MATCH (s:State) RETURN count(s) AS n"),
            ("Memory(visited)", "MATCH (m:Memory {type:'visited'}) RETURN count(m) AS n"),
            ("Memory(rumor)", "MATCH (m:Memory {type:'rumor'}) RETURN count(m) AS n"),
            ("Conversation 약속", "MATCH (c:Conversation {intent:'약속'}) RETURN count(c) AS n"),
            ("Conversation 추천", "MATCH (c:Conversation {intent:'추천'}) RETURN count(c) AS n"),
            ("Conversation 이슈", "MATCH (c:Conversation {intent:'이슈'}) RETURN count(c) AS n"),
            ("Conversation 기타", "MATCH (c:Conversation {intent:'기타'}) RETURN count(c) AS n"),
        ]:
            out[k] = s.run(q).single()["n"]
    return out


# ═══════════════════════════════════════════════════════════════
# 섹션 2 — 정책 시행 전 vs 후 매출 추이
# ═══════════════════════════════════════════════════════════════
def section2_before_after(start: date, days: int, policy_from: str, out_dir: Path) -> tuple[dict, str]:
    cutoff = date.fromisoformat(policy_from)
    target_cats = ["식사", "카페", "디저트"]
    daily: list[dict] = []
    with driver_session() as s:
        for i in range(days):
            d = start + timedelta(days=i)
            rows = s.run("""
                MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
                  <-[:HAS_DONG]-(dist:District)
                MATCH (a)-[:HAS_PLAN {day: date($day)}]->(:Plan)
                  -[i:INCLUDES]->(p:POI {type:'commerce'})
                WHERE i.category IN $cats
                RETURN dist.code AS dist, sum(coalesce(i.actual_spent,0)) AS spend,
                       count(*) AS n
            """, day=d.isoformat(), cats=target_cats).data()
            gn = sum(r["spend"] for r in rows if r["dist"] == "11680")
            ng = sum(r["spend"] for r in rows if r["dist"] != "11680")
            daily.append({
                "day": d.isoformat(),
                "phase": "after" if d >= cutoff else "before",
                "gangnam_spend": gn,
                "non_gangnam_spend": ng,
            })

    # before/after 평균
    before = [x for x in daily if x["phase"] == "before"]
    after = [x for x in daily if x["phase"] == "after"]
    summary = {}
    if before and after:
        b_gn = sum(x["gangnam_spend"] for x in before) / len(before)
        b_ng = sum(x["non_gangnam_spend"] for x in before) / len(before)
        a_gn = sum(x["gangnam_spend"] for x in after) / len(after)
        a_ng = sum(x["non_gangnam_spend"] for x in after) / len(after)
        gn_chg = (a_gn - b_gn) / max(b_gn, 1) * 100
        ng_chg = (a_ng - b_ng) / max(b_ng, 1) * 100
        summary = {
            "before_gn_daily": round(b_gn), "after_gn_daily": round(a_gn),
            "before_ng_daily": round(b_ng), "after_ng_daily": round(a_ng),
            "gangnam_change_pct": round(gn_chg, 2),
            "non_gangnam_change_pct": round(ng_chg, 2),
            "DID_pct_points": round(gn_chg - ng_chg, 2),
        }

    # 차트
    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(len(daily)))
    labels = [x["day"][5:] for x in daily]
    gn = [x["gangnam_spend"] / 1e6 for x in daily]
    ng = [x["non_gangnam_spend"] / 1e6 for x in daily]
    ax.plot(xs, gn, marker="o", label="강남구 (정책 대상)", color="#e76f51", linewidth=2)
    ax.plot(xs, ng, marker="s", label="비강남 (대조군)", color="#4cc9f0", linewidth=2)
    # 정책 시행 시점 수직선
    try:
        cut_idx = next(i for i, x in enumerate(daily) if x["day"] == policy_from)
        ax.axvline(cut_idx - 0.5, color="#888", linestyle="--", linewidth=1)
        ax.text(cut_idx - 0.4, max(gn + ng) * 0.95, f"  정책 시행 {policy_from}",
                color="#aaa", fontsize=10)
    except StopIteration:
        pass
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("날짜")
    ax.set_ylabel("일별 매출 합계 (백만원)")
    ax.set_title("정책 시행 전 vs 후 매출 추이 — 식사·카페·디저트")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig2_before_after.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"daily": daily, "summary": summary}, str(path.name)


# ═══════════════════════════════════════════════════════════════
# 섹션 3 — spillover (강남 vs 인접 자치구)
# ═══════════════════════════════════════════════════════════════
def section3_spillover(start: date, days: int, out_dir: Path) -> tuple[dict, str]:
    # 강남(11680) + 인접 서초(11650)·송파(11710), 그리고 멀리 강북(11305) 비교
    groups = {"강남": "11680", "서초": "11650", "송파": "11710", "강북": "11305"}
    target_cats = ["식사", "카페", "디저트"]
    by_group: dict = {k: [] for k in groups}
    with driver_session() as s:
        for i in range(days):
            d = start + timedelta(days=i)
            rows = s.run("""
                MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
                  <-[:HAS_DONG]-(dist:District)
                MATCH (a)-[:HAS_PLAN {day: date($day)}]->(:Plan)
                  -[i:INCLUDES]->(p:POI {type:'commerce'})
                WHERE i.category IN $cats
                RETURN dist.code AS dist, sum(coalesce(i.actual_spent,0)) AS spend
            """, day=d.isoformat(), cats=target_cats).data()
            spend_by_dist = {r["dist"]: r["spend"] for r in rows}
            for name, code in groups.items():
                by_group[name].append(spend_by_dist.get(code, 0))

    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(11, 5))
    xs = list(range(days))
    labels = [(start + timedelta(days=i)).isoformat()[5:] for i in range(days)]
    colors = {"강남": "#e76f51", "서초": "#f4a261", "송파": "#f4d35e", "강북": "#8ecae6"}
    for name, vals in by_group.items():
        ax.plot(xs, [v / 1e6 for v in vals], marker="o", label=name, color=colors[name], linewidth=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_xlabel("날짜")
    ax.set_ylabel("매출 합계 (백만원)")
    ax.set_title("자치구별 매출 추이 — 강남 정책의 간접 영향 추적")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig3_spillover.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"by_group": {k: [int(v) for v in vs] for k, vs in by_group.items()}}, str(path.name)


# ═══════════════════════════════════════════════════════════════
# 섹션 4-1 — 방문 목적·이동 패턴 (trigger 분포)
# ═══════════════════════════════════════════════════════════════
def section4_1_triggers(start: date, days: int, out_dir: Path) -> tuple[dict, str]:
    with driver_session() as s:
        rows = s.run("""
            MATCH ()-[i:INCLUDES]->()
            WHERE i.trigger IS NOT NULL AND NOT i.category IN ['집','직장']
            RETURN i.trigger AS trigger, count(*) AS n
            ORDER BY n DESC
        """).data()
    total = sum(r["n"] for r in rows) or 1
    dist = {r["trigger"]: r["n"] for r in rows}
    dist_pct = {k: round(v / total * 100, 2) for k, v in dist.items()}

    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
                 "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션",
                 "none": "기타"}
    sorted_items = sorted(dist.items(), key=lambda x: -x[1])
    xs = [labels_kr.get(k, k) for k, _ in sorted_items]
    ys = [v for _, v in sorted_items]
    bars = ax.bar(xs, ys, color=["#e76f51", "#f4a261", "#4cc9f0", "#83c5be",
                                   "#06d6a0", "#9d4edd", "#888"][:len(xs)])
    for bar, n in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{n:,}\n({n/total*100:.1f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_title("외출 이벤트의 결정 동기 (trigger 분포)")
    ax.set_ylabel("이벤트 수")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig4_1_triggers.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"distribution": dist, "distribution_pct": dist_pct, "total": total}, str(path.name)


# ═══════════════════════════════════════════════════════════════
# 섹션 4-2 — 단골 vs 신규
# ═══════════════════════════════════════════════════════════════
def section4_2_regulars(start: date, days: int, out_dir: Path) -> tuple[dict, str]:
    with driver_session() as s:
        r = s.run("""
            MATCH ()-[kp:KNOWS_POI]->()
            WHERE kp.visit_count > 0
            RETURN
              sum(CASE WHEN kp.visit_count = 1 THEN 1 ELSE 0 END) AS n_first,
              sum(CASE WHEN kp.visit_count >= 2 AND kp.visit_count <= 4 THEN 1 ELSE 0 END) AS n_repeat,
              sum(CASE WHEN kp.visit_count >= 5 THEN 1 ELSE 0 END) AS n_regular,
              count(*) AS total
        """).single()
        # source별 (initial / rumor / visit)
        sources = s.run("""
            MATCH ()-[kp:KNOWS_POI]->()
            WHERE kp.visit_count > 0
            RETURN kp.source AS src, count(*) AS n ORDER BY n DESC
        """).data()
    dist = {
        "신규 (1회 방문)": r["n_first"],
        "재방문 (2~4회)": r["n_repeat"],
        "단골 (5회+)": r["n_regular"],
    }
    src_dist = {x["src"] or "(미상)": x["n"] for x in sources}

    plt = _setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    # 좌: 방문 빈도
    axs[0].bar(dist.keys(), dist.values(),
               color=["#bdb2ff", "#83c5be", "#e76f51"])
    axs[0].set_title("방문 빈도 분포 (KNOWS_POI)")
    axs[0].set_ylabel("관계 수")
    axs[0].grid(axis="y", alpha=0.3)
    for i, (k, v) in enumerate(dist.items()):
        axs[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=10)
    # 우: source
    axs[1].pie(src_dist.values(), labels=src_dist.keys(), autopct="%1.1f%%",
               colors=["#4cc9f0", "#f4a261", "#72efdd", "#9d4edd", "#888"][:len(src_dist)])
    axs[1].set_title("KNOWS_POI 출처 (왜 알게 됐나)")
    plt.tight_layout()
    path = out_dir / "fig4_2_regulars.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {"frequency": dist, "source": src_dist, "total": r["total"]}, str(path.name)


# ═══════════════════════════════════════════════════════════════
# 섹션 4-3 — 만족도·피드백 (trigger별·카테고리별)
# ═══════════════════════════════════════════════════════════════
def section4_3_satisfaction(start: date, days: int, out_dir: Path) -> tuple[dict, str]:
    with driver_session() as s:
        by_trigger = s.run("""
            MATCH ()-[i:INCLUDES]->()
            WHERE i.actual_satisfaction IS NOT NULL AND i.trigger IS NOT NULL
              AND NOT i.category IN ['집','직장']
            RETURN i.trigger AS trigger,
                   avg(i.actual_satisfaction) AS avg_sat,
                   count(*) AS n
            ORDER BY avg_sat DESC
        """).data()
        by_cat = s.run("""
            MATCH ()-[i:INCLUDES]->()
            WHERE i.actual_satisfaction IS NOT NULL
              AND NOT i.category IN ['집','직장']
            RETURN i.category AS cat,
                   avg(i.actual_satisfaction) AS avg_sat,
                   count(*) AS n
            ORDER BY avg_sat DESC LIMIT 12
        """).data()

    plt = _setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
                 "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션",
                 "none": "기타"}
    # 좌: trigger별
    xs = [labels_kr.get(r["trigger"], r["trigger"]) for r in by_trigger]
    ys = [round(r["avg_sat"], 3) for r in by_trigger]
    axs[0].barh(xs, ys, color="#4cc9f0")
    axs[0].set_xlabel("평균 만족도")
    axs[0].set_title("결정 동기별 만족도 — 어떤 동기가 더 만족스러웠나")
    axs[0].set_xlim(0, 1)
    for i, (lbl, v, r) in enumerate(zip(xs, ys, by_trigger)):
        axs[0].text(v + 0.005, i, f"{v:.3f} (n={r['n']:,})", va="center", fontsize=9)
    axs[0].grid(axis="x", alpha=0.3)
    # 우: 카테고리별
    xs2 = [r["cat"] for r in by_cat]
    ys2 = [round(r["avg_sat"], 3) for r in by_cat]
    axs[1].barh(xs2[::-1], ys2[::-1], color="#e76f51")
    axs[1].set_xlabel("평균 만족도")
    axs[1].set_title("카테고리별 만족도 Top 12")
    axs[1].set_xlim(0, 1)
    axs[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = out_dir / "fig4_3_satisfaction.png"
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return {
        "by_trigger": [{"trigger": r["trigger"], "avg_sat": round(r["avg_sat"], 3), "n": r["n"]} for r in by_trigger],
        "by_category": [{"category": r["cat"], "avg_sat": round(r["avg_sat"], 3), "n": r["n"]} for r in by_cat],
    }, str(path.name)


# ═══════════════════════════════════════════════════════════════
# 섹션 5 — 1대1 인터뷰 자동 (positive / negative / neutral)
# ═══════════════════════════════════════════════════════════════
INTERVIEW_QUESTIONS = [
    "정책(강남구 카페·디저트 바우처)에 대해 어떻게 느끼셨나요? 직접 사용하셨다면 왜 사용했고, 안 쓰셨다면 왜 안 쓰셨는지 알려주세요.",
    "가장 자주 가신 가게는 어디고, 왜 그곳을 자주 갔나요?",
    "친구·동료가 추천한 곳에 가신 적 있나요? 있다면 누가 추천했고, 갔더니 어땠나요?",
    "약속을 잡으신 적이 있다면 그 약속이 왜 잡혔는지 설명해주세요.",
    "이번 주 가장 만족스러웠던 외출과 가장 별로였던 외출을 하나씩 꼽아주세요.",
    "다음 주에도 비슷한 패턴으로 다닐 건가요, 바꾸실 건가요? 그 이유는?",
]


def section5_interviews(start: date, days: int, out_dir: Path) -> dict:
    from interview_agent import fetch_agent_full, ask, find_label_sample
    day_strs = [(start + timedelta(days=i)).isoformat() for i in range(days)]
    last_day = day_strs[-1]
    out = {}
    for label in ["positive", "negative", "neutral"]:
        print(f"[interview] {label} ...", file=sys.stderr)
        aid = find_label_sample(label, last_day)
        if not aid:
            out[label] = {"error": "샘플 없음"}
            continue
        data = fetch_agent_full(aid, day_strs)
        qa = []
        for q in INTERVIEW_QUESTIONS:
            try:
                a = ask(data, q)
            except Exception as e:
                a = f"(인터뷰 실패: {e})"
            qa.append({"q": q, "a": a})
        out[label] = {
            "agent_id": aid,
            "persona": {
                "age": data["persona"].get("p_age_group"),
                "gender": data["persona"].get("p_gender"),
                "job": data["persona"].get("personal_job_raw"),
                "lifestyle": data["persona"].get("personality_lifestyle_raw"),
                "income": data["persona"].get("p_income_level"),
                "home_dong": data["persona"].get("home_dong_name"),
            },
            "qa": qa,
        }
    return out


# ═══════════════════════════════════════════════════════════════
# Markdown 빌더
# ═══════════════════════════════════════════════════════════════
def build_markdown(start: date, days: int, policy_from: str | None,
                   s1: dict, s2: tuple, s3: tuple, s4_1: tuple,
                   s4_2: tuple, s4_3: tuple, s5: dict,
                   chart_dir_rel: str) -> str:
    lines: list[str] = []
    lines.append(f"# 서울 상권정책 시뮬레이션 — 최종 보고서")
    lines.append("")
    lines.append(f"**기간**: {s1['기간']} ({s1['일수']}일) · "
                 f"**모델**: Qwen3-14B-AWQ · "
                 f"**정책 시행**: {s1['정책_시행일']}")
    lines.append("")

    # 1) 조건 요약
    lines.append("## 1. 시뮬레이션 조건 요약")
    lines.append("")
    lines.append("| 항목 | 값 |")
    lines.append("|---|---:|")
    for k, v in s1.items():
        lines.append(f"| {k} | {v:,} |" if isinstance(v, int) else f"| {k} | {v} |")
    lines.append("")

    # 2) 매출 추이
    s2_data, s2_fig = s2
    sm = s2_data["summary"]
    lines.append("## 2. 정책 시행 전 vs 후 매출 추이")
    lines.append("")
    lines.append(f"![매출추이]({chart_dir_rel}/{s2_fig})")
    lines.append("")
    if sm:
        lines.append("### 평균 일간 매출 비교")
        lines.append("")
        lines.append("| 자치구 | 시행 전 | 시행 후 | 변화율 |")
        lines.append("|---|---:|---:|---:|")
        lines.append(f"| 강남 (정책 대상) | {sm['before_gn_daily']:,}원 | {sm['after_gn_daily']:,}원 | "
                     f"**{sm['gangnam_change_pct']:+}%** |")
        lines.append(f"| 비강남 (대조군) | {sm['before_ng_daily']:,}원 | {sm['after_ng_daily']:,}원 | "
                     f"{sm['non_gangnam_change_pct']:+}% |")
        lines.append("")
        lines.append(f"**DID (정책 순효과)**: 강남 변화율 − 비강남 변화율 = "
                     f"**{sm['DID_pct_points']:+}%p**")
        lines.append("")

    # 3) spillover
    s3_data, s3_fig = s3
    lines.append("## 3. 간접 영향 (Spillover) — 강남 vs 인접 자치구")
    lines.append("")
    lines.append(f"![spillover]({chart_dir_rel}/{s3_fig})")
    lines.append("")
    lines.append("강남구 정책이 인접한 서초·송파에 어떤 파급을 줬는지, 멀리 떨어진 강북과 비교. "
                 "인접 자치구의 강남 대비 매출 격차가 좁혀지면 spillover로 해석.")
    lines.append("")

    # 4-1) trigger
    s41_data, s41_fig = s4_1
    lines.append("## 4. 소비자 행동·심리 분석")
    lines.append("")
    lines.append("### 4-1. 방문 목적·이동 패턴 — 결정 동기 (trigger) 분포")
    lines.append("")
    lines.append(f"![trigger]({chart_dir_rel}/{s41_fig})")
    lines.append("")
    lines.append("외출(집·직장 제외)의 결정 동기 분류:")
    lines.append("")
    lines.append("| 동기 | 건수 | 비율 |")
    lines.append("|---|---:|---:|")
    labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
                 "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션", "none": "기타"}
    for k, v in sorted(s41_data["distribution"].items(), key=lambda x: -x[1]):
        pct = s41_data["distribution_pct"].get(k, 0)
        lines.append(f"| {labels_kr.get(k, k)} | {v:,} | {pct}% |")
    lines.append("")

    # 4-2) 단골 vs 신규
    s42_data, s42_fig = s4_2
    lines.append("### 4-2. 단골 vs 신규")
    lines.append("")
    lines.append(f"![단골]({chart_dir_rel}/{s42_fig})")
    lines.append("")
    lines.append("| 구분 | 관계 수 |")
    lines.append("|---|---:|")
    for k, v in s42_data["frequency"].items():
        lines.append(f"| {k} | {v:,} |")
    lines.append("")
    lines.append(f"전체 KNOWS_POI(방문 경험 있음) 관계 수: **{s42_data['total']:,}**")
    lines.append("")

    # 4-3) 만족도
    s43_data, s43_fig = s4_3
    lines.append("### 4-3. 만족도·피드백 — 어떤 동기로 외출했을 때 더 만족했나")
    lines.append("")
    lines.append(f"![만족도]({chart_dir_rel}/{s43_fig})")
    lines.append("")
    lines.append("| 동기 | 평균 만족도 | 표본 수 |")
    lines.append("|---|---:|---:|")
    for r in s43_data["by_trigger"]:
        lines.append(f"| {labels_kr.get(r['trigger'], r['trigger'])} | {r['avg_sat']} | {r['n']:,} |")
    lines.append("")

    # 5) 인터뷰
    lines.append("## 5. 1대1 인터뷰 — 페르소나별 대표 (positive / negative / neutral)")
    lines.append("")
    label_kr = {"positive": "정책 적극 활용 + 만족도 ↑",
                "negative": "정책 무관심 + 만족도 ↓",
                "neutral":  "정책 무관심 + 만족도 보통"}
    for label in ["positive", "negative", "neutral"]:
        d = s5.get(label, {})
        if "error" in d:
            lines.append(f"### 5-{label} — 샘플 없음")
            lines.append("")
            continue
        p = d["persona"]
        lines.append(f"### [{label}] {label_kr[label]} — `{d['agent_id']}`")
        lines.append("")
        lines.append(f"- **페르소나**: {p.get('age')} {p.get('gender')} · {p.get('job')} · "
                     f"{p.get('home_dong')} 거주 · 소득 {p.get('income')}")
        lines.append(f"- **라이프스타일**: {p.get('lifestyle')}")
        lines.append("")
        for qa in d["qa"]:
            lines.append(f"**Q. {qa['q']}**")
            lines.append("")
            lines.append(f"> {qa['a']}")
            lines.append("")
        lines.append("---")
        lines.append("")

    lines.append("## 부록")
    lines.append("")
    lines.append("- 본 보고서는 `scripts/sim/generate_final_report.py`로 자동 생성됨.")
    lines.append("- 시뮬 원본 데이터는 Neo4j에 보존 (Plan/State/Memory/Conversation 노드).")
    lines.append("- 인터랙티브 시각화: `output/sim/visualization/sim_standalone.html`")
    lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--policy-from", default=None, help="정책 effective_from (ISO date)")
    ap.add_argument("--out", default="docs/FINAL_REPORT.md")
    ap.add_argument("--skip-interview", action="store_true", help="인터뷰 생략 (LLM 호출 X)")
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    out_md = Path(args.out).resolve()
    chart_dir = out_md.parent / (out_md.stem + ".d")
    chart_dir.mkdir(parents=True, exist_ok=True)
    chart_dir_rel = chart_dir.name

    print(f"[1/7] 조건 요약 ...", file=sys.stderr)
    s1 = section1_conditions(start, args.days, args.policy_from)

    if args.policy_from:
        print(f"[2/7] 정책 전/후 매출 추이 ...", file=sys.stderr)
        s2 = section2_before_after(start, args.days, args.policy_from, chart_dir)
    else:
        s2 = ({"summary": {}, "daily": []}, "fig2_skipped.png")

    print(f"[3/7] spillover ...", file=sys.stderr)
    s3 = section3_spillover(start, args.days, chart_dir)

    print(f"[4-1/7] trigger 분포 ...", file=sys.stderr)
    s4_1 = section4_1_triggers(start, args.days, chart_dir)

    print(f"[4-2/7] 단골 vs 신규 ...", file=sys.stderr)
    s4_2 = section4_2_regulars(start, args.days, chart_dir)

    print(f"[4-3/7] 만족도 ...", file=sys.stderr)
    s4_3 = section4_3_satisfaction(start, args.days, chart_dir)

    if args.skip_interview:
        s5 = {l: {"error": "skipped"} for l in ["positive", "negative", "neutral"]}
    else:
        print(f"[5/7] 인터뷰 (3 라벨 × 6 질문 = 18 LLM 호출) ...", file=sys.stderr)
        s5 = section5_interviews(start, args.days, chart_dir)

    print(f"[6/7] markdown 빌드 ...", file=sys.stderr)
    md = build_markdown(start, args.days, args.policy_from,
                        s1, s2, s3, s4_1, s4_2, s4_3, s5, chart_dir_rel)
    out_md.write_text(md, encoding="utf-8")

    print(f"\n[7/7] DONE", file=sys.stderr)
    print(f"  → markdown: {out_md}", file=sys.stderr)
    print(f"  → charts:   {chart_dir}/*.png", file=sys.stderr)


if __name__ == "__main__":
    main()
