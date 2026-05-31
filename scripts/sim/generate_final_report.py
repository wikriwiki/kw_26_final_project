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
import base64
import html as _html
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
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
    from matplotlib import font_manager
    # 사용 가능한 폰트 중 한글 지원하는 거 자동 탐색
    installed = {f.name for f in font_manager.fontManager.ttflist}
    # Windows / Mac / Linux 우선순위
    candidates = ["Malgun Gothic", "AppleGothic",
                  "Noto Sans CJK KR", "Noto Sans KR",
                  "NanumGothic", "Nanum Gothic",
                  "Noto Sans CJK SC", "Noto Sans CJK JP",
                  "DejaVu Sans"]
    chosen = next((f for f in candidates if f in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
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
        # 실측 인구 (LIVES_AT 기준 강남 vs 비강남 거주 agent 수)
        pop = s.run("""
            MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)
              <-[:HAS_DONG]-(d:District)
            RETURN
              sum(CASE WHEN d.code = '11680' THEN 1 ELSE 0 END) AS gn,
              sum(CASE WHEN d.code <> '11680' THEN 1 ELSE 0 END) AS ng
        """).single()
        GN_POP = int(pop["gn"]) or 1
        NG_POP = int(pop["ng"]) or 1
        print(f"  [pop] 강남 거주 {GN_POP:,} / 비강남 거주 {NG_POP:,}", file=sys.stderr)
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

    # ─────── 차트 — 3개 독립 PNG로 분리 (각각 충분히 크게) ───────
    # GN_POP / NG_POP은 위에서 Cypher로 실측 가져옴
    plt = _setup_mpl()
    plt.rcParams.update({
        'font.size': 14, 'axes.titlesize': 17, 'axes.labelsize': 14,
        'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13,
    })
    xs = list(range(len(daily)))
    labels = [x["day"][5:] for x in daily]
    cut_idx = None
    try:
        cut_idx = next(i for i, x in enumerate(daily) if x["day"] == policy_from)
    except StopIteration:
        pass

    # ── (A) 1인당 매출 (절대 비교) ──
    gn_per = [x["gangnam_spend"] / GN_POP for x in daily]
    ng_per = [x["non_gangnam_spend"] / NG_POP for x in daily]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.plot(xs, gn_per, marker="o", markersize=10, label=f"강남 (n={GN_POP:,})",
            color="#e76f51", linewidth=3)
    ax.plot(xs, ng_per, marker="s", markersize=10, label=f"비강남 (n={NG_POP:,})",
            color="#4cc9f0", linewidth=3)
    if cut_idx is not None:
        ax.axvline(cut_idx - 0.5, color="#888", linestyle="--", linewidth=1.5)
        ax.text(cut_idx - 0.45, max(gn_per + ng_per) * 0.95,
                f" 정책 시행 {policy_from}", color="#666", fontsize=12)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_xlabel("날짜"); ax.set_ylabel("1인당 매출 (원)")
    ax.set_title("(A) 1인당 일별 매출 — 인구 비례 환산 후 절대 비교", pad=12)
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    plt.tight_layout()
    path_a = out_dir / "fig2a_per_capita.png"
    plt.savefig(path_a, dpi=140, bbox_inches="tight")
    plt.close()

    # ── (B) baseline 대비 변화율 ──
    if daily:
        gn_base = max(daily[0]["gangnam_spend"], 1)
        ng_base = max(daily[0]["non_gangnam_spend"], 1)
        gn_chg = [(x["gangnam_spend"] - gn_base) / gn_base * 100 for x in daily]
        ng_chg = [(x["non_gangnam_spend"] - ng_base) / ng_base * 100 for x in daily]
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.plot(xs, gn_chg, marker="o", markersize=10, label="강남구", color="#e76f51", linewidth=3)
        ax.plot(xs, ng_chg, marker="s", markersize=10, label="비강남", color="#4cc9f0", linewidth=3)
        ax.axhline(0, color="#999", linewidth=1)
        if cut_idx is not None:
            ax.axvline(cut_idx - 0.5, color="#888", linestyle="--", linewidth=1.5)
            top_y = max(max(gn_chg), max(ng_chg))
            ax.text(cut_idx - 0.45, top_y * 0.95 if top_y > 0 else 3,
                    f" 정책 시행 {policy_from}", color="#666", fontsize=12)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("날짜"); ax.set_ylabel(f"{daily[0]['day'][5:]} 대비 변화율 (%)")
        ax.set_title("(B) baseline 대비 매출 변화율 — 같은 출발점에서 비교", pad=12)
        ax.legend(loc="best"); ax.grid(alpha=0.3)
        plt.tight_layout()
        path_b = out_dir / "fig2b_change_rate.png"
        plt.savefig(path_b, dpi=140, bbox_inches="tight")
        plt.close()

        # ── (C) DID ──
        did = [g - n for g, n in zip(gn_chg, ng_chg)]
        fig, ax = plt.subplots(figsize=(13, 6.5))
        colors = ['#aaa' if i == 0 else ('#38a169' if v >= 0 else '#e53e3e')
                  for i, v in enumerate(did)]
        bars = ax.bar(xs, did, color=colors, edgecolor='black', linewidth=0.8, width=0.65)
        ax.axhline(0, color="#666", linewidth=1)
        if cut_idx is not None:
            ax.axvline(cut_idx - 0.5, color="#888", linestyle="--", linewidth=1.5)
        for i, (bar, v) in enumerate(zip(bars, did)):
            if i == 0: continue
            offset = max(abs(min(did)), abs(max(did))) * 0.04
            ax.text(i, v + (offset if v >= 0 else -offset),
                    f"{v:+.1f}%p", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=13,
                    color='#1a7a3e' if v >= 0 else '#b03030', fontweight='bold')
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("날짜"); ax.set_ylabel("DID (%p)")
        ax.set_title("(C) DID — 강남 변화율 − 비강남 변화율 (정책 순효과)", pad=12)
        ax.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        path_c = out_dir / "fig2c_did.png"
        plt.savefig(path_c, dpi=140, bbox_inches="tight")
        plt.close()
    else:
        path_b = path_a
        path_c = path_a

    return {"daily": daily, "summary": summary}, {
        "per_capita": str(path_a.name),
        "change_rate": str(path_b.name),
        "did": str(path_c.name),
    }


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
    "이번 주 받으신 소득별 현금 지원금(P009 정책)에 대해 어떻게 느끼셨나요? 직접 사용하셨다면 왜 그 가게에서 쓰셨고, 안 쓰셨다면 왜 안 쓰셨는지 알려주세요.",
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
    lines.append(f"**작성일**: {datetime.now().strftime('%Y-%m-%d %H:%M KST')}")
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

    # 2) 매출 추이 — 3장
    s2_data, s2_figs = s2
    sm = s2_data["summary"]
    lines.append("## 2. 정책 시행 전 vs 후 매출 추이")
    lines.append("")
    if isinstance(s2_figs, dict):
        lines.append(f"![1인당 매출]({chart_dir_rel}/{s2_figs['per_capita']})")
        lines.append("")
        lines.append(f"![변화율]({chart_dir_rel}/{s2_figs['change_rate']})")
        lines.append("")
        lines.append(f"![DID]({chart_dir_rel}/{s2_figs['did']})")
        lines.append("")
    else:
        lines.append(f"![매출추이]({chart_dir_rel}/{s2_figs})")
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
# HTML 빌더 — 전문 보고서 웹페이지 (단일 파일, CSS·차트 임베드)
# ═══════════════════════════════════════════════════════════════
HTML_STYLE = """
:root {
  --bg-deep: #0b0d17;
  --bg-primary: #111827;
  --bg-card: #1e293b;
  --border-subtle: rgba(148, 163, 184, 0.1);
  --border-glass: rgba(148, 163, 184, 0.15);
  --cyan: #06b6d4;
  --purple: #8b5cf6;
  --gradient: linear-gradient(135deg, #06b6d4, #8b5cf6);
  --gradient-text: linear-gradient(135deg, #22d3ee, #a78bfa);
  --emerald: #10b981;
  --emerald-soft: rgba(16, 185, 129, 0.12);
  --rose: #f43f5e;
  --rose-soft: rgba(244, 63, 94, 0.12);
  --amber: #f59e0b;
  --neutral: #64748b;
  --neutral-soft: rgba(100, 116, 139, 0.12);
  --text-bright: #f1f5f9;
  --text-primary: #e2e8f0;
  --text-secondary: #94a3b8;
  --text-muted: #64748b;
  --shadow-sm: 0 2px 8px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 24px rgba(0,0,0,0.4);
  --shadow-glow-cyan: 0 0 20px rgba(6, 182, 212, 0.15);
  --shadow-glow-purple: 0 0 20px rgba(139, 92, 246, 0.15);
  --radius: 12px;
  --radius-lg: 16px;
}
body.light-theme {
  --bg-deep: #f8fafc;
  --bg-primary: #f1f5f9;
  --bg-card: #ffffff;
  --border-subtle: rgba(15, 23, 42, 0.08);
  --border-glass: rgba(15, 23, 42, 0.06);
  --cyan: #0284c7;
  --purple: #4f46e5;
  --gradient: linear-gradient(135deg, #3b82f6, #6366f1);
  --gradient-text: linear-gradient(135deg, #1e40af, #4f46e5);
  --emerald: #059669;
  --emerald-soft: rgba(5, 150, 105, 0.08);
  --rose: #e11d48;
  --rose-soft: rgba(225, 29, 72, 0.08);
  --amber: #d97706;
  --neutral: #475569;
  --neutral-soft: rgba(71, 85, 105, 0.08);
  --text-bright: #0f172a;
  --text-primary: #334155;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
  --shadow-md: 0 10px 15px -3px rgba(0,0,0,0.04), 0 4px 6px -2px rgba(0,0,0,0.02);
  --shadow-glow-cyan: 0 4px 20px rgba(59, 130, 246, 0.08);
  --shadow-glow-purple: 0 4px 20px rgba(99, 102, 241, 0.08);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Inter', 'Noto Sans KR', 'Apple SD Gothic Neo', sans-serif;
  background: var(--bg-deep);
  color: var(--text-primary);
  line-height: 1.7;
  font-size: 15px;
}
.layout { display: flex; min-height: 100vh; }

/* ─── sidebar ─── */
.sidebar {
  width: 272px;
  background: rgba(15, 23, 42, 0.85);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-right: 1px solid var(--border-subtle);
  padding: 32px 20px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}
.sidebar::-webkit-scrollbar { width: 4px; }
.sidebar::-webkit-scrollbar-thumb { background: var(--text-muted); border-radius: 2px; }
.sidebar .brand {
  background: var(--gradient-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 4px;
}
.sidebar h2 {
  color: var(--text-bright);
  font-size: 14px;
  font-weight: 600;
  letter-spacing: 0.5px;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 16px;
}
.sidebar nav a {
  display: block;
  color: var(--text-secondary);
  text-decoration: none;
  padding: 9px 14px;
  font-size: 13px;
  border-radius: 8px;
  margin: 2px 0;
  border-left: 2px solid transparent;
  transition: all 0.25s ease;
}
.sidebar nav a:hover {
  background: rgba(6, 182, 212, 0.08);
  color: var(--text-bright);
  border-left-color: var(--cyan);
}
.sidebar nav a.active {
  background: rgba(6, 182, 212, 0.12);
  color: var(--cyan);
  border-left-color: var(--cyan);
  font-weight: 500;
}
.sidebar nav .lvl2 { padding-left: 28px; font-size: 12px; }
.sidebar .tech-info {
  margin-top: 40px;
  padding: 16px;
  background: rgba(139, 92, 246, 0.06);
  border: 1px solid rgba(139, 92, 246, 0.15);
  border-radius: 10px;
  font-size: 11px;
  color: var(--text-muted);
  line-height: 1.7;
}
.sidebar .tech-info span { color: var(--purple); font-weight: 500; }

/* ─── main ─── */
.main {
  flex: 1;
  max-width: 1080px;
  margin: 0 auto;
  padding: 56px 64px 80px;
}

/* ─── cover / hero ─── */
header.cover {
  margin-bottom: 56px;
  padding-bottom: 36px;
  border-bottom: 1px solid var(--border-subtle);
  position: relative;
}
.cover::before {
  content: '';
  position: absolute;
  top: -56px; left: -64px; right: -64px;
  height: 400px;
  background:
    radial-gradient(ellipse at 30% 0%, rgba(6,182,212,0.08) 0%, transparent 60%),
    radial-gradient(ellipse at 70% 0%, rgba(139,92,246,0.06) 0%, transparent 60%);
  pointer-events: none;
  z-index: 0;
}
.cover > * { position: relative; z-index: 1; }
.cover .meta {
  background: var(--gradient-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 4px;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.cover h1 {
  color: var(--text-bright);
  font-size: 36px;
  font-weight: 800;
  line-height: 1.3;
  margin-bottom: 18px;
  letter-spacing: -0.5px;
}
.cover .subtitle {
  color: var(--text-secondary);
  font-size: 15px;
  line-height: 1.7;
}
.cover .badges { margin-top: 20px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  background: rgba(6,182,212,0.08);
  color: var(--cyan);
  border: 1px solid rgba(6,182,212,0.2);
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
  transition: all 0.2s;
}
.badge:hover { background: rgba(6,182,212,0.15); }
.badge.alt { background: rgba(16,185,129,0.08); color: var(--emerald); border-color: rgba(16,185,129,0.2); }
.badge.alt:hover { background: rgba(16,185,129,0.15); }
.badge.purple { background: rgba(139,92,246,0.08); color: var(--purple); border-color: rgba(139,92,246,0.2); }
.badge.purple:hover { background: rgba(139,92,246,0.15); }

/* ─── sections ─── */
section { margin-bottom: 56px; }
.reveal {
  opacity: 0;
  transform: translateY(24px);
  transition: all 0.7s cubic-bezier(0.16, 1, 0.3, 1);
}
.reveal.visible { opacity: 1; transform: translateY(0); }

h2 {
  color: var(--text-bright);
  font-size: 22px;
  font-weight: 700;
  padding-bottom: 12px;
  border-bottom: 1px solid var(--border-subtle);
  margin-bottom: 20px;
  scroll-margin-top: 24px;
  position: relative;
}
h2::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 0;
  width: 60px;
  height: 2px;
  background: var(--gradient);
  border-radius: 1px;
}
h3 {
  color: var(--text-bright);
  font-size: 17px;
  font-weight: 600;
  margin: 28px 0 14px;
  scroll-margin-top: 24px;
}
h4 { font-size: 14px; color: var(--text-secondary); margin: 20px 0 10px; }
p  { margin-bottom: 14px; color: var(--text-secondary); }

/* ─── tables ─── */
table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  margin: 16px 0 24px;
  font-size: 14px;
  background: var(--bg-card);
  border-radius: var(--radius);
  border: 1px solid var(--border-glass);
  overflow: hidden;
}
thead {
  background: linear-gradient(135deg, rgba(6,182,212,0.12), rgba(139,92,246,0.12));
}
th, td {
  padding: 12px 16px;
  text-align: left;
  border-bottom: 1px solid var(--border-subtle);
}
th {
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  color: var(--text-secondary);
}
tbody tr { transition: background 0.15s ease; }
tbody tr:hover { background: rgba(6,182,212,0.04); }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td strong { color: var(--cyan); }

/* ─── figures ─── */
figure {
  margin: 24px 0;
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  padding: 20px;
  border-radius: var(--radius-lg);
  text-align: center;
  transition: all 0.3s ease;
}
figure:hover {
  border-color: rgba(6,182,212,0.2);
  box-shadow: var(--shadow-glow-cyan);
}
figure img {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
  transition: transform 0.3s ease;
}
figure img.chart-light { display: none; }
figure img.chart-dark { display: block; margin: 0 auto; }
body.light-theme figure img.chart-dark { display: none; }
body.light-theme figure img.chart-light { display: block; margin: 0 auto; }
figure:hover img { transform: scale(1.01); }
figcaption {
  margin-top: 14px;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 500;
}

/* ─── interview cards ─── */
.interview-card {
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  padding: 28px 32px;
  margin: 24px 0;
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.interview-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; bottom: 0;
  width: 4px;
}
.interview-card:hover {
  border-color: rgba(148,163,184,0.2);
  transform: translateY(-1px);
}
.interview-card.positive::before { background: linear-gradient(180deg, var(--emerald), #34d399); }
.interview-card.positive:hover { box-shadow: 0 0 24px rgba(16,185,129,0.1); }
.interview-card.negative::before { background: linear-gradient(180deg, var(--rose), #fb7185); }
.interview-card.negative:hover { box-shadow: 0 0 24px rgba(244,63,94,0.1); }
.interview-card.neutral::before  { background: linear-gradient(180deg, var(--neutral), #94a3b8); }

.label-tag {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-bottom: 12px;
}
.interview-card.positive .label-tag { background: var(--emerald-soft); color: var(--emerald); }
.interview-card.negative .label-tag { background: var(--rose-soft); color: var(--rose); }
.interview-card.neutral  .label-tag { background: var(--neutral-soft); color: var(--neutral); }

.interview-card .persona {
  font-size: 13px;
  color: var(--text-secondary);
  background: rgba(139,92,246,0.04);
  border: 1px solid rgba(139,92,246,0.1);
  padding: 14px 18px;
  border-radius: 10px;
  margin-bottom: 20px;
  line-height: 1.7;
}
.interview-card .persona strong { color: var(--purple); }

.qa { margin: 16px 0; }
.qa .q {
  font-weight: 600;
  color: var(--text-bright);
  margin-bottom: 8px;
  font-size: 14px;
  display: flex;
  align-items: flex-start;
  gap: 8px;
}
.qa .q::before {
  content: 'Q';
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 22px;
  height: 22px;
  background: var(--gradient);
  color: #fff;
  border-radius: 6px;
  font-size: 11px;
  font-weight: 700;
  flex-shrink: 0;
  margin-top: 2px;
}
.qa .a {
  background: rgba(6,182,212,0.04);
  border: 1px solid rgba(6,182,212,0.08);
  padding: 14px 18px;
  margin: 8px 0 16px 30px;
  border-radius: 4px 12px 12px 12px;
  color: var(--text-primary);
  font-size: 14px;
  line-height: 1.8;
}

/* ─── KPI cards ─── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 16px;
  margin: 28px 0;
}
.kpi {
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  padding: 22px 24px;
  border-radius: var(--radius);
  position: relative;
  overflow: hidden;
  transition: all 0.3s ease;
}
.kpi::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--gradient);
}
.kpi:hover {
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow-cyan);
  border-color: rgba(6,182,212,0.25);
}
.kpi .icon {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 12px;
  background: rgba(6,182,212,0.1);
}
.kpi .icon svg {
  width: 18px;
  height: 18px;
  stroke: var(--cyan);
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}
.kpi .value {
  font-size: 28px;
  font-weight: 800;
  background: var(--gradient-text);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
.kpi .label {
  font-size: 12px;
  color: var(--text-muted);
  margin-top: 6px;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  font-weight: 500;
}
.kpi.pos::before { background: linear-gradient(135deg, var(--emerald), #34d399); }
.kpi.pos .value {
  background: linear-gradient(135deg, var(--emerald), #34d399);
  -webkit-background-clip: text;
  background-clip: text;
}
.kpi.pos .icon { background: var(--emerald-soft); }
.kpi.pos .icon svg { stroke: var(--emerald); }

/* ─── code · callout · footer ─── */
code {
  background: rgba(139,92,246,0.1);
  border: 1px solid rgba(139,92,246,0.15);
  padding: 2px 8px;
  border-radius: 6px;
  font-family: 'JetBrains Mono', 'D2Coding', 'Consolas', monospace;
  font-size: 13px;
  color: var(--purple);
}
.callout {
  background: rgba(6,182,212,0.06);
  border: 1px solid rgba(6,182,212,0.15);
  border-left: 4px solid var(--cyan);
  padding: 16px 20px;
  border-radius: 4px 10px 10px 4px;
  margin: 18px 0;
  font-size: 14px;
  color: var(--text-primary);
}
.callout strong { color: var(--cyan); }
.callout.warn {
  border-left-color: var(--amber);
  border-color: rgba(245,158,11,0.15);
  background: rgba(245,158,11,0.06);
}
.callout.warn strong { color: var(--amber); }

.footer {
  margin-top: 72px;
  padding-top: 28px;
  border-top: 1px solid var(--border-subtle);
  color: var(--text-muted);
  font-size: 12px;
  text-align: center;
  line-height: 1.8;
}

/* ─── animations ─── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(20px); }
  to   { opacity: 1; transform: translateY(0); }
}

/* ─── responsive ─── */
@media (max-width: 900px) {
  .layout { flex-direction: column; }
  .sidebar {
    width: 100%; height: auto; position: relative;
    backdrop-filter: none; -webkit-backdrop-filter: none;
  }
  .sidebar nav { display: flex; flex-wrap: wrap; gap: 4px; }
  .sidebar nav a { padding: 6px 12px; }
  .main { padding: 32px 20px; }
  .cover h1 { font-size: 26px; }
  .cover::before { display: none; }
  .kpi-grid { grid-template-columns: repeat(2, 1fr); }
}

/* ─── print ─── */
@media print {
  :root {
    --bg-deep: #fff; --bg-primary: #fff; --bg-card: #fff;
    --text-bright: #111; --text-primary: #333; --text-secondary: #555;
    --border-subtle: #ddd; --border-glass: #ddd;
  }
  .sidebar { display: none; }
  .main { padding: 0; max-width: 100%; }
  .cover::before { display: none; }
  figure, .interview-card, table, .kpi {
    box-shadow: none; border: 1px solid #ddd;
  }
  .kpi .value, .cover .meta, .sidebar .brand {
    -webkit-text-fill-color: initial;
    background: none; color: #111;
  }
  .reveal { opacity: 1 !important; transform: none !important; }
  * { animation: none !important; transition: none !important; }
}

/* ─── theme toggle button ─── */
.theme-toggle-btn {
  background: transparent;
  border: 1px solid var(--border-subtle);
  color: var(--text-secondary);
  border-radius: 8px;
  width: 32px;
  height: 32px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: all 0.2s ease;
}
.theme-toggle-btn:hover {
  background: var(--border-subtle);
  color: var(--text-bright);
}
.theme-toggle-btn svg {
  stroke: currentColor;
  fill: none;
  stroke-width: 2;
  stroke-linecap: round;
  stroke-linejoin: round;
}

/* ─── light theme overrides ─── */
body.light-theme {
  background: var(--bg-deep);
  color: var(--text-primary);
}
body.light-theme .sidebar {
  background: linear-gradient(185deg, #eff6ff 0%, #e0e7ff 100%);
  border-right: 1px solid rgba(226, 232, 240, 0.8);
}
body.light-theme .sidebar h2 {
  border-bottom-color: rgba(15, 23, 42, 0.08);
}
body.light-theme .sidebar nav a:hover {
  background: rgba(59, 130, 246, 0.06);
  color: var(--text-bright);
}
body.light-theme .sidebar nav a.active {
  background: rgba(59, 130, 246, 0.10);
  color: var(--cyan);
}
body.light-theme .sidebar .tech-info {
  background: rgba(99, 102, 241, 0.04);
  border: 1px solid rgba(99, 102, 241, 0.1);
}
body.light-theme .sidebar .tech-info span {
  color: var(--purple);
}
body.light-theme thead {
  background: linear-gradient(135deg, rgba(59, 130, 246, 0.06), rgba(99, 102, 241, 0.06));
}
body.light-theme .qa .a {
  background: rgba(59, 130, 246, 0.03);
  border-color: rgba(59, 130, 246, 0.06);
}
body.light-theme .cover::before {
  background:
    radial-gradient(ellipse at 30% 0%, rgba(59, 130, 246, 0.04) 0%, transparent 60%),
    radial-gradient(ellipse at 70% 0%, rgba(99, 102, 241, 0.03) 0%, transparent 60%);
}
body.light-theme .interview-card .persona {
  background: rgba(99, 102, 241, 0.03);
  border-color: rgba(99, 102, 241, 0.06);
}

/* ─── interview chat UI ─── */
.chat-container {
  display: flex;
  flex-direction: column;
  height: 400px;
  background: rgba(15, 23, 42, 0.45);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius);
  overflow: hidden;
  margin-top: 20px;
}
body.light-theme .chat-container {
  background: rgba(241, 245, 249, 0.6);
}
.chat-messages {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 16px;
  scroll-behavior: smooth;
}
.chat-messages::-webkit-scrollbar {
  width: 6px;
}
.chat-messages::-webkit-scrollbar-thumb {
  background: var(--border-glass);
  border-radius: 3px;
}
.chat-bubble {
  max-width: 80%;
  display: flex;
  flex-direction: column;
  padding: 12px 16px;
  border-radius: 14px;
  font-size: 13.5px;
  line-height: 1.6;
  position: relative;
  animation: fadeInBubble 0.25s ease-out forwards;
}
@keyframes fadeInBubble {
  from { opacity: 0; transform: translateY(8px); }
  to { opacity: 1; transform: translateY(0); }
}
.chat-bubble.agent {
  align-self: flex-start;
  background: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border-glass);
  border-top-left-radius: 4px;
}
body.light-theme .chat-bubble.agent {
  background: #ffffff;
  border-color: rgba(15, 23, 42, 0.08);
}
.chat-bubble.user {
  align-self: flex-end;
  background: var(--gradient);
  color: #ffffff;
  border-top-right-radius: 4px;
}
.chat-bubble .meta {
  font-size: 10px;
  color: var(--text-muted);
  margin-bottom: 4px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.chat-bubble.user .meta {
  color: rgba(255, 255, 255, 0.7);
}
.chat-bubble .text {
  word-break: break-all;
  white-space: pre-wrap;
}
.chat-input-area {
  display: flex;
  align-items: center;
  padding: 12px 16px;
  background: var(--bg-card);
  border-top: 1px solid var(--border-subtle);
  gap: 12px;
}
body.light-theme .chat-input-area {
  background: #ffffff;
}
.chat-input {
  flex: 1;
  background: rgba(15, 23, 42, 0.3);
  border: 1px solid var(--border-glass);
  color: var(--text-bright);
  padding: 10px 16px;
  border-radius: 20px;
  font-size: 13.5px;
  outline: none;
  transition: all 0.2s ease;
}
body.light-theme .chat-input {
  background: #f8fafc;
}
.chat-input:focus {
  border-color: var(--cyan);
  box-shadow: 0 0 0 2px rgba(6, 182, 212, 0.2);
}
.chat-send-btn {
  background: var(--gradient);
  border: none;
  color: white;
  width: 36px;
  height: 36px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  transition: transform 0.2s ease, opacity 0.2s ease;
  flex-shrink: 0;
}
.chat-send-btn:hover {
  transform: scale(1.05);
  opacity: 0.95;
}
.chat-send-btn:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}
.chat-send-btn:active {
  transform: scale(0.95);
}
.chat-send-btn svg {
  width: 16px;
  height: 16px;
  fill: currentColor;
}
/* Typing Indicator */
.typing-indicator {
  display: none;
  align-items: center;
  gap: 4px;
  padding: 12px 16px;
  background: var(--bg-card);
  border: 1px solid var(--border-glass);
  border-radius: 12px;
  border-top-left-radius: 4px;
  align-self: flex-start;
  margin-top: 4px;
}
body.light-theme .typing-indicator {
  background: #ffffff;
}
.typing-indicator span {
  width: 6px;
  height: 6px;
  background-color: var(--text-muted);
  border-radius: 50%;
  display: inline-block;
  animation: typingBounce 1.4s infinite ease-in-out both;
}
.typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
.typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
@keyframes typingBounce {
  0%, 80%, 100% { transform: scale(0); }
  40% { transform: scale(1.0); }
}
.fallback-banner {
  font-size: 11px;
  color: var(--amber);
  background: rgba(245, 158, 11, 0.08);
  border: 1px solid rgba(245, 158, 11, 0.15);
  padding: 6px 12px;
  border-radius: 6px;
  margin-bottom: 12px;
  display: none;
  align-items: center;
  gap: 6px;
}
.fallback-banner svg {
  width: 14px;
  height: 14px;
  fill: currentColor;
}

"""


def _img_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _h(t: str) -> str:
    return _html.escape(str(t) if t is not None else "")


def _figure(path: Path | None, caption: str) -> str:
    if not path or not path.exists():
        return ""
    return (f'<figure><img src="{_img_data_uri(path)}" alt="{_h(caption)}"/>'
            f'<figcaption>{_h(caption)}</figcaption></figure>')


def build_html(start: date, days: int, policy_from: str | None,
               s1: dict, s2: tuple, s3: tuple, s4_1: tuple,
               s4_2: tuple, s4_3: tuple, s5: dict, chart_dir: Path) -> str:
    s2_data, s2_figs = s2
    s3_data, s3_fig = s3
    s41_data, s41_fig = s4_1
    s42_data, s42_fig = s4_2
    s43_data, s43_fig = s4_3
    sm = s2_data.get("summary", {})

    # Cover KPI 4종
    n_agent = s1.get("Agent 수", "—")
    n_plan = s1.get("Plan 수", "—")
    n_conv = (s1.get("Conversation 약속", 0) + s1.get("Conversation 추천", 0)
              + s1.get("Conversation 이슈", 0) + s1.get("Conversation 기타", 0))
    did_str = f"{sm.get('DID_pct_points', 0):+}%p" if sm else "—"

    # Sidebar 목차
    toc = [
        ("s-summary", "1. 시뮬레이션 개요"),
        ("s-sales", "2. 정책 시행 전 vs 후"),
        ("s-spillover", "3. 간접 영향 (Spillover)"),
        ("s-behavior", "4. 소비자 행동 분석"),
        ("s-trigger", "4-1. 결정 동기 분포", True),
        ("s-regular", "4-2. 단골 vs 신규", True),
        ("s-satisfaction", "4-3. 만족도", True),
        ("s-interview", "5. 1대1 인터뷰"),
        ("s-appendix", "부록"),
    ]
    nav = "".join(
        f'<a class="{"lvl2" if (len(t)>2 and t[2]) else ""}" href="#{tid}">{label}</a>'
        for t in toc for tid, label in [(t[0], t[1])]
    )

    labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
                 "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션",
                 "none": "기타"}
    label_kr_intv = {"positive": "정책 적극 활용 + 만족도 ↑",
                     "negative": "정책 무관심 + 만족도 ↓",
                     "neutral":  "정책 무관심 + 만족도 보통"}

    # Section 1
    cond_rows = "".join(
        f"<tr><td>{_h(k)}</td><td class='num'>{v:,}</td></tr>"
        if isinstance(v, int) else
        f"<tr><td>{_h(k)}</td><td>{_h(v)}</td></tr>"
        for k, v in s1.items()
    )

    # Section 2 — 매출 표
    sales_rows = ""
    if sm:
        sales_rows = (
            f"<tr><td><strong>강남 (정책 대상)</strong></td>"
            f"<td class='num'>{sm['before_gn_daily']:,}원</td>"
            f"<td class='num'>{sm['after_gn_daily']:,}원</td>"
            f"<td class='num'><strong>{sm['gangnam_change_pct']:+}%</strong></td></tr>"
            f"<tr><td>비강남 (대조군)</td>"
            f"<td class='num'>{sm['before_ng_daily']:,}원</td>"
            f"<td class='num'>{sm['after_ng_daily']:,}원</td>"
            f"<td class='num'>{sm['non_gangnam_change_pct']:+}%</td></tr>"
        )
        did_callout = (
            f"<div class='callout'><strong>DID (정책 순효과)</strong>: "
            f"강남 변화율 − 비강남 변화율 = "
            f"<strong style='font-size:18px;color:var(--accent);'>"
            f"{sm['DID_pct_points']:+}%p</strong></div>"
        )
    else:
        did_callout = ""

    # Section 4-1 — trigger 표
    trigger_rows = ""
    for k, v in sorted(s41_data["distribution"].items(), key=lambda x: -x[1]):
        pct = s41_data["distribution_pct"].get(k, 0)
        trigger_rows += (f"<tr><td>{_h(labels_kr.get(k, k))}</td>"
                         f"<td class='num'>{v:,}</td>"
                         f"<td class='num'>{pct}%</td></tr>")

    # Section 4-2 — 단골
    regular_rows = "".join(
        f"<tr><td>{_h(k)}</td><td class='num'>{v:,}</td></tr>"
        for k, v in s42_data["frequency"].items()
    )

    # Section 4-3 — 만족도
    sat_rows = "".join(
        f"<tr><td>{_h(labels_kr.get(r['trigger'], r['trigger']))}</td>"
        f"<td class='num'>{r['avg_sat']}</td>"
        f"<td class='num'>{r['n']:,}</td></tr>"
        for r in s43_data["by_trigger"]
    )

    # Section 5 — 인터뷰 카드
    interview_html = ""
    for label in ["positive", "negative", "neutral"]:
        d = s5.get(label, {})
        if "error" in d:
            interview_html += (
                f'<div class="interview-card {label}">'
                f'<span class="label-tag">{_h(label)}</span>'
                f'<p style="color:var(--text-muted)">샘플 없음 — {_h(d.get("error",""))}</p>'
                f'</div>'
            )
            continue
        p = d["persona"]
        agent_id = d.get("agent_id", "")
        label_kr = label_kr_intv.get(label, label)
        age = p.get("age", "")
        gender = p.get("gender", "")
        job = p.get("job", "")
        home_dong = p.get("home_dong", "")
        income = p.get("income", "")
        lifestyle = p.get("lifestyle", "")

        initial_bubbles = ""
        for qa in d["qa"]:
            q_text = qa["q"]
            a_text = qa["a"]
            initial_bubbles += (
                f'<div class="chat-bubble user">'
                f'<div class="meta"><span class="sender">인터뷰어</span></div>'
                f'<div class="text">{_h(q_text)}</div>'
                f'</div>'
                f'<div class="chat-bubble agent">'
                f'<div class="meta"><span class="sender">에이전트 ({_h(agent_id)})</span></div>'
                f'<div class="text">{_h(a_text)}</div>'
                f'</div>'
            )

        interview_html += f"""
        <div class="interview-card {label}" data-agent-id="{_h(agent_id)}">
          <span class="label-tag">{_h(label)}</span>
          <h3 style="margin:0 0 12px 0;color:var(--text-bright)">
            {_h(label_kr)} <span style="font-weight:400;font-size:13px;color:var(--text-muted)">— <code>{_h(agent_id)}</code></span>
          </h3>
          <div class="persona">
            <strong>페르소나:</strong> {_h(age)} {_h(gender)}
            · {_h(job)} · {_h(home_dong)} 거주 · 소득 {_h(income)}<br/>
            <strong>라이프스타일:</strong> {_h(lifestyle)}
          </div>

          <div class="fallback-banner" id="fallback-banner-{label}">
            <svg viewBox="0 0 20 20"><path d="M10 2a8 8 0 100 16 8 8 0 000-16zm1 11H9v-2h2v2zm0-4H9V5h2v4z"/></svg>
            <span>API 서버가 실행 중이지 않아 시뮬레이션 모드로 답변합니다.</span>
          </div>

          <div class="chat-container">
            <div class="chat-messages" id="chat-messages-{label}">
              {initial_bubbles}
            </div>
            
            <div class="typing-indicator" id="typing-indicator-{label}">
              <span></span><span></span><span></span>
            </div>

            <div class="chat-input-area">
              <input type="text" class="chat-input" id="chat-input-{label}" placeholder="{_h(label_kr)} 에이전트에게 질문을 입력해보세요..." />
              <button class="chat-send-btn" id="chat-send-{label}" onclick="sendChatMessage('{label}')">
                <svg viewBox="0 0 24 24">
                  <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
                </svg>
              </button>
            </div>
          </div>
        </div>
        """

    js_code = """
    document.addEventListener('DOMContentLoaded', function() {
      // IntersectionObserver for reveal animations
      var obs = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
          if (e.isIntersecting) e.target.classList.add('visible');
        });
      }, { threshold: 0.1 });
      document.querySelectorAll('.reveal').forEach(function(el) { obs.observe(el); });

      // Scroll-spy for sidebar navigation
      var sections = document.querySelectorAll('section[id]');
      var navLinks = document.querySelectorAll('.sidebar nav a');
      var spy = new IntersectionObserver(function(entries) {
        entries.forEach(function(e) {
          if (e.isIntersecting) {
            navLinks.forEach(function(a) { a.classList.remove('active'); });
            var link = document.querySelector('.sidebar nav a[href=\"#' + e.target.id + '\"]');
            if (link) link.classList.add('active');
          }
        });
      }, { rootMargin: '-20% 0px -80% 0px' });
      sections.forEach(function(s) { spy.observe(s); });

      // Theme toggle functionality
      var toggleBtn = document.getElementById('theme-toggle');
      var sunIcon = toggleBtn.querySelector('.sun-icon');
      var moonIcon = toggleBtn.querySelector('.moon-icon');

      function updateTheme(isLight) {
        if (isLight) {
          document.body.classList.add('light-theme');
          sunIcon.style.display = 'block';
          moonIcon.style.display = 'none';
        } else {
          document.body.classList.remove('light-theme');
          sunIcon.style.display = 'none';
          moonIcon.style.display = 'block';
        }
      }

      toggleBtn.addEventListener('click', function() {
        var isLight = !document.body.classList.contains('light-theme');
        updateTheme(isLight);
        localStorage.setItem('theme', isLight ? 'light' : 'dark');
      });

      var savedTheme = localStorage.getItem('theme') || 'dark';
      updateTheme(savedTheme === 'light');

      // Initialize Chat inputs and scroll to bottom
      document.querySelectorAll('.chat-input').forEach(function(input) {
        input.addEventListener('keypress', function(e) {
          if (e.key === 'Enter') {
            const label = this.id.replace('chat-input-', '');
            sendChatMessage(label);
          }
        });
      });

      document.querySelectorAll('.chat-messages').forEach(function(box) {
        box.scrollTop = box.scrollHeight;
      });
    });

    // Send Chat Message
    window.sendChatMessage = async function(label) {
      const inputEl = document.getElementById('chat-input-' + label);
      const messagesEl = document.getElementById('chat-messages-' + label);
      const sendBtnEl = document.getElementById('chat-send-' + label);
      const typingEl = document.getElementById('typing-indicator-' + label);
      const cardEl = messagesEl.closest('.interview-card');
      const agentId = cardEl.getAttribute('data-agent-id');
      const text = inputEl.value.trim();
      
      if (!text) return;
      
      // Clear input and disable UI
      inputEl.value = '';
      inputEl.disabled = true;
      sendBtnEl.disabled = true;
      
      // Append user bubble
      appendBubble(messagesEl, 'user', '인터뷰어', text);
      
      // Show typing indicator
      typingEl.style.display = 'flex';
      messagesEl.scrollTop = messagesEl.scrollHeight;
      
      // Prepare chat history
      const history = getChatHistory(messagesEl);
      
      let reply = '';
      let isFallback = false;
      
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 6000); // 6s timeout
        
        const response = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            agent_id: agentId,
            message: text,
            history: history
          }),
          signal: controller.signal
        });
        clearTimeout(timeoutId);
        
        if (!response.ok) throw new Error('Server returned error status');
        const data = await response.json();
        reply = data.reply;
      } catch (err) {
        console.warn('Failed to call API chat backend, using local simulation:', err);
        isFallback = true;
        
        // Add a small natural-feeling delay for fallback thinking
        await new Promise(resolve => setTimeout(resolve, 800 + Math.random() * 800));
        reply = generateMockReply(label, text);
      }
      
      // Hide typing indicator
      typingEl.style.display = 'none';
      
      // Show warning banner if fallback was used
      if (isFallback) {
        const banner = document.getElementById('fallback-banner-' + label);
        if (banner) banner.style.display = 'flex';
      }
      
      // Append agent reply
      appendBubble(messagesEl, 'agent', '에이전트 (' + agentId + ')', reply);
      messagesEl.scrollTop = messagesEl.scrollHeight;
      
      // Re-enable UI
      inputEl.disabled = false;
      sendBtnEl.disabled = false;
      inputEl.focus();
    };

    function appendBubble(container, role, sender, text) {
      const bubble = document.createElement('div');
      bubble.className = 'chat-bubble ' + role;
      
      const meta = document.createElement('div');
      meta.className = 'meta';
      
      const senderSpan = document.createElement('span');
      senderSpan.className = 'sender';
      senderSpan.textContent = sender;
      
      const timeSpan = document.createElement('span');
      const now = new Date();
      timeSpan.textContent = now.getHours().toString().padStart(2, '0') + ':' + now.getMinutes().toString().padStart(2, '0');
      
      meta.appendChild(senderSpan);
      meta.appendChild(timeSpan);
      
      const textDiv = document.createElement('div');
      textDiv.className = 'text';
      textDiv.textContent = text;
      
      bubble.appendChild(meta);
      bubble.appendChild(textDiv);
      container.appendChild(bubble);
    }

    function getChatHistory(messagesEl) {
      const history = [];
      messagesEl.querySelectorAll('.chat-bubble').forEach(bubble => {
        const role = bubble.classList.contains('user') ? 'user' : 'assistant';
        const text = bubble.querySelector('.text').textContent;
        history.push({ role: role, content: text });
      });
      return history;
    }

    function generateMockReply(label, text) {
      const lower = text.toLowerCase();
      
      if (label === 'positive') {
        if (lower.includes('가게') || lower.includes('카페') || lower.includes('자주') || lower.includes('어디')) {
          return "저는 주로 역삼역 2번 출구 근처의 '블루보틀'을 매일 방문했어요. 분위기가 마음에 들고 바우처 할인이 적용되니까 매일 가기에 정말 좋았거든요!";
        }
        if (lower.includes('정책') || lower.includes('바우처') || lower.includes('할인') || lower.includes('혜택')) {
          return "바우처 정책 덕분에 30%나 할인받을 수 있었던 건 진짜 최고였어요! 부담 없이 맛있는 음료와 디저트를 마음껏 즐겼습니다.";
        }
        if (lower.includes('친구') || lower.includes('추천') || lower.includes('동료')) {
          return "네, 직장 동료가 선릉역 주변의 분위기 좋은 디저트 카페를 추천해줘서 다녀왔는데, 바우처 사용이 가능해서 아주 기분 좋게 다녀왔어요.";
        }
        return "바우처 혜택 덕분에 최근 제 일상과 카페 탐방에 큰 보탬이 되었답니다. 다음 주에도 이 혜택을 계속 이용할 생각이에요. 혹시 다른 것도 알고 싶으신가요?";
      } else if (label === 'negative') {
        if (lower.includes('가게') || lower.includes('카페') || lower.includes('자주') || lower.includes('어디')) {
          return "저는 굳이 멀리 안 나가고 집 근처 편의점이나 아는 동네 식당 위주로 다녀요. 이동하기 번거로워서 집 주변이 가장 마음 편합니다.";
        }
        if (lower.includes('정책') || lower.includes('바우처') || lower.includes('할인') || lower.includes('혜택')) {
          return "강남까지 갈 일이 없는 저 같은 거주자들에겐 전혀 혜택이 없었어요. 특정 자치구에만 쏠린 바우처 정책은 실효성이 크지 않다고 봅니다.";
        }
        if (lower.includes('친구') || lower.includes('추천') || lower.includes('동료')) {
          return "주변 사람들이 아무리 좋다고 추천을 해줘도, 거리가 멀다면 귀찮아서 결국 가던 곳만 가게 되더라고요.";
        }
        return "저는 강남까지 갈 동기가 없어서 정책 혜택을 받지 못했어요. 서울시 전반의 균형 잡힌 골목 상권 활성화 정책이 필요하다고 생각합니다.";
      } else {
        // neutral
        if (lower.includes('가게') || lower.includes('카페') || lower.includes('자주') || lower.includes('어디')) {
          return "저는 특별히 한 가게를 고집하기보다, 학교 근처의 밥집이나 커피점들을 동선 맞춰서 가끔 가는 편이에요.";
        }
        if (lower.includes('정책') || lower.includes('바우처') || lower.includes('할인') || lower.includes('혜택')) {
          return "정책의 취지는 대충 들었지만, 제가 자주 가는 구역이 아니기도 하고 신청 절차도 눈에 띄지 않아서 굳이 찾아 쓰지는 않았습니다.";
        }
        if (lower.includes('친구') || lower.includes('추천') || lower.includes('동료')) {
          return "가끔 동기들이나 친구들의 추천으로 가까운 음식점에 가거나 약속을 잡을 때는 있어요. 그럴 때는 그냥 무난한 식사를 해요.";
        }
        return "특별한 혜택을 직접적으로 느끼진 못했던 기간이었습니다. 정책이 있다는 건 긍정적이지만 저와 같은 동선의 대학원생들에겐 체감이 덜 되는 편이네요.";
      }
    }
    """

    # 최종 HTML
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>서울 상권정책 시뮬레이션 — 최종 보고서</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{HTML_STYLE}</style>
</head>
<body>
<div class="layout">

  <aside class="sidebar">
    <div class="sidebar-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom: 24px;">
      <div class="brand" style="margin-bottom:0;">SEOUL POLICY SIMULATION</div>
      <button id="theme-toggle" class="theme-toggle-btn" aria-label="Toggle theme">
        <svg class="sun-icon" viewBox="0 0 24 24" style="display:none; width:16px; height:16px;"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
        <svg class="moon-icon" viewBox="0 0 24 24" style="width:16px; height:16px;"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
      </button>
    </div>
    <h2>목차</h2>
    <nav>{nav}</nav>
    <div class="tech-info">
      <span>Qwen3-14B-AWQ</span><br/>
      <span>Neo4j 5.x</span><br/>
      ABM × Generative Agent
    </div>
  </aside>

  <main class="main">
    <header class="cover">
      <div class="meta">FINAL REPORT · {datetime.now().strftime('%Y-%m-%d')}</div>
      <h1>서울시 상권 활성화 정책<br/>시뮬레이션 결과 보고서</h1>
      <div class="subtitle">
        <strong>{_h(s1.get('기간'))}</strong> · 에이전트 {n_agent:,}명 · 정책 발효 {_h(s1.get('정책_시행일'))}<br/>
        강남구 여름 카페·디저트 원소 바우처 (자연어 정책 자동 주입) 효과 분석
      </div>
      <div class="badges">
        <span class="badge alt">Qwen3-14B-AWQ</span>
        <span class="badge purple">Neo4j Graph DB</span>
        <span class="badge">DID 분석</span>
        <span class="badge">설명가능 AI</span>
      </div>

      <div class="kpi-grid" style="margin-top:32px;">
        <div class="kpi">
          <div class="icon"><svg viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg></div>
          <div class="value">{n_agent:,}</div><div class="label">에이전트</div>
        </div>
        <div class="kpi">
          <div class="icon"><svg viewBox="0 0 24 24"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg></div>
          <div class="value">{n_plan:,}</div><div class="label">생성된 Plan</div>
        </div>
        <div class="kpi">
          <div class="icon"><svg viewBox="0 0 24 24"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg></div>
          <div class="value">{n_conv:,}</div><div class="label">사회적 상호작용</div>
        </div>
        <div class="kpi pos">
          <div class="icon"><svg viewBox="0 0 24 24"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg></div>
          <div class="value">{did_str}</div><div class="label">정책 DID 순효과</div>
        </div>
      </div>
    </header>

    <section id="s-summary" class="reveal">
      <h2>1. 시뮬레이션 개요</h2>
      <p>본 보고서는 서울시 25개 자치구의 에이전트 {n_agent:,}명을 대상으로,
        강남구 단일 정책의 효과를 정책 시행 전·후 시간축으로 분석한 결과를 정리합니다.
        각 에이전트는 매일 LLM이 생성한 일정에 따라 외출·소비를 하며, 그 결정의
        사유(reasoning)와 동기(trigger)가 그래프 DB에 영구 저장되어 사후 인터뷰가 가능합니다.</p>
      <table>
        <thead><tr><th>항목</th><th class="num">값</th></tr></thead>
        <tbody>{cond_rows}</tbody>
      </table>
    </section>

    <section id="s-sales" class="reveal">
      <h2>2. 정책 시행 전 vs 후 매출 추이</h2>
      <p>강남구 카페·디저트 바우처(30% 환급, 인당 5만원 한도)가
        시행된 <code>{_h(policy_from or '—')}</code> 시점을 기준으로
        정책 대상 카테고리(식사·카페·디저트)의 일별 매출을 비교합니다.
        세 관점 — (A) 인구 비례 1인당 매출, (B) baseline 대비 변화율, (C) DID — 으로 분석합니다.</p>
      {_figure(chart_dir / s2_figs['per_capita'] if isinstance(s2_figs, dict) else chart_dir / s2_figs,
               '(A) 1인당 일별 매출 — 인구 비례 환산 후 강남 vs 비강남 절대 비교')}
      {_figure(chart_dir / s2_figs['change_rate'], '(B) baseline 대비 매출 변화율 — 같은 0% 출발점에서 패턴 비교') if isinstance(s2_figs, dict) else ''}
      {_figure(chart_dir / s2_figs['did'], '(C) DID — 강남 변화율 − 비강남 변화율 (정책 순효과)') if isinstance(s2_figs, dict) else ''}
      {'<h3>평균 일간 매출 비교</h3><table><thead><tr><th>자치구</th><th class="num">시행 전</th><th class="num">시행 후</th><th class="num">변화율</th></tr></thead><tbody>' + sales_rows + '</tbody></table>' + did_callout if sm else ''}
    </section>

    <section id="s-spillover" class="reveal">
      <h2>3. 간접 영향 (Spillover)</h2>
      <p>강남 정책이 직접 적용되지 않은 인접 자치구(서초·송파)와 멀리 떨어진 강북에 미친 영향을 비교합니다.
        인접 자치구의 매출 변화가 강북보다 두드러지면 spillover로 해석할 수 있습니다.</p>
      {_figure(chart_dir / s3_fig, '자치구별 매출 추이 — 강남 정책의 간접 영향 추적')}
    </section>

    <section id="s-behavior" class="reveal">
      <h2>4. 소비자 행동 분석</h2>

      <h3 id="s-trigger">4-1. 결정 동기 분포</h3>
      <p>외출(집·직장 제외) 시 어떤 요인이 결정을 이끌었는지 LLM이 trigger 라벨로 분류한 분포입니다.
        7-라벨 enum(약속·소문·정책·습관·Top 카테고리·컨디션·기타)로 정량화됩니다.</p>
      {_figure(chart_dir / s41_fig, '외출 이벤트의 결정 동기 (trigger) 분포')}
      <table>
        <thead><tr><th>동기</th><th class="num">건수</th><th class="num">비율</th></tr></thead>
        <tbody>{trigger_rows}</tbody>
      </table>

      <h3 id="s-regular">4-2. 단골 vs 신규</h3>
      <p>각 에이전트의 POI 인지 관계(KNOWS_POI) 빈도 분포와, 그 관계가 어떻게 형성되었는지(출처) 분석합니다.</p>
      {_figure(chart_dir / s42_fig, '방문 빈도 분포 + KNOWS_POI 출처')}
      <table>
        <thead><tr><th>구분</th><th class="num">관계 수</th></tr></thead>
        <tbody>{regular_rows}</tbody>
      </table>
      <p style="color:var(--text-muted);font-size:13px">전체 KNOWS_POI 관계 수: <strong style="color:var(--cyan)">{s42_data['total']:,}</strong></p>

      <h3 id="s-satisfaction">4-3. 만족도 — 어떤 동기가 더 만족스러웠나</h3>
      <p>결정 동기(trigger)별 평균 만족도를 비교해 어떤 동기로 외출했을 때 가장 만족도가 높은지 측정합니다.</p>
      {_figure(chart_dir / s43_fig, '결정 동기별 + 카테고리별 평균 만족도')}
      <table>
        <thead><tr><th>동기</th><th class="num">평균 만족도</th><th class="num">표본 수</th></tr></thead>
        <tbody>{sat_rows}</tbody>
      </table>
    </section>

    <section id="s-interview" class="reveal">
      <h2>5. 1대1 인터뷰 — 페르소나별 대표</h2>
      <p>정책 사용액과 만족도(mood)를 기준으로 세 유형의 군집을 정의하고, 각 군집에서 대표 1명씩 자동
        추출하여 LLM이 그 페르소나로 인터뷰에 응답합니다. 답변은 시뮬 시점에 적재된 reasoning을 인용해
        실제 의사결정을 추적합니다.</p>
      {interview_html}
    </section>

    <section id="s-appendix" class="reveal">
      <h2>부록</h2>
      <ul>
        <li>본 보고서는 <code>scripts/sim/generate_final_report.py</code>로 자동 생성되었습니다.</li>
        <li>시뮬 원본 데이터는 Neo4j에 보존됩니다 (Plan / State / Memory / Conversation 노드).</li>
        <li>인터랙티브 시각화: <code>output/sim/visualization/sim_standalone.html</code></li>
        <li>인터뷰 LLM 모듈: <code>scripts/sim/interview_agent.py</code> (라벨별 또는 특정 agent ID로 호출 가능)</li>
      </ul>
    </section>

    <div class="footer">
      Generated by <code>generate_final_report.py</code> · {datetime.now().strftime('%Y-%m-%d %H:%M KST')}<br/>
      Kw Capstone · 서울시 상권정책 시뮬레이션 프로젝트
    </div>
  </main>
</div>
<script>{js_code}</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# P009 income bucket별 DID 분석 (정책 적용 그룹 vs baseline)
# ═══════════════════════════════════════════════════════════════
def section2b_income_did_p009(start: date, days: int, policy_from: str, out_dir: Path) -> dict:
    """P009 적용 효과: income bucket(중상/중/중하/하)별 baseline(Day 1·2) vs treatment(Day 3) DID.

    grant 받은 agent만 treatment 그룹 (jsonl grant_applied_today > 0).
    grant=0 agent (정책 적용 실패 + income='상' 자연 비대상) 모두 제외 → 정책 효과만 측정.
    """
    import json
    cutoff = date.fromisoformat(policy_from)
    treatment_day = cutoff.isoformat()

    # 정책 적용 그룹 aid set — jsonl 메트릭에서 fetch
    grant_aids: set[str] = set()
    metrics_path = Path(os.path.expanduser("~/sim_output/metrics")) / f"day_{treatment_day}.jsonl"
    if not metrics_path.exists():
        # SIM_OUTPUT_DIR 환경변수
        sim_dir = os.environ.get("SIM_OUTPUT_DIR")
        if sim_dir:
            metrics_path = Path(sim_dir) / "metrics" / f"day_{treatment_day}.jsonl"
    if metrics_path.exists():
        with open(metrics_path, encoding='utf-8') as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get('status') == 'ok' and (r.get('grant_applied_today', 0) or 0) > 0:
                        grant_aids.add(r['aid'])
                except Exception:
                    pass
    print(f"  [P009 분석] 정책 적용 그룹: {len(grant_aids)}명", file=sys.stderr)

    income_buckets = ["중상", "중", "중하", "하"]
    grant_amounts = {"중상": 100000, "중": 250000, "중하": 450000, "하": 600000}
    baseline_days = [(start + timedelta(days=i)).isoformat() for i in range(days)
                     if (start + timedelta(days=i)) < cutoff]

    per_bucket: dict[str, dict] = {}
    with driver_session() as s:
        for bucket in income_buckets:
            # baseline: 해당 income bucket agent들의 Day 1·2 평균 일소비 (commerce)
            baseline_avg = 0.0
            baseline_agents = 0
            if baseline_days:
                row = s.run("""
                    MATCH (a:Agent {p_income_level: $lv})
                    MATCH (a)-[:HAS_PLAN]->(p:Plan)-[i:INCLUDES]->(:POI {type:'commerce'})
                    WHERE p.day IN [date(d) | d IN $days]
                    WITH a, p.day AS day, sum(coalesce(i.actual_spent, 0)) AS daily_spend
                    WITH avg(daily_spend) AS avg_per_day, count(DISTINCT a) AS n
                    RETURN avg_per_day AS avg, n
                """, lv=bucket, days=baseline_days).single()
                baseline_avg = float(row['avg'] or 0)
                baseline_agents = int(row['n'] or 0)

            # treatment: 정책 적용 agent + 같은 bucket의 Day 3 평균 일소비
            treatment_avg = 0.0
            treatment_agents = 0
            if grant_aids:
                row = s.run("""
                    MATCH (a:Agent {p_income_level: $lv})
                    WHERE a.id IN $aids
                    MATCH (a)-[:HAS_PLAN {day: date($day)}]->(p:Plan)-[i:INCLUDES]->(:POI {type:'commerce'})
                    WITH a, sum(coalesce(i.actual_spent, 0)) AS daily_spend
                    RETURN avg(daily_spend) AS avg, count(DISTINCT a) AS n
                """, lv=bucket, aids=list(grant_aids), day=treatment_day).single()
                treatment_avg = float(row['avg'] or 0)
                treatment_agents = int(row['n'] or 0)

            # 정책 사용액 평균 (treatment 그룹) — apoc 의존 없이 Python 후처리
            policy_spend_avg = 0.0
            if treatment_agents > 0:
                rows = s.run("""
                    MATCH (a:Agent {p_income_level: $lv})
                    WHERE a.id IN $aids
                    MATCH (a)-[:HAS_PLAN {day: date($day)}]->(p:Plan)-[i:INCLUDES]->(:POI)
                    WHERE i.spent_from_policy IS NOT NULL
                      AND i.spent_from_policy <> '{}'
                      AND i.spent_from_policy <> 'null'
                    RETURN a.id AS aid, i.spent_from_policy AS sp
                """, lv=bucket, aids=list(grant_aids), day=treatment_day).data()
                agent_spend: dict[str, int] = {}
                for r in rows:
                    try:
                        sp_dict = json.loads(r['sp']) if r['sp'] else {}
                        aid = r['aid']
                        agent_spend[aid] = agent_spend.get(aid, 0) + int(sp_dict.get('P009', 0))
                    except Exception:
                        pass
                if agent_spend:
                    # 정책 사용 agent만 집계 vs 전체 treatment agent 분모? 후자가 의미 있음
                    policy_spend_avg = sum(agent_spend.values()) / max(treatment_agents, 1)

            change_pct = ((treatment_avg - baseline_avg) / max(baseline_avg, 1)) * 100
            grant = grant_amounts[bucket]
            usage_rate = (policy_spend_avg / grant * 100) if grant > 0 else 0

            per_bucket[bucket] = {
                "baseline_avg": round(baseline_avg),
                "treatment_avg": round(treatment_avg),
                "change_pct": round(change_pct, 2),
                "grant_amount": grant,
                "policy_spend_avg": round(policy_spend_avg),
                "usage_rate_pct": round(usage_rate, 2),
                "baseline_n": baseline_agents,
                "treatment_n": treatment_agents,
            }

    # 차트 — bucket별 baseline vs treatment 막대
    plt = _setup_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    buckets = income_buckets
    baseline_vals = [per_bucket[b]['baseline_avg'] for b in buckets]
    treatment_vals = [per_bucket[b]['treatment_avg'] for b in buckets]
    grant_vals = [per_bucket[b]['grant_amount'] for b in buckets]
    spend_vals = [per_bucket[b]['policy_spend_avg'] for b in buckets]

    # (A) baseline vs treatment per income
    ax = axes[0]
    x = range(len(buckets))
    w = 0.35
    ax.bar([xi - w/2 for xi in x], baseline_vals, w, label='baseline (Day 1·2 평균)', color='#4cc9f0')
    ax.bar([xi + w/2 for xi in x], treatment_vals, w, label='treatment (Day 3 정책 적용)', color='#e76f51')
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Income bucket'); ax.set_ylabel('1인 1일 평균 소비 (원)')
    ax.set_title('(A) 정책 적용 그룹의 baseline vs treatment per income bucket')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))

    # (B) grant 지급액 vs 사용액
    ax = axes[1]
    ax.bar([xi - w/2 for xi in x], grant_vals, w, label='Grant 지급액', color='#a3d977')
    ax.bar([xi + w/2 for xi in x], spend_vals, w, label='Grant 평균 사용액', color='#fb8500')
    ax.set_xticks(list(x))
    ax.set_xticklabels(buckets)
    ax.set_xlabel('Income bucket'); ax.set_ylabel('금액 (원)')
    ax.set_title('(B) Grant 차등 지급액 vs 평균 사용액')
    ax.legend()
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v):,}'))

    plt.tight_layout()
    fig_path = out_dir / "fig2b_income_did_p009.png"
    plt.savefig(fig_path, dpi=140, bbox_inches='tight')
    plt.close()

    return {
        "per_bucket": per_bucket,
        "grant_aids_count": len(grant_aids),
        "fig": fig_path.name,
    }


def _insert_p009_section(md: str, s2b: dict, chart_dir_rel: str) -> str:
    """section 2 직후에 P009 income bucket 분석 섹션 삽입."""
    pb = s2b["per_bucket"]
    fig = s2b["fig"]
    n_grant = s2b["grant_aids_count"]

    lines = []
    lines.append("## 2-B. P009 정책 효과 — Income Bucket별 분리 분석")
    lines.append("")
    lines.append(f"- 정책 적용 그룹(treatment): grant 받은 agent **{n_grant:,}명**")
    lines.append(f"- baseline: Day 1·2 (정책 미주입) per income bucket 평균 일소비")
    lines.append(f"- treatment: Day 3 (정책 주입일) 같은 income bucket의 정책 적용 agent 평균 일소비")
    lines.append("")
    lines.append(f"![P009 income bucket DID]({chart_dir_rel}/{fig})")
    lines.append("")
    lines.append("### Income bucket별 정책 효과")
    lines.append("")
    lines.append("| Income | Grant | baseline 평균 | treatment 평균 | 변화율 | 평균 grant 사용 | 사용률 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for b in ["중상", "중", "중하", "하"]:
        d = pb[b]
        lines.append(
            f"| **{b}** | {d['grant_amount']:,}원 | {d['baseline_avg']:,}원 | "
            f"{d['treatment_avg']:,}원 | **{d['change_pct']:+}%** | "
            f"{d['policy_spend_avg']:,}원 | {d['usage_rate_pct']}% |"
        )
    lines.append("")
    lines.append("**해석**:")
    lines.append("- 변화율 = (treatment - baseline) / baseline × 100. 양수면 정책으로 소비 증가.")
    lines.append("- 사용률 = 평균 grant 사용 / 지급액. 받은 사람들이 얼마나 활용했는지.")
    lines.append("- income bucket별 분리 비교라 income 분포 mismatch 영향 없음.")
    lines.append("")

    # section 3 앞에 삽입
    marker = "## 3. 간접 영향"
    if marker in md:
        return md.replace(marker, "\n".join(lines) + "\n" + marker)
    # marker 없으면 끝에 추가
    return md + "\n\n" + "\n".join(lines)


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

    # P009 income bucket DID 분석 (정책 적용 그룹 vs baseline, per income)
    if args.policy_from:
        print(f"[6/8] P009 income bucket DID 분석 ...", file=sys.stderr)
        s2b = section2b_income_did_p009(start, args.days, args.policy_from, chart_dir)
    else:
        s2b = None

    print(f"[7/8] markdown 빌드 ...", file=sys.stderr)
    md = build_markdown(start, args.days, args.policy_from,
                        s1, s2, s3, s4_1, s4_2, s4_3, s5, chart_dir_rel)
    # P009 분석 markdown 삽입 (section 2 직후)
    if s2b:
        md = _insert_p009_section(md, s2b, chart_dir_rel)
    out_md.write_text(md, encoding="utf-8")

    print(f"[7/8] HTML 빌드 (차트 PNG → base64 임베드, 단일 파일) ...", file=sys.stderr)
    html_doc = build_html(start, args.days, args.policy_from,
                          s1, s2, s3, s4_1, s4_2, s4_3, s5, chart_dir)
    out_html = out_md.with_suffix(".html")
    out_html.write_text(html_doc, encoding="utf-8")

    print(f"\n[8/8] DONE", file=sys.stderr)
    print(f"  → markdown: {out_md}", file=sys.stderr)
    print(f"  → HTML:     {out_html} ({out_html.stat().st_size/1024:.0f} KB)", file=sys.stderr)
    print(f"  → charts:   {chart_dir}/*.png (HTML 안엔 base64 임베드)", file=sys.stderr)


if __name__ == "__main__":
    main()
