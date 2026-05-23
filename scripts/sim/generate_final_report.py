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
      --out output/sim/report/FINAL_REPORT_7D.md
"""
from __future__ import annotations

import argparse
import base64
import html as _html
import json
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
from stage1_intent import normalize_trigger  # noqa: E402


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
    # 라이프스타일·기타 환각 라벨은 normalize_trigger 가 흡수.
    dist: dict[str, int] = {}
    for r in rows:
        k = normalize_trigger(r["trigger"]) or r["trigger"]
        dist[k] = dist.get(k, 0) + r["n"]
    dist_pct = {k: round(v / total * 100, 2) for k, v in dist.items()}

    plt = _setup_mpl()
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
                 "lifestyle": "라이프스타일", "top_category": "Top 카테고리", "mood": "컨디션",
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
        by_trigger_raw = s.run("""
            MATCH ()-[i:INCLUDES]->()
            WHERE i.actual_satisfaction IS NOT NULL AND i.trigger IS NOT NULL
              AND NOT i.category IN ['집','직장']
            RETURN i.trigger AS trigger,
                   sum(i.actual_satisfaction) AS sum_sat,
                   count(*) AS n
        """).data()
        # normalize_trigger 로 habit / life_style → lifestyle 등 통합 후 재집계
        agg: dict[str, dict] = {}
        for r in by_trigger_raw:
            k = normalize_trigger(r["trigger"]) or r["trigger"]
            slot = agg.setdefault(k, {"sum": 0.0, "n": 0})
            slot["sum"] += float(r["sum_sat"] or 0)
            slot["n"]   += int(r["n"] or 0)
        by_trigger = sorted(
            ({"trigger": k, "avg_sat": (v["sum"] / v["n"] if v["n"] else 0.0), "n": v["n"]}
             for k, v in agg.items()),
            key=lambda x: -x["avg_sat"],
        )
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
                 "lifestyle": "라이프스타일", "top_category": "Top 카테고리", "mood": "컨디션",
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
                 "lifestyle": "라이프스타일", "top_category": "Top 카테고리", "mood": "컨디션", "none": "기타"}
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
  --primary: #1a365d;
  --accent: #3182ce;
  --accent-soft: #ebf4ff;
  --success: #38a169;
  --warning: #d69e2e;
  --danger: #e53e3e;
  --neutral-50: #f7fafc;
  --neutral-100: #edf2f7;
  --neutral-200: #e2e8f0;
  --neutral-400: #a0aec0;
  --neutral-600: #4a5568;
  --neutral-800: #2d3748;
  --neutral-900: #1a202c;
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.08);
}
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body {
  font-family: 'Noto Sans KR', 'Apple SD Gothic Neo', 'Malgun Gothic', sans-serif;
  background: var(--neutral-50);
  color: var(--neutral-800);
  line-height: 1.7;
  font-size: 15px;
}
.layout { display: flex; min-height: 100vh; }

/* ─────────────────── 사이드바 목차 ─────────────────── */
.sidebar {
  width: 260px;
  background: var(--neutral-900);
  color: #cbd5e0;
  padding: 28px 20px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
}
.sidebar h2 {
  color: #fff;
  font-size: 15px;
  letter-spacing: 0.5px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(255,255,255,0.1);
  margin-bottom: 14px;
}
.sidebar .brand {
  color: #63b3ed;
  font-size: 11px;
  letter-spacing: 2px;
  margin-bottom: 4px;
  text-transform: uppercase;
}
.sidebar nav a {
  display: block;
  color: #cbd5e0;
  text-decoration: none;
  padding: 8px 10px;
  font-size: 13px;
  border-radius: 4px;
  margin: 2px 0;
  border-left: 2px solid transparent;
  transition: all 0.15s;
}
.sidebar nav a:hover {
  background: rgba(255,255,255,0.05);
  color: #fff;
  border-left-color: #63b3ed;
}
.sidebar nav .lvl2 { padding-left: 22px; font-size: 12px; color: #a0aec0; }

/* ─────────────────── 메인 ─────────────────── */
.main {
  flex: 1;
  max-width: 1024px;
  margin: 0 auto;
  padding: 56px 64px 80px;
}
header.cover {
  margin-bottom: 48px;
  padding-bottom: 28px;
  border-bottom: 3px solid var(--primary);
}
.cover .meta {
  color: var(--accent);
  font-size: 12px;
  letter-spacing: 3px;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.cover h1 {
  color: var(--primary);
  font-size: 34px;
  font-weight: 700;
  line-height: 1.3;
  margin-bottom: 16px;
  letter-spacing: -0.5px;
}
.cover .subtitle {
  color: var(--neutral-600);
  font-size: 15px;
  line-height: 1.6;
}
.cover .badges { margin-top: 18px; display: flex; gap: 8px; flex-wrap: wrap; }
.badge {
  display: inline-block;
  padding: 5px 12px;
  background: var(--accent-soft);
  color: var(--accent);
  border-radius: 4px;
  font-size: 12px;
  font-weight: 500;
}
.badge.alt { background: #f0fff4; color: var(--success); }
.badge.warn { background: #fffbeb; color: var(--warning); }

/* ─────────────────── 섹션 ─────────────────── */
section { margin-bottom: 48px; }
h2 {
  color: var(--primary);
  font-size: 22px;
  font-weight: 700;
  padding-bottom: 10px;
  border-bottom: 2px solid var(--neutral-200);
  margin-bottom: 20px;
  scroll-margin-top: 20px;
}
h3 {
  color: var(--neutral-800);
  font-size: 17px;
  font-weight: 600;
  margin: 24px 0 12px;
  scroll-margin-top: 20px;
}
h4 { font-size: 14px; color: var(--neutral-600); margin: 16px 0 8px; }
p { margin-bottom: 12px; }

/* ─────────────────── 표 ─────────────────── */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 20px;
  font-size: 14px;
  background: white;
  border-radius: 6px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
thead { background: var(--primary); color: white; }
th, td {
  padding: 10px 14px;
  text-align: left;
  border-bottom: 1px solid var(--neutral-200);
}
th { font-weight: 600; font-size: 13px; letter-spacing: 0.3px; }
tbody tr:nth-child(odd) { background: var(--neutral-50); }
tbody tr:hover { background: var(--accent-soft); }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td strong { color: var(--primary); }

/* ─────────────────── 그림 ─────────────────── */
figure {
  margin: 20px 0;
  background: white;
  padding: 16px;
  border-radius: 8px;
  box-shadow: var(--shadow-md);
  text-align: center;
}
figure img { max-width: 100%; height: auto; border-radius: 4px; }
figcaption {
  margin-top: 10px;
  color: var(--neutral-600);
  font-size: 13px;
  font-style: italic;
}

/* ─────────────────── 인터뷰 ─────────────────── */
.interview-card {
  background: white;
  border-radius: 10px;
  padding: 24px 28px;
  margin: 20px 0;
  box-shadow: var(--shadow-md);
  border-left: 4px solid var(--accent);
}
.interview-card.positive { border-left-color: var(--success); }
.interview-card.negative { border-left-color: var(--danger); }
.interview-card.neutral  { border-left-color: var(--neutral-400); }
.interview-card .label {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.interview-card.positive .label { background: #f0fff4; color: var(--success); }
.interview-card.negative .label { background: #fff5f5; color: var(--danger); }
.interview-card.neutral  .label { background: var(--neutral-100); color: var(--neutral-600); }
.interview-card .persona {
  font-size: 13px;
  color: var(--neutral-600);
  background: var(--neutral-50);
  padding: 12px 14px;
  border-radius: 6px;
  margin-bottom: 16px;
  line-height: 1.6;
}
.interview-card .persona strong { color: var(--primary); }
.qa { margin: 14px 0; }
.qa .q {
  font-weight: 600;
  color: var(--neutral-900);
  margin-bottom: 6px;
  font-size: 14px;
}
.qa .a {
  background: var(--neutral-50);
  border-left: 3px solid var(--neutral-200);
  padding: 12px 16px;
  margin: 6px 0 14px 0;
  border-radius: 0 6px 6px 0;
  font-style: normal;
  color: var(--neutral-800);
  font-size: 14px;
  line-height: 1.75;
}

/* ─────────────────── KPI 카드 ─────────────────── */
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin: 20px 0;
}
.kpi {
  background: white;
  padding: 18px 20px;
  border-radius: 8px;
  box-shadow: var(--shadow-sm);
  border-top: 3px solid var(--accent);
}
.kpi .value {
  font-size: 26px;
  font-weight: 700;
  color: var(--primary);
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
.kpi .label {
  font-size: 12px;
  color: var(--neutral-600);
  margin-top: 4px;
  letter-spacing: 0.3px;
}
.kpi.pos { border-top-color: var(--success); }
.kpi.pos .value { color: var(--success); }
.kpi.neg { border-top-color: var(--danger); }
.kpi.neg .value { color: var(--danger); }

/* ─────────────────── code·footnote ─────────────────── */
code {
  background: var(--neutral-100);
  padding: 2px 6px;
  border-radius: 3px;
  font-family: 'JetBrains Mono', 'D2Coding', 'Consolas', monospace;
  font-size: 13px;
  color: var(--primary);
}
.footer {
  margin-top: 64px;
  padding-top: 24px;
  border-top: 1px solid var(--neutral-200);
  color: var(--neutral-400);
  font-size: 12px;
  text-align: center;
}
.callout {
  background: var(--accent-soft);
  border-left: 4px solid var(--accent);
  padding: 14px 18px;
  border-radius: 0 6px 6px 0;
  margin: 16px 0;
  font-size: 14px;
}
.callout.warn { background: #fffbeb; border-left-color: var(--warning); }

@media (max-width: 900px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: relative; }
  .main { padding: 32px 24px; }
}
@media print {
  .sidebar { display: none; }
  .main { padding: 0; max-width: 100%; }
  figure, .interview-card, table, .kpi { box-shadow: none; border: 1px solid var(--neutral-200); }
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
                 "lifestyle": "라이프스타일", "top_category": "Top 카테고리", "mood": "컨디션",
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
                f'<span class="label">{_h(label)}</span>'
                f'<p style="color:var(--neutral-400)">샘플 없음 — {_h(d.get("error",""))}</p>'
                f'</div>'
            )
            continue
        p = d["persona"]
        qa_html = "".join(
            f'<div class="qa"><div class="q">Q. {_h(qa["q"])}</div>'
            f'<div class="a">{_h(qa["a"])}</div></div>'
            for qa in d["qa"]
        )
        interview_html += f"""
        <div class="interview-card {label}">
          <span class="label">{_h(label)}</span>
          <h3 style="margin:0 0 12px 0;color:var(--primary)">
            {_h(label_kr_intv[label])} <span style="font-weight:400;font-size:13px;color:var(--neutral-400)">— <code>{_h(d['agent_id'])}</code></span>
          </h3>
          <div class="persona">
            <strong>페르소나:</strong> {_h(p.get('age'))} {_h(p.get('gender'))}
            · {_h(p.get('job'))} · {_h(p.get('home_dong'))} 거주 · 소득 {_h(p.get('income'))}<br/>
            <strong>라이프스타일:</strong> {_h(p.get('lifestyle'))}
          </div>
          {qa_html}
        </div>
        """

    # 최종 HTML
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<title>서울 상권정책 시뮬레이션 — 최종 보고서</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{HTML_STYLE}</style>
</head>
<body>
<div class="layout">

  <aside class="sidebar">
    <div class="brand">SEOUL POLICY SIMULATION</div>
    <h2>목차</h2>
    <nav>{nav}</nav>
    <div style="margin-top:32px; font-size:11px; color:#718096; line-height:1.6">
      Qwen3-14B-AWQ<br/>
      Neo4j 5.x<br/>
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
        <span class="badge">Neo4j Graph DB</span>
        <span class="badge">DID 분석</span>
        <span class="badge">설명가능 AI</span>
      </div>

      <div class="kpi-grid" style="margin-top:32px;">
        <div class="kpi"><div class="value">{n_agent:,}</div><div class="label">에이전트</div></div>
        <div class="kpi"><div class="value">{n_plan:,}</div><div class="label">생성된 Plan</div></div>
        <div class="kpi"><div class="value">{n_conv:,}</div><div class="label">사회적 상호작용</div></div>
        <div class="kpi pos"><div class="value">{did_str}</div><div class="label">정책 DID 순효과</div></div>
      </div>
    </header>

    <section id="s-summary">
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

    <section id="s-sales">
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

    <section id="s-spillover">
      <h2>3. 간접 영향 (Spillover)</h2>
      <p>강남 정책이 직접 적용되지 않은 인접 자치구(서초·송파)와 멀리 떨어진 강북에 미친 영향을 비교합니다.
        인접 자치구의 매출 변화가 강북보다 두드러지면 spillover로 해석할 수 있습니다.</p>
      {_figure(chart_dir / s3_fig, '자치구별 매출 추이 — 강남 정책의 간접 영향 추적')}
    </section>

    <section id="s-behavior">
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
      <p style="color:var(--neutral-600);font-size:13px">전체 KNOWS_POI 관계 수: <strong>{s42_data['total']:,}</strong></p>

      <h3 id="s-satisfaction">4-3. 만족도 — 어떤 동기가 더 만족스러웠나</h3>
      <p>결정 동기(trigger)별 평균 만족도를 비교해 어떤 동기로 외출했을 때 가장 만족도가 높은지 측정합니다.</p>
      {_figure(chart_dir / s43_fig, '결정 동기별 + 카테고리별 평균 만족도')}
      <table>
        <thead><tr><th>동기</th><th class="num">평균 만족도</th><th class="num">표본 수</th></tr></thead>
        <tbody>{sat_rows}</tbody>
      </table>
    </section>

    <section id="s-interview">
      <h2>5. 1대1 인터뷰 — 페르소나별 대표</h2>
      <p>정책 사용액과 만족도(mood)를 기준으로 세 유형의 군집을 정의하고, 각 군집에서 대표 1명씩 자동
        추출하여 LLM이 그 페르소나로 인터뷰에 응답합니다. 답변은 시뮬 시점에 적재된 reasoning을 인용해
        실제 의사결정을 추적합니다.</p>
      {interview_html}
    </section>

    <section id="s-appendix">
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
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--policy-from", default=None, help="정책 effective_from (ISO date)")
    ap.add_argument("--out", default="output/sim/report/FINAL_REPORT.md")
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

    print(f"[6/8] markdown 빌드 ...", file=sys.stderr)
    md = build_markdown(start, args.days, args.policy_from,
                        s1, s2, s3, s4_1, s4_2, s4_3, s5, chart_dir_rel)
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
