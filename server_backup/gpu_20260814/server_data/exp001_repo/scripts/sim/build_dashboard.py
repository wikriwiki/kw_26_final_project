"""v4 시뮬 인터랙티브 HTML 대시보드.

Plotly 사용 — 단일 HTML 파일 (외부 의존 X, 브라우저로 열기만 하면 됨).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import mean

import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
from _common import driver_session  # noqa: E402

SIM_DIR = Path("C:/Users/Administrator/sim_output/metrics")
DATA_JSON = Path("G:/내 드라이브/Kw/final_project/output/sim/report/REPORT_3D_V4_MIDM.d/data.json")
OUT_HTML = Path("G:/내 드라이브/Kw/final_project/output/sim/dashboard_v4.html")
OUT_HTML.parent.mkdir(parents=True, exist_ok=True)

DAYS = ["2026-05-01", "2026-05-02", "2026-05-03"]
DAY_LABELS = ["Day 0 (금)", "Day 1 (토)", "Day 2 (일)"]


def load_jsonl(day):
    seen = {}
    with open(SIM_DIR / f"day_{day}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            try:
                j = json.loads(line)
                if j.get("status") == "ok": seen[j["aid"]] = j
            except: pass
    return list(seen.values())


def main():
    with open(DATA_JSON, encoding="utf-8") as f:
        d = json.load(f)

    # ─── Figure 1: 시뮬 전체 통계 (요약 카드용) ───
    summary = d["summary"]
    total_ok = sum(s["n"] for s in summary)
    total_err = 100  # 알려진 값
    err_rate = total_err / (total_ok + total_err) * 100

    # ─── Figure 2: factor 진화 (stacked bar) ───
    factors = ["distance","satisfaction","known","appointment","rumor","random","review"]
    colors = ["#5B8FF9","#5AD8A6","#5D7092","#F6BD16","#E8684A","#9270CA","#FF6B6B"]
    factor_data = {f: [] for f in factors}
    for day in DAYS:
        n = d["factor"][day]["n"]
        for f in factors:
            factor_data[f].append(d["factor"][day]["counter"].get(f, 0)*100/max(n,1))

    fig_factor = go.Figure()
    for f, c in zip(factors, colors):
        fig_factor.add_trace(go.Bar(name=f, x=DAY_LABELS, y=factor_data[f], marker_color=c))
    fig_factor.update_layout(barmode="stack", title="Day별 pick_factor 분포 진화",
                             yaxis_title="비율 (%)", height=400)

    # ─── Figure 3: dong 분포 (grouped bar) ───
    dong = d["dong"]
    home = [dong[day]['home']*100/dong[day]['total'] for day in DAYS]
    work = [dong[day]['work']*100/dong[day]['total'] for day in DAYS]
    other = [dong[day]['other']*100/dong[day]['total'] for day in DAYS]
    fig_dong = go.Figure(data=[
        go.Bar(name="거주 행정동", x=DAY_LABELS, y=home, marker_color="#5B8FF9",
               text=[f"{v:.1f}%" for v in home], textposition="outside"),
        go.Bar(name="직장 행정동", x=DAY_LABELS, y=work, marker_color="#F6BD16",
               text=[f"{v:.1f}%" for v in work], textposition="outside"),
        go.Bar(name="그 외 (광역·타동)", x=DAY_LABELS, y=other, marker_color="#5AD8A6",
               text=[f"{v:.1f}%" for v in other], textposition="outside"),
    ])
    fig_dong.update_layout(barmode="group", title="Day별 외출 행정동 분포",
                           yaxis_title="비율 (%)", height=400)

    # ─── Figure 4: Conversation intent 진화 ───
    intent_cats = ["약속", "추천", "기타"]
    intent_colors = ["#F6BD16", "#5AD8A6", "#5D7092"]
    intent_data = {c: [] for c in intent_cats}
    for day in DAYS:
        info = d["conversation"][day]
        if info.get('n', 0) == 0:
            for c in intent_cats: intent_data[c].append(0)
            continue
        intent_dist = info.get('intent_dist', {})
        for c in intent_cats: intent_data[c].append(intent_dist.get(c, 0))
    fig_conv = go.Figure()
    for c, col in zip(intent_cats, intent_colors):
        fig_conv.add_trace(go.Bar(name=c, x=DAY_LABELS, y=intent_data[c], marker_color=col,
                                   text=intent_data[c], textposition="outside"))
    fig_conv.update_layout(barmode="group", title="Night Phase intent 분류 (약속·추천 폭발 발현)",
                           yaxis_title="Conversation 수", height=400)

    # ─── Figure 5: 단계별 timing ───
    timing = d["timing"]
    stages = ["dawn", "s1", "s2", "write_plan", "night_finalize"]
    stage_colors = ["#5D7092", "#5B8FF9", "#F6BD16", "#5AD8A6", "#E8684A"]
    fig_timing = go.Figure()
    for stg, c in zip(stages, stage_colors):
        vals = [timing[day][stg] for day in DAYS]
        fig_timing.add_trace(go.Bar(name=f"t_{stg}", x=DAY_LABELS, y=vals, marker_color=c))
    fig_timing.update_layout(barmode="stack", title="Day별 단계별 elapsed (sec) — 병목 분석",
                              yaxis_title="시간 (sec)", height=400)

    # ─── Figure 6: review_lookup 발동률 + 카테고리 ───
    rl_pct = [s["review_lookup_pct"] for s in summary]
    fig_rl = go.Figure()
    fig_rl.add_trace(go.Bar(x=DAY_LABELS, y=rl_pct, marker_color="#FF6B6B",
                            text=[f"{v:.1f}%" for v in rl_pct], textposition="outside"))
    fig_rl.update_layout(title="Day별 review_lookup 발동률",
                         yaxis_title="발동 agent 비율 (%)", height=350)

    # ─── Figure 7: 소비 ───
    spending = d["spending"]
    totals_m = [spending[day]['total']/1e6 for day in DAYS]
    avgs = [spending[day]['avg']/1000 for day in DAYS]
    fig_spend = make_subplots(specs=[[{"secondary_y": True}]])
    fig_spend.add_trace(go.Bar(x=DAY_LABELS, y=totals_m, marker_color="#5B8FF9",
                               name="총 소비 (M원)",
                               text=[f"{v:.0f}M" for v in totals_m], textposition="outside"),
                        secondary_y=False)
    fig_spend.add_trace(go.Scatter(x=DAY_LABELS, y=avgs, mode="lines+markers",
                                   marker=dict(size=12, color="#E8684A"),
                                   line=dict(width=3), name="평균 거래액 (천원)"),
                        secondary_y=True)
    fig_spend.update_layout(title="Day별 소비 패턴", height=400)
    fig_spend.update_yaxes(title_text="총 소비 (백만원)", secondary_y=False)
    fig_spend.update_yaxes(title_text="평균 거래액 (천원)", secondary_y=True)

    # ─── Figure 8: relationship_score 진화 (산점도+라인) ───
    conv = d["conversation"]
    rel_avg = [conv[day].get("avg_rel", 0) for day in DAYS]
    ambient_pct = [conv[day]["ambient_n"]*100/max(conv[day]["n"],1) for day in DAYS]
    fig_rel = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rel.add_trace(go.Scatter(x=DAY_LABELS, y=rel_avg, mode="lines+markers",
                                  name="평균 relationship_score",
                                  marker=dict(size=14, color="#5AD8A6"),
                                  line=dict(width=3)),
                       secondary_y=False)
    fig_rel.add_trace(go.Scatter(x=DAY_LABELS, y=ambient_pct, mode="lines+markers",
                                  name="ambient 적용 비율 (%)",
                                  marker=dict(size=14, color="#E8684A"),
                                  line=dict(width=3, dash="dash")),
                       secondary_y=True)
    fig_rel.update_layout(title="사회적 관계 누적 (relationship_score ↑ + ambient ↓)",
                          height=400)
    fig_rel.update_yaxes(title_text="평균 relationship_score", secondary_y=False)
    fig_rel.update_yaxes(title_text="ambient %", secondary_y=True)

    # ─── Figure 9: v3 vs v4 비교 ───
    fig_compare = go.Figure()
    categories = ["에러율 (%)", "Conversation 총", "약속 발현", "추천 발현", "KNOWS_POI 누적"]
    v3_vals = [12.0, 463, 0, 1, 4]
    v4_vals = [0.44, 11827, 785, 4438, 67040]
    # 로그 스케일 비교 — 비율 표시
    ratios = [f"{v4/max(v3,0.01):.1f}x" if v3 > 0 else "신규" for v3, v4 in zip(v3_vals, v4_vals)]
    fig_compare = go.Figure(data=[
        go.Bar(name="v3 (EXAONE)", x=categories, y=v3_vals, marker_color="#888"),
        go.Bar(name="v4 (Midm + fix)", x=categories, y=v4_vals, marker_color="#5AD8A6",
               text=[f"{v:,}<br>({r})" for v,r in zip(v4_vals,ratios)], textposition="outside")
    ])
    fig_compare.update_layout(barmode="group", title="v3 vs v4 비교 — fix 효과",
                              yaxis_type="log", height=450)

    # ─── HTML 조립 ───
    cards_html = f"""
    <div style="display:flex; gap:1.5rem; flex-wrap:wrap; margin:1.5rem 0;">
        <div class="card"><div class="num">22,400</div><div class="lbl">총 처리 (3일)</div></div>
        <div class="card"><div class="num">{err_rate:.2f}%</div><div class="lbl">에러율</div></div>
        <div class="card"><div class="num">11,827</div><div class="lbl">Conversation</div></div>
        <div class="card"><div class="num">785</div><div class="lbl">약속 발현</div></div>
        <div class="card"><div class="num">4,438</div><div class="lbl">추천 발현</div></div>
        <div class="card"><div class="num">76.1%</div><div class="lbl">리뷰 활용 agent</div></div>
        <div class="card"><div class="num">55 hr</div><div class="lbl">총 소요</div></div>
    </div>
    """

    figs = [fig_compare, fig_factor, fig_dong, fig_conv, fig_rel, fig_rl, fig_spend, fig_timing]
    fig_htmls = "\n".join(f.to_html(include_plotlyjs="cdn" if i==0 else False, full_html=False, default_height=420)
                          for i, f in enumerate(figs))

    html = f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8"><title>v4 시뮬 대시보드</title>
<style>
body {{ font-family: -apple-system, "Malgun Gothic", sans-serif; margin: 0; padding: 2rem; background:#f5f7fa; color:#222; }}
h1 {{ font-size: 1.8rem; margin: 0 0 0.5rem; }}
h2 {{ font-size: 1.1rem; margin: 2.5rem 0 0.5rem; color:#5B8FF9; border-bottom:2px solid #5B8FF9; padding-bottom:.3rem; }}
.sub {{ color: #666; margin-bottom: 1rem; }}
.card {{ background:white; padding: 1.2rem 1.8rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.05); min-width:140px; }}
.num {{ font-size: 1.8rem; font-weight: bold; color:#5B8FF9; }}
.lbl {{ font-size: .85rem; color:#666; margin-top: .3rem; }}
.section {{ background:white; padding:1.5rem; border-radius:12px; box-shadow:0 2px 8px rgba(0,0,0,.05); margin-bottom:1.5rem; }}
.footer {{ text-align:center; color:#999; margin-top:3rem; font-size:.85rem; }}
</style></head><body>
<h1>📊 7,500 Agent × 3일 시뮬 v4 대시보드</h1>
<div class="sub">Model: Midm-2.0-Base-Instruct (BF16) · 2026-05-01~03 (금/토/일) · 모든 fix 적용</div>

{cards_html}

<div class="section"><h2>1. v3 vs v4 — fix 효과 (로그 스케일)</h2>{fig_htmls.split("</div>")[0]}</div></div>
<div class="section"><h2>2. 의사결정 factor 진화</h2>{"</div>".join(fig_htmls.split("</div>")[1:2])}</div></div>
<div class="section"><h2>3. 외출 행정동 분포</h2>{"</div>".join(fig_htmls.split("</div>")[2:3])}</div></div>
<div class="section"><h2>4. Night Phase — 사회적 상호작용</h2>{"</div>".join(fig_htmls.split("</div>")[3:4])}</div></div>
<div class="section"><h2>5. 관계 누적 — relationship_score ↑ + ambient ↓</h2>{"</div>".join(fig_htmls.split("</div>")[4:5])}</div></div>
<div class="section"><h2>6. 별점·리뷰 lookup 발동률</h2>{"</div>".join(fig_htmls.split("</div>")[5:6])}</div></div>
<div class="section"><h2>7. 소비 패턴</h2>{"</div>".join(fig_htmls.split("</div>")[6:7])}</div></div>
<div class="section"><h2>8. 단계별 병목 (timing)</h2>{"</div>".join(fig_htmls.split("</div>")[7:])}</div>

<div class="footer">v4 시뮬 (Midm + 모든 fix) — 55시간 처리, 에러 0.44%, KNOWS·workplace 데이터 fix + Stage2 strict satisfaction</div>
</body></html>
"""
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"\n=== Dashboard: {OUT_HTML} ===")
    print(f"  size: {OUT_HTML.stat().st_size/1024:.0f} KB")
    print(f"  브라우저로 직접 열기 (인터넷 연결 시 plotly CDN 로드)")


if __name__ == "__main__":
    main()
