#!/usr/bin/env python3
"""Generate a standalone test HTML to verify the redesign — no Neo4j needed."""
import sys, os, re, base64
from datetime import date, datetime, timedelta
from pathlib import Path
import html as _html

# ── Read generate_final_report.py and extract HTML_STYLE + build_html ──
src = Path(__file__).parent / "generate_final_report.py"
code = src.read_text(encoding="utf-8")

# Extract HTML_STYLE string (between triple quotes)
m = re.search(r'HTML_STYLE\s*=\s*"""(.*?)"""', code, re.DOTALL)
if not m:
    print("ERROR: Could not find HTML_STYLE"); sys.exit(1)
HTML_STYLE = m.group(1)

# ── Helpers ──
def _h(t):
    return _html.escape(str(t) if t is not None else "")

def _img_data_uri(path):
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _figure_dual(path_dark, path_light, caption):
    dark_exists = path_dark and path_dark.exists()
    light_exists = path_light and path_light.exists()
    if not dark_exists and not light_exists:
        return ""
    dark_uri = _img_data_uri(path_dark) if dark_exists else ""
    light_uri = _img_data_uri(path_light) if light_exists else dark_uri
    if not dark_uri:
        dark_uri = light_uri
    return f"""<figure>
  <img class="chart-dark" src="{dark_uri}" alt="{_h(caption)}"/>
  <img class="chart-light" src="{light_uri}" alt="{_h(caption)}"/>
  <figcaption>{_h(caption)}</figcaption>
</figure>"""

# ── Dummy data ──
s1 = {
    "기간": "2026-05-01 ~ 2026-05-07",
    "일수": 7,
    "정책_시행일": "2026-05-02",
    "Agent 수": 1247,
    "Plan 수": 8729,
    "INCLUDES 엣지": 43645,
    "State 수": 8729,
    "Memory(visited)": 12380,
    "Memory(rumor)": 3456,
    "Conversation 약속": 892,
    "Conversation 추천": 1234,
    "Conversation 이슈": 567,
    "Conversation 기타": 345,
}
n_agent = s1["Agent 수"]
n_plan = s1["Plan 수"]
n_conv = s1.get("Conversation 약속",0)+s1.get("Conversation 추천",0)+s1.get("Conversation 이슈",0)+s1.get("Conversation 기타",0)

s2_data = {"summary": {
    "before_gn_daily": 4523000, "after_gn_daily": 5891000,
    "before_ng_daily": 12340000, "after_ng_daily": 13120000,
    "gangnam_change_pct": 30.25, "non_gangnam_change_pct": 6.32,
    "DID_pct_points": 23.93,
}, "daily": []}
s2_figs = {
    "per_capita": {"dark": "fig2a_per_capita_dark.png", "light": "fig2a_per_capita_light.png"},
    "change_rate": {"dark": "fig2b_change_rate_dark.png", "light": "fig2b_change_rate_light.png"},
    "did": {"dark": "fig2c_did_dark.png", "light": "fig2c_did_light.png"}
}
sm = s2_data["summary"]
did_str = f"{sm.get('DID_pct_points',0):+}%p" if sm else "—"
policy_from = "2026-05-02"

s41_data = {
    "distribution": {"appointment": 2345, "rumor": 1890, "policy": 1456,
                     "habit": 3210, "top_category": 987, "mood": 654, "none": 321},
    "distribution_pct": {"appointment": 21.5, "rumor": 17.3, "policy": 13.4,
                         "habit": 29.5, "top_category": 9.1, "mood": 6.0, "none": 2.9},
}
s42_data = {
    "frequency": {"신규 (1회 방문)": 5432, "재방문 (2~4회)": 3210, "단골 (5회+)": 1098},
    "total": 9740
}
s43_data = {
    "by_trigger": [
        {"trigger": "appointment", "avg_sat": 0.823, "n": 2345},
        {"trigger": "policy", "avg_sat": 0.791, "n": 1456},
        {"trigger": "rumor", "avg_sat": 0.756, "n": 1890},
        {"trigger": "habit", "avg_sat": 0.712, "n": 3210},
        {"trigger": "mood", "avg_sat": 0.689, "n": 654},
        {"trigger": "top_category", "avg_sat": 0.654, "n": 987},
    ],
}

labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책",
             "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션", "none": "기타"}
label_kr_intv = {"positive": "정책 적극 활용 + 만족도 ↑",
                 "negative": "정책 무관심 + 만족도 ↓",
                 "neutral":  "정책 무관심 + 만족도 보통"}

# ── Build table rows ──
cond_rows = "".join(
    f"<tr><td>{_h(k)}</td><td class='num'>{v:,}</td></tr>"
    if isinstance(v, int) else
    f"<tr><td>{_h(k)}</td><td>{_h(v)}</td></tr>"
    for k, v in s1.items()
)
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
    f"강남 변화율 - 비강남 변화율 = "
    f"<strong style='font-size:18px;color:var(--cyan);'>"
    f"{sm['DID_pct_points']:+}%p</strong></div>"
)

trigger_rows = ""
for k, v in sorted(s41_data["distribution"].items(), key=lambda x: -x[1]):
    pct = s41_data["distribution_pct"].get(k, 0)
    trigger_rows += (f"<tr><td>{_h(labels_kr.get(k, k))}</td>"
                     f"<td class='num'>{v:,}</td>"
                     f"<td class='num'>{pct}%</td></tr>")

regular_rows = "".join(
    f"<tr><td>{_h(k)}</td><td class='num'>{v:,}</td></tr>"
    for k, v in s42_data["frequency"].items()
)
sat_rows = "".join(
    f"<tr><td>{_h(labels_kr.get(r['trigger'], r['trigger']))}</td>"
    f"<td class='num'>{r['avg_sat']}</td>"
    f"<td class='num'>{r['n']:,}</td></tr>"
    for r in s43_data["by_trigger"]
)

# ── TOC ──
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

# ── Interview cards ──
s5 = {
    "positive": {
        "agent_id": "agent_0042",
        "persona": {"age":"30대","gender":"여성","job":"마케터","lifestyle":"트렌디한 카페 탐방, SNS 맛집 리뷰","income":"중상","home_dong":"역삼1동"},
        "qa": [
            {"q": "정책에 대해 어떻게 느끼셨나요?", "a": "정말 좋았어요! 평소 자주 가던 역삼역 근처 카페에서 30% 할인을 받으니 부담 없이 더 자주 가게 되었습니다."},
            {"q": "가장 자주 가신 가게는?", "a": "역삼역 2번 출구 근처 '블루보틀'이요. 원래 단골이었는데 바우처 덕분에 거의 매일 갔어요."},
            {"q": "친구가 추천한 곳에 가신 적 있나요?", "a": "네! 직장 동료가 선릉역 근처 새로 생긴 디저트 카페를 추천해줘서 갔는데, 바우처도 되고 맛도 좋아서 단골이 됐습니다."},
        ]
    },
    "negative": {
        "agent_id": "agent_0891",
        "persona": {"age":"50대","gender":"남성","job":"자영업자","lifestyle":"집 근처 위주 활동, 익숙한 곳 선호","income":"중","home_dong":"도봉1동"},
        "qa": [
            {"q": "정책에 대해 어떻게 느끼셨나요?", "a": "솔직히 저랑은 관련이 없었습니다. 강남까지 카페 마시러 갈 이유가 없어요."},
            {"q": "가장 자주 가신 가게는?", "a": "집 앞 편의점이랑 동네 분식집이요. 가깝고 편하니까요."},
        ]
    },
    "neutral": {
        "agent_id": "agent_0567",
        "persona": {"age":"20대","gender":"남성","job":"대학원생","lifestyle":"학교와 연구실 중심 생활","income":"하","home_dong":"신림동"},
        "qa": [
            {"q": "정책에 대해 어떻게 느끼셨나요?", "a": "존재는 알고 있었는데 적극적으로 활용하진 않았어요. 강남에 갈 일이 가끔 있긴 한데, 바우처를 쓸 만큼 자주 가지는 않았습니다."},
            {"q": "이번 주 가장 만족스러웠던 외출은?", "a": "연구실 동기들과 함께 학교 앞 삼겹살집에서 저녁 먹은 게 제일 좋았어요."},
        ]
    }
}

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
    initial_bubbles = ""
    for qa in d["qa"]:
        initial_bubbles += f"""
        <div class="chat-bubble user">
          <div class="meta"><span class="sender">인터뷰어</span></div>
          <div class="text">{_h(qa["q"])}</div>
        </div>
        <div class="chat-bubble agent">
          <div class="meta"><span class="sender">에이전트 ({_h(d['agent_id'])})</span></div>
          <div class="text">{_h(qa["a"])}</div>
        </div>
        """

    interview_html += f"""
    <div class="interview-card {label}" data-agent-id="{_h(d['agent_id'])}">
      <span class="label-tag">{_h(label)}</span>
      <h3 style="margin:0 0 12px 0;color:var(--text-bright)">
        {_h(label_kr_intv[label])} <span style="font-weight:400;font-size:13px;color:var(--text-muted)">— <code>{_h(d['agent_id'])}</code></span>
      </h3>
      <div class="persona">
        <strong>페르소나:</strong> {_h(p.get('age'))} {_h(p.get('gender'))}
        · {_h(p.get('job'))} · {_h(p.get('home_dong'))} 거주 · 소득 {_h(p.get('income'))}<br/>
        <strong>라이프스타일:</strong> {_h(p.get('lifestyle'))}
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
          <input type="text" class="chat-input" id="chat-input-{label}" placeholder="{_h(label_kr_intv[label])} 에이전트에게 질문을 입력해보세요..." />
          <button class="chat-send-btn" id="chat-send-{label}" onclick="sendChatMessage('{label}')">
            <svg viewBox="0 0 24 24">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/>
            </svg>
          </button>
        </div>
      </div>
    </div>
    """

def generate_dummy_charts(chart_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    installed = {f.name for f in font_manager.fontManager.ttflist}
    candidates = ["Malgun Gothic", "AppleGothic",
                  "Noto Sans CJK KR", "Noto Sans KR",
                  "NanumGothic", "Nanum Gothic",
                  "Noto Sans CJK SC", "Noto Sans CJK JP",
                  "DejaVu Sans"]
    chosen = next((f for f in candidates if f in installed), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False

    def _apply_premium_theme(fig, axs, is_light=False):
        if is_light:
            bg_color = "#ffffff"
            text_color = "#334155"
            grid_color = "#e2e8f0"
        else:
            bg_color = "#1e293b" # Matches var(--bg-card)
            text_color = "#f1f5f9" # Matches var(--text-bright)
            grid_color = "#334155" # Matches var(--border-glass)
        
        fig.patch.set_facecolor(bg_color)
        
        if hasattr(axs, "flat"):
            ax_list = list(axs.flat)
        elif isinstance(axs, (list, tuple)):
            ax_list = list(axs)
        else:
            ax_list = [axs]
            
        for ax in ax_list:
            ax.set_facecolor(bg_color)
            
            # Hide top/right spines
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["bottom", "left"]:
                ax.spines[spine].set_color(grid_color)
                ax.spines[spine].set_linewidth(1.2)
                
            ax.tick_params(colors=text_color, which='both', labelsize=11, length=4, width=1.2)
            ax.xaxis.label.set_color(text_color)
            ax.xaxis.label.set_size(12)
            ax.yaxis.label.set_color(text_color)
            ax.yaxis.label.set_size(12)
            ax.title.set_color(text_color)
            ax.title.set_size(14)
            ax.title.set_weight("bold")
            
            # Grid
            ax.grid(True, color=grid_color, linestyle=':', linewidth=0.8, alpha=0.7)
            
            # Legend styling if exists
            legend = ax.get_legend()
            if legend:
                legend.get_frame().set_facecolor(bg_color)
                legend.get_frame().set_edgecolor(grid_color)
                legend.get_frame().set_linewidth(1)
                for text in legend.get_texts():
                    text.set_color(text_color)
                    text.set_size(10)

    for is_light in [False, True]:
        suffix = "_light" if is_light else "_dark"
        cyan = "#0284c7" if is_light else "#06b6d4"
        purple = "#4f46e5" if is_light else "#8b5cf6"
        emerald = "#059669" if is_light else "#10b981"
        rose = "#e11d48" if is_light else "#f43f5e"
        txt_color = "#334155" if is_light else "#f1f5f9"
        border_clr = "#e2e8f0" if is_light else "#334155"
        bg_pie = "#ffffff" if is_light else "#1e293b"
        lbl_ann_clr = "#475569" if is_light else "#94a3b8"

        GN_POP = 1247
        NG_POP = 3500
        daily = [
            {"day": "2026-05-01", "phase": "before", "gangnam_spend": 4500000, "non_gangnam_spend": 12000000},
            {"day": "2026-05-02", "phase": "after", "gangnam_spend": 5200000, "non_gangnam_spend": 12500000},
            {"day": "2026-05-03", "phase": "after", "gangnam_spend": 5500000, "non_gangnam_spend": 12800000},
            {"day": "2026-05-04", "phase": "after", "gangnam_spend": 5800000, "non_gangnam_spend": 13100000},
            {"day": "2026-05-05", "phase": "after", "gangnam_spend": 6000000, "non_gangnam_spend": 13300000},
            {"day": "2026-05-06", "phase": "after", "gangnam_spend": 6100000, "non_gangnam_spend": 13500000},
            {"day": "2026-05-07", "phase": "after", "gangnam_spend": 6200000, "non_gangnam_spend": 13700000},
        ]
        xs = list(range(len(daily)))
        labels = [x["day"][5:] for x in daily]
        cut_idx = 1
        policy_from = "2026-05-02"

        # ── (A) 1인당 매출 ──
        gn_per = [x["gangnam_spend"] / GN_POP for x in daily]
        ng_per = [x["non_gangnam_spend"] / NG_POP for x in daily]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(xs, gn_per, marker="o", markersize=8, label=f"강남 (n={GN_POP:,})", color=cyan, linewidth=2.5)
        ax.plot(xs, ng_per, marker="s", markersize=8, label=f"비강남 (n={NG_POP:,})", color=purple, linewidth=2.5)
        ax.axvline(cut_idx - 0.5, color="#64748b", linestyle="--", linewidth=1.5)
        ax.text(cut_idx - 0.45, max(gn_per + ng_per) * 0.95, f" 정책 시행 {policy_from}", color=lbl_ann_clr, fontsize=11)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("날짜"); ax.set_ylabel("1인당 매출 (원)")
        ax.set_title("(A) 1인당 일별 매출 — 인구 비례 환산 후 절대 비교", pad=12)
        ax.legend(loc="upper left")
        ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
        _apply_premium_theme(fig, ax, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig2a_per_capita{suffix}.png", dpi=140, bbox_inches="tight")
        plt.close()

        # ── (B) baseline 대비 변화율 ──
        gn_base = max(daily[0]["gangnam_spend"], 1)
        ng_base = max(daily[0]["non_gangnam_spend"], 1)
        gn_chg = [(x["gangnam_spend"] - gn_base) / gn_base * 100 for x in daily]
        ng_chg = [(x["non_gangnam_spend"] - ng_base) / ng_base * 100 for x in daily]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        ax.plot(xs, gn_chg, marker="o", markersize=8, label="강남구", color=cyan, linewidth=2.5)
        ax.plot(xs, ng_chg, marker="s", markersize=8, label="비강남", color=purple, linewidth=2.5)
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.axvline(cut_idx - 0.5, color="#64748b", linestyle="--", linewidth=1.5)
        top_y = max(max(gn_chg), max(ng_chg))
        ax.text(cut_idx - 0.45, top_y * 0.95 if top_y > 0 else 3, f" 정책 시행 {policy_from}", color=lbl_ann_clr, fontsize=11)
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("날짜"); ax.set_ylabel(f"{daily[0]['day'][5:]} 대비 변화율 (%)")
        ax.set_title("(B) baseline 대비 매출 변화율", pad=12)
        ax.legend(loc="upper left")
        _apply_premium_theme(fig, ax, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig2b_change_rate{suffix}.png", dpi=140, bbox_inches="tight")
        plt.close()

        # ── (C) DID ──
        did = [g - n for g, n in zip(gn_chg, ng_chg)]
        fig, ax = plt.subplots(figsize=(11, 5.5))
        colors_list = ['#64748b' if i == 0 else (emerald if v >= 0 else rose) for i, v in enumerate(did)]
        bars = ax.bar(xs, did, color=colors_list, edgecolor=border_clr, linewidth=1, width=0.6)
        ax.axhline(0, color="#64748b", linewidth=1)
        ax.axvline(cut_idx - 0.5, color="#64748b", linestyle="--", linewidth=1.5)
        for i, (bar, v) in enumerate(zip(bars, did)):
            if i == 0: continue
            offset = max(abs(min(did)), abs(max(did))) * 0.04
            ax.text(i, v + (offset if v >= 0 else -offset), f"{v:+.1f}%p", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=11,
                    color=emerald if v >= 0 else rose, fontweight='bold')
        ax.set_xticks(xs); ax.set_xticklabels(labels)
        ax.set_xlabel("날짜"); ax.set_ylabel("DID (%p)")
        ax.set_title("(C) DID — 강남 변화율 - 비강남 변화율 (정책 순효과)", pad=12)
        _apply_premium_theme(fig, ax, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig2c_did{suffix}.png", dpi=140, bbox_inches="tight")
        plt.close()

        # ── Section 3: Spillover ──
        by_group = {
            "강남": [4.5e6, 5.2e6, 5.5e6, 5.8e6, 6.0e6, 6.1e6, 6.2e6],
            "서초": [3.8e6, 4.0e6, 4.2e6, 4.3e6, 4.4e6, 4.5e6, 4.6e6],
            "송파": [3.5e6, 3.6e6, 3.7e6, 3.8e6, 3.9e6, 4.0e6, 4.1e6],
            "강북": [1.2e6, 1.2e6, 1.3e6, 1.3e6, 1.3e6, 1.4e6, 1.4e6]
        }
        fig, ax = plt.subplots(figsize=(11, 5))
        xs_spill = list(range(7))
        colors_spill = {"강남": cyan, "서초": purple, "송파": emerald, "강북": "#64748b"}
        for name, vals in by_group.items():
            ax.plot(xs_spill, [v / 1e6 for v in vals], marker="o", label=name, color=colors_spill[name], linewidth=2.5, markersize=6)
        ax.set_xticks(xs_spill)
        ax.set_xticklabels(labels)
        ax.set_xlabel("날짜")
        ax.set_ylabel("매출 합계 (백만원)")
        ax.set_title("자치구별 매출 추이 — 강남 정책의 간접 영향 추적")
        ax.legend(loc="upper left")
        _apply_premium_theme(fig, ax, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig3_spillover{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # ── Section 4-1: Triggers ──
        dist = {"habit": 3210, "appointment": 2345, "rumor": 1890, "policy": 1456, "top_category": 987, "mood": 654, "none": 321}
        total = sum(dist.values()) or 1
        fig, ax = plt.subplots(figsize=(9, 5))
        labels_kr = {"appointment": "약속", "rumor": "소문", "policy": "정책", "habit": "습관", "top_category": "Top 카테고리", "mood": "컨디션", "none": "기타"}
        sorted_items = sorted(dist.items(), key=lambda x: -x[1])
        xs_trig = [labels_kr.get(k, k) for k, _ in sorted_items]
        ys_trig = [v for _, v in sorted_items]
        
        trig_colors = [purple, cyan, emerald, rose, "#d97706" if is_light else "#f59e0b", "#6366f1" if is_light else "#a78bfa", "#475569" if is_light else "#64748b"]
        bars = ax.bar(xs_trig, ys_trig, color=trig_colors[:len(xs_trig)], edgecolor=border_clr, linewidth=1, width=0.6)
        for bar, n in zip(bars, ys_trig):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.01, f"{n:,}\n({n/total*100:.1f}%)", ha="center", va="bottom", fontsize=10, color=txt_color)
        ax.set_title("외출 이벤트의 결정 동기 (trigger 분포)")
        ax.set_ylabel("이벤트 수")
        _apply_premium_theme(fig, ax, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig4_1_triggers{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # ── Section 4-2: regulars ──
        dist_reg = {"신규 (1회 방문)": 5432, "재방문 (2~4회)": 3210, "단골 (5회+)": 1098}
        src_dist = {"initial": 4000, "visit": 3500, "rumor": 2240}
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.8))
        axs[0].bar(dist_reg.keys(), dist_reg.values(), color=[purple, cyan, emerald], edgecolor=border_clr, linewidth=1, width=0.55)
        axs[0].set_title("방문 빈도 분포 (KNOWS_POI)")
        axs[0].set_ylabel("관계 수")
        for i, (k, v) in enumerate(dist_reg.items()):
            axs[0].text(i, v + 100, f"{v:,}", ha="center", va="bottom", fontsize=11, color=txt_color)
            
        wedges, texts, autotexts = axs[1].pie(src_dist.values(), labels=src_dist.keys(), autopct="%1.1f%%",
                                              colors=[cyan, purple, emerald, "#6366f1" if is_light else "#a78bfa"][:len(src_dist)],
                                              wedgeprops=dict(edgecolor=bg_pie, linewidth=2),
                                              textprops=dict(color=txt_color, fontsize=11))
        axs[1].set_title("KNOWS_POI 출처 (왜 알게 됐나)")
        _apply_premium_theme(fig, axs, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig4_2_regulars{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close()

        # ── Section 4-3: satisfaction ──
        by_trigger = [
            {"trigger": "appointment", "avg_sat": 0.823, "n": 2345},
            {"trigger": "policy", "avg_sat": 0.791, "n": 1456},
            {"trigger": "rumor", "avg_sat": 0.756, "n": 1890},
            {"trigger": "habit", "avg_sat": 0.712, "n": 3210},
            {"trigger": "mood", "avg_sat": 0.689, "n": 654},
            {"trigger": "top_category", "avg_sat": 0.654, "n": 987},
        ]
        by_cat = [
            {"cat": "한식", "avg_sat": 0.85, "n": 1200},
            {"cat": "카페", "avg_sat": 0.82, "n": 2500},
            {"cat": "양식", "avg_sat": 0.80, "n": 900},
            {"cat": "디저트", "avg_sat": 0.79, "n": 1500},
            {"cat": "일식", "avg_sat": 0.77, "n": 800},
            {"cat": "중식", "avg_sat": 0.75, "n": 600},
        ]
        fig, axs = plt.subplots(1, 2, figsize=(13, 5))
        xs_sat = [labels_kr.get(r["trigger"], r["trigger"]) for r in by_trigger]
        ys_sat = [round(r["avg_sat"], 3) for r in by_trigger]
        axs[0].barh(xs_sat, ys_sat, color=cyan, edgecolor=border_clr, linewidth=1, height=0.55)
        axs[0].set_xlabel("평균 만족도")
        axs[0].set_title("결정 동기별 만족도")
        axs[0].set_xlim(0, 1)
        for i, (lbl, v, r) in enumerate(zip(xs_sat, ys_sat, by_trigger)):
            axs[0].text(v + 0.02, i, f"{v:.3f} (n={r['n']:,})", va="center", fontsize=10, color=txt_color)
        
        xs_cat = [r["cat"] for r in by_cat]
        ys_cat = [round(r["avg_sat"], 3) for r in by_cat]
        axs[1].barh(xs_cat[::-1], ys_cat[::-1], color=purple, edgecolor=border_clr, linewidth=1, height=0.55)
        axs[1].set_xlabel("평균 만족도")
        axs[1].set_title("카테고리별 만족도 Top")
        axs[1].set_xlim(0, 1)
        for i, (lbl, v, r) in enumerate(zip(xs_cat[::-1], ys_cat[::-1], by_cat[::-1])):
             axs[1].text(v + 0.02, i, f"{v:.3f}", va="center", fontsize=10, color=txt_color)
             
        _apply_premium_theme(fig, axs, is_light)
        plt.tight_layout()
        plt.savefig(chart_dir / f"fig4_3_satisfaction{suffix}.png", dpi=120, bbox_inches="tight")
        plt.close()

# ── Chart dir setup ──
chart_dir = Path(__file__).parent / "_test_charts"
chart_dir.mkdir(exist_ok=True)
s3_figs = {"dark": "fig3_spillover_dark.png", "light": "fig3_spillover_light.png"}
s41_figs = {"dark": "fig4_1_triggers_dark.png", "light": "fig4_1_triggers_light.png"}
s42_figs = {"dark": "fig4_2_regulars_dark.png", "light": "fig4_2_regulars_light.png"}
s43_figs = {"dark": "fig4_3_satisfaction_dark.png", "light": "fig4_3_satisfaction_light.png"}

generate_dummy_charts(chart_dir)

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
        var link = document.querySelector('.sidebar nav a[href="#' + e.target.id + '"]');
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
    const timeoutId = setTimeout(() => controller.abort(), 1500); // 1.5s timeout for fast test report response
    
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

# ── Build final HTML ──
html = f"""<!DOCTYPE html>
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
        시행된 <code>{_h(policy_from)}</code> 시점을 기준으로
        정책 대상 카테고리(식사·카페·디저트)의 일별 매출을 비교합니다.
        세 관점 — (A) 인구 비례 1인당 매출, (B) baseline 대비 변화율, (C) DID — 으로 분석합니다.</p>
      {_figure_dual(chart_dir / s2_figs['per_capita']['dark'], chart_dir / s2_figs['per_capita']['light'],
               '(A) 1인당 일별 매출 — 인구 비례 환산 후 강남 vs 비강남 절대 비교') if isinstance(s2_figs, dict) else _figure_dual(chart_dir / s2_figs, chart_dir / s2_figs, '정책 전/후 분석 생략')}
      {_figure_dual(chart_dir / s2_figs['change_rate']['dark'], chart_dir / s2_figs['change_rate']['light'], '(B) baseline 대비 매출 변화율 — 같은 0% 출발점에서 패턴 비교') if isinstance(s2_figs, dict) else ''}
      {_figure_dual(chart_dir / s2_figs['did']['dark'], chart_dir / s2_figs['did']['light'], '(C) DID — 강남 변화율 - 비강남 변화율 (정책 순효과)') if isinstance(s2_figs, dict) else ''}
      <h3>평균 일간 매출 비교</h3>
      <table>
        <thead><tr><th>자치구</th><th class="num">시행 전</th><th class="num">시행 후</th><th class="num">변화율</th></tr></thead>
        <tbody>{sales_rows}</tbody>
      </table>
      {did_callout}
    </section>

    <section id="s-spillover" class="reveal">
      <h2>3. 간접 영향 (Spillover)</h2>
      <p>강남 정책이 직접 적용되지 않은 인접 자치구(서초·송파)와 멀리 떨어진 강북에 미친 영향을 비교합니다.
        인접 자치구의 매출 변화가 강북보다 두드러지면 spillover로 해석할 수 있습니다.</p>
      {_figure_dual(chart_dir / s3_figs['dark'], chart_dir / s3_figs['light'], '자치구별 매출 추이 — 강남 정책의 간접 영향 추적')}
    </section>

    <section id="s-behavior" class="reveal">
      <h2>4. 소비자 행동 분석</h2>

      <h3 id="s-trigger">4-1. 결정 동기 분포</h3>
      <p>외출(집·직장 제외) 시 어떤 요인이 결정을 이끌었는지 LLM이 trigger 라벨로 분류한 분포입니다.</p>
      {_figure_dual(chart_dir / s41_figs['dark'], chart_dir / s41_figs['light'], '외출 이벤트의 결정 동기 (trigger) 분포')}
      <table>
        <thead><tr><th>동기</th><th class="num">건수</th><th class="num">비율</th></tr></thead>
        <tbody>{trigger_rows}</tbody>
      </table>

      <h3 id="s-regular">4-2. 단골 vs 신규</h3>
      <p>각 에이전트의 POI 인지 관계(KNOWS_POI) 빈도 분포와, 그 관계가 어떻게 형성되었는지(출처) 분석합니다.</p>
      {_figure_dual(chart_dir / s42_figs['dark'], chart_dir / s42_figs['light'], '방문 빈도 분포 + KNOWS_POI 출처')}
      <table>
        <thead><tr><th>구분</th><th class="num">관계 수</th></tr></thead>
        <tbody>{regular_rows}</tbody>
      </table>
      <p style="color:var(--text-muted);font-size:13px">전체 KNOWS_POI 관계 수: <strong style="color:var(--cyan)">{s42_data['total']:,}</strong></p>

      <h3 id="s-satisfaction">4-3. 만족도 — 어떤 동기가 더 만족스러웠나</h3>
      <p>결정 동기(trigger)별 평균 만족도를 비교해 어떤 동기로 외출했을 때 가장 만족도가 높은지 측정합니다.</p>
      {_figure_dual(chart_dir / s43_figs['dark'], chart_dir / s43_figs['light'], '결정 동기별 + 카테고리별 평균 만족도')}
      <table>
        <thead><tr><th>동기</th><th class="num">평균 만족도</th><th class="num">표본 수</th></tr></thead>
        <tbody>{sat_rows}</tbody>
      </table>
    </section>

    <section id="s-interview" class="reveal">
      <h2>5. 1대1 인터뷰 — 페르소나별 대표</h2>
      <p>정책 사용액과 만족도(mood)를 기준으로 세 유형의 군집을 정의하고, 각 군집에서 대표 1명씩 자동
        추출하여 LLM이 그 페르소나로 인터뷰에 응답합니다.</p>
      {interview_html}
    </section>

    <section id="s-appendix" class="reveal">
      <h2>부록</h2>
      <ul>
        <li>본 보고서는 <code>scripts/sim/generate_final_report.py</code>로 자동 생성되었습니다.</li>
        <li>시뮬 원본 데이터는 Neo4j에 보존됩니다 (Plan / State / Memory / Conversation 노드).</li>
        <li>인터랙티브 시각화: <code>output/sim/visualization/sim_standalone.html</code></li>
        <li>인터뷰 LLM 모듈: <code>scripts/sim/interview_agent.py</code></li>
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

out = Path(__file__).parent / "_test_report.html"
out.write_text(html, encoding="utf-8")
print(f"✅ Test report: {out}")
print(f"   Size: {out.stat().st_size / 1024:.0f} KB")
print(f"   Open in browser: file://{out.resolve()}")
