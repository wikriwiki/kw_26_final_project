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

def _figure(path, caption):
    if not path or not path.exists():
        return ""
    return (f'<figure><img src="{_img_data_uri(path)}" alt="{_h(caption)}"/>'
            f'<figcaption>{_h(caption)}</figcaption></figure>')

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
s2_figs = "fig2_dummy.png"
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
    f"강남 변화율 − 비강남 변화율 = "
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
    qa_html = "".join(
        f'<div class="qa"><div class="q">{_h(qa["q"])}</div>'
        f'<div class="a">{_h(qa["a"])}</div></div>'
        for qa in d["qa"]
    )
    interview_html += f"""
    <div class="interview-card {label}">
      <span class="label-tag">{_h(label)}</span>
      <h3 style="margin:0 0 12px 0;color:var(--text-bright)">
        {_h(label_kr_intv[label])} <span style="font-weight:400;font-size:13px;color:var(--text-muted)">— <code>{_h(d['agent_id'])}</code></span>
      </h3>
      <div class="persona">
        <strong>페르소나:</strong> {_h(p.get('age'))} {_h(p.get('gender'))}
        · {_h(p.get('job'))} · {_h(p.get('home_dong'))} 거주 · 소득 {_h(p.get('income'))}<br/>
        <strong>라이프스타일:</strong> {_h(p.get('lifestyle'))}
      </div>
      {qa_html}
    </div>
    """

# ── Chart dir (no actual charts — figures will just be empty) ──
chart_dir = Path(__file__).parent / "_test_charts"
chart_dir.mkdir(exist_ok=True)
s3_fig = "fig3_dummy.png"
s41_fig = "fig41_dummy.png"
s42_fig = "fig42_dummy.png"
s43_fig = "fig43_dummy.png"

# JavaScript
js_code = """
document.addEventListener('DOMContentLoaded', function() {
  var obs = new IntersectionObserver(function(entries) {
    entries.forEach(function(e) {
      if (e.isIntersecting) e.target.classList.add('visible');
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach(function(el) { obs.observe(el); });

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
});
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
      <p style="color:var(--text-muted);font-style:italic">(차트는 실제 시뮬레이션 데이터로 생성됩니다)</p>
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
      <p style="color:var(--text-muted);font-style:italic">(차트는 실제 시뮬레이션 데이터로 생성됩니다)</p>
    </section>

    <section id="s-behavior" class="reveal">
      <h2>4. 소비자 행동 분석</h2>

      <h3 id="s-trigger">4-1. 결정 동기 분포</h3>
      <p>외출(집·직장 제외) 시 어떤 요인이 결정을 이끌었는지 LLM이 trigger 라벨로 분류한 분포입니다.</p>
      <table>
        <thead><tr><th>동기</th><th class="num">건수</th><th class="num">비율</th></tr></thead>
        <tbody>{trigger_rows}</tbody>
      </table>

      <h3 id="s-regular">4-2. 단골 vs 신규</h3>
      <p>각 에이전트의 POI 인지 관계(KNOWS_POI) 빈도 분포와, 그 관계가 어떻게 형성되었는지(출처) 분석합니다.</p>
      <table>
        <thead><tr><th>구분</th><th class="num">관계 수</th></tr></thead>
        <tbody>{regular_rows}</tbody>
      </table>
      <p style="color:var(--text-muted);font-size:13px">전체 KNOWS_POI 관계 수: <strong style="color:var(--cyan)">{s42_data['total']:,}</strong></p>

      <h3 id="s-satisfaction">4-3. 만족도 — 어떤 동기가 더 만족스러웠나</h3>
      <p>결정 동기(trigger)별 평균 만족도를 비교해 어떤 동기로 외출했을 때 가장 만족도가 높은지 측정합니다.</p>
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
