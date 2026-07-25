"""build_html.py -- fixed-design, self-contained HTML report builder.

The visual design is intentionally kept in report_template.css so every run
uses the same layout, spacing, colors, cards, tables, and interview treatment.
"""
from __future__ import annotations

import base64
import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_CSS_PATH = _HERE / "report_template.css"


def _h(value: Any) -> str:
    return _html.escape("" if value is None else str(value))


def _img_data_uri(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _report_css() -> str:
    return _CSS_PATH.read_text(encoding="utf-8")


def _report_js() -> str:
    return """
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
  }, { rootMargin: '-20% 0px -70% 0px' });
  sections.forEach(function(s) { spy.observe(s); });

  var toggleBtn = document.getElementById('theme-toggle');
  if (!toggleBtn) return;
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

  updateTheme((localStorage.getItem('theme') || 'dark') === 'light');
});
"""


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table(rows: list[dict] | None, *, class_name: str = "") -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(
        f'<th class="{"num" if all(_is_num(r.get(c)) for r in rows) else ""}">{_h(c)}</th>'
        for c in cols
    )
    body_rows = []
    for row in rows:
        cells = []
        for col in cols:
            value = row.get(col, "")
            cls = ' class="num"' if _is_num(value) else ""
            cells.append(f"<td{cls}>{_h(_format_value(value))}</td>")
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    table_class = f' class="{_h(class_name)}"' if class_name else ""
    return (
        f'<div class="table-scroll"><table{table_class}>'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody>"
        "</table></div>"
    )


def _summary_table(s1: dict) -> str:
    rows = [{"항목": k, "값": v} for k, v in s1.items()]
    return _table(rows, class_name="cond-table")


def _figure(path: Path | None, caption: str) -> str:
    if not path or not path.exists():
        return ""
    return (
        f'<figure><img class="chart-dark" src="{_img_data_uri(path)}" '
        f'alt="{_h(caption)}"/><figcaption>{_h(caption)}</figcaption></figure>'
    )


def _svg_path(kind: str) -> str:
    icons = {
        "users": '<path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/>',
        "plans": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
        "chat": '<path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>',
        "trend": '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
        "moon": '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
        "sun": '<circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>',
    }
    return icons[kind]


def _svg(kind: str) -> str:
    return f'<svg viewBox="0 0 24 24">{_svg_path(kind)}</svg>'


def _conversation_count(s1: dict) -> int:
    return sum(v for k, v in s1.items() if k.startswith("Conversation") and isinstance(v, int))


def _did_value(sections: list[dict]) -> tuple[str, str]:
    for section in sections:
        data = section.get("data") or {}
        if not isinstance(data, dict):
            continue
        summary = data.get("summary") if isinstance(data.get("summary"), dict) else {}
        if "DID_pct_points" in summary:
            return f"{summary['DID_pct_points']:+}%p", "정책 DID 순효과"
    return f"{len(sections)}개", "선택 분석"


def _kpi(icon: str, value: Any, label: str, *, positive: bool = False) -> str:
    cls = "kpi pos" if positive else "kpi"
    return (
        f'<div class="{cls}"><div class="icon">{_svg(icon)}</div>'
        f'<div class="value">{_h(_format_value(value))}</div>'
        f'<div class="label">{_h(label)}</div></div>'
    )


def _kpi_grid(s1: dict, sections: list[dict]) -> str:
    did, did_label = _did_value(sections)
    return (
        '<div class="kpi-grid">'
        f'{_kpi("users", s1.get("Agent 수", "-"), "에이전트")}'
        f'{_kpi("plans", s1.get("Plan 수", "-"), "생성된 Plan")}'
        f'{_kpi("chat", _conversation_count(s1), "사회적 상호작용")}'
        f'{_kpi("trend", did, did_label, positive=True)}'
        '</div>'
    )


def _label_class(label: str) -> str:
    lowered = label.lower()
    if "positive" in lowered or "긍정" in label or "수혜" in label:
        return "positive"
    if "negative" in lowered or "부정" in label:
        return "negative"
    return "neutral"


def _interview_html(interview: dict | None) -> str:
    if not interview:
        return ""
    blocks = []
    for label, person in interview.items():
        card_class = _label_class(label)
        if "error" in person:
            blocks.append(
                f'<div class="interview-card {card_class}">'
                f'<span class="label-tag">{_h(label)}</span>'
                f'<p>{_h(person["error"])}</p></div>'
            )
            continue

        p = person.get("persona", {})
        agent_id = person.get("agent_id", "")
        profile_bits = [
            f"{p.get('age', '?')} {p.get('gender', '?')}",
            p.get("job", "?"),
            f"소득 {p.get('income', '?')}",
            f"거주 {p.get('home_dong', '?')}",
        ]
        lifestyle = p.get("lifestyle")
        profile = " · ".join(profile_bits)
        if lifestyle:
            profile += f"<br/><strong>라이프스타일</strong> {_h(lifestyle)}"

        fallback = ""
        if label == "대표 시민":
            fallback = (
                '<div class="fallback-banner" style="display:flex;">'
                "정책 target 샘플이 없어 행동 trace가 풍부한 대표 시민을 사용했습니다."
                "</div>"
            )

        bubbles = []
        for qa in person.get("qa", []):
            bubbles.append(
                '<div class="chat-bubble user">'
                '<div class="meta"><span class="sender">인터뷰어</span></div>'
                f'<div class="text">{_h(qa.get("q", ""))}</div></div>'
            )
            bubbles.append(
                '<div class="chat-bubble agent">'
                f'<div class="meta"><span class="sender">에이전트 ({_h(agent_id)})</span></div>'
                f'<div class="text">{_h(qa.get("a", ""))}</div></div>'
            )

        blocks.append(
            f'<div class="interview-card {card_class}" data-agent-id="{_h(agent_id)}">'
            f'<span class="label-tag">{_h(label)}</span>'
            f'<h3>{_h(agent_id)}</h3>'
            f'<div class="persona">{profile}</div>'
            f'{fallback}'
            '<div class="chat-container static-chat"><div class="chat-messages">'
            f'{"".join(bubbles)}'
            '</div></div></div>'
        )
    return "".join(blocks)


def _nav(sections: list[dict], has_interview: bool) -> str:
    links = ['<a href="#s-summary">1. 시뮬레이션 개요</a>']
    for i, section in enumerate(sections, start=2):
        links.append(f'<a href="#s-analysis-{i}">{i}. {_h(section["title"])}</a>')
    if has_interview:
        links.append(f'<a href="#s-interview">{len(sections) + 2}. 1대1 인터뷰</a>')
    links.append('<a href="#s-appendix">부록</a>')
    return "".join(links)


def build_report_html(ctx: dict, s1: dict, sections: list[dict],
                       interview: dict | None = None) -> str:
    """Build the final fixed-template report HTML."""
    policy_name = ctx.get("name") or ctx.get("id") or "정책"
    policy_id = ctx.get("id") or "POLICY"
    policy_type = ctx.get("type") or "policy"
    policy_from = s1.get("정책_시행일") or ctx.get("effective_from") or "-"
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M KST")

    body_sections = []
    for i, section in enumerate(sections, start=2):
        figs = "".join(_figure(path, section["title"]) for path in section.get("chart_paths", []))
        table = _table(section.get("table_rows"))
        body_sections.append(f"""
    <section id="s-analysis-{i}" class="reveal">
      <h2>{i}. {_h(section["title"])}</h2>
      <div class="callout">{_h(section.get("narration", ""))}</div>
      {figs}
      {table}
    </section>""")

    interview_section = ""
    if interview:
        interview_section = f"""
    <section id="s-interview" class="reveal">
      <h2>{len(sections) + 2}. 1대1 인터뷰</h2>
      {_interview_html(interview)}
    </section>"""

    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{_h(policy_name)} — 최종 보고서</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Noto+Sans+KR:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>{_report_css()}</style>
</head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="sidebar-header" style="display:flex; justify-content:space-between; align-items:center; margin-bottom:24px;">
      <div class="brand" style="margin-bottom:0;">SEOUL POLICY SIMULATION</div>
      <button id="theme-toggle" class="theme-toggle-btn" aria-label="Toggle theme">
        <svg class="sun-icon" viewBox="0 0 24 24" style="display:none; width:16px; height:16px;">{_svg_path("sun")}</svg>
        <svg class="moon-icon" viewBox="0 0 24 24" style="width:16px; height:16px;">{_svg_path("moon")}</svg>
      </button>
    </div>
    <h2>최종 보고서</h2>
    <nav>{_nav(sections, bool(interview))}</nav>
    <div class="tech-info">
      분석 계산: <span>Python deterministic functions</span><br/>
      문장 생성: <span>K-EXAONE narration only</span><br/>
      저장 형식: <span>Self-contained HTML</span>
    </div>
  </aside>
  <main class="main">
    <header class="cover">
      <div class="meta">FINAL REPORT · {_h(policy_id)}</div>
      <h1>{_h(policy_name)}</h1>
      <div class="subtitle">{_h(ctx.get("description", ""))}</div>
      <div class="badges">
        <span class="badge alt">{_h(policy_type)}</span>
        <span class="badge purple">Neo4j Graph DB</span>
        <span class="badge">시행일 {_h(policy_from)}</span>
        <span class="badge">고정 HTML 템플릿</span>
      </div>
      {_kpi_grid(s1, sections)}
    </header>

    <section id="s-summary" class="reveal">
      <h2>1. 시뮬레이션 개요</h2>
      {_summary_table(s1)}
    </section>
    {"".join(body_sections)}
    {interview_section}
    <section id="s-appendix" class="reveal">
      <h2>부록</h2>
      <div class="callout">
        본 HTML은 <code>scripts/report/build_html.py</code>와
        <code>scripts/report/report_template.css</code>의 고정 템플릿으로 생성됩니다.
      </div>
    </section>
    <div class="footer">
      Generated by <code>scripts/report/menu.py</code> · {_h(generated_at)}<br/>
      Kw Capstone · 서울시 상권정책 시뮬레이션 프로젝트
    </div>
  </main>
</div>
<script>{_report_js()}</script>
</body>
</html>"""
