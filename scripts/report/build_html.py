"""build_html.py — 선택된 분석 결과를 self-contained HTML 보고서로 조립.

숫자·차트는 catalog.py가 만든 그대로 쓰고, 여기서는 레이아웃만 담당한다.
CSS/이미지-임베드 헬퍼는 generate_final_report.py 것을 재사용해 기존 보고서와
톤을 맞춘다.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))

from generate_final_report import HTML_STYLE, _figure, _h  # noqa: E402

_EXTRA_CSS = """
table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 14px; }
th, td { padding: 8px 12px; border-bottom: 1px solid var(--border-subtle); text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
figure { margin: 20px 0; }
figure img { max-width: 100%; border-radius: var(--radius); }
figcaption { color: var(--text-muted); font-size: 12px; margin-top: 6px; }
.narration { color: var(--text-primary); margin: 12px 0; line-height: 1.8; }
.cond-table td:first-child { color: var(--text-secondary); width: 40%; }
.interview-card { background: var(--bg-card); border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg); padding: 24px; margin-bottom: 20px; }
.interview-card .profile { color: var(--text-secondary); font-size: 13px; margin-bottom: 16px; }
.qa { margin-bottom: 14px; }
.qa .q { color: var(--cyan); font-weight: 600; margin-bottom: 4px; }
.qa .a { color: var(--text-primary); }
"""


def _table(rows: list[dict] | None) -> str:
    if not rows:
        return ""
    cols = list(rows[0].keys())
    head = "".join(f"<th>{_h(c)}</th>" for c in cols)
    body = "".join(
        "<tr>" + "".join(f"<td>{_h(r.get(c, ''))}</td>" for c in cols) + "</tr>"
        for r in rows
    )
    return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def _interview_html(interview: dict | None) -> str:
    if not interview:
        return ""
    blocks = []
    for label, person in interview.items():
        if "error" in person:
            blocks.append(f'<p>({_h(label)}: {_h(person["error"])})</p>')
            continue
        p = person["persona"]
        profile = (f"{p.get('age', '?')} {p.get('gender', '?')} · {p.get('job', '?')} · "
                   f"소득 {p.get('income', '?')} · 거주 {p.get('home_dong', '?')}")
        qa_html = "".join(
            f'<div class="qa"><p class="q">Q. {_h(qa["q"])}</p>'
            f'<p class="a">A. {_h(qa["a"])}</p></div>'
            for qa in person["qa"]
        )
        blocks.append(
            f'<div class="interview-card"><h3>{_h(label)} — {_h(person.get("agent_id", ""))}</h3>'
            f'<p class="profile">{_h(profile)}</p>{qa_html}</div>'
        )
    return "".join(blocks)


def build_report_html(ctx: dict, s1: dict, sections: list[dict],
                       interview: dict | None = None) -> str:
    """sections: [{"title": str, "narration": str,
                    "table_rows": list[dict] | None, "chart_paths": list[Path]}]"""
    nav = "".join(f'<a href="#s-{i}">{_h(s["title"])}</a>' for i, s in enumerate(sections))
    if interview:
        nav += '<a href="#s-interview">1대1 인터뷰</a>'

    body_sections = []
    for i, s in enumerate(sections):
        figs = "".join(_figure(p, s["title"]) for p in s.get("chart_paths", []))
        table = _table(s.get("table_rows"))
        body_sections.append(f'''
<section id="s-{i}" class="reveal">
  <h2>{i + 1}. {_h(s["title"])}</h2>
  <p class="narration">{_h(s.get("narration", ""))}</p>
  {table}
  {figs}
</section>''')

    interview_section = ""
    if interview:
        interview_section = f'''
<section id="s-interview" class="reveal">
  <h2>1대1 인터뷰</h2>
  {_interview_html(interview)}
</section>'''

    cond_rows = "".join(f"<tr><td>{_h(k)}</td><td>{_h(v)}</td></tr>" for k, v in s1.items())
    policy_name = ctx.get("name") or ctx.get("id") or "정책"

    return f'''<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<title>{_h(policy_name)} — 최종 보고서</title>
<style>{HTML_STYLE}{_EXTRA_CSS}</style></head>
<body>
<div class="layout">
  <aside class="sidebar">
    <div class="brand">POLICY REPORT</div>
    <h2>목차</h2>
    <nav>
      <a href="#s-cond">시뮬레이션 조건</a>
      {nav}
    </nav>
  </aside>
  <main class="main">
    <header class="cover">
      <div class="meta">시뮬레이션 최종 보고서</div>
      <h1>{_h(policy_name)}</h1>
      <p class="subtitle">{_h(ctx.get("description", ""))}</p>
      <p class="subtitle">작성일 {datetime.now().strftime("%Y-%m-%d %H:%M KST")}</p>
    </header>
    <section id="s-cond" class="reveal">
      <h2>시뮬레이션 조건</h2>
      <table class="cond-table"><tbody>{cond_rows}</tbody></table>
    </section>
    {"".join(body_sections)}
    {interview_section}
  </main>
</div>
</body></html>'''
