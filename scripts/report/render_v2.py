"""보고서 v2 — 단일 HTML 렌더러.

콘솔(`web/ui`)과 같은 디자인 규칙을 따른다: 그라디언트·글래스·큰 그림자 없음,
0~4px radius, 1px 헤어라인, 의미가 있을 때만 색, 숫자는 monospace.
색은 전부 CSS 변수로 두어 **라이트/다크가 같은 그림 하나로** 동작한다.

콘솔의 `ReportScreen` 은 이 파일을 파싱해 `<style>` 을 `.reportdoc` 아래로
스코프하고 `<script>` 를 제거한다. 그래서 **모든 그림과 표는 JS 없이 보여야 한다.**
테마 토글만 진행적 향상(progressive enhancement)으로 붙인다.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from html import escape
from typing import Any, Sequence
from urllib.parse import quote

try:
    from . import charts
except ImportError:  # menu.py 실행 경로
    import charts  # type: ignore

krw = charts.krw
pct = charts.pct
num = charts.num

TOP_CATEGORIES = 8  # 소형 겹쳐보기·히트맵에 쓸 상위 업종 수

# f-string 안에서 역슬래시를 쓸 수 없어(3.10) 배지 조각은 상수로 둔다.
TAG_TARGET = '<span class="tag tag--t">대상</span>'
TAG_GRANT = '<span class="tag tag--t">지급</span>'

CSS = """
:root {
  color-scheme: light dark;
}
.doc {
  /* 표면 */
  --bg: #f7f7f5;
  --surface: #ffffff;
  --surface-2: #f2f2ef;
  --border: #d9d9d3;
  --border-strong: #b9b9b1;
  --fg: #16181c;
  --fg-muted: #62666e;
  --fg-faint: #8b8f97;
  /* 상태 */
  --ok: #2f6f4f;
  --warn: #8a6100;
  --danger: #a03028;
  --info: #1f5c8a;
  --sel: #2c4f7c;
  /* 차트 */
  --k1: #2c4f7c;
  --k2: #7a6a3f;
  --k3: #2f6f4f;
  --k4: #8a4b62;
  --k5: #46617a;
  --k6: #6b5b8a;
  --k7: #8a6100;
  --k8: #4a6b6b;
  --k-pos: #2f6f4f;
  --k-neg: #a03028;
  --k-grid: #e4e4de;
  --k-axis: #b9b9b1;
  --k-ink: #16181c;
  --k-muted: #62666e;
  --k-band: #ecece6;

  background: var(--bg);
  color: var(--fg);
  font-family: "Pretendard", "Inter", -apple-system, "Segoe UI", "Noto Sans KR", system-ui, sans-serif;
  font-size: 14px;
  line-height: 1.62;
  -webkit-font-smoothing: antialiased;
}
@media (prefers-color-scheme: dark) {
  .doc:not([data-theme="light"]) {
    --bg: #121316;
    --surface: #191b1f;
    --surface-2: #202328;
    --border: #2d3138;
    --border-strong: #414751;
    --fg: #e6e8ec;
    --fg-muted: #9aa0aa;
    --fg-faint: #737a85;
    --ok: #6cbf95;
    --warn: #d7a53f;
    --danger: #e07a70;
    --info: #6aa9d8;
    --sel: #7aa2d8;
    --k1: #7aa2d8;
    --k2: #cbb173;
    --k3: #6cbf95;
    --k4: #d68fa6;
    --k5: #8fa8bf;
    --k6: #a795cf;
    --k7: #d7a53f;
    --k8: #7fb3b3;
    --k-pos: #6cbf95;
    --k-neg: #e07a70;
    --k-grid: #262a30;
    --k-axis: #414751;
    --k-ink: #e6e8ec;
    --k-muted: #9aa0aa;
    --k-band: #1e2126;
  }
}
.doc[data-theme="dark"] {
  --bg: #121316;
  --surface: #191b1f;
  --surface-2: #202328;
  --border: #2d3138;
  --border-strong: #414751;
  --fg: #e6e8ec;
  --fg-muted: #9aa0aa;
  --fg-faint: #737a85;
  --ok: #6cbf95;
  --warn: #d7a53f;
  --danger: #e07a70;
  --info: #6aa9d8;
  --sel: #7aa2d8;
  --k1: #7aa2d8;
  --k2: #cbb173;
  --k3: #6cbf95;
  --k4: #d68fa6;
  --k5: #8fa8bf;
  --k6: #a795cf;
  --k7: #d7a53f;
  --k8: #7fb3b3;
  --k-pos: #6cbf95;
  --k-neg: #e07a70;
  --k-grid: #262a30;
  --k-axis: #414751;
  --k-ink: #e6e8ec;
  --k-muted: #9aa0aa;
  --k-band: #1e2126;
}
.doc * { box-sizing: border-box; }
.doc .wrap { max-width: 1120px; margin: 0 auto; padding: 28px 24px 72px; }
.doc .num, .doc .tick, .doc code, .doc pre {
  font-family: "JetBrains Mono", "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums;
}

/* 머리말 */
.doc .masthead {
  border: 1px solid var(--border);
  background: var(--surface);
  padding: 20px 22px;
  margin-bottom: 20px;
}
.doc .masthead__kicker {
  font-size: 11px; letter-spacing: .09em; text-transform: uppercase;
  color: var(--fg-faint); margin: 0 0 6px;
}
.doc .masthead__title { font-size: 22px; font-weight: 650; margin: 0 0 4px; letter-spacing: -.01em; }
.doc .masthead__sub { color: var(--fg-muted); margin: 0 0 16px; font-size: 13px; }
.doc .factbar {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 0; border: 1px solid var(--border); border-bottom: 0;
}
.doc .fact { border-bottom: 1px solid var(--border); border-right: 1px solid var(--border); padding: 10px 12px; }
.doc .fact:last-child { border-right: 0; }
.doc .fact__k { font-size: 11px; color: var(--fg-faint); margin: 0 0 3px; }
.doc .fact__v { font-size: 15px; font-weight: 600; margin: 0; }
.doc .fact__m { font-size: 11px; color: var(--fg-muted); margin: 2px 0 0; }

/* 목차 */
.doc .toc { border: 1px solid var(--border); background: var(--surface); padding: 14px 18px; margin-bottom: 22px; }
.doc .toc h2 { font-size: 12px; letter-spacing: .06em; text-transform: uppercase; color: var(--fg-faint); margin: 0 0 8px; }
.doc .toc ol { margin: 0; padding-left: 20px; columns: 2; column-gap: 32px; font-size: 13px; }
.doc .toc a { color: var(--fg); text-decoration: none; border-bottom: 1px solid transparent; }
.doc .toc a:hover { border-bottom-color: var(--border-strong); }

/* 절 */
.doc .sec { border: 1px solid var(--border); background: var(--surface); margin-bottom: 18px; }
.doc .sec__head { padding: 14px 18px; border-bottom: 1px solid var(--border); background: var(--surface-2); }
.doc .sec__n { font-size: 11px; color: var(--fg-faint); letter-spacing: .08em; margin: 0 0 2px; }
.doc .sec__t { font-size: 16px; font-weight: 620; margin: 0; }
.doc .sec__p { font-size: 12.5px; color: var(--fg-muted); margin: 5px 0 0; max-width: 78ch; }
.doc .sec__body { padding: 18px; }
.doc .sec__body > * + * { margin-top: 16px; }
.doc h3 { font-size: 13.5px; font-weight: 620; margin: 0 0 8px; }
.doc p { margin: 0 0 10px; max-width: 84ch; }
.doc p:last-child { margin-bottom: 0; }

/* 해설 */
.doc .note { border-left: 2px solid var(--border-strong); padding: 2px 0 2px 14px; color: var(--fg); }
.doc .note__src { display: block; font-size: 11px; color: var(--fg-faint); margin-top: 6px; }
.doc .callout { border: 1px solid var(--border); border-left-width: 3px; padding: 11px 14px; font-size: 13px; }
.doc .callout--warn { border-left-color: var(--warn); }
.doc .callout--danger { border-left-color: var(--danger); }
.doc .callout--ok { border-left-color: var(--ok); }
.doc .callout--info { border-left-color: var(--info); }

/* 그림 */
.doc figure { margin: 0; border: 1px solid var(--border); background: var(--surface); }
.doc figure > svg { display: block; width: 100%; height: auto; padding: 12px 12px 4px; }
.doc figcaption { border-top: 1px solid var(--border); padding: 8px 12px; font-size: 12px; color: var(--fg-muted); }
.doc figcaption b { color: var(--fg); font-weight: 600; }
.doc figcaption .src { display: block; font-size: 11px; color: var(--fg-faint); margin-top: 3px; }
.doc .figgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }

/* 표 */
.doc .tablewrap { border: 1px solid var(--border); overflow-x: auto; }
.doc table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.doc thead th {
  text-align: left; font-weight: 600; color: var(--fg-muted); background: var(--surface-2);
  padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; font-size: 11.5px;
}
.doc tbody td { padding: 7px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
.doc tbody tr:last-child td { border-bottom: 0; }
.doc tbody tr[data-target="1"] { background: color-mix(in srgb, var(--sel) 7%, transparent); }
.doc td.n, .doc th.n { text-align: right; font-variant-numeric: tabular-nums; font-family: "JetBrains Mono", ui-monospace, monospace; }
.doc tfoot td { padding: 8px 10px; border-top: 1px solid var(--border-strong); font-weight: 600; background: var(--surface-2); }
.doc .pos { color: var(--ok); }
.doc .neg { color: var(--danger); }
.doc .tag {
  display: inline-block; font-size: 10.5px; padding: 1px 6px; border: 1px solid var(--border-strong);
  color: var(--fg-muted); white-space: nowrap;
}
.doc .tag--t { border-color: var(--sel); color: var(--sel); }
.doc .st { font-size: 11px; padding: 1px 7px; border: 1px solid currentColor; white-space: nowrap; }
.doc .st--pass { color: var(--ok); }
.doc .st--fail { color: var(--danger); }
.doc .st--skip { color: var(--fg-faint); }

/* 정의 목록 */
.doc .dl { display: grid; grid-template-columns: minmax(140px, 190px) 1fr; gap: 0; border: 1px solid var(--border); }
.doc .dl dt { padding: 8px 12px; border-bottom: 1px solid var(--border); background: var(--surface-2); font-size: 12px; color: var(--fg-muted); }
.doc .dl dd { padding: 8px 12px; border-bottom: 1px solid var(--border); margin: 0; font-size: 12.5px; }
.doc .dl dt:last-of-type, .doc .dl dd:last-of-type { border-bottom: 0; }

/* 수식 */
.doc .formula {
  border: 1px solid var(--border); background: var(--surface-2); padding: 12px 14px;
  font-family: "JetBrains Mono", ui-monospace, monospace; font-size: 12.5px; line-height: 1.9;
  overflow-x: auto; white-space: pre;
}

.doc .foot { margin-top: 26px; padding-top: 14px; border-top: 1px solid var(--border); font-size: 11.5px; color: var(--fg-faint); }
.doc .foot ul { padding-left: 18px; margin: 6px 0 0; }
.doc .foot a { color: var(--info); }
.doc .themebtn {
  border: 1px solid var(--border-strong); background: var(--surface); color: var(--fg);
  font: inherit; font-size: 12px; padding: 4px 10px; cursor: pointer;
}
.doc .headrow { display: flex; justify-content: space-between; align-items: flex-start; gap: 16px; }
@media print {
  .doc .themebtn, .doc .toc { display: none; }
  .doc .sec { break-inside: avoid; }
}
"""

THEME_SCRIPT = """
(function () {
  var doc = document.querySelector('.doc');
  var btn = document.getElementById('themebtn');
  if (!doc || !btn) return;
  var KEY = 'dasol-report-theme';
  function apply(v) {
    if (v === 'dark' || v === 'light') { doc.setAttribute('data-theme', v); }
    else { doc.removeAttribute('data-theme'); }
    btn.textContent = v === 'dark' ? '라이트 테마' : (v === 'light' ? '시스템 테마' : '다크 테마');
    btn.setAttribute('data-mode', v || 'system');
  }
  var saved = null;
  try { saved = localStorage.getItem(KEY); } catch (e) { saved = null; }
  apply(saved);
  btn.addEventListener('click', function () {
    var cur = btn.getAttribute('data-mode');
    var next = cur === 'system' ? 'dark' : (cur === 'dark' ? 'light' : 'system');
    try { localStorage.setItem(KEY, next === 'system' ? '' : next); } catch (e) {}
    apply(next === 'system' ? null : next);
  });
})();
"""


# --------------------------------------------------------------------------- #
# 조각
# --------------------------------------------------------------------------- #


def esc(value: Any) -> str:
    return escape("" if value is None else str(value))


def _signed(value: float | None, formatter=krw) -> str:
    if value is None:
        return '<span class="n">—</span>'
    cls = "pos" if value >= 0 else "neg"
    prefix = "+" if value >= 0 else ""
    return f'<span class="{cls}">{prefix}{esc(formatter(value))}</span>'


def fact(key: str, value: str, meta: str = "") -> str:
    meta_html = f'<p class="fact__m">{esc(meta)}</p>' if meta else ""
    return f'<div class="fact"><p class="fact__k">{esc(key)}</p><p class="fact__v num">{value}</p>{meta_html}</div>'


def section(
    number: str,
    title: str,
    purpose: str,
    body: str,
    *,
    anchor: str,
) -> str:
    return (
        f'<section class="sec" id="{esc(anchor)}">'
        f'<div class="sec__head"><p class="sec__n">{esc(number)}</p>'
        f'<h2 class="sec__t">{esc(title)}</h2>'
        f'<p class="sec__p">{esc(purpose)}</p></div>'
        f'<div class="sec__body">{body}</div></section>'
    )


def figure(svg: str, caption: str, source: str = "") -> str:
    src = f'<span class="src">출처: {esc(source)}</span>' if source else ""
    return f"<figure>{svg}<figcaption>{caption}{src}</figcaption></figure>"


def table(
    headers: Sequence[tuple[str, str]],
    rows: Sequence[Sequence[str]],
    *,
    foot: Sequence[str] | None = None,
    row_attrs: Sequence[str] | None = None,
) -> str:
    head = "".join(f'<th class="{cls}" scope="col">{esc(label)}</th>' for label, cls in headers)
    body_rows = []
    for index, row in enumerate(rows):
        attrs = row_attrs[index] if row_attrs and index < len(row_attrs) else ""
        cells = "".join(
            f'<td class="{headers[ci][1]}">{cell}</td>' if ci < len(headers) else f"<td>{cell}</td>"
            for ci, cell in enumerate(row)
        )
        body_rows.append(f"<tr {attrs}>{cells}</tr>")
    foot_html = ""
    if foot:
        cells = "".join(
            f'<td class="{headers[ci][1]}">{cell}</td>' if ci < len(headers) else f"<td>{cell}</td>"
            for ci, cell in enumerate(foot)
        )
        foot_html = f"<tfoot><tr>{cells}</tr></tfoot>"
    return (
        f'<div class="tablewrap"><table><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>{foot_html}</table></div>'
    )


def note(narration: dict[str, Any], key: str) -> str:
    entry = (narration.get("sections") or {}).get(key)
    if not entry or not entry.get("text"):
        return ""
    source = entry.get("source", "deterministic")
    if source.startswith("llm"):
        label = f"해설: {esc(entry.get('model') or source)} · 숫자 검증 통과"
    elif entry.get("guard") == "rejected":
        label = "해설: 규칙 기반 서술 (LLM 문장에 계산 결과에 없는 숫자가 있어 채택하지 않음)"
    elif entry.get("guard") == "llm_error":
        label = "해설: 규칙 기반 서술 (LLM 호출 실패)"
    else:
        label = "해설: 규칙 기반 서술 (해설 LLM 미설정)"
    paragraphs = "".join(f"<p>{esc(line)}</p>" for line in str(entry["text"]).split("\n") if line.strip())
    return f'<div class="note">{paragraphs}<span class="note__src">{label}</span></div>'


def dl(items: Sequence[tuple[str, str]]) -> str:
    body = "".join(f"<dt>{esc(k)}</dt><dd>{v}</dd>" for k, v in items)
    return f'<dl class="dl">{body}</dl>'


# --------------------------------------------------------------------------- #
# 본문
# --------------------------------------------------------------------------- #


def _masthead(bundle: dict[str, Any], consistency: dict[str, Any]) -> str:
    meta = bundle["meta"]
    period = bundle["period"]
    totals = bundle["totals"]
    did = bundle.get("did") or {}
    counts = consistency.get("counts", {})
    verdict_cls = "ok" if consistency.get("consistent") else "danger"
    facts = [
        fact("분석 실행", esc(meta.get("run_id")), f"{meta.get('day_count')}일 · {meta.get('event_rows'):,}건"),
        fact(
            "정책",
            esc(meta.get("policy_id") or "미지정"),
            esc(meta.get("policy_name") or "정책 메타 없음"),
        ),
        fact(
            "시행일",
            esc(period.get("policy_from") or "미지정"),
            f"사전 {len(period.get('pre') or [])}일 · 사후 {len(period.get('post') or [])}일",
        ),
        fact("총 소비금액", esc(krw(totals.get("amt"))), f"{int(totals.get('events') or 0):,}건"),
        fact(
            "정책 지급액",
            esc(krw(totals.get("policy_paid"))),
            f"자기부담 {krw(totals.get('self_paid'))}",
        ),
        fact(
            "이중차분 추정",
            _signed(did.get("did_absolute")) if did else "—",
            (f"일평균 · 반사실 대비 {pct(did.get('did_pct_of_counterfactual'))}" if did else "계산 불가"),
        ),
        fact(
            "일관성 검증",
            f'<span class="{verdict_cls}">{counts.get("pass", 0)}/{counts.get("total", 0)} 통과</span>',
            f"실패 {counts.get('fail', 0)} · 미검사 {counts.get('skip', 0)}",
        ),
    ]
    return (
        '<header class="masthead">'
        '<div class="headrow"><div>'
        '<p class="masthead__kicker">정책 시뮬레이션 최종 분석 보고서 · DASOL v2</p>'
        f'<h1 class="masthead__title">{esc(meta.get("policy_name") or meta.get("policy_id") or "정책")} '
        f'효과 분석</h1>'
        f'<p class="masthead__sub">{esc(meta.get("run_id"))} · '
        f'{esc((meta.get("days") or ["—"])[0])} ~ {esc((meta.get("days") or ["—"])[-1])} · '
        f'생성 {esc(datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))}</p>'
        "</div>"
        '<button type="button" class="themebtn" id="themebtn" data-mode="system">다크 테마</button>'
        "</div>"
        f'<div class="factbar">{"".join(facts)}</div>'
        "</header>"
    )


def _toc(entries: Sequence[tuple[str, str]]) -> str:
    items = "".join(f'<li><a href="#{esc(anchor)}">{esc(title)}</a></li>' for anchor, title in entries)
    return f'<nav class="toc"><h2>목차</h2><ol>{items}</ol></nav>'


def _sec_overview(bundle: dict[str, Any], narration: dict[str, Any], consistency: dict[str, Any]) -> str:
    meta, period, mix = bundle["meta"], bundle["period"], bundle["mix"]
    targets = bundle.get("targets", {})
    body = [note(narration, "overview")]
    body.append(
        dl(
            [
                ("실행 ID", f'<span class="num">{esc(meta.get("run_id"))}</span>'),
                ("산출물 경로", f'<code>{esc(meta.get("run_root"))}</code>'),
                ("분석 구간", f'<span class="num">{esc((meta.get("days") or ["—"])[0])} ~ '
                             f'{esc((meta.get("days") or ["—"])[-1])} ({meta.get("day_count")}일)</span>'),
                ("정책 시행일", f'<span class="num">{esc(period.get("policy_from") or "미지정")}</span>'),
                (
                    "처치군(정책 대상 업종)",
                    (", ".join(esc(c) for c in targets.get("categories") or []) or "없음")
                    + f'<br><span class="tag">판정 근거: {esc(targets.get("source") or "판정 불가")}</span>',
                ),
                (
                    "대조군(비대상 업종)",
                    ", ".join(esc(c) for c in bundle.get("control_categories") or []) or "없음",
                ),
                ("소비 이벤트", f'<span class="num">{int(meta.get("event_rows") or 0):,}건</span>'),
                (
                    "정책 지렛대",
                    (
                        f'<span class="num">{esc(num(mix.get("leverage"), digits=2))}배</span> '
                        f'— 정책지급 1원당 발생한 총 소비금액'
                        if mix.get("leverage")
                        else "정책 지급액이 0이라 계산할 수 없습니다"
                    ),
                ),
                (
                    "사중손실 추정",
                    (
                        f'<span class="num">{int(mix.get("deadweight_events") or 0):,}건 '
                        f'({esc(num(mix.get("deadweight_share_pct"), digits=2))}%)</span> '
                        "— 정책이 없어도 구매했을 것이라고 응답한 이벤트"
                    ),
                ),
            ]
        )
    )
    if not period.get("usable"):
        body.append(
            f'<div class="callout callout--warn">이 실행에서는 이중차분을 계산할 수 없습니다. '
            f'{esc(period.get("reason") or "")} 아래의 전후 비교는 인과 효과가 아니라 단순 관측치입니다.</div>'
        )
    if not consistency.get("consistent"):
        body.append(
            f'<div class="callout callout--danger">일관성 검증에서 '
            f'{len(consistency.get("failed_ids") or [])}건이 실패했습니다 '
            f'({", ".join(esc(i) for i in consistency.get("failed_ids") or [])}). '
            "이 보고서의 수치를 그대로 인용하기 전에 마지막 절을 확인하세요.</div>"
        )
    return "".join(body)


def _sec_policy(bundle: dict[str, Any], policy: dict[str, Any]) -> str:
    grants = policy.get("decile_grants") or policy.get("income_grants") or {}
    body: list[str] = []
    body.append(
        dl(
            [
                ("정책 ID", f'<span class="num">{esc(policy.get("id"))}</span>'),
                ("정책명", esc(policy.get("name"))),
                ("유형", esc(policy.get("type"))),
                ("시행 기간", f'<span class="num">{esc(policy.get("effective_from"))} ~ '
                             f'{esc(policy.get("effective_until"))}</span>'),
                ("대상 지역", ", ".join(esc(x) for x in (policy.get("target_districts") or [])) or "지정 없음"),
                ("대상 업종", ", ".join(esc(x) for x in (policy.get("benefit_categories") or [])) or "지정 없음"),
                ("사용처 제한", "쿠폰 가맹점으로 제한" if policy.get("poi_restricted") else "제한 없음"),
                ("지급 기준", esc(policy.get("grant_key") or "기본값")),
                ("설명", esc(policy.get("description") or "—")),
            ]
        )
    )
    if grants:
        keys = sorted(grants, key=lambda k: (len(str(k)), str(k)))
        body.append(
            figure(
                charts.grouped_bar(
                    [str(k) for k in keys],
                    [{"name": "지급액", "values": [float(grants[k]) for k in keys], "color": "var(--k1)"}],
                    title="지급 구간별 금액",
                    value_labels=True,
                    height=260,
                ),
                "<b>지급 구간별 금액.</b> 정책 JSON 에 적힌 값 그대로이며 시뮬레이션이 실제로 배선한 금액과 "
                "같아야 합니다 (사전 검증 preflight 항목).",
                "정책 JSON",
            )
        )
    paid = bundle.get("policy_paid_by_policy_id") or {}
    if paid:
        body.append(
            table(
                [("정책 ID", ""), ("실제 지급액", "n"), ("전체 대비", "n")],
                [
                    [
                        f'<span class="num">{esc(pid)}</span>',
                        f'<span class="num">{esc(krw(amount))}</span>',
                        f'<span class="num">{amount / (bundle["totals"]["policy_paid"] or 1) * 100:.1f}%</span>',
                    ]
                    for pid, amount in sorted(paid.items(), key=lambda kv: -kv[1])
                ],
            )
        )
    return "".join(body)


def _sec_timeseries(bundle: dict[str, Any]) -> str:
    daily = bundle["daily"]
    labels = [row["day"][5:] for row in daily]
    marker = next((i for i, row in enumerate(daily) if row["phase"] == "post"), None)
    body = [
        figure(
            charts.line_chart(
                labels,
                [
                    {"name": "총 소비금액", "values": [row["amt"] for row in daily], "color": "var(--k1)"},
                    {"name": "자기부담", "values": [row["self_paid"] for row in daily], "color": "var(--k5)"},
                    {"name": "정책지급", "values": [row["policy_paid"] for row in daily], "color": "var(--k3)"},
                    {"name": "추가지출", "values": [row["extra"] for row in daily], "color": "var(--k7)"},
                ],
                marker_index=marker,
                marker_label="시행일",
                height=320,
            ),
            "<b>일자별 소비 총량.</b> 붉은 점선이 정책 시행일입니다. 정책지급과 자기부담을 함께 그려, "
            "총액 증가가 지급액 그 자체인지 자기부담이 함께 늘어난 결과인지 구분할 수 있게 했습니다.",
            "events.jsonl",
        )
    ]
    body.append(
        figure(
            charts.line_chart(
                labels,
                [
                    {"name": "이벤트 수", "values": [row["events"] for row in daily], "color": "var(--k2)"},
                    {"name": "객단가", "values": [row["avg_ticket"] for row in daily], "color": "var(--k4)"},
                ],
                marker_index=marker,
                marker_label="시행일",
                formatter=lambda v: num(v),
                height=260,
            ),
            "<b>건수와 객단가.</b> 총액이 늘었을 때 사람들이 <b>더 자주</b> 샀는지 <b>더 비싸게</b> 샀는지를 "
            "가른다. 둘의 움직임이 다르면 정책의 작동 방식이 다르다는 뜻입니다.",
            "events.jsonl",
        )
    )
    rows = []
    for row in daily:
        rows.append(
            [
                f'<span class="num">{esc(row["day"])}</span>',
                f'<span class="tag">{"사후" if row["phase"] == "post" else "사전"}</span>',
                f'<span class="num">{row["events"]:,}</span>',
                f'<span class="num">{esc(krw(row["amt"]))}</span>',
                f'<span class="num">{esc(krw(row["policy_paid"]))}</span>',
                f'<span class="num">{esc(krw(row["self_paid"]))}</span>',
                f'<span class="num">{esc(krw(row["extra"]))}</span>',
                f'<span class="num">{esc(num(row["avg_ticket"]))}</span>',
                f'<span class="num">{esc(num(row["per_capita"]))}</span>',
                f'<span class="num">{esc(num(row["avg_satisfaction"], digits=3))}</span>',
            ]
        )
    totals = bundle["totals"]
    body.append(
        table(
            [
                ("일자", ""),
                ("구간", ""),
                ("건수", "n"),
                ("소비금액", "n"),
                ("정책지급", "n"),
                ("자기부담", "n"),
                ("추가지출", "n"),
                ("객단가", "n"),
                ("1인당", "n"),
                ("만족도", "n"),
            ],
            rows,
            foot=[
                "합계",
                "",
                f'<span class="num">{int(totals["events"]):,}</span>',
                f'<span class="num">{esc(krw(totals["amt"]))}</span>',
                f'<span class="num">{esc(krw(totals["policy_paid"]))}</span>',
                f'<span class="num">{esc(krw(totals["self_paid"]))}</span>',
                f'<span class="num">{esc(krw(totals["extra"]))}</span>',
                "",
                "",
                "",
            ],
        )
    )
    return "".join(body)


def _sec_overlay(bundle: dict[str, Any], narration: dict[str, Any]) -> str:
    overlay = bundle["overlay"]
    if not overlay.get("available"):
        return f'<div class="callout callout--warn">{esc(overlay.get("reason") or "겹쳐 그릴 구간이 없습니다.")}</div>'
    overall = overlay["overall"]
    body = [note(narration, "overlay")]
    body.append(
        figure(
            charts.overlay_chart(
                overall["labels"],
                overall["pre"],
                overall["post"],
                title="시행 전후 소비 겹쳐보기",
                height=340,
            ),
            "<b>시행 전/후 겹쳐보기 — 전체.</b> 시행 직전 "
            f'{overlay["window_days"]}일과 시행 직후 {overlay["window_days"]}일을 같은 축에 올렸습니다. '
            "두 곡선 사이의 칠해진 면적이 <b>소비가 달라진 크기</b>이고, 색이 방향입니다. "
            + esc(overlay.get("note") or ""),
            "events.jsonl · 일자별 합계",
        )
    )
    rows = []
    for index, label in enumerate(overall["labels"]):
        pre_v, post_v = overall["pre"][index], overall["post"][index]
        growth = ((post_v / pre_v - 1) * 100) if pre_v else None
        rows.append(
            [
                f'<span class="num">{esc(label)}</span>',
                f'<span class="num">{esc(overall["pre_days"][index])}</span>',
                f'<span class="num">{esc(krw(pre_v))}</span>',
                f'<span class="num">{esc(overall["post_days"][index])}</span>',
                f'<span class="num">{esc(krw(post_v))}</span>',
                _signed(post_v - pre_v),
                _signed(growth, formatter=lambda v: f"{abs(v):.1f}%") if growth is not None else "—",
            ]
        )
    body.append(
        table(
            [
                ("상대 일차", ""),
                ("시행 전 일자", ""),
                ("시행 전 금액", "n"),
                ("시행 후 일자", ""),
                ("시행 후 금액", "n"),
                ("차이", "n"),
                ("증감률", "n"),
            ],
            rows,
            foot=[
                "합계",
                "",
                f'<span class="num">{esc(krw(sum(overall["pre"])))}</span>',
                "",
                f'<span class="num">{esc(krw(sum(overall["post"])))}</span>',
                _signed(sum(overall["post"]) - sum(overall["pre"])),
                "",
            ],
        )
    )

    # 상위 업종 소형 겹쳐보기
    top = [row["l1"] for row in bundle["categories"][:TOP_CATEGORIES]]
    smalls = []
    for l1 in top:
        series = overlay["by_category"].get(l1)
        if not series:
            continue
        smalls.append(
            figure(
                charts.overlay_chart(
                    overall["labels"],
                    series["pre"],
                    series["post"],
                    title=f"{l1} 전후 겹쳐보기",
                    width=520,
                    height=250,
                ),
                f"<b>{esc(l1)}</b> — 시행 전 {krw(sum(series['pre']))} → 시행 후 {krw(sum(series['post']))} "
                f"({_signed(sum(series['post']) - sum(series['pre']))})",
            )
        )
    if smalls:
        body.append(f'<div class="figgrid">{"".join(smalls)}</div>')
    return "".join(body)


def _sec_categories(bundle: dict[str, Any], narration: dict[str, Any]) -> str:
    categories = bundle["categories"]
    if not categories:
        return '<div class="callout callout--warn">업종 정보를 가진 이벤트가 없습니다.</div>'
    labels = [row["l1"] for row in categories[:TOP_CATEGORIES]]
    period = bundle.get("period") or {}
    # 시행일이 run 첫날이면 사전 기간이 비어 `pre_daily_amt` 가 전부 0 이 된다.
    # 그 0 을 막대로 그리면 "정책 전에는 소비가 없었다"로 읽힌다 — 사실이 아니다.
    # 전후 비교 자체가 성립하지 않으므로 사후 한 계열만 그리고 그 사실을 적는다.
    no_pre = not period.get("pre")
    body = [note(narration, "categories")]
    if no_pre:
        body.append(
            '<div class="callout callout--warn">이 실행에는 <b>시행 전 기간이 없습니다.</b> '
            f'{esc(period.get("reason") or "시행일이 run 첫날입니다.")} '
            "따라서 아래는 전후 비교가 아니라 <b>시행 후 일평균</b>만 보여줍니다. "
            "증감률·차이·이중차분은 계산하지 않았습니다.</div>"
        )
    series = [
        {
            "name": "시행 후 일평균",
            "values": [row["post_daily_amt"] for row in categories[:TOP_CATEGORIES]],
            "color": "var(--k1)",
        }
    ]
    if not no_pre:
        series.insert(
            0,
            {
                "name": "시행 전 일평균",
                "values": [row["pre_daily_amt"] for row in categories[:TOP_CATEGORIES]],
                "color": "var(--k5)",
            },
        )
    body.append(
        figure(
            charts.grouped_bar(
                labels,
                series,
                title="업종별 시행 후 일평균 소비금액"
                if no_pre
                else "업종별 시행 전후 일평균 소비금액",
                height=330,
            ),
            (
                "<b>업종별 시행 후 일평균.</b> 시행 전 기간이 없어 비교 대상이 없습니다. "
                "여기서 큰 업종은 '정책으로 늘어난 업종'이 아니라 '금액이 큰 업종'입니다."
                if no_pre
                else "<b>업종별 시행 전/후 일평균.</b> 기간 길이가 다르므로 총액이 아니라 <b>일평균</b>으로 맞췄습니다. "
                "총액으로 비교하면 사후 기간이 길다는 이유만으로 모든 업종이 늘어난 것처럼 보입니다."
            ),
            "events.jsonl · 업종×일자 교차표",
        )
    )
    # 사전 기간이 없으면 증감률이 전부 None 이다. 빈 그림을 남기지 않는다
    if not no_pre:
        body.append(
            figure(
                charts.diverging_bar(
                    [
                        {
                            "label": row["l1"],
                            "value": row["growth_pct"],
                            "targeted": row["targeted"],
                        }
                        for row in sorted(
                            [c for c in categories if c["growth_pct"] is not None],
                            key=lambda c: -(c["growth_pct"] or 0),
                        )
                    ],
                    title="업종별 증감률",
                    formatter=lambda v: f"{v:+.1f}%",
                ),
                "<b>업종별 증감률(단순 전후비교).</b> 진한 막대가 정책 대상 업종입니다. "
                "여기에는 시장 전체의 추세가 섞여 있으므로, 정책 효과는 다음 절의 이중차분으로 판단해야 합니다.",
                "events.jsonl",
            )
        )
    body.append(
        figure(
            charts.stacked_bar(
                labels,
                [
                    {
                        "name": "정책지급",
                        "values": [row["policy_paid"] for row in categories[:TOP_CATEGORIES]],
                        "color": "var(--k3)",
                    },
                    {
                        "name": "자기부담",
                        "values": [row["self_paid"] for row in categories[:TOP_CATEGORIES]],
                        "color": "var(--k5)",
                    },
                ],
                title="업종별 지급·자기부담 구성",
                normalize=True,
                height=280,
            ),
            "<b>업종별 결제 구성.</b> 정책지급이 차지하는 비중이 높을수록 그 업종의 소비가 "
            "정책에 직접 기대고 있다는 뜻입니다.",
            "events.jsonl",
        )
    )
    rows = []
    attrs = []
    for row in categories:
        attrs.append('data-target="1"' if row["targeted"] else "")
        rows.append(
            [
                esc(row["l1"]) + (" " + TAG_TARGET if row["targeted"] else ""),
                f'<span class="num">{row["events"]:,}</span>',
                f'<span class="num">{esc(krw(row["amt"]))}</span>',
                f'<span class="num">{esc(num(row["share"], digits=1))}%</span>',
                # 사전 기간이 없으면 0 이 아니라 "없음"이다. 0 으로 적으면 뺄셈이 성립한 것처럼 보인다
                "—" if no_pre else f'<span class="num">{esc(krw(row["pre_daily_amt"]))}</span>',
                f'<span class="num">{esc(krw(row["post_daily_amt"]))}</span>',
                "—" if no_pre else _signed(row["delta_daily_amt"]),
                _signed(row["growth_pct"], formatter=lambda v: f"{abs(v):.1f}%")
                if row["growth_pct"] is not None
                else "—",
                f'<span class="num">{esc(num(row["policy_share_pct"], digits=1))}%</span>',
                f'<span class="num">{esc(num(row["avg_ticket"]))}</span>',
            ]
        )
    body.append(
        table(
            [
                ("업종", ""),
                ("건수", "n"),
                ("총 금액", "n"),
                ("구성비", "n"),
                ("시행 전 일평균", "n"),
                ("시행 후 일평균", "n"),
                ("차이", "n"),
                ("증감률", "n"),
                ("정책지급 비중", "n"),
                ("객단가", "n"),
            ],
            rows,
            row_attrs=attrs,
        )
    )
    return "".join(body)


def _sec_did(bundle: dict[str, Any], narration: dict[str, Any]) -> str:
    did = bundle.get("did")
    body = [note(narration, "did")]
    body.append(
        '<div class="formula">'
        "처치군 T = 정책 대상 업종        대조군 C = 비대상 업종\n"
        "사전 P0 = 시행일 이전            사후 P1 = 시행일 당일 이후\n"
        "모든 값은 기간 길이를 맞추기 위한 <b>일평균 금액</b>\n\n"
        "  반사실  T1* = T0 × (C1 / C0)\n"
        "  DID(절대) = T1 − T1*\n"
        "  DID(상대) = (T1/T0) − (C1/C0)"
        "</div>"
    )
    if not did:
        body.append(
            f'<div class="callout callout--warn">이중차분을 계산할 수 없습니다. '
            f'{esc((bundle.get("period") or {}).get("reason") or "사전 또는 사후 기간이 없습니다.")}</div>'
        )
        return "".join(body)

    body.append(
        figure(
            charts.slope_chart(
                treat_pre=did["treat_pre"],
                treat_post=did["treat_post"],
                control_pre=did["control_pre"],
                control_post=did["control_post"],
                counterfactual=did["counterfactual_post"],
            ),
            "<b>이중차분 슬로프 차트.</b> 회색 점선이 <b>반사실</b>(정책이 없었다면 대조군과 같은 속도로 "
            "변했을 처치군)입니다. 실제 사후값과 반사실의 세로 간격이 이중차분 추정치입니다.",
            "events.jsonl · 업종×일자 교차표",
        )
    )
    body.append(
        table(
            [("집단", ""), ("사전 일평균", "n"), ("사후 일평균", "n"), ("차이", "n"), ("성장률", "n")],
            [
                [
                    f'처치군 — 정책 대상 업종 {len(did["treat_categories"])}개',
                    f'<span class="num">{esc(krw(did["treat_pre"]))}</span>',
                    f'<span class="num">{esc(krw(did["treat_post"]))}</span>',
                    _signed(did["treat_diff"]),
                    f'<span class="num">{esc(num((did["treat_growth"] or 0) * 100, digits=1))}%</span>',
                ],
                [
                    f'대조군 — 비대상 업종 {len(did["control_categories"])}개',
                    f'<span class="num">{esc(krw(did["control_pre"]))}</span>',
                    f'<span class="num">{esc(krw(did["control_post"]))}</span>',
                    _signed(did["control_diff"]),
                    f'<span class="num">{esc(num((did["control_growth"] or 0) * 100, digits=1))}%</span>',
                ],
                [
                    "반사실 (대조군 성장률을 처치군에 적용)",
                    f'<span class="num">{esc(krw(did["treat_pre"]))}</span>',
                    f'<span class="num">{esc(krw(did["counterfactual_post"]))}</span>',
                    _signed((did["counterfactual_post"] or 0) - did["treat_pre"]),
                    f'<span class="num">{esc(num((did["control_growth"] or 0) * 100, digits=1))}%</span>',
                ],
            ],
            foot=[
                "<b>이중차분 (DID)</b>",
                "",
                "",
                f'<b>{_signed(did["did_absolute"])}</b>',
                f'<b>{_signed(did["did_pct_of_counterfactual"], formatter=lambda v: f"{abs(v):.2f}%")}</b>',
            ],
        )
    )
    body.append(
        figure(
            charts.waterfall(
                [
                    {"label": "시행 전 일평균", "value": did["treat_pre"], "absolute": True},
                    {
                        "label": "시장 전체 추세",
                        "value": (did["counterfactual_post"] or 0) - did["treat_pre"],
                    },
                    {"label": "정책 순효과 (DID)", "value": did["did_absolute"] or 0},
                    {"label": "시행 후 일평균", "value": did["treat_post"], "absolute": True},
                ],
                title="처치군 변화 분해",
                height=300,
            ),
            "<b>변화 분해.</b> 단순 전후비교("
            + krw(did["naive_before_after"])
            + ")를 <b>시장 전체 추세</b>와 <b>정책 순효과</b>로 나눴습니다. "
            f'두 값의 차이 {krw(did["bias_removed"])} 가 전후비교만 했을 때 정책 몫으로 잘못 계산되는 부분입니다.',
            "events.jsonl",
        )
    )

    study = bundle["event_study"]
    if study.get("available"):
        body.append(
            figure(
                charts.event_study_chart(study["points"], title="사전추세 검증"),
                "<b>사전추세(parallel trends) 검증.</b> 처치군과 대조군의 로그 격차를 시행일 기준 상대일로 "
                "그렸습니다. 회색 구간(시행 전)이 <b>평평할수록</b> 이중차분의 대조군 가정이 성립합니다. "
                "시행 전부터 격차가 벌어지고 있었다면 DID 추정치를 정책 효과로 읽으면 안 됩니다.",
                "events.jsonl · 상대일 정규화",
            )
        )
    else:
        body.append(
            f'<div class="callout callout--warn">사전추세를 검증하지 못했습니다. '
            f'{esc(study.get("reason") or "")} 대조군 가정이 확인되지 않은 상태이므로 '
            "이중차분 값을 인과 효과로 단정할 수 없습니다.</div>"
        )
    return "".join(body)


def _sec_did_by_category(bundle: dict[str, Any]) -> str:
    rows = bundle.get("did_by_category") or []
    if not rows:
        return '<div class="callout callout--warn">업종별 이중차분을 계산할 수 없습니다.</div>'
    body = [
        figure(
            charts.diverging_bar(
                [
                    {"label": row["l1"], "value": row["did_absolute"], "targeted": row["targeted"]}
                    for row in rows
                    if row["did_absolute"] is not None
                ],
                title="업종별 이중차분 추정치",
            ),
            "<b>어떤 업종에서 금액이 늘었는가 — 이중차분 기준.</b> 각 업종의 사후 일평균에서, "
            "대조군 성장률로 만든 그 업종의 반사실값을 뺀 값입니다. 진한 막대가 정책 대상 업종이며 "
            "<b>대상 업종들의 값을 모두 더하면 앞 절의 처치군 전체 DID 와 정확히 같습니다</b> "
            "(마지막 절 일관성 검증 <code>did_category_sum</code> 참고).",
            "events.jsonl · 업종별 반사실",
        )
    ]
    table_rows = []
    attrs = []
    for row in rows:
        attrs.append('data-target="1"' if row["targeted"] else "")
        table_rows.append(
            [
                esc(row["l1"]) + (" " + TAG_TARGET if row["targeted"] else ""),
                f'<span class="num">{esc(krw(row["pre_daily"]))}</span>',
                f'<span class="num">{esc(krw(row["post_daily"]))}</span>',
                f'<span class="num">{esc(krw(row["counterfactual_post"]))}</span>',
                _signed(row["did_absolute"]),
                _signed(row["did_pct"], formatter=lambda v: f"{abs(v):.1f}%") if row["did_pct"] is not None else "—",
                _signed(row["growth_pct"], formatter=lambda v: f"{abs(v):.1f}%")
                if row["growth_pct"] is not None
                else "—",
                f'<span class="num">{int(row["pre_events"]):,} → {int(row["post_events"]):,}</span>',
            ]
        )
    targeted_sum = sum(r["did_absolute"] for r in rows if r["targeted"] and r["did_absolute"] is not None)
    body.append(
        table(
            [
                ("업종", ""),
                ("사전 일평균", "n"),
                ("사후 일평균", "n"),
                ("반사실", "n"),
                ("DID", "n"),
                ("DID 비율", "n"),
                ("단순 증감률", "n"),
                ("건수 변화", "n"),
            ],
            table_rows,
            row_attrs=attrs,
            foot=[
                "<b>정책 대상 업종 합계</b>",
                "",
                "",
                "",
                f"<b>{_signed(targeted_sum)}</b>",
                "",
                "",
                "",
            ],
        )
    )

    # 업종 × 일자 히트맵 (사전 일평균 대비 증감률)
    daily_index = {row["day"]: row for row in bundle["daily"]}
    days = [row["day"] for row in bundle["daily"]]
    top = [row["l1"] for row in bundle["categories"][:TOP_CATEGORIES]]
    pre_days = set(bundle["period"]["pre"])
    matrix: list[list[float | None]] = []
    overlay_cat = bundle["overlay"].get("by_category", {})
    for l1 in top:
        base_row = next((r for r in bundle["categories"] if r["l1"] == l1), None)
        base = base_row["pre_daily_amt"] if base_row else 0
        line: list[float | None] = []
        for day in days:
            value = None
            cell = bundle.get("_day_l1_lookup", {}).get((day, l1))
            if cell is None:
                # 히트맵은 일자별 합계에서 다시 만든다 (원장과 같은 값)
                cell = _lookup_day_l1(bundle, day, l1)
            if base and cell is not None:
                value = (cell / base - 1) * 100
            line.append(value)
        matrix.append(line)
    if matrix and any(any(v is not None for v in row) for row in matrix):
        body.append(
            figure(
                charts.heatmap(top, days, matrix, title="업종×일자 증감 히트맵"),
                "<b>업종 × 일자 히트맵.</b> 각 칸은 그 업종의 <b>사전 일평균 대비</b> 그날의 증감률입니다. "
                "시행일 이후 특정 업종의 열이 통째로 진해지면 그 업종에서 소비가 옮겨붙었다는 신호입니다.",
                "events.jsonl · 업종×일자 교차표",
            )
        )
    if pre_days and daily_index and overlay_cat:
        pass
    return "".join(body)


def _lookup_day_l1(bundle: dict[str, Any], day: str, l1: str) -> float | None:
    """겹쳐보기 시리즈와 일자 목록으로 업종×일자 값을 되찾는다 (원장과 같은 출처)."""
    overlay = bundle.get("overlay") or {}
    if not overlay.get("available"):
        return None
    overall = overlay["overall"]
    series = overlay["by_category"].get(l1)
    if not series:
        return None
    if day in overall["pre_days"]:
        return series["pre"][overall["pre_days"].index(day)]
    if day in overall["post_days"]:
        return series["post"][overall["post_days"].index(day)]
    return None


def _sec_deciles(bundle: dict[str, Any]) -> str:
    deciles = bundle.get("deciles") or {}
    if not deciles.get("available"):
        return f'<div class="callout callout--warn">{esc(deciles.get("reason") or "분위별 데이터가 없습니다.")}</div>'
    items = deciles["items"]
    labels = [f'{item["decile"]}분위' for item in items]
    body = [
        figure(
            charts.grouped_bar(
                labels,
                [
                    {"name": "시행 전 1인당", "values": [i["per_capita_pre"] for i in items], "color": "var(--k5)"},
                    {"name": "시행 후 1인당", "values": [i["per_capita_post"] for i in items], "color": "var(--k1)"},
                ],
                title="분위별 1인당 소비",
                height=300,
            ),
            "<b>분위별 1인당 소비.</b> 지급을 받은 분위와 받지 않은 분위의 변화 폭이 다르면, "
            "정책이 의도한 계층에 도달했는지 판단할 수 있습니다.",
            "metrics/day_*.jsonl",
        ),
        figure(
            charts.diverging_bar(
                [
                    {
                        "label": f'{i["decile"]}분위',
                        "value": i["per_capita_growth_pct"],
                        "targeted": i["treated"],
                    }
                    for i in items
                    if i["per_capita_growth_pct"] is not None
                ],
                title="분위별 1인당 소비 증감률",
                formatter=lambda v: f"{v:+.1f}%",
            ),
            "<b>분위별 증감률.</b> 진한 막대가 지급 대상 분위입니다.",
            "metrics/day_*.jsonl",
        ),
    ]
    rows = []
    attrs = []
    for item in items:
        attrs.append('data-target="1"' if item["treated"] else "")
        rows.append(
            [
                f'<span class="num">{item["decile"]}분위</span> ' + (TAG_GRANT if item["treated"] else ""),
                f'<span class="num">{item["agents_post"] or item["agents_pre"]:,}</span>',
                f'<span class="num">{esc(krw(item["grant_total"]))}</span>',
                f'<span class="num">{esc(krw(item["policy_spend_total"]))}</span>',
                f'<span class="num">{esc(krw(item["per_capita_pre"]))}</span>',
                f'<span class="num">{esc(krw(item["per_capita_post"]))}</span>',
                _signed(item["per_capita_delta"]),
                _signed(item["per_capita_growth_pct"], formatter=lambda v: f"{abs(v):.1f}%")
                if item["per_capita_growth_pct"] is not None
                else "—",
            ]
        )
    body.append(
        table(
            [
                ("분위", ""),
                ("관측 인원", "n"),
                ("지급 총액", "n"),
                ("정책 소진액", "n"),
                ("시행 전 1인당", "n"),
                ("시행 후 1인당", "n"),
                ("차이", "n"),
                ("증감률", "n"),
            ],
            rows,
            row_attrs=attrs,
        )
    )
    return "".join(body)


def _sec_structure(bundle: dict[str, Any]) -> str:
    mix = bundle["mix"]
    categories = bundle["categories"]
    body = []
    donut_items = [
        {"label": "정책지급", "value": mix["policy_paid"], "color": "var(--k3)"},
        {"label": "자기부담", "value": mix["self_paid"], "color": "var(--k5)"},
    ]
    cat_donut = [
        {"label": row["l1"], "value": row["amt"]} for row in categories[:TOP_CATEGORIES]
    ]
    body.append(
        '<div class="figgrid">'
        + figure(
            charts.donut(donut_items, title="결제 구성"),
            "<b>결제 구성.</b> 전체 소비금액 중 정책이 직접 부담한 몫.",
            "events.jsonl",
        )
        + figure(
            charts.donut(cat_donut, title="업종 구성"),
            f"<b>업종 구성.</b> 상위 {len(cat_donut)}개 업종의 소비금액.",
            "events.jsonl",
        )
        + "</div>"
    )
    daytype = bundle.get("daytype") or {}
    if daytype:
        labels = list(daytype)
        body.append(
            figure(
                charts.grouped_bar(
                    labels,
                    [
                        {"name": "소비금액", "values": [daytype[k]["amt"] for k in labels], "color": "var(--k1)"},
                        {
                            "name": "정책지급",
                            "values": [daytype[k]["policy_paid"] for k in labels],
                            "color": "var(--k3)",
                        },
                    ],
                    title="요일 유형별 소비",
                    height=250,
                ),
                "<b>주중·주말 비교.</b> 사후 기간에 주말이 몇 개 더 들어갔는지에 따라 단순 전후비교가 "
                "왜곡될 수 있습니다. 이중차분은 이 왜곡을 대조군으로 상쇄합니다.",
                "events.jsonl",
            )
        )
    scatter_points = [
        {
            "x": row["events"],
            "y": row["avg_ticket"],
            "size": row["amt"] / 1e5 if row["amt"] else 0,
            "label": row["l1"],
            "color": "var(--k1)" if row["targeted"] else "var(--k5)",
        }
        for row in categories[: TOP_CATEGORIES + 4]
        if row["avg_ticket"]
    ]
    if scatter_points:
        body.append(
            figure(
                charts.scatter(
                    scatter_points,
                    x_label="이벤트 수(건)",
                    y_label="객단가(원)",
                    title="업종 위치",
                ),
                "<b>업종 위치.</b> 오른쪽일수록 자주 사고, 위쪽일수록 비싸게 삽니다. 원의 크기는 총 소비금액. "
                "진한 원이 정책 대상 업종입니다.",
                "events.jsonl",
            )
        )
    districts = bundle.get("districts") or {}
    if districts:
        top_d = sorted(districts.items(), key=lambda kv: -kv[1]["amt"])[:12]
        body.append(
            figure(
                charts.diverging_bar(
                    [{"label": name, "value": cell["amt"], "targeted": True} for name, cell in top_d],
                    title="지역별 소비금액",
                ),
                "<b>지역별 소비금액.</b> 정책 대상 지역이 지정된 경우 인접 지역의 값과 함께 보면 "
                "간접 영향(spillover)의 단서를 얻을 수 있습니다.",
                "events.jsonl",
            )
        )
    return "".join(body)


def _sec_consistency(bundle: dict[str, Any], consistency: dict[str, Any], narration: dict[str, Any]) -> str:
    counts = consistency.get("counts", {})
    tone = "ok" if consistency.get("consistent") else "danger"
    body = [
        f'<div class="callout callout--{tone}"><b>{esc(consistency.get("verdict"))}</b> — '
        f'검사 {counts.get("total", 0)}건 중 통과 {counts.get("pass", 0)} · '
        f'실패 {counts.get("fail", 0)} · 미검사 {counts.get("skip", 0)}</div>',
        note(narration, "consistency"),
    ]
    rows = []
    for check in consistency.get("checks", []):
        status = check["status"]
        rows.append(
            [
                f'<span class="num">{esc(check["id"])}</span>',
                esc(check["label"]),
                f'<span class="st st--{status}">{"통과" if status == "pass" else ("실패" if status == "fail" else "미검사")}</span>',
                f'<span class="num">{esc(num(check["expected"], digits=2)) if isinstance(check["expected"], (int, float)) else esc(check["expected"])}</span>',
                f'<span class="num">{esc(num(check["actual"], digits=2)) if isinstance(check["actual"], (int, float)) else esc(check["actual"])}</span>',
                f'<span class="num">{esc(num(check["diff"], digits=4)) if isinstance(check["diff"], (int, float)) else "—"}</span>',
                esc(check.get("note") or ""),
            ]
        )
    body.append(
        table(
            [
                ("검사 ID", ""),
                ("항등식", ""),
                ("결과", ""),
                ("기준값", "n"),
                ("비교값", "n"),
                ("차이", "n"),
                ("비고", ""),
            ],
            rows,
        )
    )
    return "".join(body)


def _sec_provenance(
    bundle: dict[str, Any],
    narration: dict[str, Any],
    provenance: str,
    run_id: str,
    source_paths: Sequence[str],
) -> str:
    meta = bundle["meta"]
    llm = narration.get("llm", {})
    links = "".join(
        f'<li><a href="/api/runs/{quote(run_id, safe="")}/artifacts/{quote(path, safe="/")}">{esc(path)}</a></li>'
        for path in source_paths
    )
    body = [
        dl(
            [
                ("생성 기록", f'<span class="num">{esc(provenance)}</span>'),
                ("계산 입력", esc(meta.get("generated_from"))),
                ("events.jsonl 필드", f'<code>{esc(", ".join(meta.get("event_keys") or []))}</code>'),
                (
                    "해설 LLM",
                    (
                        f'{esc(llm.get("provider"))} · {esc(llm.get("model") or "—")} · '
                        f'{"연결됨" if llm.get("configured") else "미설정"}'
                        + (f'<br><span class="tag">{esc(llm.get("reason"))}</span>' if llm.get("reason") else "")
                    ),
                ),
                (
                    "해설 검증",
                    (
                        f'LLM 문장 채택 {"예" if narration.get("used_llm") else "아니오"} · '
                        f'숫자 검증 거절 {len(narration.get("guard_rejected") or [])}건'
                        + (
                            f'<br><span class="tag">거절 섹션: {esc(", ".join(narration.get("guard_rejected") or []))}</span>'
                            if narration.get("guard_rejected")
                            else ""
                        )
                    ),
                ),
                ("미확인 항목", ", ".join(esc(x) for x in bundle.get("unknown") or []) or "없음"),
            ]
        )
    ]
    if links:
        body.append(f"<h3>원본 근거 (run snapshot)</h3><ul>{links}</ul>")
    body.append(
        '<div class="callout callout--info"><b>읽을 때의 한계.</b> '
        "① 이중차분은 대조군이 처치군과 같은 추세를 따랐을 것이라는 가정 위에서만 인과로 읽힙니다. "
        "사전추세 검증 그림에서 시행 전 구간이 평평하지 않다면 이 가정이 깨진 것입니다. "
        "② 이 결과는 에이전트 시뮬레이션의 산출물이며 실측 통계가 아닙니다. "
        "③ 표본이 작은 업종은 며칠의 우연으로 증감률이 크게 흔들립니다 — 건수 열을 함께 보세요. "
        "④ 해설 문장은 계산 결과에 있는 숫자만 쓰도록 검증했지만, 해석의 책임은 사람에게 있습니다.</div>"
    )
    return "".join(body)


# --------------------------------------------------------------------------- #
# 조립
# --------------------------------------------------------------------------- #

SECTION_PLAN = [
    ("s1", "1", "분석 개요", "이 보고서가 무엇을 어떤 자료로 계산했는지 먼저 고정한다."),
    ("s2", "2", "정책 사양", "시뮬레이션에 실제로 배선된 정책 파일의 내용."),
    ("s3", "3", "소비 총량 추이", "시행일을 기준으로 총액·건수·객단가가 어떻게 움직였는가."),
    ("s4", "4", "시행 전후 겹쳐보기", "같은 길이의 두 구간을 한 축에 올려 소비가 얼마나 달라졌는지 본다."),
    ("s5", "5", "업종별 전후 비교", "어떤 업종에서 금액이 늘고 줄었는가 (단순 비교)."),
    ("s6", "6", "이중차분 (DID)", "시장 전체 추세를 걷어낸 정책 순효과."),
    ("s7", "7", "업종별 이중차분", "어떤 업종에서 정책 때문에 금액이 늘었는가."),
    ("s8", "8", "분위별 효과", "지급 대상 분위에 실제로 도달했는가."),
    ("s9", "9", "소비 구조", "결제 구성·요일·업종 위치·지역."),
    ("s10", "10", "일관성 검증", "이 보고서 안의 숫자들이 서로 어긋나지 않는지 다시 계산해 대조한다."),
    ("s11", "11", "근거와 한계", "출처, 해설 생성 방식, 해석상의 한계."),
]


ALWAYS_ON = {"s1", "s10", "s11"}


def build_html(
    bundle: dict[str, Any],
    consistency: dict[str, Any],
    narration: dict[str, Any],
    *,
    policy: dict[str, Any],
    provenance: str = "",
    run_id: str = "",
    source_paths: Sequence[str] = (),
    sections: Sequence[str] | None = None,
) -> str:
    selected = ALWAYS_ON | set(sections) if sections is not None else {a for a, _, _, _ in SECTION_PLAN}
    builders = {
        "s1": lambda: _sec_overview(bundle, narration, consistency),
        "s2": lambda: _sec_policy(bundle, policy),
        "s3": lambda: _sec_timeseries(bundle),
        "s4": lambda: _sec_overlay(bundle, narration),
        "s5": lambda: _sec_categories(bundle, narration),
        "s6": lambda: _sec_did(bundle, narration),
        "s7": lambda: _sec_did_by_category(bundle),
        "s8": lambda: _sec_deciles(bundle),
        "s9": lambda: _sec_structure(bundle),
        "s10": lambda: _sec_consistency(bundle, consistency, narration),
        "s11": lambda: _sec_provenance(bundle, narration, provenance, run_id, source_paths),
    }
    plan = [item for item in SECTION_PLAN if item[0] in selected]
    # 선택된 절만 다시 번호를 매긴다 — 목차와 본문의 번호가 어긋나면 안 된다.
    numbered = [(anchor, str(index + 1), title, purpose) for index, (anchor, _, title, purpose) in enumerate(plan)]
    sections = "".join(
        section(f"{number}절", title, purpose, builders[anchor](), anchor=anchor)
        for anchor, number, title, purpose in numbered
    )
    toc = _toc([(anchor, f"{number}. {title}") for anchor, number, title, _ in numbered])
    title = f'{bundle["meta"].get("policy_name") or bundle["meta"].get("policy_id") or "정책"} 효과 분석 보고서'
    payload = json.dumps(
        {"bundle": bundle, "consistency": consistency, "narration": narration},
        ensure_ascii=False,
        default=str,
    )
    return (
        "<!DOCTYPE html>"
        '<html lang="ko"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{esc(title)}</title>"
        f"<style>{CSS}</style>"
        "</head><body>"
        '<div class="doc"><div class="wrap">'
        + _masthead(bundle, consistency)
        + toc
        + sections
        + '<div class="foot"><p>이 문서는 run snapshot 파일에서 직접 계산해 생성되었습니다. '
        "모든 그림은 인라인 SVG 이며 외부 리소스를 불러오지 않습니다.</p></div>"
        "</div></div>"
        f'<script type="application/json" id="dasol-report-data">{escape(payload)}</script>'
        f"<script>{THEME_SCRIPT}</script>"
        "</body></html>"
    )


def build_markdown(
    bundle: dict[str, Any],
    consistency: dict[str, Any],
    narration: dict[str, Any],
    *,
    policy: dict[str, Any],
) -> str:
    meta, period = bundle["meta"], bundle["period"]
    did = bundle.get("did") or {}
    lines: list[str] = [
        f'# {meta.get("policy_name") or meta.get("policy_id") or "정책"} 효과 분석 보고서',
        "",
        f'- 실행: `{meta.get("run_id")}` · 구간 {(meta.get("days") or ["—"])[0]} ~ {(meta.get("days") or ["—"])[-1]} ({meta.get("day_count")}일)',
        f'- 정책: `{meta.get("policy_id")}` · 시행일 {period.get("policy_from") or "미지정"}',
        f'- 소비 이벤트 {int(meta.get("event_rows") or 0):,}건 · 총 소비금액 {krw(bundle["totals"].get("amt"))}',
        f'- 일관성 검증: {consistency.get("verdict")}',
        "",
        "## 1. 분석 개요",
        "",
        narration["sections"].get("overview", {}).get("text", ""),
        "",
        "## 2. 이중차분 (DID)",
        "",
        narration["sections"].get("did", {}).get("text", ""),
        "",
    ]
    if did:
        lines += [
            "| 집단 | 사전 일평균 | 사후 일평균 | 차이 |",
            "|---|---:|---:|---:|",
            f'| 처치군(정책 대상 업종) | {did["treat_pre"]:,.0f} | {did["treat_post"]:,.0f} | {did["treat_diff"]:+,.0f} |',
            f'| 대조군(비대상 업종) | {did["control_pre"]:,.0f} | {did["control_post"]:,.0f} | {did["control_diff"]:+,.0f} |',
            f'| 반사실 | {did["treat_pre"]:,.0f} | {(did["counterfactual_post"] or 0):,.0f} | — |',
            f'| **이중차분** | | | **{(did["did_absolute"] or 0):+,.0f}** |',
            "",
        ]
    lines += ["## 3. 업종별 이중차분", "", narration["sections"].get("categories", {}).get("text", ""), ""]
    rows = bundle.get("did_by_category") or []
    if rows:
        lines += [
            "| 업종 | 대상 | 사전 일평균 | 사후 일평균 | 반사실 | DID | 단순 증감률 |",
            "|---|:--:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                f'| {row["l1"]} | {"O" if row["targeted"] else ""} | {row["pre_daily"]:,.0f} | '
                f'{row["post_daily"]:,.0f} | {(row["counterfactual_post"] or 0):,.0f} | '
                f'{(row["did_absolute"] or 0):+,.0f} | '
                f'{("%+.1f%%" % row["growth_pct"]) if row["growth_pct"] is not None else "—"} |'
            )
        lines.append("")
    lines += ["## 4. 시행 전후 겹쳐보기", "", narration["sections"].get("overlay", {}).get("text", ""), ""]
    overlay = bundle["overlay"]
    if overlay.get("available"):
        overall = overlay["overall"]
        lines += ["| 상대 일차 | 시행 전 | 시행 후 | 차이 |", "|---|---:|---:|---:|"]
        for index, label in enumerate(overall["labels"]):
            lines.append(
                f'| {label} | {overall["pre"][index]:,.0f} | {overall["post"][index]:,.0f} | '
                f'{overall["delta"][index]:+,.0f} |'
            )
        lines.append("")
    lines += ["## 5. 일관성 검증", "", narration["sections"].get("consistency", {}).get("text", ""), ""]
    lines += ["| 검사 | 결과 | 차이 |", "|---|:--:|---:|"]
    for check in consistency.get("checks", []):
        mark = {"pass": "통과", "fail": "실패", "skip": "미검사"}[check["status"]]
        diff = f'{check["diff"]:+,.4f}' if isinstance(check["diff"], (int, float)) else "—"
        lines.append(f'| {check["label"]} | {mark} | {diff} |')
    lines += [
        "",
        "## 6. 근거와 한계",
        "",
        f'- 계산 입력: `{meta.get("generated_from")}`',
        f'- 산출물 경로: `{meta.get("run_root")}`',
        f'- 해설 LLM: {narration["llm"].get("provider")} / {narration["llm"].get("model") or "—"} '
        f'({"연결됨" if narration["llm"].get("configured") else "미설정"})',
        f'- 미확인 항목: {", ".join(bundle.get("unknown") or []) or "없음"}',
        "- 이중차분은 대조군 가정 위에서만 인과로 읽을 수 있습니다. 사전추세 검증을 함께 확인하세요.",
        "",
    ]
    return "\n".join(lines)
