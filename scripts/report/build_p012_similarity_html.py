"""similarity_p012.py 의 결과를 한 장의 HTML 로 — 실측과 얼마나 닮았나.

    python scripts/report/build_p012_similarity_html.py \
        --similarity <similarity.json> \
        [--preservation preservation_check.json] [--restore-dir restore_check] \
        [--dossier-manifest on=<..> --dossier-manifest off=<..>] \
        [--cashback-manifest on=<..> --cashback-manifest off=<..>] \
        [--offsite offsite_copy.json] --out page.html [--standalone]

숫자는 모두 입력 json 에서 온다. 이 파일은 그리기만 한다 — 값을 새로 만들지 않는다.
--standalone 은 파일로 열어 볼 때를 위해 문서 머리(doctype·charset)를 붙인다. 없으면
게시 도구가 머리를 붙이는 형식(본문만)으로 쓴다.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
from pathlib import Path

E = html.escape


def load(p):
    if not p:
        return None
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def kv_args(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        out[k] = v
    return out


def pct(v, d=1, sign=True):
    if v is None:
        return "–"
    return ("%+.*f%%" if sign else "%.*f%%") % (d, v)


def num(v, d=2):
    return "–" if v is None else "%.*f" % (d, v)


def won(v):
    return "–" if v is None else "{:,.0f}원".format(v)


def plain(s):
    return re.sub(r"\*\*", "", s or "")


ARM = {"on": "지원금 있음", "off": "지원금 없음"}

CSS = """
:root{
  --bg:#F2F5F4; --surface:#FFFFFF; --ink:#15202A; --muted:#55616B; --faint:#8A949C;
  --rule:#D6DDE0; --grid:#E6EBED;
  --sim:#1F67A8; --sim-band:rgba(31,103,168,.28); --truth:#C4561C;
  --good:#2B7A4B; --good-bg:#E3F1E8; --bad:#B42318; --bad-bg:#FBE6E4;
  --unsure:#5F6B75; --unsure-bg:#ECEFF1; --bar:#1F67A8; --bar-track:#E4E9EC;
  --display:"Hahmlet", "Noto Serif KR", "Batang", serif;
  --text:"IBM Plex Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", system-ui, sans-serif;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    color-scheme:dark;
    --bg:#0E1317; --surface:#151C22; --ink:#E3E8EC; --muted:#9AA6B0; --faint:#6E7A84;
    --rule:#2A343C; --grid:#212A31;
    --sim:#62A6E6; --sim-band:rgba(98,166,230,.30); --truth:#F08B4D;
    --good:#5CBF86; --good-bg:#16301F; --bad:#F26B5F; --bad-bg:#3A1A17;
    --unsure:#A3AEB7; --unsure-bg:#232C33; --bar:#62A6E6; --bar-track:#26313A;
  }
}
:root[data-theme="dark"]{
  color-scheme:dark;
  --bg:#0E1317; --surface:#151C22; --ink:#E3E8EC; --muted:#9AA6B0; --faint:#6E7A84;
  --rule:#2A343C; --grid:#212A31;
  --sim:#62A6E6; --sim-band:rgba(98,166,230,.30); --truth:#F08B4D;
  --good:#5CBF86; --good-bg:#16301F; --bad:#F26B5F; --bad-bg:#3A1A17;
  --unsure:#A3AEB7; --unsure-bg:#232C33; --bar:#62A6E6; --bar-track:#26313A;
}
*{box-sizing:border-box}
body{background:var(--bg); color:var(--ink); font-family:var(--text); font-size:15px; line-height:1.6;
  margin:0; padding-inline:20px; padding-block:28px 56px}
.wrap{max-width:1040px; margin:0 auto; display:flex; flex-direction:column; gap:40px}
h1,h2,h3{text-wrap:balance; margin:0}
h1{font-family:var(--display); font-weight:600; font-size:clamp(26px,4.2vw,38px); line-height:1.25; letter-spacing:-.01em}
h2{font-size:20px; font-weight:600; letter-spacing:-.005em}
h3{font-size:16px; font-weight:600}
p{margin:0; max-width:68ch}
.eyebrow{font-size:12px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:500}
header{display:flex; flex-direction:column; gap:12px}
.meta{color:var(--muted); font-size:14px}
section{display:flex; flex-direction:column; gap:16px}
.lede{color:var(--muted)}
.tiles{display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px}
@media (max-width:860px){.tiles{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media (max-width:440px){.tiles{grid-template-columns:1fr}}
.tile{background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:16px 18px;
  display:flex; flex-direction:column; gap:6px}
.tile.lead{border-color:var(--sim)}
.tile .k{font-size:13px; color:var(--muted); font-weight:500}
.tile .v{font-family:var(--display); font-size:34px; font-weight:600; line-height:1.1; font-variant-numeric:tabular-nums}
.tile .ci{font-size:13px; color:var(--muted); font-variant-numeric:tabular-nums}
.tile .x{font-size:13px; color:var(--muted); line-height:1.5}
.note{background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:14px 18px; color:var(--ink)}
.note b{font-weight:600}
.legend{display:flex; flex-wrap:wrap; gap:18px; font-size:13px; color:var(--muted); align-items:center}
.legend span{display:inline-flex; gap:6px; align-items:center}
.scroll{overflow-x:auto; -webkit-overflow-scrolling:touch}
.panel{background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:12px 8px}
svg text{font-family:var(--text)}
svg .lab{fill:var(--ink); font-size:13px}
svg .sub{fill:var(--muted); font-size:11.5px}
svg .tick{fill:var(--muted); font-size:11px; font-variant-numeric:tabular-nums}
svg .val{fill:var(--ink); font-size:12px; font-variant-numeric:tabular-nums}
svg .grid{stroke:var(--grid); stroke-width:1}
svg .zero{stroke:var(--faint); stroke-width:1.2}
svg .ci{stroke:var(--sim); stroke-width:2; stroke-linecap:round}
svg .simdot{fill:var(--sim); stroke:var(--surface); stroke-width:2}
svg .truth{fill:var(--truth); stroke:var(--surface); stroke-width:2}
svg .slope{stroke:var(--faint); stroke-width:1.5; fill:none}
svg .slope.far{stroke:var(--truth)}
svg .rowhit{fill:transparent}
svg .rowhit:hover{fill:var(--grid)}
table{border-collapse:collapse; width:100%; font-size:14px; font-variant-numeric:tabular-nums}
th,td{text-align:left; padding:9px 10px; border-bottom:1px solid var(--rule); vertical-align:top}
th{font-size:12.5px; color:var(--muted); font-weight:500; white-space:nowrap}
td.n{text-align:right; white-space:nowrap}
td .s{display:block; font-size:12.5px; color:var(--muted)}
.bar{display:flex; align-items:center; gap:8px; min-width:120px}
.bar i{display:block; height:8px; border-radius:4px; background:var(--bar-track); flex:1; position:relative; overflow:hidden}
.bar i b{position:absolute; inset:0 auto 0 0; background:var(--bar); border-radius:4px}
.chip{display:inline-flex; align-items:center; gap:5px; padding:2px 9px; border-radius:999px; font-size:12.5px; font-weight:500; white-space:nowrap}
.chip.good{background:var(--good-bg); color:var(--good)}
.chip.bad{background:var(--bad-bg); color:var(--bad)}
.chip.unsure{background:var(--unsure-bg); color:var(--unsure)}
.chip::before{font-size:11px}
.chip.good::before{content:"●"}
.chip.bad::before{content:"✕"}
.chip.unsure::before{content:"○"}
.grid2{display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}
.card{background:var(--surface); border:1px solid var(--rule); border-radius:10px; padding:16px 18px; display:flex; flex-direction:column; gap:10px}
.card .big{font-family:var(--display); font-size:26px; font-weight:600; font-variant-numeric:tabular-nums}
ul.plain{margin:0; padding-left:1.1em; display:flex; flex-direction:column; gap:6px; max-width:72ch}
dl.method{display:grid; grid-template-columns:minmax(120px,180px) 1fr; gap:10px 18px; margin:0}
dl.method dt{font-weight:600}
dl.method dd{margin:0; color:var(--ink); max-width:68ch}
@media (max-width:560px){dl.method{grid-template-columns:1fr} dl.method dd{margin-bottom:6px}}
code{font-size:13px; background:var(--grid); padding:1px 5px; border-radius:4px}
footer{color:var(--muted); font-size:13px}
a{color:var(--sim)}
:focus-visible{outline:2px solid var(--sim); outline-offset:2px}
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=Hahmlet:wght@500;600&family=IBM+Plex+Sans+KR:wght@400;500;600&display=swap">')


def status_chip(x):
    if x.get("direction_certain") and x.get("direction_match"):
        return '<span class="chip good">방향 일치 · 확실</span>'
    if x.get("direction_certain") and not x.get("direction_match"):
        return '<span class="chip bad">방향 반대 · 확실</span>'
    if x.get("direction_match"):
        return '<span class="chip unsure">방향 같음 · 불확실</span>'
    return '<span class="chip unsure">방향 다름 · 불확실</span>'


def nice_step(span):
    raw = span / 6.0
    p = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * p:
            return m * p
    return 10 * p


def forest(rows, unit, title_id):
    """실측(마름모)과 시뮬(점 + 95% 구간)을 한 축에. 축 밖 값은 가장자리에 화살표와 값."""
    W, LW, RW = 780, 210, 120
    PW = W - LW - RW
    RH, TOP = 34, 30
    H = TOP + RH * len(rows) + 34
    truths = [r["truth"] for r in rows]
    sims = [r["sim"] for r in rows]
    cap = max(40.0, 2.5 * max(abs(t) for t in truths))
    hi = min(max(truths + sims + [0.0]) * 1.12 + 1, cap)
    lo = max(min(truths + sims + [0.0]) * 1.12 - 1, -0.6 * cap)
    step = nice_step(hi - lo)
    lo = math.floor(lo / step) * step
    hi = math.ceil(hi / step) * step
    X = lambda v: LW + (min(max(v, lo), hi) - lo) / (hi - lo) * PW
    out = ['<svg viewBox="0 0 %d %d" width="100%%" style="min-width:640px;max-width:%dpx" '
           'role="img" aria-labelledby="%s">' % (W, H, W, title_id)]
    t = lo
    while t <= hi + 1e-9:
        x = X(t)
        out.append('<line class="%s" x1="%.1f" y1="%d" x2="%.1f" y2="%d"/>'
                   % ("zero" if abs(t) < 1e-9 else "grid", x, TOP - 8, x, H - 30))
        out.append('<text class="tick" x="%.1f" y="%d" text-anchor="middle">%s%s</text>'
                   % (x, H - 12, ("%+g" % t) if t else "0", unit))
        t += step
    out.append('<text class="sub" x="%d" y="16">지표</text>' % 4)
    out.append('<text class="sub" x="%d" y="16">유사도</text>' % (W - RW + 18))
    for i, r in enumerate(rows):
        y = TOP + i * RH + RH / 2
        tip = ("%s %s — 실측 %s · 시뮬 %s (95%% %s ~ %s) · 유사도 %s"
               % (r["id"], r["name"], pct(r["truth"], 2), pct(r["sim"], 2),
                  pct((r.get("ci") or [None])[0], 1), pct((r.get("ci") or [None, None])[1], 1),
                  num(r.get("similarity"))))
        out.append('<g><title>%s</title><rect class="rowhit" x="0" y="%.1f" width="%d" height="%d" rx="4"/>'
                   % (E(tip), y - RH / 2, W, RH))
        out.append('<text class="lab" x="4" y="%.1f">%s</text>' % (y + 1, E(r["id"])))
        out.append('<text class="lab" x="40" y="%.1f">%s</text>' % (y + 1, E(short(r["name"]))))
        ci = r.get("ci") or [None, None]
        if ci[0] is not None and ci[1] is not None:
            x1, x2 = X(ci[0]), X(ci[1])
            out.append('<line class="ci" x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f"/>' % (x1, y + 4, x2, y + 4))
            for v, xe, d in ((ci[0], x1, -1), (ci[1], x2, 1)):
                if (v < lo and d < 0) or (v > hi and d > 0):
                    out.append('<path d="M%.1f %.1f l%d -4 l0 8 z" fill="var(--sim)"/>' % (xe + 6 * d, y + 4, -6 * d))
        tx = X(r["truth"])
        out.append('<path class="truth" d="M%.1f %.1f l6 6 l-6 6 l-6 -6 z"/>' % (tx, y - 10))
        sx = X(r["sim"])
        out.append('<circle class="simdot" cx="%.1f" cy="%.1f" r="5.5"/>' % (sx, y + 4))
        if r["sim"] > hi or r["sim"] < lo:
            out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="%s">%s</text>'
                       % (sx + (-9 if r["sim"] > hi else 9), y - 4,
                          "end" if r["sim"] > hi else "start", pct(r["sim"], 0)))
        sim_ = r.get("similarity")
        if sim_ is not None:
            bx = W - RW + 18
            out.append('<rect x="%d" y="%.1f" width="70" height="7" rx="3.5" fill="var(--bar-track)"/>' % (bx, y - 3))
            out.append('<rect x="%d" y="%.1f" width="%.1f" height="7" rx="3.5" fill="var(--bar)"/>'
                       % (bx, y - 3, 70 * max(0.0, min(1.0, sim_))))
            out.append('<text class="val" x="%d" y="%.1f">%.2f</text>' % (bx + 76, y + 4, sim_))
        out.append('</g>')
    out.append('</svg>')
    return "".join(out)


def short(name):
    return re.sub(r"\s*\((?:삼중차분, )?10월(?:, 적립)?\)", "", name)


def slope(rows):
    """실측 크기 순위 ↔ 시뮬 크기 순위. 두 칸 넘게 움직인 지표는 주황."""
    n = len(rows)
    W, RH, TOP = 640, 30, 36
    H = TOP + RH * n + 10
    lt = sorted(rows, key=lambda r: -r["truth"])
    ls = sorted(rows, key=lambda r: -r["sim"])
    pt = {r["id"]: i for i, r in enumerate(lt)}
    ps = {r["id"]: i for i, r in enumerate(ls)}
    xl, xr = 250, 390
    out = ['<svg viewBox="0 0 %d %d" width="100%%" style="min-width:560px;max-width:%dpx" role="img" '
           'aria-label="실측과 시뮬레이션의 지표 크기 순위 비교">' % (W, H, W)]
    out.append('<text class="sub" x="%d" y="18" text-anchor="end">실측 순위</text>' % (xl - 14))
    out.append('<text class="sub" x="%d" y="18">시뮬레이션 순위</text>' % (xr + 14))
    for r in rows:
        y1 = TOP + pt[r["id"]] * RH
        y2 = TOP + ps[r["id"]] * RH
        far = abs(pt[r["id"]] - ps[r["id"]]) > 2
        out.append('<g><title>%s — 실측 %d위 (%s) · 시뮬 %d위 (%s)</title>'
                   % (E(r["name"]), pt[r["id"]] + 1, pct(r["truth"], 2), ps[r["id"]] + 1, pct(r["sim"], 2)))
        out.append('<path class="slope%s" d="M%d %.1f L%d %.1f"/>' % (" far" if far else "", xl, y1, xr, y2))
        out.append('<circle cx="%d" cy="%.1f" r="4" fill="var(--truth)"/>' % (xl, y1))
        out.append('<circle cx="%d" cy="%.1f" r="4" fill="var(--sim)"/>' % (xr, y2))
        out.append('</g>')
    for i, r in enumerate(lt):
        y = TOP + i * RH
        out.append('<text class="lab" x="%d" y="%.1f" text-anchor="end">%s <tspan class="sub">%s</tspan></text>'
                   % (xl - 14, y + 4, E(short(r["name"])), pct(r["truth"], 1)))
    for i, r in enumerate(ls):
        y = TOP + i * RH
        out.append('<text class="lab" x="%d" y="%.1f">%s <tspan class="sub">%s</tspan></text>'
                   % (xr + 14, y + 4, E(short(r["name"])), pct(r["sim"], 1)))
    out.append('</svg>')
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--similarity", required=True)
    ap.add_argument("--preservation", default="")
    ap.add_argument("--restore-dir", default="")
    ap.add_argument("--dossier-manifest", action="append", default=[], help="on=경로 / off=경로")
    ap.add_argument("--cashback-manifest", action="append", default=[], help="on=경로 / off=경로")
    ap.add_argument("--offsite", default="", help="서버 밖 사본 확인 결과 json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--standalone", action="store_true")
    a = ap.parse_args()

    s = load(a.similarity)
    if not s:
        raise SystemExit("유사도 json 을 읽을 수 없다: %s" % a.similarity)
    h = s["headline"]
    ind = s["indicators"]
    rk = s["rank_indicators"]
    rows = {r["id"]: r for r in s.get("score_rows") or []}
    days = s.get("window_days")

    P: list[str] = []
    w = P.append
    if a.standalone:
        w('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">')
    w("<title>상생소비지원금 실측 대조</title>")
    w(FONTS)
    w("<style>%s</style>" % CSS)
    if a.standalone:
        w("</head><body>")
    w('<div class="wrap">')

    # 머리
    w("<header>")
    w('<div class="eyebrow">P012 상생소비지원금 · KDI 정책연구(2022.9) 실측과 대조</div>')
    w("<h1>시뮬레이션은 실제 정책 효과와 얼마나 닮았나</h1>")
    w('<p class="meta">서울 시민 에이전트 <b>{:,}명</b> · 시뮬레이션 날짜 {} ({}일) · 같은 사람을 지원금이 있을 때와 '
      '없을 때 두 번 돌려 차이를 쟀다. 실측은 KDI 가 2021년 10월 한 달 카드 결제로 잰 값이다.</p>'
      .format(s["n_agents"], E(str(s.get("window") or "-")), days or "-"))
    w("</header>")

    # 요약 숫자
    msim, mci = h["magnitude_similarity"], h["magnitude_similarity_ci"]
    w('<section aria-labelledby="h-sum"><h2 id="h-sum">요약</h2>')
    w('<div class="tiles">')
    w('<div class="tile lead"><div class="k">크기 유사도</div><div class="v">%d%%</div>'
      '<div class="ci">95%% 구간 %s ~ %s</div>'
      '<div class="x">효과 지표 %d개에서 시뮬레이션 값이 실측 값에 얼마나 가까운지. 100%%면 같고, 부호가 반대면 0%%.</div></div>'
      % (round(100 * msim), pct(100 * mci[0], 0, False) if mci[0] is not None else "–",
         pct(100 * mci[1], 0, False) if mci[1] is not None else "–", s["n_effect"]))
    w('<div class="tile"><div class="k">방향 일치</div><div class="v">%d / %d</div>'
      '<div class="ci">그중 확실한 일치 %d · 확실한 반대 %d</div>'
      '<div class="x">지원금이 소비를 늘렸는지 줄였는지, 어느 업종이 더 늘었는지의 방향이 실측과 같은 지표 수.</div></div>'
      % (h["direction_matched"], s["n_direction"], h["direction_certain_matched"],
         h["direction_certain_mismatched"]))
    rc, rci = h["rank_correlation"], h["rank_correlation_ci"]
    w('<div class="tile"><div class="k">순위 상관</div><div class="v">%s</div>'
      '<div class="ci">95%% 구간 %s ~ %s · 우연일 확률 %s</div>'
      '<div class="x">어느 업종이 더 크게 반응했는지의 순서가 실측과 닮은 정도. 1이면 순서가 같고 0이면 무관.</div></div>'
      % (num(rc), num(rci[0]), num(rci[1]),
         ("%.1f%%" % (100 * h["rank_correlation_p"])) if h.get("rank_correlation_p") is not None else "–"))
    w('<div class="tile"><div class="k">평균 절대 오차</div><div class="v">%s<span style="font-size:18px">%%p</span></div>'
      '<div class="ci">95%% 구간 %s ~ %s%%p</div>'
      '<div class="x">효과 지표에서 시뮬레이션과 실측의 차이를 평균낸 값. 작을수록 가깝다.</div></div>'
      % (num(h["mae_pp"], 1), num(h["mae_pp_ci"][0], 1), num(h["mae_pp_ci"][1], 1)))
    w("</div>")
    w('<div class="note"><b>읽는 법.</b> 구간은 같은 시민들을 다시 뽑아 {:,}번 다시 계산했을 때 값이 움직인 범위다. '
      '구간이 넓으면 그 숫자는 표본 크기에 따라 크게 달라질 수 있다는 뜻이다. '
      '실측이 시뮬레이션의 95% 구간 안에 든 지표는 {}개 중 <b>{}개</b>다.</div>'
      .format(s["boot"], h["truth_in_ci_of"], h["truth_in_ci"]))
    w('<div class="note"><b>시뮬레이션 기간의 한계.</b> 실측은 10월 한 달 누적이고 시뮬레이션은 {}일이다. '
      '이것은 시뮬레이션 시간의 물리적 한계다. 같은 조건의 앞선 시뮬레이션에서 총소비 증가율은 7일 +18.9%, '
      '14일 +15.5%, 31일 +11.3%로 기간이 짧을수록 크게 나왔다. 그래서 크기는 실측보다 크게 나올 수 있고, '
      '방향과 순위가 이 기간에서 더 믿을 만한 비교다. 값을 기간에 맞춰 고치지 않았다.</div>'.format(days or "-"))
    w("</section>")

    # 지표별 그림
    w('<section aria-labelledby="h-forest"><h2 id="h-forest">지표별 비교 — 지원금이 없을 때보다 몇 % 더 썼나</h2>')
    w('<div class="legend"><span><svg width="14" height="14" aria-hidden="true"><path d="M7 1 l6 6 l-6 6 l-6 -6 z" fill="var(--truth)"/></svg>실측 (KDI)</span>'
      '<span><svg width="30" height="14" aria-hidden="true"><line x1="2" y1="7" x2="28" y2="7" stroke="var(--sim)" stroke-width="2"/>'
      '<circle cx="15" cy="7" r="5" fill="var(--sim)"/></svg>시뮬레이션과 95% 구간</span>'
      '<span>오른쪽 막대: 지표별 크기 유사도 (0~1)</span></div>')
    w('<div class="panel scroll">%s</div>' % forest(ind, "%", "h-forest"))
    if rk:
        w("<h3>어느 쪽이 더 늘었나 — 두 증가율의 차이 (%p)</h3>")
        rk_rows = [{"id": r["id"], "name": r["name"], "truth": r["truth_gap"], "sim": r["sim_gap"],
                    "ci": r.get("ci"), "similarity": None} for r in rk]
        w('<div class="panel scroll">%s</div>' % forest(rk_rows, "%p", "h-forest"))
    w("</section>")

    # 순위
    w('<section aria-labelledby="h-rank"><h2 id="h-rank">어느 업종이 더 크게 반응했나 — 순서 비교</h2>')
    w('<p class="lede">왼쪽은 실측에서 증가율이 큰 순서, 오른쪽은 시뮬레이션에서 큰 순서다. 선이 평행에 가까울수록 '
      '순서가 닮았다. 세 칸 이상 자리를 옮긴 지표는 주황 선이다. 순위 상관 {} (우연히 이만큼 닮을 확률 {}).</p>'
      .format(num(rc), ("%.1f%%" % (100 * h["rank_correlation_p"])) if h.get("rank_correlation_p") is not None else "–"))
    w('<div class="panel scroll">%s</div>' % slope(ind))
    w("</section>")

    # 표
    w('<section aria-labelledby="h-table"><h2 id="h-table">지표별 수치</h2>')
    w('<div class="scroll"><table><thead><tr><th>지표</th><th class="n">실측</th><th class="n">시뮬레이션</th>'
      '<th class="n">95% 구간</th><th>크기 유사도</th><th class="n">부호 확실성</th><th>판정</th></tr></thead><tbody>')
    for x in ind:
        ci = x.get("ci") or [None, None]
        scj = x.get("similarity_ci") or [None, None]
        w('<tr><td><b>%s</b> %s<span class="s">%s</span></td><td class="n">%s</td><td class="n">%s</td>'
          '<td class="n">%s ~ %s</td><td><div class="bar"><i><b style="width:%.0f%%"></b></i>%s</div>'
          '<span class="s">구간 %s ~ %s</span></td><td class="n">%s</td><td>%s</td></tr>'
          % (E(x["id"]), E(short(x["name"])),
             "실측이 통계적으로 유의" if x.get("truth_significant") else "실측도 유의하지 않음",
             pct(x["truth"], 2), pct(x["sim"], 2), pct(ci[0], 1), pct(ci[1], 1),
             100 * max(0, min(1, x["similarity"])), num(x["similarity"]),
             num(scj[0]), num(scj[1]),
             ("%.1f%%" % (100 * x["sign_conf"])) if x.get("sign_conf") is not None else "–",
             status_chip(x)))
    for x in rk:
        ci = x.get("ci") or [None, None]
        w('<tr><td><b>%s</b> %s<span class="s">두 증가율의 차이</span></td><td class="n">%s</td><td class="n">%s</td>'
          '<td class="n">%s ~ %s</td><td><span class="s">방향만 본다</span></td><td class="n">%s</td><td>%s</td></tr>'
          % (E(x["id"]), E(x["name"]), pct(x["truth_gap"], 2).replace("%", "%p"),
             pct(x["sim_gap"], 2).replace("%", "%p"), pct(ci[0], 1).replace("%", "%p"),
             pct(ci[1], 1).replace("%", "%p"),
             ("%.1f%%" % (100 * x["sign_conf"])) if x.get("sign_conf") is not None else "–",
             status_chip(x)))
    w("</tbody></table></div>")
    w('<p class="lede">부호 확실성은 시민을 다시 뽑아 계산했을 때 부호가 그대로인 비율이다. 97.5% 이상이면 '
      '“확실”로 표시한다. 그 미만이면 방향이 맞아도 우연일 수 있다.</p>')
    w("</section>")

    # 수준·이질성
    w('<section aria-labelledby="h-level"><h2 id="h-level">금액과 집단별 반응</h2><div class="grid2">')
    k13 = rows.get("K13")
    if k13 and k13.get("sim") is not None:
        tw = k13.get("truth_window")
        ratio = k13.get("ratio")
        lsim = (min(ratio, 1 / ratio) if ratio and ratio > 0 else None)
        w('<div class="card"><h3>K13 1인당 캐시백</h3><div class="big">%s</div>'
          '<p>실측 {:,}원은 10·11월 두 달 합이다. 한 달 평균 {:,.0f}원을 시뮬레이션 {}일로 나눈 <b>{}</b>과 비교한다. '
          '시뮬레이션은 그 <b>%s배</b>다 (유사도 %s).</p></div>'
          .format(int(k13["truth"]), k13.get("truth_month") or 0, k13.get("window_days") or days, won(tw))
          % (won(k13["sim"]), num(ratio), num(lsim)))
    k15 = rows.get("K15")
    if k15 and k15.get("sim") is not None:
        r15 = (k15["sim"] / k15["truth"]) if k15.get("truth") else None
        w('<div class="card"><h3>K15 투입 재원 대비 소비 증가</h3><div class="big">%s</div>'
          '<p>캐시백 1원이 소비를 얼마나 늘렸는지. 실측 %s. 시뮬레이션은 실측의 <b>%s배</b>다. %s</p></div>'
          % (pct(k15["sim"], 0, False), pct(k15["truth"], 0, False), num(r15), E(plain(k15.get("note")))))
    k17 = rows.get("K17")
    if k17 and k17.get("cells"):
        want = {"가전·가구": "고", "학원": "고", "요식": "저", "유통": "저"}
        trs = "".join(
            "<tr><td>%s</td><td class='n'>%s</td><td class='n'>%s</td><td>%s 쪽이 커야</td></tr>"
            % (E(k), pct(v[0], 1), pct(v[1], 1), "소비 기준액 상위" if want.get(k) == "고" else "하위")
            for k, v in k17["cells"].items())
        w('<div class="card"><h3>K17 소득 수준별 업종 반응</h3>'
          '<p>소비 기준액 중위값으로 나눈 상위·하위 집단의 증가율. 실측은 고소득은 가전·가구·학원, 저소득은 요식·유통에서 컸다. '
          '<b>방향 %d/%d칸 일치.</b></p><div class="scroll"><table><thead><tr><th>업종</th><th class="n">상위</th>'
          '<th class="n">하위</th><th>실측</th></tr></thead><tbody>%s</tbody></table></div></div>'
          % (k17.get("dir_ok", 0), k17.get("dir_total", 0), trs))
    k18 = rows.get("K18")
    if k18 and k18.get("gap") is not None:
        w('<div class="card"><h3>K18 가구 규모별 반응</h3><div class="big">%s</div>'
          '<p>가구 구성으로 인원이 분명한 사람만 썼다: 3인 이상 %d명 %s, 1~2인 %d명 %s (제외 %d명). '
          '실측은 4인 이상이 1~3인보다 컸다. 부호 확실성 %s.</p></div>'
          % (pct(k18["gap"], 1).replace("%", "%p"), k18["n_large"], pct(k18["large_pct"], 1),
             k18["n_small"], pct(k18["small_pct"], 1), k18["n_excluded"],
             ("%.1f%%" % (100 * k18["sign_conf"])) if k18.get("sign_conf") is not None else "–"))
    k19 = rows.get("K19")
    if k19 and k19.get("spread") is not None:
        even = k19.get("p") is not None and k19["p"] >= 0.05
        w('<div class="card"><h3>K19 지역별 고른 정도</h3><div class="big">%s</div>'
          '<p>자치구 %d개의 증가율 퍼짐 %.1f%%p. 시민을 무작위로 나눠도 95%%가 %.1f%%p 안에 들어간다(p=%.2f). '
          '실측은 “지역별로 고르게” 나타났다.</p></div>'
          % ('<span class="chip good">고르다 · 실측과 같음</span>' if even else '<span class="chip bad">고르지 않다</span>',
             k19["n_gu"], k19["spread"], k19.get("null_p95") or 0, k19.get("p") or 0))
    w("</div></section>")

    # 맞대지 못한 것
    miss = [r for r in (s.get("score_rows") or [])
            if r.get("status") in ("해당없음", "관측없음", "출력부족", "미구현")
            or r.get("id") in ("K14", "K20")]
    if miss:
        w('<section aria-labelledby="h-miss"><h2 id="h-miss">비교하지 못한 지표</h2><ul class="plain">')
        for r in miss:
            why = r.get("why") or plain(r.get("note")) or r.get("status")
            w("<li><b>%s</b> %s — %s</li>" % (E(r["id"]), E(r.get("name") or ""), E(plain(why))))
        w("</ul></section>")

    # 보존
    pres = load(a.preservation)
    dman = {k: load(v) for k, v in kv_args(a.dossier_manifest).items()}
    cman = {k: load(v) for k, v in kv_args(a.cashback_manifest).items()}
    off = load(a.offsite)
    if pres or any(dman.values()):
        w('<section aria-labelledby="h-keep"><h2 id="h-keep">시뮬레이션 결과 보존</h2>')
        w('<p class="lede">그래프, 에이전트별 기억·계획·지출, 결제 원장을 시뮬레이션마다 따로 저장하고 검사로 확인했다.</p>')
        w('<div class="scroll"><table><thead><tr><th>시뮬레이션</th><th>그래프 덤프</th><th>에이전트 기억·계획·상태</th>'
          '<th>업종 결제 원장</th><th>캐시백 원장</th><th>복원 대조</th></tr></thead><tbody>')
        for arm in ("on", "off"):
            pa = ((pres or {}).get("arms") or {}).get(arm) or {}
            def cell(k):
                c = pa.get(k)
                if not c:
                    return "–"
                return ('<span class="chip %s">%s</span><span class="s">%s</span>'
                        % ("good" if c["ok"] else "bad", "통과" if c["ok"] else "실패",
                           E(" · ".join(c.get("notes") or [])[:160])))
            dm = dman.get(arm) or {}
            tot = dm.get("totals") or {}
            mem = ("기억 {:,} · 계획 항목 {:,} · 상태 {:,}".format(tot.get("memories", 0), tot.get("plan_items", 0),
                                                          tot.get("states", 0)) if tot else "")
            rest = load(str(Path(a.restore_dir) / ("%s.json" % arm))) if a.restore_dir else None
            rcell = ("–" if not rest else '<span class="chip %s">%s</span><span class="s">%d명 표본 · 불일치 %d명</span>'
                     % ("good" if rest.get("ok") else "bad", "일치" if rest.get("ok") else "불일치",
                        len(rest.get("sampled") or []), len(rest.get("mismatch") or [])))
            mcell = cell("메모리·인터뷰")
            if mem:
                mcell += '<span class="s">%s</span>' % E(mem)
            cb = cman.get(arm) or {}
            ccell = cell("캐시백 원장")
            if cb.get("no_anchor_citizens"):
                ccell += '<span class="s">소비 기준액 없는 %d명은 캐시백 0으로 포함</span>' % len(cb["no_anchor_citizens"])
            w("<tr><td><b>%s</b></td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
              % (ARM[arm], cell("그래프"), mcell, cell("업종 원장"), ccell, rcell))
        w("</tbody></table></div>")
        if off:
            w('<div class="note"><b>서버 밖 사본.</b> %s 에 %d개 파일 %.1fGB 를 복사했고, 모든 파일의 SHA-256 이 서버 값과 '
              '%s. 확인 시각 %s.</div>'
              % (E(off.get("location", "")), off.get("files", 0), off.get("bytes", 0) / 1e9,
                 "일치한다" if off.get("verified") else "<b>일치하지 않는다</b>", E(off.get("at", ""))))
        w("</section>")

    # 계산법
    w('<section aria-labelledby="h-how"><h2 id="h-how">계산 방법</h2><dl class="method">')
    w("<dt>비교 설계</dt><dd>같은 3,000명을 같은 날짜(2021년 10월 1일~7일)로 두 번 시뮬레이션했다. 한 번은 상생소비지원금이 "
      "있고 한 번은 없다. 두 결과의 차이가 지원금의 효과다. 같은 사람·같은 날짜라 날씨·요일·명절 같은 차이가 섞이지 않는다.</dd>")
    w("<dt>크기 유사도</dt><dd>지표마다 <code>1 − |시뮬 − 실측| ÷ (|시뮬| + |실측|)</code>를 구해 평균했다. "
      "같으면 1, 한쪽이 다른 쪽의 절반이면 0.67, 부호가 반대면 0이다. 단위가 없어 업종끼리 평균낼 수 있다. "
      "대칭 평균 백분율 오차(sMAPE)를 0~1 유사도로 바꾼 값이다(1 − sMAPE/2).</dd>")
    w("<dt>방향 일치</dt><dd>증가율 지표 %d개와, 어느 쪽이 더 늘었는지를 보는 순위 지표 %d개에서 부호가 실측과 같은 지표 수.</dd>"
      % (s["n_effect"], len(rk)))
    w("<dt>순위 상관</dt><dd>증가율 지표 %d개를 실측 크기 순과 시뮬레이션 크기 순으로 줄 세운 뒤 스피어만 상관을 구했다. "
      "‘우연일 확률’은 가능한 모든 줄 세우기(%d!가지) 중 이만큼 이상 닮은 비율이다.</dd>" % (s["n_effect"], s["n_effect"]))
    w("<dt>구간</dt><dd>시민을 복원 추출로 다시 뽑되 한 사람의 ‘지원금 있음·없음’ 결과는 함께 뽑아 %s번 다시 계산했다. "
      "채점과 같은 방식이다.</dd>" % "{:,}".format(s["boot"]))
    w("<dt>실측 값</dt><dd>KDI 「상생소비지원금의 소비 진작 효과」(2022.9)의 수치를 지표 계약서에 원문 인용과 함께 고정해 두고 "
      "그 파일에서만 읽었다. 시뮬레이션 프롬프트에는 실측 수치나 방향을 넣지 않았다.</dd>")
    w("<dt>원 자료</dt><dd>모든 수치는 두 시뮬레이션의 결제 원장(시민 × 날짜 행)에서 계산했다. 원장과 그래프 덤프, 에이전트별 "
      "기억은 보존돼 있어 같은 계산을 다시 할 수 있다.</dd>")
    w("</dl></section>")
    w('<footer>생성: scripts/report/similarity_p012.py → build_p012_similarity_html.py · 시민 {:,}명 · 재표집 {:,}번</footer>'
      .format(s["n_agents"], s["boot"]))
    w("</div>")
    if a.standalone:
        w("</body></html>")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(P) + "\n")
    print("→ %s (%.0f KB)" % (out, out.stat().st_size / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
