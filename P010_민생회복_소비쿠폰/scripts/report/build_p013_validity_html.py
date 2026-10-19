"""score_p013_two_arm.py 의 결과를 한 장의 HTML 로 — P013 시뮬레이션이 KDI 정답지와 얼마나 닮았나.

    python scripts/report/build_p013_validity_html.py --score <score_p013.json> \
        [--dossier-manifest on=<..> --dossier-manifest off=<..>] [--title-note <설명>] \
        --out page.html [--standalone]

숫자는 모두 입력 json 과 지표 계약서에서 온다. 이 파일은 그리기만 한다.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_p012_similarity_html as B  # noqa: E402  (CSS·글꼴·숲그림·서식 재사용)

E = html.escape
ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data/experiments/P013_indicator_contract.json"


def sym(s, t):
    den = abs(s) + abs(t)
    return 1.0 if den == 0 else 1.0 - abs(s - t) / den


def chip(status):
    s = status or ""
    if s.startswith("일치") or s.startswith("수준 안") or s.startswith("같은 방향"):
        return '<span class="chip good">%s</span>' % E(s)
    if s.startswith("수준 밖") or s.startswith("반대 방향"):
        return '<span class="chip bad">%s</span>' % E(s)
    if s.startswith("불일치") or s.startswith("순서 다름") or "아님" in s:
        return '<span class="chip bad">%s</span>' % E(s)
    return '<span class="chip unsure">%s</span>' % E(s or "–")


def weekly_chart(d2rows, truth_weeks):
    """주별 누적 사용처 추가 지출 / 지원금 — 시뮬(점·구간) vs 정답지 범위(막대)."""
    weeks = sorted({int(r["weeks"]) for r in d2rows} | {1, 2})
    weeks = [w for w in weeks if w <= 4]
    W, H, L, R, T, Bm = 640, 300, 56, 20, 24, 44
    vals = [x for r in d2rows for x in (r["sim"], r["ci"][0], r["ci"][1]) if x is not None]
    tv = [x for w in weeks for x in (truth_weeks.get("%d주" % w) or truth_weeks.get("%d주(5/11~)" % w) or [])]
    top = max(vals + tv + [10]) * 1.15
    step = B.nice_step(top)
    top = math.ceil(top / step) * step
    Y = lambda v: T + (H - T - Bm) * (1 - max(0, min(v, top)) / top)
    bw = (W - L - R) / len(weeks)
    out = ['<svg viewBox="0 0 %d %d" width="100%%" style="min-width:520px;max-width:%dpx" role="img" '
           'aria-label="주별 누적 사용처 추가 지출 대 지원금">' % (W, H, W)]
    v = 0
    while v <= top + 1e-9:
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (L, W - R, Y(v), Y(v)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%g%%</text>' % (L - 6, Y(v) + 4, v))
        v += step
    for i, w in enumerate(weeks):
        cx = L + bw * (i + 0.5)
        tr = truth_weeks.get("%d주" % w) or truth_weeks.get("%d주(5/11~)" % w)
        if tr:
            out.append('<g><title>정답지 %d주 누적 %.1f~%.1f%%</title><rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" rx="4" '
                       'fill="var(--truth)" opacity="0.28"/></g>' % (w, tr[0], tr[1], cx - bw * 0.3, Y(tr[1]), bw * 0.6, Y(tr[0]) - Y(tr[1])))
        r = next((r for r in d2rows if int(r["weeks"]) == w), None)
        if r:
            out.append('<g><title>시뮬 %d주 누적 %.2f%% (95%% %.2f ~ %.2f)</title>'
                       '<line class="ci" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>'
                       '<circle class="simdot" cx="%.1f" cy="%.1f" r="6"/></g>'
                       % (w, r["sim"], r["ci"][0], r["ci"][1], cx, cx, Y(r["ci"][0]), Y(r["ci"][1]), cx, Y(r["sim"])))
            out.append('<text class="val" x="%.1f" y="%.1f" text-anchor="start">%.1f%%</text>' % (cx + 10, Y(r["sim"]) + 4, r["sim"]))
        out.append('<text class="lab" x="%.1f" y="%d" text-anchor="middle">%d주</text>' % (cx, H - 22, w))
        out.append('<text class="sub" x="%.1f" y="%d" text-anchor="middle">지급 뒤 누적</text>' % (cx, H - 6))
    out.append("</svg>")
    return "".join(out)


def usage_chart(u1rows):
    """주별 누적 소진율 — 시뮬(점·구간) vs 서울 실측(마름모)·전국 실측(빈 마름모)."""
    W, H, L, R, T, Bm = 640, 280, 56, 20, 20, 44
    top = 100.0
    Y = lambda v: T + (H - T - Bm) * (1 - max(0, min(v, top)) / top)
    bw = (W - L - R) / max(1, len(u1rows))
    out = ['<svg viewBox="0 0 %d %d" width="100%%" style="min-width:420px;max-width:%dpx" role="img" '
           'aria-label="주별 누적 지원금 소진율">' % (W, H, W)]
    for v in range(0, 101, 20):
        out.append('<line class="grid" x1="%d" x2="%d" y1="%.1f" y2="%.1f"/>' % (L, W - R, Y(v), Y(v)))
        out.append('<text class="tick" x="%d" y="%.1f" text-anchor="end">%d%%</text>' % (L - 6, Y(v) + 4, v))
    for i, r in enumerate(u1rows):
        cx = L + bw * (i + 0.5)
        if r.get("truth") is not None:
            out.append('<g><title>서울 실측 %d주 누적 %.1f%%</title><path class="truth" d="M%.1f %.1f l7 7 l-7 7 l-7 -7 z"/></g>'
                       % (r["weeks"], r["truth"], cx - 18, Y(r["truth"]) - 7))
        if r.get("truth_national") is not None:
            out.append('<g><title>전국 실측 %d주 누적 %.1f%%</title><path d="M%.1f %.1f l6 6 l-6 6 l-6 -6 z" fill="none" '
                       'stroke="var(--truth)" stroke-width="1.5"/></g>' % (r["weeks"], r["truth_national"], cx + 18, Y(r["truth_national"]) - 6))
        out.append('<g><title>시뮬 %d주 누적 %.1f%% (95%% %.1f ~ %.1f)</title><line class="ci" x1="%.1f" x2="%.1f" y1="%.1f" y2="%.1f"/>'
                   '<circle class="simdot" cx="%.1f" cy="%.1f" r="6"/></g>'
                   % (r["weeks"], r["sim"], r["ci"][0], r["ci"][1], cx, cx, Y(r["ci"][0]), Y(r["ci"][1]), cx, Y(r["sim"])))
        out.append('<text class="val" x="%.1f" y="%.1f">%.1f%%</text>' % (cx + 9, Y(r["sim"]) - 8, r["sim"]))
        out.append('<text class="lab" x="%.1f" y="%d" text-anchor="middle">%d주</text>' % (cx, H - 22, r["weeks"]))
        out.append('<text class="sub" x="%.1f" y="%d" text-anchor="middle">지급 뒤 누적</text>' % (cx, H - 6))
    out.append("</svg>")
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--dossier-manifest", action="append", default=[])
    ap.add_argument("--title-note", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--standalone", action="store_true")
    a = ap.parse_args()
    s = json.loads(Path(a.score).read_text(encoding="utf-8"))
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    ind = {i["id"]: i for i in c["indicators"]}
    rows = s["rows"]
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    g = lambda k: (by.get(k) or [None])[0]
    gate = s.get("gate") or {}

    # 방향 셈: 실측이 방향을 말한 지표만
    dirs = []
    for k in ("T-total", "T-eligible", "S1", "S2", "S3", "S4", "S6"):
        r = g(k)
        if r and r.get("sim") is not None:
            dirs.append((k, r.get("name"), r["sim"] > 0, (r.get("sign_conf") or 0) >= 0.975))
    for r in by.get("D2", []):
        dirs.append(("D2-%dw" % r["weeks"], r["name"], r["sim"] > 0, (r.get("sign_conf") or 0) >= 0.975))
    d3 = g("D3")
    if d3 and d3.get("status") not in (None, "미측정"):
        dirs.append(("D3", d3.get("name"), d3["status"].startswith("일치"), (d3.get("sign_conf") or 0) >= 0.975))
    for k in ("H1", "H2"):
        r = g(k)
        if r and r.get("status") not in (None, "미측정"):
            dirs.append((k, r.get("name"), r["status"].startswith("일치"), None))
    for r in rows:
        if r["id"] in ("U2", "U3", "U4") and r.get("status", "").split(" ")[0] in ("일치", "불일치"):
            dirs.append((r["id"], r.get("name"), r["status"].startswith("일치"),
                         None if r["id"] == "U2" else True))
    matched = sum(1 for d in dirs if d[2])
    certain_m = sum(1 for d in dirs if d[2] and d[3])
    certain_x = sum(1 for d in dirs if (not d[2]) and d[3])

    sims = [sym(g(k)["sim"], ind[k]["truth"]) for k in ("S1", "S2", "S3", "S4") if g(k) and g(k).get("sim") is not None]
    d2 = by.get("D2", [])
    d2last = d2[-1] if d2 else None

    P = []
    w = P.append
    if a.standalone:
        w('<!doctype html><html lang="ko"><head><meta charset="utf-8">'
          '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">')
    w("<title>긴급재난지원금 실측 대조</title>")
    w(B.FONTS)
    w("<style>%s</style>" % B.CSS)
    if a.standalone:
        w("</head><body>")
    w('<div class="wrap"><header>')
    w('<div class="eyebrow">P013 1차 긴급재난지원금 · KDI 정책포럼 제281호(2020) 실측과 대조</div>')
    w("<h1>시뮬레이션은 2020년 긴급재난지원금의 효과와 얼마나 닮았나</h1>")
    w('<p class="meta">서울 시민 에이전트 <b>{:,}명</b> · 2020-{}~{} · 같은 사람을 지원금이 있을 때와 없을 때 같은 날짜로 '
      '두 번 돌렸다. 지원금은 실제 신청·충전 일정대로 사람마다 다른 날 받았다. {}</p>'
      .format(s["n_agents"], s["days"][0][5:], s["days"][-1][5:], E(a.title_note)))
    w("</header>")

    w('<section aria-labelledby="h-sum"><h2 id="h-sum">요약</h2><div class="tiles">')
    w('<div class="tile lead"><div class="k">방향 일치</div><div class="v">%d / %d</div>'
      '<div class="ci">확실한 일치 %d · 확실한 반대 %d</div><div class="x">실측이 방향을 말한 지표 중 시뮬레이션의 방향이 같은 수.</div></div>'
      % (matched, len(dirs), certain_m, certain_x))
    if d2last:
        tr = d2last.get("truth_range") or []
        w('<div class="tile"><div class="k">%d주 누적 추가 지출 / 지원금</div><div class="v">%.1f%%</div>'
          '<div class="ci">95%% %.1f ~ %.1f · 실측 %s</div><div class="x">사용처에서 더 쓴 금액을 배정된 지원금으로 나눈 값. 같은 기간의 정답지 값과 맞댄다.</div></div>'
          % (d2last["weeks"], d2last["sim"], d2last["ci"][0], d2last["ci"][1],
             ("%.1f~%.1f%%" % tuple(tr)) if tr else "–"))
    w('<div class="tile"><div class="k">업종 크기 유사도</div><div class="v">%d%%</div>'
      '<div class="ci">업종 묶음 %d개 평균</div><div class="x">1 − |시뮬 − 실측| ÷ (|시뮬| + |실측|). 같으면 100%%, 부호가 반대면 0%%.</div></div>'
      % (round(100 * sum(sims) / len(sims)) if sims else 0, len(sims)))
    d4 = g("D4")
    if d4:
        w('<div class="tile"><div class="k">지급 전 격차 (잡음)</div><div class="v">%+.1f%%</div>'
          '<div class="ci">95%% %+.1f ~ %+.1f</div><div class="x">지원금이 들어오기 전 두 시뮬레이션의 총지출 차이. 0 근처면 지급 뒤 차이를 지원금 효과로 읽을 수 있다.</div></div>'
          % (d4["sim"], d4["ci"][0], d4["ci"][1]))
    w("</div></section>")

    if d2:
        w('<section aria-labelledby="h-week"><h2 id="h-week">주별 누적 효과 — 정답지 그림 4와 같은 기간끼리</h2>')
        w('<div class="legend"><span><svg width="14" height="14" aria-hidden="true"><rect width="14" height="14" rx="3" fill="var(--truth)" opacity="0.4"/></svg>정답지 범위 (투입재원 11.1~15.3조원 기준)</span>'
          '<span><svg width="30" height="14" aria-hidden="true"><line x1="15" y1="1" x2="15" y2="13" stroke="var(--sim)" stroke-width="2"/><circle cx="15" cy="7" r="5" fill="var(--sim)"/></svg>시뮬레이션과 95% 구간</span></div>')
        w('<div class="panel scroll">%s</div>' % weekly_chart(d2, ind["D2"]["truth_by_weeks"]))
        w("</section>")

    srows = []
    for k in ("S1", "S2", "S3", "S4"):
        r = g(k)
        if r and r.get("sim") is not None:
            srows.append({"id": k, "name": ind[k]["name"].replace(" 매출 증대 효과", ""), "truth": ind[k]["truth"],
                          "sim": r["sim"], "ci": r.get("ci"), "similarity": sym(r["sim"], ind[k]["truth"])})
    if srows:
        w('<section aria-labelledby="h-sec"><h2 id="h-sec">업종 묶음별 효과 — 지원금이 없을 때보다 몇 % 더 썼나</h2>')
        w('<div class="legend"><span><svg width="14" height="14" aria-hidden="true"><path d="M7 1 l6 6 l-6 6 l-6 -6 z" fill="var(--truth)"/></svg>실측 (%p)</span>'
          '<span><svg width="30" height="14" aria-hidden="true"><line x1="2" y1="7" x2="28" y2="7" stroke="var(--sim)" stroke-width="2"/><circle cx="15" cy="7" r="5" fill="var(--sim)"/></svg>시뮬레이션과 95% 구간</span></div>')
        w('<div class="panel scroll">%s</div>' % B.forest(srows, "%", "h-sec"))
        w('<p class="lede">실측은 전년 동기 대비 증가율의 차이(%p)이고 시뮬레이션은 같은 사람의 지원금 없을 때 대비 증가율(%)이라 자가 조금 다르다. 방향과 순서를 먼저 본다.</p>')
        w("</section>")

    w('<section aria-labelledby="h-tab"><h2 id="h-tab">지표별 수치</h2><div class="scroll"><table><thead><tr>'
      '<th>지표</th><th class="n">시뮬레이션</th><th class="n">95% 구간</th><th class="n">실측</th><th class="n">부호 확실성</th><th>판정</th></tr></thead><tbody>')
    for r in rows:
        if r["id"] in ("S5", "H1", "H2", "U1", "U2", "U3", "U4", "H3", "R1"):
            continue
        if r.get("sim") is None and r.get("why"):
            w("<tr><td><b>%s</b> %s</td><td colspan='4'>%s</td><td>%s</td></tr>" % (E(r["id"]), E(r.get("name") or ""), E(r["why"]), chip(r.get("status"))))
            continue
        tr = r.get("truth_range") or ([r["truth"]] if r.get("truth") is not None else ([r["truth_gap"]] if r.get("truth_gap") else []))
        ci = r.get("ci") or [None, None]
        w("<tr><td><b>%s</b> %s</td><td class='n'>%s</td><td class='n'>%s</td><td class='n'>%s</td><td class='n'>%s</td><td>%s</td></tr>"
          % (E(r["id"]), E(r.get("name") or ""),
             ("{:+,.0f}원/인".format(r["sim"]) if r["id"] == "D3" else B.pct(r["sim"], 2)) if r.get("sim") is not None else "–",
             ("%s ~ %s" % (B.pct(ci[0], 1), B.pct(ci[1], 1))) if ci and ci[0] is not None else "–",
             " ~ ".join(("%g" % x) if isinstance(x, (int, float)) else str(x) for x in tr) if tr else "–",
             ("%.1f%%" % (100 * r["sign_conf"])) if r.get("sign_conf") is not None else "–", chip(r.get("status"))))
    s5 = g("S5")
    if s5:
        w("<tr><td><b>S5</b> 업종 순서</td><td class='n' colspan='3'>시뮬 %s · 실측 %s</td><td></td><td>%s</td></tr>"
          % (E(" > ".join(s5["sim_order"])), E(" > ".join(s5["truth_order"])), chip(s5["status"])))
    w("</tbody></table></div></section>")

    urows = [r for r in rows if r["id"] in ("U1", "U2", "U3", "U4", "H3", "R1")]
    if urows:
        w('<section aria-labelledby="h-use"><h2 id="h-use">지원금은 어떻게 쓰였나 — 행정안전부·서울연구원·KDI 연구 Ⅱ 실측과</h2>')
        w('<p class="lede">효과(있음 − 없음)가 아니라 지원금이 있는 시뮬레이션에서 지원금이 언제·어디에·얼마나 가까이 쓰였는지를 본다. '
          '실측은 카드사 자료이고, 수치는 원문을 직접 열어 대조했다(계약서 _sources_added_20261003).</p>')
        u1r = [r for r in urows if r["id"] == "U1" and r.get("sim") is not None]
        if u1r:
            w('<h3>U1 누적 소진율</h3><div class="legend">'
              '<span><svg width="14" height="14" aria-hidden="true"><path d="M7 0 l7 7 l-7 7 l-7 -7 z" fill="var(--truth)"/></svg>서울 실측(서울연구원, 신한카드)</span>'
              '<span><svg width="14" height="14" aria-hidden="true"><path d="M7 1 l6 6 l-6 6 l-6 -6 z" fill="none" stroke="var(--truth)" stroke-width="1.5"/></svg>전국 실측(행안부 주별 사용액)</span>'
              '<span><svg width="30" height="14" aria-hidden="true"><line x1="15" y1="1" x2="15" y2="13" stroke="var(--sim)" stroke-width="2"/><circle cx="15" cy="7" r="5" fill="var(--sim)"/></svg>시뮬레이션과 95% 구간</span></div>')
            w('<div class="panel scroll">%s</div>' % usage_chart(u1r))
        u2r = next((r for r in urows if r["id"] == "U2" and r.get("share_sim")), None)
        if u2r:
            cats = sorted(u2r["share_truth"], key=lambda c: -u2r["share_truth"][c])
            mx = max(max(u2r["share_truth"].values()), max(u2r["share_sim"].values())) or 1
            body = "".join(
                '<tr><td>%s</td><td><div class="bar"><i><b style="width:%.0f%%"></b></i>%.1f%%</div></td>'
                '<td><div class="bar"><i><b style="width:%.0f%%;background:var(--truth)"></b></i>%.1f%%</div></td></tr>'
                % (E(c), 100 * u2r["share_sim"][c] / mx, u2r["share_sim"][c], 100 * u2r["share_truth"][c] / mx, u2r["share_truth"][c])
                for c in cats)
            w('<h3>U2 지원금 사용처 구성 — 지급 뒤 %d주</h3>' % u2r["weeks"])
            w('<p class="lede">유사도 %.1f%% (95%% %.1f ~ %.1f) · 상한 %.1f%% (안경·자동차정비·서점 POI 가 그래프에 없다) · 상위 2개: 시뮬 %s / 실측 %s %s</p>'
              % (u2r["sim"], u2r["ci"][0], u2r["ci"][1], u2r["ceiling"], E("·".join(u2r["top_sim"][:2])), E("·".join(u2r["top_truth"][:2])), chip(u2r["status"])))
            w('<div class="scroll"><table><thead><tr><th>업종(행안부 분류)</th><th>시뮬레이션</th><th>실측(행안부, 같은 주)</th></tr></thead><tbody>%s</tbody></table></div>' % body)
        w('<h3>나머지 사용 지표</h3><div class="scroll"><table><thead><tr><th>지표</th><th class="n">시뮬레이션</th><th class="n">실측</th><th>판정</th></tr></thead><tbody>')
        for r in urows:
            if r["id"] in ("U1", "U2"):
                if r["id"] == "U1" and r.get("sim") is not None:
                    w("<tr><td><b>U1</b> %s</td><td class='n'>%.1f%% (%.1f ~ %.1f)</td><td class='n'>서울 %s%% · 전국 %s%%</td><td>%s</td></tr>"
                      % (E(r["name"]), r["sim"], r["ci"][0], r["ci"][1], r.get("truth"), r.get("truth_national"), chip(r["status"])))
                continue
            if r.get("sim") is None:
                w("<tr><td><b>%s</b> %s</td><td colspan='2'>%s</td><td>%s</td></tr>" % (E(r["id"]), E(r.get("name") or ""), E(r.get("why") or ""), chip(r.get("status"))))
            elif r["id"] == "U4":
                w("<tr><td><b>U4</b> %s</td><td class='n'>지원금 %.1f%% (%.1f ~ %.1f) · 일반 %.1f%%</td><td class='n'>지원금 %s%% · 일반 %s%%</td><td>%s</td></tr>"
                  % (E(r["name"]), r["sim"], r["ci"][0], r["ci"][1], r["sim_general"], r["truth"], r["truth_general"], chip(r["status"])))
            elif r["id"] == "U3":
                w("<tr><td><b>U3</b> %s</td><td class='n'>스피어만 %+.2f (%+.2f ~ %+.2f) · 업종 %d개</td><td class='n'>C1 표 8-10</td><td>%s</td></tr>"
                  % (E(r["name"]), r["sim"], r["ci"][0], r["ci"][1], r.get("n_categories", 0), chip(r["status"])))
            elif r["id"] == "H3":
                w("<tr><td><b>H3</b> %s</td><td class='n'>%+.2f%%p (65세 미만 − 이상)</td><td class='n'>0.251 vs 0.204 (현금 수급가구)</td><td>%s <span class='s'>참고 — 집계 제외</span></td></tr>"
                  % (E(r["name"]), r["sim"], chip(r["status"])))
            elif r["id"] == "R1":
                w("<tr><td><b>R1</b> %s</td><td class='n'>스피어만 %+.2f · 업종 %d개</td><td class='n'>K2 표 2-6</td><td>%s</td></tr>"
                  % (E(r["name"]), r["sim"], r.get("n_categories", 0), chip(r["status"])))
        w("</tbody></table></div></section>")

    h1 = g("H1")
    if h1 and h1.get("by_quintile"):
        q = h1["by_quintile"]
        mx = max(abs(v) for v in q.values()) or 1
        tq = ind["H1"]["truth"]
        tmx = max(tq.values())
        bars = "".join(
            '<tr><td>%s분위</td><td><div class="bar"><i><b style="width:%.0f%%"></b></i>%s원</div></td>'
            '<td><div class="bar"><i><b style="width:%.0f%%;background:var(--truth)"></b></i>%s만원</div></td></tr>'
            % (k, 100 * max(0, v) / mx, "{:+,.0f}".format(v), 100 * tq[k + "분위"] / tmx, tq[k + "분위"])
            for k, v in q.items())
        w('<section aria-labelledby="h-inc"><h2 id="h-inc">소득분위별 반응</h2>')
        w('<p class="lede">시뮬레이션은 지급 뒤 1인 추가 총지출(원), 실측은 1인가구의 지급 전후 소비 증감 차이(만원, p6). 단위가 달라 모양만 본다. '
          '정답지: “1분위(35.9만원) 가구에서 가장 크나, 소득분위에 따른 일정한 경향성은 발견되지 않음”. 판정 %s</p>' % chip(h1["status"]))
        w('<div class="scroll"><table><thead><tr><th>분위(정책 전 소비 기준액)</th><th>시뮬레이션</th><th>실측</th></tr></thead><tbody>%s</tbody></table></div></section>' % bars)

    w('<section aria-labelledby="h-miss"><h2 id="h-miss">비교하지 못한 것</h2><ul class="plain">')
    for k, why in (c.get("_not_compared") or {}).items():
        w("<li><b>%s</b> — %s</li>" % (E(k), E(why)))
    w("</ul></section>")

    dm = {}
    for item in a.dossier_manifest:
        arm, _, p = item.partition("=")
        try:
            dm[arm] = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    if dm:
        w('<section aria-labelledby="h-keep"><h2 id="h-keep">보존</h2><div class="scroll"><table><thead><tr><th>시뮬레이션</th>'
          '<th class="n">사람</th><th class="n">상태</th><th class="n">기억</th><th class="n">계획 항목</th><th class="n">빠진 날</th></tr></thead><tbody>')
        for arm, label in (("on", "지원금 있음"), ("off", "지원금 없음")):
            m = dm.get(arm)
            if m:
                t = m["totals"]
                w("<tr><td>%s</td><td class='n'>%s</td><td class='n'>%s</td><td class='n'>%s</td><td class='n'>%s</td><td class='n'>%d</td></tr>"
                  % (label, "{:,}".format(m["agents"]), "{:,}".format(t["states"]), "{:,}".format(t["memories"]),
                     "{:,}".format(t["plan_items"]), len(m.get("state_day_gaps") or [])))
        w("</tbody></table></div><p class='lede'>그래프 덤프·결제 원장·에이전트별 기억 모음은 체크섬으로 확인해 보존했다.</p></section>")

    w('<section aria-labelledby="h-how"><h2 id="h-how">계산 방법과 한계</h2><dl class="method">')
    w("<dt>비교 설계</dt><dd>같은 사람을 같은 날짜로 두 번 돌린다. 지원금이 있는 쪽과 없는 쪽의 차이가 효과다. 정답지는 합성대조법(이중차분)으로 같은 효과를 추정했다.</dd>")
    w("<dt>결제 규칙</dt><dd>카드에 충전된 지원금은 사용처에서 결제하면 자동으로 먼저 차감된다(정책브리핑 2020-05). 사용처 밖(대형마트·백화점·온라인·유흥)에는 쓸 수 없다.</dd>")
    w("<dt>지급 일정</dt><dd>정부 집계의 누적 지급률(5/17 65.7% · 5/24 92.6% · 6/2 98.6%, 신청 다음 날 사용)대로 사람마다 지급일을 정했다. 현금으로 받은 취약계층(13.2%, 5/4)은 따로 두지 않았다.</dd>")
    w("<dt>계획이 지출로</dt><dd>그날 사용처 지출 = 소비 기준액 × 사용처 몫 × (오늘 계획 ÷ 평소 계획, 0.5~2.0). 평소 계획은 정책이 없는 날들의 사람별 평일·주말 평균이다.</dd>")
    w("<dt>실측 값</dt><dd>KDI 정책포럼 제281호의 수치를 지표 계약서에 원문 인용과 함께 고정해 두고 그 파일에서만 읽었다. 시뮬레이션 프롬프트에는 실측 수치나 방향을 넣지 않았다.</dd>")
    w("</dl></section>")
    w('<footer>생성: scripts/report/score_p013_two_arm.py → build_p013_validity_html.py</footer></div>')
    if a.standalone:
        w("</body></html>")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    io.open(out, "w", encoding="utf-8", newline="\n").write("\n".join(P) + "\n")
    print("→ %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
