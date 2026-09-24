"""반응이 **문턱 가까이에 몰려 있는가** — 부호가 아니라 모양을 본다.

    python scripts/report/threshold_response_shape.py <metrics_dir> \
        --off 2021-10-21:2021-10-22 --on 2021-10-25:2021-10-26

## 왜 모양을 보나

정책 반응의 **부호**는 이미 확인됐다(계획액 +9.76%, p=0.0015). 그러나 정답지가
재는 제도는 **문턱 제도**다 — 이번 달 실적이 2분기 월평균의 103% 를 넘겨야
비로소 적립이 시작된다. 그러면 반응은 **문턱 가까이에 몰려야** 한다.

    문턱 훨씬 아래   넘길 가망이 없다 -> 반응 작음
    문턱 언저리      조금만 더 쓰면 넘는다 -> **반응 커야 한다**
    문턱 이미 초과   넘겼으니 유인이 약해진다 -> 반응 작음

소비분위로 갈라 보니 증가 쌍 비율이 58~64% 로 **거의 균일**했다. 그것은
"정책이 있으니 조금 더" 에 가깝고 문턱 제도의 모양이 아니다. 다만 분위는
문턱 거리의 대용이 못 되므로, State 의 `sangsaeng_month_spent` 로 제대로 가른다.

## 이것이 가리는 것

    몰려 있다   프롬프트가 문턱 사실을 쓰고 있다. 더 손댈 자리가 적다
    균일하다    **계산된 사실이 안 쓰이고 있다.** 다음 후보의 표적이 여기다

## 순환이 아니다

문턱 거리는 정책 파라미터와 에이전트 자신의 누적에서 나오고, 재는 것은 계획
금액이다. 측정 공식을 프롬프트에 넣지 않는다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from math import comb


def _days(spec: str) -> list[str]:
    a, b = spec.split(":")
    return [a, b] if a != b else [a]


def load(mdir: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(mdir, "*.jsonl"))):
        day = os.path.basename(f).replace("day_", "").replace(".jsonl", "")
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("status") == "ok":
                d["_day"] = day
                rows.append(d)
    return rows


def _threshold_from_graph(ref_day: str, ratio: float) -> dict:
    """State.sangsaeng_month_spent 와 개인 앵커로 문턱 대비 위치를 만든다.

    앵커는 dawn_context._sangsaeng_monthly_anchor 를 **그대로** 쓴다 — 손으로
    다시 쓰면 런타임과 어긋난다.
    """
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "sim"))
    try:
        from neo4j import GraphDatabase
        import dawn_context as DC
    except ImportError:
        return {}
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    auth = (os.environ.get("NEO4J_USER", "neo4j"), os.environ.get("NEO4J_PASSWORD", ""))
    out = {}
    try:
        drv = GraphDatabase.driver(uri, auth=auth)
        with drv.session() as s:
            q = ("MATCH (a:Agent)-[:HAS_STATE]->(st:State) "
                 "WHERE toString(st.day) = $d AND st.sangsaeng_month_spent IS NOT NULL "
                 "RETURN a.id AS aid, toFloat(st.sangsaeng_month_spent) AS spent")
            spent = {r["aid"]: r["spent"] for r in s.run(q, d=ref_day)}
            if not spent:
                return {}
            pq = DC.PERSONA_CYPHER
            for aid, sp in spent.items():
                rec = s.run(pq, aid=aid).single()
                if not rec:
                    continue
                anchor = DC._sangsaeng_monthly_anchor(dict(rec))
                z = float(anchor) * ratio
                if z > 0:
                    out[aid] = sp / z
        drv.close()
    except Exception as e:
        print("  (그래프 조회 실패: %s)" % e)
        return {}
    return out


def sign(pairs):
    up = sum(1 for x, y in pairs if y > x)
    dn = sum(1 for x, y in pairs if y < x)
    tie = len(pairs) - up - dn
    n = up + dn
    p = min(1.0, sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n * 2) if n else 1.0
    return up, dn, tie, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_dir")
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--ratio", type=float, default=1.03)
    a = ap.parse_args()
    off_d, on_d = _days(a.off), _days(a.on)
    rows = load(a.metrics_dir)
    if not rows:
        print("metrics 가 비었다")
        return 2

    # 문턱 거리 — 정책 시작 직전(off 창 마지막 날)의 누적으로 잰다.
    ref_day = off_d[-1]
    thr = {}
    for r in rows:
        if r["_day"] != ref_day:
            continue
        spent = r.get("cm_sangsaeng_month_spent")
        anchor = r.get("cm_sangsaeng_anchor")
        if spent is None or anchor is None:
            continue
        z = float(anchor) * a.ratio
        if z > 0:
            thr[r["aid"]] = float(spent) / z      # 1.0 = 문턱

    P = lambda r: float(r.get("cm_planned_total") or 0)

    def per(days):
        acc = {}
        for r in rows:
            if r["_day"] in days:
                acc.setdefault(r["aid"], []).append(P(r))
        return {k: st.mean(v) for k, v in acc.items() if v}

    off, on = per(off_d), per(on_d)
    aids = [x for x in off if x in on]
    print("# 반응이 문턱 가까이에 몰려 있는가")
    print()
    print("  계획 %d · 쌍 %d · 문턱 거리 있는 사람 %d" % (len(rows), len(aids), len(thr)))
    if not thr:
        # metrics 에는 그 둘이 없다(run_simulation 이 안 적는다). **그래프에서 읽는다** —
        # State 노드에 sangsaeng_month_spent 가 매일 적힌다.
        thr = _threshold_from_graph(ref_day, a.ratio)
    if not thr:
        print()
        print("  **문턱 거리를 못 잰다** — metrics 에도 그래프에도 없다.")
        print("  metrics 에 cm_sangsaeng_month_spent 를 남기도록 고치는 것이 옳다.")
        return 3

    both = [x for x in aids if x in thr]
    bands = [("문턱의 60% 미만", 0.0, 0.6), ("60~85%", 0.6, 0.85),
             ("85~100% (언저리)", 0.85, 1.0), ("문턱 초과", 1.0, 9.9)]
    print()
    print("  %-18s %6s %10s %10s %8s  %s" % ("구간", "n", "정책전", "정책후", "변화", "쌍별"))
    for label, lo, hi in bands:
        S = [x for x in both if lo <= thr[x] < hi]
        if not S:
            print("  %-18s %6d  —" % (label, 0))
            continue
        x = [off[i] for i in S]
        y = [on[i] for i in S]
        up, dn, tie, p = sign(list(zip(x, y)))
        ch = (st.mean(y) - st.mean(x)) / st.mean(x) * 100 if st.mean(x) else 0
        print("  %-18s %6d %10.0f %10.0f %+7.2f%%  %d:%d:%d p=%.3f"
              % (label, len(S), st.mean(x), st.mean(y), ch, up, dn, tie, p))
    print()
    print("  읽는 법: **언저리 구간(85~100%)이 가장 커야** 문턱 제도의 모양이다.")
    print("          구간별로 고르면 '정책이 있으니 조금 더' 이지 문턱 반응이 아니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
