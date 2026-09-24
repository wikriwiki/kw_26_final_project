"""후보 6 (v5self) — `daily_propensity` 가 **사람을 타는가**.

사전등록: `experiments/plan_channel/prereg_v5self.md`

    python scripts/sim/s1_propensity_probe.py --out DIR --build --n 120
    python scripts/sim/s1_propensity_probe.py --out DIR --responses DIR/seed_X.jsonl

## 왜 이 지표인가

앞선 후보 넷은 **이벤트 구성**(내구재 건수·이벤트 수)으로 판정했다. 그런데 총액으로
가는 길은 `_anchor_total = spend_today(p, daily) × basket` 이라 **p 가 곧 총액**이다.
아무도 p 자체를 재지 않았다.

재 보니 p 는 정책에 반응한다(원값 +5.72% · 133:45 · p=0.0000). 그런데 **사람을
안 탄다**:

    원값 vs 그 사람의 중심값   r = -0.155
    고유값 17개 · 0.68 이 800칸 중 383칸(48%)
    클램프가 위로 올린 칸 30% — 그 메움이 정책 반응까지 함께 누른다(38%)

## 판정 (사전등록에서 옮김 — 여기서 바꾸지 않는다)

    1차  r >= 0.30  **그리고** 고유값이 17개보다 는다
    2차  1차를 통과했을 때만 본다
    받아쓰기 방어: 고유값이 분위 개수(10)보다 많아야 한다. 주어진 분위를
                   그대로 베끼면 r 은 오르지만 그건 반응이 아니다.

셀 구성은 `s1_ownership_probe.build` 를 **그대로 쓴다** — 두 탐침이 다른 맥락을
보면 비교가 안 된다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics as st
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from s1_ownership_probe import build  # noqa: E402  (셀 구성 공유)


def parse_propensity(raw: str) -> float | None:
    """최상위 `daily_propensity`. events 안의 것이 아니라 최상위를 집는다."""
    if not raw:
        return None
    m = re.search(r'"daily_propensity"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None


def centers(aids: list) -> dict:
    """{aid: 그 사람의 성향 중심값}. 그래프에서 소득계층·평소 지출을 읽는다."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
    from _common import driver_session  # noqa: E402
    from consumption import propensity_center  # noqa: E402

    out = {}
    with driver_session() as s:
        q = ("MATCH (a:Agent) WHERE a.id IN $ids "
             "RETURN a.id AS aid, a.income AS inc, a.s_daily_wd AS wd")
        for r in s.run(q, ids=list(aids)):
            try:
                out[r["aid"]] = float(propensity_center(
                    r["inc"], None, r["wd"], None))
            except Exception:
                continue
    return out


def pearson(xs: list, ys: list) -> float:
    if len(xs) < 3:
        return 0.0
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return (num / den) if den else 0.0


def sign_p(up: int, dn: int) -> float:
    n = up + dn
    if n == 0:
        return 1.0
    k = min(up, dn)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def shape(vals: list) -> dict:
    """흩어짐·고유값·최빈 비중. **동점을 숨기지 않는다.**"""
    if not vals:
        return {"n": 0}
    from collections import Counter
    c = Counter(vals)
    top, top_n = c.most_common(1)[0]
    return {"n": len(vals), "mean": st.mean(vals), "sd": st.pstdev(vals),
            "uniq": len(c), "modal": top, "modal_share": top_n / len(vals)}


def report(rows: list) -> int:
    by = {}
    for r in rows:
        if r.get("error"):
            continue
        p = parse_propensity(r.get("raw") or "")
        if p is None:
            continue
        key = (r.get("aid"), r.get("case"), r.get("date"), r.get("seed"))
        by.setdefault(key, {})[r.get("variant")] = p

    pairs = [(v["v5"], v["v5self"]) for v in by.values()
             if "v5" in v and "v5self" in v]
    print("# 후보 6 (v5self) — 소비성향이 사람을 타는가")
    print()
    print("  쌍 %d  (응답 %d · 파싱 실패 %d)"
          % (len(pairs), len(rows), sum(1 for r in rows if not r.get("error")
                                        and parse_propensity(r.get("raw") or "") is None)))
    if not pairs:
        print("  ** 쌍이 없다 — variant 표시를 확인하라")
        return 1

    cen = {}
    try:
        cen = centers(sorted({k[0] for k in by}))
    except Exception as e:
        print("  (중심값 조회 실패: %s — 1차 판정 중 r 은 건너뛴다)" % type(e).__name__)

    print()
    print("  %-10s %8s %8s %7s %7s %9s" % ("", "평균", "표준편차", "고유값", "최빈", "최빈비중"))
    out = {}
    for i, name in ((0, "v5"), (1, "v5self")):
        s = shape([p[i] for p in pairs])
        out[name] = s
        print("  %-10s %8.3f %8.4f %7d %7.2f %8.1f%%"
              % (name, s["mean"], s["sd"], s["uniq"], s["modal"], 100 * s["modal_share"]))

    print()
    if cen:
        for i, name in ((0, "v5"), (1, "v5self")):
            xs, ys = [], []
            for k, v in by.items():
                if "v5" in v and "v5self" in v and k[0] in cen:
                    xs.append((v["v5"], v["v5self"])[i])
                    ys.append(cen[k[0]])
            out[name]["r"] = pearson(xs, ys)
            print("  %-10s 중심값과의 상관 r = %+.3f  (n=%d)" % (name, out[name]["r"], len(xs)))

    up = sum(1 for a, b in pairs if b > a)
    dn = sum(1 for a, b in pairs if b < a)
    print()
    print("  쌍별 이동  늘 %d : 줄 %d : 동점 %d   양측 p=%.4f"
          % (up, dn, len(pairs) - up - dn, sign_p(up, dn)))

    print()
    print("## 등록한 1차 기준에 대면")
    r_new = out.get("v5self", {}).get("r")
    ok_r = (r_new is not None and r_new >= 0.30)
    ok_u = out["v5self"]["uniq"] > out["v5"]["uniq"]
    ok_c = out["v5self"]["uniq"] > 10
    print("  r >= 0.30                 %s" % ("통과 (%.3f)" % r_new if ok_r
                                              else "미달 (%s)" % ("없음" if r_new is None
                                                                 else "%.3f" % r_new)))
    print("  고유값이 v5 보다 는다       %s (%d -> %d)"
          % ("통과" if ok_u else "미달", out["v5"]["uniq"], out["v5self"]["uniq"]))
    print("  고유값 > 10 (받아쓰기 아님)  %s (%d)" % ("통과" if ok_c else "미달",
                                                    out["v5self"]["uniq"]))
    print()
    print("  ** %s **" % ("1차 통과 — 2차(클램프 비율)로 간다"
                          if (ok_r and ok_u and ok_c)
                          else "1차 미달 — 등록대로 여기서 멈춘다. 2·3차를 보지 않는다"))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--day", default="2021-10-21")
    ap.add_argument("--policy", default="data/neo4j_load/policies/P012.json")
    ap.add_argument("--frozen",
                    default="/data/validation_v3/pilot_registered/frozen_inputs.json")
    ap.add_argument("--responses", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.build:
        build(a.n, out, a.day, Path(a.policy), Path(a.frozen))
        return 0
    rows = [json.loads(l) for l in io.open(a.responses, encoding="utf-8") if l.strip()]
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
