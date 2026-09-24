"""후보 3 — `계기:policy` 한 낱말이 **그 이벤트의 금액**을 바꾸는가.

    python scripts/sim/s2_trigger_probe.py --build --out <dir> --n 24
    python scripts/sim/s2_trigger_probe.py --out <dir> --responses <jsonl>

사전등록 `experiments/plan_channel/s2_trigger_candidate.md`

## 이 탐침의 진짜 관문

**표시된 이벤트만 움직여야 한다.** 표시 없는 이벤트까지 같이 커지면 그건 그
낱말의 효과가 아니라 프롬프트가 길어져 생긴 잡음이다. 그래서 **이벤트 단위**로
재고, 표시된 것과 안 된 것을 **같은 응답 안에서** 갈라 본다.

후보 1 에서 배운 것도 반영했다 — 그때는 이벤트를 고정해 놓고 "금액이 안
달라진다" 를 봤는데, 이번에는 **금액이 달라져야 하는 것이 가설**이므로
이벤트 고정이 옳다. 고정한 것을 묻지 않는다.

## 돌고 있는 코드를 안 건드린다

본런이 /data/repo 에서 도는 중이라 그 저장소의 stage2_poi.py 를 못 고친다.
프롬프트를 평소대로 만들고 **그 낱말만 끼워 넣는다.** 코드가 넣는 문구와
글자까지 같은지는 단위시험이 지킨다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import statistics as st
import sys
from datetime import date
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# 어느 이벤트에 계기를 붙일지 — order 기준. 나머지는 대조군이다.
# 코드는 **모든** 이벤트에 자기 trigger 를 붙인다(policy·lifestyle …).
# 탐침도 그렇게 해야 본런과 같은 프롬프트가 된다 — 시험이 이 일치를 지킨다.
# 대조는 "어디에 안 붙이나" 가 아니라 **붙은 낱말이 무엇이냐**로 만든다.
#   표시군   계기:policy    가 붙은 이벤트
#   대조군   계기:lifestyle 가 붙은 이벤트 (같은 응답 안에 있다)
# policy 쪽만 오르면 그 낱말의 효과이고, 둘 다 오르면 프롬프트가 길어진 잡음이다.
POLICY_ORDERS = (3,)
TAG = " | 계기:policy"
TRIGGERS = {0: "lifestyle", 1: "lifestyle", 2: "lifestyle",
            3: "policy", 4: "lifestyle"}


def build(n: int, out: Path, day: str) -> None:
    from neo4j import GraphDatabase
    import dawn_context as DC
    import stage2_poi as S2
    from stage1_intent import Stage1Event

    drv = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "")))
    with drv.session() as s:
        ids = [r["aid"] for r in s.run(
            "MATCH (a:Agent) WHERE a.s_daily_wd IS NOT NULL "
            "RETURN a.id AS aid ORDER BY a.id LIMIT $n", n=n)]
        rows = []
        for aid in ids:
            rec = s.run(DC.PERSONA_CYPHER, aid=aid).single()
            if rec:
                rows.append({"aid": aid, "p": dict(rec)})
    drv.close()
    today = date.fromisoformat(day)

    def events(dong: str):
        z = "zone:%s" % dong if dong else "residence"
        raw = [
            {"time": "08:10", "anchor": "residence", "category": "집",
             "sub_category": "", "intent": "기상", "trigger": "lifestyle"},
            {"time": "12:30", "anchor": z, "category": "식사",
             "sub_category": "한식", "intent": "점심", "trigger": "lifestyle"},
            {"time": "18:10", "anchor": z, "category": "마트",
             "sub_category": "슈퍼마켓", "intent": "장보기", "trigger": "lifestyle"},
            {"time": "18:50", "anchor": z, "category": "쇼핑",
             "sub_category": "가전", "intent": "드라이기 바꾸기", "trigger": "policy"},
            {"time": "22:30", "anchor": "residence", "category": "집",
             "sub_category": "", "intent": "취침", "trigger": "lifestyle"},
        ]
        return [Stage1Event(**e) for e in raw]

    cells = []
    for r in rows:
        persona = dict(r["p"])
        persona["id"] = r["aid"]
        persona["sangsaeng_active"] = True
        dong = str(persona.get("home_dong_code") or "")
        ev = events(dong)
        try:
            cands = S2.fetch_candidates_for_events(r["aid"], ev, persona, today)
        except Exception as e:
            print("  후보 실패 %s: %s" % (r["aid"], e))
            continue
        if not any(cands.values()):
            continue
        base = S2.build_stage2_prompt(ev, cands, persona=persona,
                                      state={"balance": 3_000_000})
        # **모든** 이벤트 줄에 자기 계기를 끼운다 — 코드와 같은 모양이어야 한다.
        # 코드는 intent 뒤·가격앵커 앞에 넣는다(f"...{ev.intent}{trig_s}{anchor_s}").
        # 정규식을 쓰지 않는다 — 역슬래시가 배포 과정에서 몇 번 먹혔다.
        out_lines = []
        for ln in base.split(chr(10)):
            trg = None
            for o, t in TRIGGERS.items():
                if ln.startswith("### 이벤트 %d " % o):
                    trg = t
            if trg:
                tag = " | 계기:%s" % trg
                for marker in (" | 바꾸려는 물건 시세", " | 동네 평균단가"):
                    if marker in ln:
                        ln = ln.replace(marker, tag + marker, 1)
                        break
                else:
                    ln = ln + tag
            out_lines.append(ln)
        on = chr(10).join(out_lines)
        if on == base:
            print("  삽입 실패 %s — 건너뜀" % r["aid"])
            continue
        cells.append({"aid": r["aid"], "side": "off", "user": base})
        cells.append({"aid": r["aid"], "side": "on", "user": on})
    io.open(out / "cells.json", "w", encoding="utf-8", newline=chr(10)).write(
        json.dumps({"cells": cells}, ensure_ascii=False, indent=1))
    io.open(out / "system_s2.txt", "w", encoding="utf-8", newline=chr(10)).write(S2.SYSTEM_S2)
    print("시민 %d · 칸 %d · 표시한 order %s"
          % (len({c["aid"] for c in cells}), len(cells), list(POLICY_ORDERS)))
    print("wrote", out / "cells.json")


def picks(raw: str) -> dict[int, float]:
    """pick 의 order -> actual_spent. **정규식으로 못 긁는다.**

    두 가지에 걸렸다.
      1. `policy_spend: {"P009": 5000}` 처럼 중괄호가 중첩돼 있어
         중괄호 하나만 세는 정규식이 pick 을 통째로 못 잡는다
      2. pick 의 `order` 는 **외출 이벤트만 0부터** 센다. 프롬프트 머리글의
         "### 이벤트 3" (Stage1 인덱스)과 번호가 다르다 — 실제 응답에서
         order 0 이 점심(Stage1 인덱스 1)이었다.

    그래서 JSON 을 제대로 파싱하고, 번호는 부르는 쪽에서 옮긴다.
    """
    if not raw:
        return {}
    t = raw.strip()
    a = t.find("{")
    b = t.rfind("}")
    if a < 0 or b <= a:
        # picks 배열만 온 경우
        a = t.find("[")
        b = t.rfind("]")
        if a < 0 or b <= a:
            return {}
        try:
            arr = json.loads(t[a:b + 1])
        except ValueError:
            return {}
        items = arr
    else:
        try:
            obj = json.loads(t[a:b + 1])
        except ValueError:
            return {}
        items = obj.get("picks") if isinstance(obj, dict) else None
        if items is None:
            items = obj if isinstance(obj, list) else []
    out: dict[int, float] = {}
    for it in items or []:
        if not isinstance(it, dict):
            continue
        o, amt = it.get("order"), it.get("actual_spent")
        if isinstance(o, int) and isinstance(amt, (int, float)):
            out[o] = float(amt)
    return out


# pick 의 order(외출 0-base) -> 우리가 만든 Stage1 인덱스
OUTING_TO_STAGE1 = {0: 1, 1: 2, 2: 3}


def sign(pairs):
    up = sum(1 for x, y in pairs if y > x)
    dn = sum(1 for x, y in pairs if y < x)
    tie = len(pairs) - up - dn
    n = up + dn
    p = min(1.0, sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n * 2) if n else 1.0
    return up, dn, tie, p


def report(rows: list[dict]) -> int:
    per: dict[tuple, dict[str, dict[int, float]]] = {}
    for r in rows:
        if r.get("error"):
            continue
        # **시드까지 키에 넣는다.** 후보 2 에서 aid 만으로 묶어 쌍이
        # 5개로 줄어든 적이 있다.
        per.setdefault((r["aid"], r.get("seed")), {})[r["side"]] = picks(r.get("raw") or "")
    both = {k: v for k, v in per.items() if "off" in v and "on" in v}
    print("응답 %d · 쌍 %d" % (len(rows), len(both)))
    if not both:
        print("  **쌍이 없다**")
        return 2
    tagged, other = [], []
    for v in both.values():
        for o, amt in v["off"].items():
            if o not in v["on"]:
                continue
            s1 = OUTING_TO_STAGE1.get(o)
            if s1 is None:
                continue
            (tagged if s1 in POLICY_ORDERS else other).append((amt, v["on"][o]))
    for label, P in (("표시된 이벤트(계기:policy)", tagged), ("표시 없는 이벤트(대조)", other)):
        if not P:
            print()
            print("  %-28s 표본 없음" % label)
            continue
        up, dn, tie, p = sign(P)
        ma, mb = st.mean([x for x, _ in P]), st.mean([y for _, y in P])
        print()
        print("  %-28s n=%d  %,.0f -> %,.0f  (%+.1f%%)".replace(",.0f", ".0f")
              % (label, len(P), ma, mb, (mb - ma) / ma * 100 if ma else 0))
        print("  %-28s   늘 %d : 줄 %d : 동점 %d · 양측 p=%.4f %s"
              % ("", up, dn, tie, p, "**다르다**" if p < 0.05 else ""))
    print()
    print("  판정 규칙: **표시된 쪽만** 움직여야 한다. 대조군까지 움직이면")
    print("            그 낱말의 효과가 아니라 프롬프트가 길어진 잡음이다")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--day", default="2021-10-26")
    ap.add_argument("--responses", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.build:
        build(a.n, out, a.day)
        return 0
    rows = [json.loads(l) for l in io.open(a.responses, encoding="utf-8") if l.strip()]
    return report(rows)


if __name__ == "__main__":
    raise SystemExit(main())
