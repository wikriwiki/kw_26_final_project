"""문턱 줄이 **Stage2 의 금액을 바꾸기는 하는가** — 본런 전에 싸게 본다.

    python scripts/sim/s2_threshold_probe.py --build --out <dir> --n 24
    python scripts/sim/s2_threshold_probe.py --out <dir> --responses <jsonl>

사전등록: `experiments/plan_channel/s2_threshold.md`

## 왜 Stage2 를 따로 재나

정책 반응은 **계획 금액**에 있다(+9.76%, p=0.0015). 소비성향 스칼라는 안
움직이고 이벤트 수도 그대로다. 개수가 같은데 금액이 는다면 **금액을 정하는
자리**가 반응하는 것이고, 그 자리는 Stage2 다.

그런데 Stage2 가 정책에 대해 받는 것은 후보 줄의 `[적립]` 다섯 글자뿐이었다.
문턱·남은 금액·남은 일수는 Stage1 에만 있었다. 그 줄을 Stage2 로 옮기는 것이
후보이고, 이 탐침은 **그 줄 하나만** 넣고 빼서 같은 시민을 두 번 부른다.

## 공정한 대조

두 팔 모두 `sangsaeng_active=True` 다 — 후보 목록의 `[적립]` 꼬리표도 양쪽에
똑같이 붙는다. **다른 것은 헤더의 문턱 줄 하나뿐이다.** 정책 유무를 가르는
탐침이 아니라 **그 줄의 효과**를 가르는 탐침이다.

## 읽는 것

    금액 합   picks 의 actual_spent 합 = 계획액. 이것이 총액으로 가는 값이다
    쌍별 부호검정 **양측** — 방향을 미리 고정하지 않는다. 덧댄 문장이 사실대로
    김을 빼므로("오늘 쓸 돈이 는 것은 아니다") 내려갈 수도 있다. 동점도 적는다.
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

# 문턱 줄은 **런타임이 쓰는 그 함수**로 만든다. 손으로 쓰면 탐침과 본런이 달라진다.
CASHBACK_RULE = {"threshold_ratio": 1.03, "rate": 0.10, "cap": 100000}
# 끼울 자리 — 페르소나 헤더의 "평소 1일 소비규모" 줄 바로 뒤.
# 코드(stage2_poi.py)가 넣는 자리와 같아야 한다. 시험이 그 일치를 지킨다.
ANCHOR_RE = re.compile("평소 1일 소비규모.*")


def build(n: int, out: Path, day: str) -> None:
    from neo4j import GraphDatabase
    import dawn_context as DC
    import stage2_poi as S2
    from stage1_intent import Stage1Output  # noqa: F401  (형상 참고)

    drv = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"),
              os.environ.get("NEO4J_PASSWORD", "")))
    # **런타임이 쓰는 그 쿼리를 그대로 쓴다.** 그래프의 키(s_daily_wd)와 페르소나의
    # 키(daily_wd)가 달라서, 직접 properties(a) 를 뜨면 Stage2 가 못 읽는다.
    # 실제로 그렇게 짰다가 시민 0명이 나왔다.
    pq = DC.PERSONA_CYPHER
    with drv.session() as s:
        ids = [r["aid"] for r in s.run(
            "MATCH (a:Agent) WHERE a.s_daily_wd IS NOT NULL "
            "RETURN a.id AS aid ORDER BY a.id LIMIT $n", n=n)]
        rows = []
        for aid in ids:
            rec = s.run(pq, aid=aid).single()
            if rec:
                rows.append({"aid": aid, "p": dict(rec)})
    drv.close()
    today = date.fromisoformat(day)

    from stage1_intent import Stage1Event

    # 모두에게 같은 하루 — 외출 셋. 이벤트가 같아야 **금액만** 견줄 수 있다.
    # anchor 는 "zone:<행정동코드>" 여야 한다(Stage1Event 계약). 사는 동을 쓴다.
    def events_for(dong: str):
        z = "zone:%s" % dong if dong else "residence"
        raw = [
            {"order": 0, "time": "08:10", "anchor": "residence", "category": "집",
             "sub_category": "", "intent": "기상"},
            {"order": 1, "time": "12:30", "anchor": "zone", "category": "식사",
             "sub_category": "한식", "intent": "점심"},
            {"order": 2, "time": "18:40", "anchor": "zone", "category": "마트",
             "sub_category": "슈퍼마켓", "intent": "장보기"},
            {"order": 3, "time": "20:10", "anchor": "zone", "category": "쇼핑",
             "sub_category": "의류", "intent": "필요한 것 사기"},
            {"order": 4, "time": "22:30", "anchor": "residence", "category": "집",
             "sub_category": "", "intent": "취침"},
        ]
        for e in raw:
            if e["anchor"] == "zone":
                e["anchor"] = z
        return [Stage1Event(**e) for e in raw]

    cells = []
    for r in rows:
        persona = dict(r["p"])
        persona["id"] = r["aid"]
        persona["sangsaeng_active"] = True
        dong = str(persona.get("home_dong_code") or persona.get("dong_code")
                   or persona.get("residence_dong_code") or "")
        ev = events_for(dong)
        try:
            cands = S2.fetch_candidates_for_events(r["aid"], ev, persona, today)
        except Exception as e:  # 후보를 못 뽑으면 그 시민은 건너뛴다
            print("  후보 실패 %s: %s" % (r["aid"], e))
            continue
        if not any(cands.values()):
            continue
        # 문턱 줄 — 런타임과 같은 함수로, 이 사람의 실제 앵커로 만든다.
        # 이번달 누적을 **문턱 대비 비율로** 깔아 둔다. 한 규제에만 몰리면
        # 그 규제에서만 재게 된다 — 정책이 실제로 무는 곳은 **문턱 아래**다.
        _anchor = DC._sangsaeng_monthly_anchor(persona)
        _thr = _anchor * float(CASHBACK_RULE["threshold_ratio"])
        _f = (0.55, 0.70, 0.85, 1.05)[len(cells) // 2 % 4]
        st_node = {"sangsaeng_month_spent": int(_thr * _f)}
        line = DC._format_cashback_status("P012", CASHBACK_RULE, persona, st_node, today)
        # **돌고 있는 코드를 건드리지 않는다.** 본런이 /data/repo 에서 도는 중이라
        # 그 저장소의 stage2_poi.py 를 고칠 수 없다. 그래서 프롬프트를 평소대로
        # 만들고 **그 줄만 끼워 넣는다** — 앞선 탐침들과 같은 방식이다.
        # 코드가 같은 자리에 같은 문구를 넣는지는 단위시험이 따로 지킨다.
        base = S2.build_stage2_prompt(ev, cands, persona=persona,
                                      state={"balance": 3_000_000})
        NL = chr(10)
        tail = NL + "적립 정책 상태: " + line.lstrip("- ") + (
            NL + "  (돌아오는 돈은 다음 달이므로 오늘 쓸 수 있는 돈이 는 것은 아니다."
            " 어차피 할 지출로 문턱이 저절로 넘어가는 사람도, 넘길 일이 없어"
            " 신경 쓰지 않는 사람도 있다.)")
        if ANCHOR_RE.search(base) is None:
            print("  닻 없음 %s — 건너뜀" % r["aid"])
            continue
        on = ANCHOR_RE.sub(lambda m: m.group(0) + tail, base, count=1)
        if on == base:
            print("  삽입 실패 %s — 건너뜀" % r["aid"])
            continue
        cells.append({"aid": r["aid"], "side": "off", "user": base})
        cells.append({"aid": r["aid"], "side": "on", "user": on})
    io.open(out / "cells.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"cells": cells}, ensure_ascii=False, indent=1))
    # SYSTEM 프롬프트를 파일로 떨군다 — 호출하는 쪽 venv 에는 neo4j 가 없어서
    # stage2_poi 를 import 할 수 없다(실제로 여기서 한 번 죽었다).
    io.open(out / "system_s2.txt", "w", encoding="utf-8",
            newline=chr(10)).write(S2.SYSTEM_S2)
    n_on = sum(1 for c in cells if c["side"] == "on")
    print("시민 %d · 칸 %d (off %d / on %d)"
          % (len({c["aid"] for c in cells}), len(cells), len(cells) - n_on, n_on))
    # 두 팔이 **그 줄 하나만** 다른지 확인한다 — 다른 게 섞이면 탐침이 무의미하다.
    byaid = {}
    for c in cells:
        byaid.setdefault(c["aid"], {})[c["side"]] = c["user"]
    bad = 0
    for aid, d in byaid.items():
        if "off" not in d or "on" not in d:
            continue
        extra = len(d["on"]) - len(d["off"])
        if extra <= 0:
            bad += 1
    print("off/on 길이 역전 %d (0 이어야 한다)" % bad)
    print("wrote", out / "cells.json")


def total_spend(raw: str) -> float | None:
    """응답에서 actual_spent 합."""
    if not raw:
        return None
    vals = [float(x) for x in re.findall(r'"actual_spent"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)]
    return sum(vals) if vals else None


def report(rows: list[dict]) -> int:
    per: dict[str, dict[str, float]] = {}
    for r in rows:
        if r.get("error"):
            continue
        v = total_spend(r.get("raw") or "")
        if v is not None:
            per.setdefault(r["aid"], {})[r["side"]] = v
    pairs = [(d["off"], d["on"]) for d in per.values() if "off" in d and "on" in d]
    print("응답 %d · 쌍 %d" % (len(rows), len(pairs)))
    if not pairs:
        print("  **쌍이 없다** — 파싱이 안 됐다")
        return 2
    a = [x for x, _ in pairs]
    b = [y for _, y in pairs]
    up = sum(1 for x, y in pairs if y > x)
    dn = sum(1 for x, y in pairs if y < x)
    tie = len(pairs) - up - dn
    n = up + dn
    p = min(1.0, sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n * 2) if n else 1.0
    print()
    print("  계획액 합  문턱줄 없음 %9.0f · 있음 %9.0f · 차이 %+.2f%%"
          % (st.mean(a), st.mean(b), (st.mean(b) - st.mean(a)) / st.mean(a) * 100))
    print("  중앙       없음 %9.0f · 있음 %9.0f" % (st.median(a), st.median(b)))
    print()
    print("  쌍별 부호검정 (**양측** — 방향 미고정)")
    print("    늘어남 %d · 줄어듦 %d · 동점 %d" % (up, dn, tie))
    print("    p=%.4f  %s" % (p, "구별 안 됨" if p > 0.05 else "**다르다**"))
    if tie / len(pairs) >= 0.8:
        print("    동점 %.0f%% — 그 줄이 금액 판단에 안 닿는다" % (100 * tie / len(pairs)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--day", default="2021-10-21")
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
