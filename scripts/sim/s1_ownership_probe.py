"""후보 2 — **Stage1 의 이벤트가 달라지는가.** 사전등록 `experiments/plan_channel/s1_ownership.md`

    python scripts/sim/s1_ownership_probe.py --build --out <dir> --n 24
    python scripts/sim/s1_ownership_probe.py --out <dir> --responses <jsonl>

## 후보 1 에서 배운 것

**고정한 것은 못 잰다.** 후보 1 은 이벤트를 고정해 놓고 금액만 봤는데, 실제
정책 반응은 **이벤트가 달라져서** 나온다(수 11.5 -> 11.5 그대로, 금액 +9.76%).
그래서 이번엔 Stage1 을 부르고 **이벤트 자체**를 본다.

## 두 팔

같은 사용자 맥락(정책 블록 포함)을 **SYSTEM 만 바꿔** 두 번 부른다.

    v5      현행
    v5own   + "적립형은 고를 결제수단이 없다. 제도가 닿는 자리는 무엇을 하기로
              하는가 하나뿐이다" 한 줄 (감쇠 문장은 그대로 둔다)

## 맥락을 왜 합성하나

동결 맥락은 전부 "(활성 정책 없음)" 이다. 정책이 켜진 맥락은 그래프에서
만들어지는데 지금 그래프에는 P013 이 올라가 있고 본런이 돌고 있다. 그래서
**런타임이 쓰는 그 함수들**로 정책 블록을 만들어 끼운다
(`_format_policy_facts` · `_format_cashback_status`). 손으로 쓰지 않는다.

## 재는 것

    이벤트 수        늘어나는가 (실제 런에서는 **안 늘었다** — 이것이 대조군이다)
    policy 트리거    제도를 이유로 든 이벤트 수
    내구재 이벤트    쇼핑·가전·의류 — 금액을 끌어올리는 것은 이쪽이다
    쌍별 부호검정 **양측** · 동점 공개
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

NO_FACTS = "(활성 정책 없음)"
NO_MINE = "(해당 없음 — 정책·지원금·바우처·쿠폰을 임의로 언급하지 말 것)"
DURABLE = ("쇼핑", "가전", "의류", "가구", "전자")


def _policy_row(pol: dict) -> dict:
    """POLICY_CYPHER 가 돌려주는 모양으로 맞춘다."""
    return {
        "id": pol["id"], "name": pol["name"], "type": pol["type"],
        "description": pol.get("description") or "",
        "rate": pol.get("benefit_rate"), "cap": pol.get("cap_per_agent"),
        "threshold_ratio": pol.get("threshold_ratio"),
        "eligible_marker": pol.get("eligible_marker"),
        "mech_params": None, "poi_restricted": bool(pol.get("poi_restricted")),
        "effective_from": pol.get("effective_from"),
        "effective_until": pol.get("effective_until"),
        # from_/until_ 이 None 이면 렌더가 "None~None" 으로 찍힌다 — 실제로 그렇게 나왔다.
        "from_": date.fromisoformat(pol["effective_from"]),
        "until_": date.fromisoformat(pol["effective_until"]),
        "income_grants": {}, "excluded_income": [], "decile_grants": {},
        "regions": ["서울특별시"], "region_codes": [""], "target_l1s": [],
    }


def build(n: int, out: Path, day: str, pol_path: Path, frozen: Path) -> None:
    import dawn_context as DC

    pol = json.loads(io.open(pol_path, encoding="utf-8").read())
    row = _policy_row(pol)
    facts = DC._format_policy_facts([row])
    rule = {"threshold_ratio": pol.get("threshold_ratio", 1.03),
            "rate": pol.get("benefit_rate", 0.10), "cap": pol.get("cap_per_agent", 100000)}
    today = date.fromisoformat(day)
    cells_in = json.loads(io.open(frozen, encoding="utf-8").read()).get("cells") or []
    out_cells = []
    for c in cells_in:
        if len(out_cells) >= n:
            break
        user = c["user"]
        if NO_FACTS not in user or NO_MINE not in user:
            continue
        # 개인 문턱 상태 — 규제를 갈라 깐다(문턱 아래가 정책이 무는 곳).
        persona = {"daily_wd": 45000, "daily_we": 60000}
        anchor = DC._sangsaeng_monthly_anchor(persona)
        f = (0.55, 0.70, 0.85, 1.05)[len(out_cells) % 4]
        stn = {"sangsaeng_month_spent": int(anchor * rule["threshold_ratio"] * f)}
        mine = DC._format_cashback_status(pol["id"], rule, persona, stn, today)
        u = user.replace(NO_FACTS, facts, 1).replace(NO_MINE, mine, 1)
        out_cells.append({"aid": c["aid"], "case": c.get("case"), "user": u})
    io.open(out / "cells.json", "w", encoding="utf-8", newline=chr(10)).write(
        json.dumps({"cells": out_cells}, ensure_ascii=False, indent=1))
    print("정책이 켜진 맥락 %d칸 (합성 — 런타임 함수로 렌더)" % len(out_cells))
    print("  정책 사실 %d자 · 개인 상태 예시: %s" % (len(facts), mine[:70]))
    print("wrote", out / "cells.json")


def parse_events(raw: str) -> list[dict] | None:
    if not raw:
        return None
    m = re.search(r'"events"\s*:\s*\[', raw)
    if not m:
        return None
    i = raw.index("[", m.start())
    depth, j = 0, i
    for j in range(i, len(raw)):
        if raw[j] == "[":
            depth += 1
        elif raw[j] == "]":
            depth -= 1
            if depth == 0:
                break
    try:
        return json.loads(raw[i:j + 1])
    except ValueError:
        return None


def measure(ev: list[dict]) -> dict:
    out = {"n": 0, "policy": 0, "durable": 0}
    for e in ev or []:
        cat = str(e.get("category") or "")
        if cat in ("집", "직장"):
            continue
        out["n"] += 1
        if str(e.get("trigger") or "") == "policy":
            out["policy"] += 1
        sub = str(e.get("sub_category") or "")
        if any(d in cat or d in sub for d in DURABLE):
            out["durable"] += 1
    return out


def sign(pairs):
    up = sum(1 for x, y in pairs if y > x)
    dn = sum(1 for x, y in pairs if y < x)
    tie = len(pairs) - up - dn
    n = up + dn
    p = min(1.0, sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n * 2) if n else 1.0
    return up, dn, tie, p


def report(rows: list[dict]) -> int:
    per: dict[str, dict[str, dict]] = {}
    bad = 0
    for r in rows:
        if r.get("error"):
            continue
        ev = parse_events(r.get("raw") or "")
        if ev is None:
            bad += 1
            continue
        per.setdefault(r["aid"], {})[r["arm"]] = measure(ev)
    pairs = {k: (d["v5"], d["v5own"]) for k, d in per.items() if "v5" in d and "v5own" in d}
    print("응답 %d · 파싱 실패 %d · 쌍 %d" % (len(rows), bad, len(pairs)))
    if not pairs:
        print("  **쌍이 없다**")
        return 2
    for key, label in (("n", "이벤트 수"), ("policy", "policy 트리거"),
                       ("durable", "내구재(쇼핑·가전·의류)")):
        P = [(a[key], b[key]) for a, b in pairs.values()]
        up, dn, tie, p = sign(P)
        ma, mb = st.mean([x for x, _ in P]), st.mean([y for _, y in P])
        print()
        print("  %-22s v5 %.2f -> v5own %.2f  (%+.1f%%)"
              % (label, ma, mb, (mb - ma) / ma * 100 if ma else 0))
        print("  %-22s   늘 %d : 줄 %d : 동점 %d · 양측 p=%.4f %s"
              % ("", up, dn, tie, p, "**다르다**" if p < 0.05 else ""))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--day", default="2021-10-21")
    ap.add_argument("--policy", default="data/neo4j_load/policies/P012.json")
    ap.add_argument("--frozen", default="/data/validation_v3/pilot_registered/frozen_inputs.json")
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
