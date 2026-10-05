"""탐침용 맥락을 **그래프에서 새로 만든다** — 고정 셀의 12명 한계를 푼다.

    python scripts/sim/make_probe_cells.py --out DIR --n 120 --day 2021-10-03

## 왜 필요한가

`/data/validation_v3/pilot_registered/frozen_inputs.json` 의 셀은 60칸이지만
**고유 에이전트가 12명뿐**이다. 사람 단위 지표(예: 소비성향이 그 사람의 중심값을
따라가는가)는 독립한 점이 12개라 아무것도 판정할 수 없다 —
`experiments/plan_channel/v5self_probe_aborted.md`.

여기서는 런타임과 **같은 함수로** 맥락을 만든다.

    build_dawn_context(aid, day)        그래프에서 그 사람의 그날 맥락
    stage1_intent._format_dawn_blocks   런타임이 쓰는 그 렌더

그러므로 셀은 본런이 실제로 모델에 주는 것과 같은 모양이다.

## 정책은 꺼진 날로 만든다 — 그리고 탐침이 주입한다

정책이 켜진 날을 쓰면 그래프의 State 가 있어야 하는데, 본런이 아직 거기까지 안
갔을 수 있다. 그래서 **정책이 꺼진 날**로 만들고, 탐침 쪽에서
`s1_ownership_probe.build` 와 같은 방식으로 정책 사실·개인 문턱 상태를 끼워 넣는다.
꺼진 날의 렌더에는 자리표시가 그대로 남는다.

    (활성 정책 없음)
    (해당 없음 — 정책·지원금·바우처·쿠폰을 임의로 언급하지 말 것)

**둘 중 하나라도 없는 칸은 버린다** — 정책이 이미 껴 있는 칸에 또 끼우면 두 번
들어간다.

## 표본

`fetch_agents(limit)` 는 **소비 10분위 비례 층화표본**이다. 사람 단위 지표에는
이것이 맞다 — 분위가 한쪽으로 쏠리면 상관이 그 쏠림을 재게 된다.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

NO_FACTS = "(활성 정책 없음)"
NO_MINE = "(해당 없음 — 정책·지원금·바우처·쿠폰을 임의로 언급하지 말 것)"

_DOW = "월화수목금토일"


def day_parts(d: date) -> tuple:
    """(day_type, dow_kr) — 런타임과 같은 규칙."""
    wd = d.weekday()
    return ("weekend" if wd >= 5 else "weekday"), _DOW[wd]


def build_cells(n: int, day: str, out: Path, env: str = "") -> int:
    from dawn_context import build_dawn_context
    from run_simulation import fetch_agents
    import stage1_intent as S1

    today = date.fromisoformat(day)
    day_type, dow = day_parts(today)
    aids = fetch_agents(n)
    print("층화표본 %d명 요청 → %d명" % (n, len(aids)))

    try:
        from run_simulation import build_environment
        environment = build_environment(env, today) if env else ""
    except Exception:
        environment = ""

    cells, skipped, failed = [], 0, 0
    for aid in aids:
        try:
            ctx = build_dawn_context(aid, today)
        except Exception:
            failed += 1
            continue
        if not getattr(ctx, "persona", None):
            failed += 1
            continue
        try:
            ctx.environment = environment
            user = S1._format_dawn_blocks(ctx, today, day_type)
        except Exception:
            failed += 1
            continue
        # 자리표시가 남아 있어야 탐침이 정책을 끼울 수 있다.
        if NO_FACTS not in user or NO_MINE not in user:
            skipped += 1
            continue
        cells.append({"aid": aid, "case": "graph", "date": day, "user": user,
                      "dow": dow, "day_type": day_type})

    out.mkdir(parents=True, exist_ok=True)
    io.open(out / "cells_raw.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"cells": cells}, ensure_ascii=False, indent=1))
    print("  쓸 수 있는 칸 %d · 정책이 이미 껴 있어 버린 칸 %d · 실패 %d"
          % (len(cells), skipped, failed))
    print("  **고유 에이전트 %d명**  <- 사람 단위 지표의 진짜 표본" % len({c["aid"] for c in cells}))
    print("wrote", out / "cells_raw.json")
    return len(cells)


def inject_policy(out: Path, pol_path: Path, day: str, fracs: list) -> int:
    """자리표시 자리에 정책 사실·**그 사람 자신의** 문턱 상태를 끼운다.

    `s1_ownership_probe.build` 는 문턱을 고정 페르소나(45,000/60,000원)로 계산한다.
    여기서는 **그 사람의 실제 앵커**를 쓴다 — 사람 단위 지표를 재는 탐침이라
    문턱까지의 거리도 사람마다 달라야 맥락이 일관된다. 두 팔에 같은 값이 들어가므로
    비교 자체에는 영향이 없고, 다만 맥락이 덜 인위적이다.

    문턱 대비 위치(`fracs`)는 돌아가며 준다 — 한 위치만 깔면 그 위치의 반응만 본다.
    """
    import dawn_context as DC
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
    from _common import driver_session

    pol = json.loads(io.open(pol_path, encoding="utf-8").read())
    row = {"id": pol.get("id"), "name": pol.get("name"), "type": pol.get("type"),
           "description": pol.get("description"),
           "benefit_rate": pol.get("benefit_rate"),
           "cap_per_agent": pol.get("cap_per_agent"),
           "threshold_ratio": pol.get("threshold_ratio"),
           "eligible_marker": pol.get("eligible_marker"),
           "effective_from": pol.get("effective_from"),
           "effective_until": pol.get("effective_until")}
    facts = DC._format_policy_facts([row])
    rule = {"threshold_ratio": pol.get("threshold_ratio", 1.03),
            "rate": pol.get("benefit_rate", 0.10),
            "cap": pol.get("cap_per_agent", 100000)}
    today = date.fromisoformat(day)

    cells = json.loads(io.open(out / "cells_raw.json", encoding="utf-8").read())["cells"]
    aids = sorted({c["aid"] for c in cells})
    with driver_session() as s:
        anc = {r["aid"]: (float(r["wd"] or 0), float(r["we"] or 0)) for r in s.run(
            "MATCH (a:Agent) WHERE a.id IN $ids "
            "RETURN a.id AS aid, a.s_daily_wd AS wd, a.s_daily_we AS we", ids=aids)}

    done = []
    for i, c in enumerate(cells):
        wd, we = anc.get(c["aid"], (0.0, 0.0))
        if wd <= 0:
            wd = we
        if wd <= 0:
            continue
        persona = {"daily_wd": wd, "daily_we": we or wd}
        anchor = DC._sangsaeng_monthly_anchor(persona)
        f = fracs[i % len(fracs)]
        stn = {"sangsaeng_month_spent": int(anchor * rule["threshold_ratio"] * f)}
        mine = DC._format_cashback_status(pol["id"], rule, persona, stn, today)
        u = c["user"].replace(NO_FACTS, facts, 1).replace(NO_MINE, mine, 1)
        done.append({**c, "user": u, "thr_frac": f})

    io.open(out / "cells.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"cells": done}, ensure_ascii=False, indent=1))
    print("정책을 끼운 칸 %d · 고유 에이전트 %d명 · 문턱 위치 %s"
          % (len(done), len({c["aid"] for c in done}), fracs))
    print("  정책 사실 %d자 · 개인 상태 예시: %s" % (len(facts), done[0]["user"][:0] or ""))
    print("wrote", out / "cells.json")
    return len(done)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--day", default="2021-10-03")
    ap.add_argument("--environment", default=os.environ.get("SIM_ENVIRONMENT", ""))
    ap.add_argument("--policy", default="data/neo4j_load/policies/P012.json")
    ap.add_argument("--policy-day", default="2021-10-21",
                    help="문턱 상태를 렌더할 날 (정책이 켜진 날)")
    ap.add_argument("--thr-fracs", default="0.55,0.70,0.85,1.05")
    ap.add_argument("--no-inject", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    got = build_cells(a.n, a.day, out, a.environment)
    if not got:
        return 1
    if a.no_inject:
        return 0
    fr = [float(x) for x in a.thr_fracs.split(",") if x.strip()]
    return 0 if inject_policy(out, Path(a.policy), a.policy_day, fr) else 1


if __name__ == "__main__":
    raise SystemExit(main())
