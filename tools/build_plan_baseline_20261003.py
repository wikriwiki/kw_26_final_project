"""정책이 없는 날들의 계획액으로 사람별 '평소 계획' 기준선을 만든다 — 계획 통로(EXP_PLAN_DRIVES_TOTAL)용.

    python tools/build_plan_baseline_20261003.py --metrics <run>/metrics --days 2020-05-04,...,2020-05-10 \
        --out <baseline.json>

출력: {"<aid>|wd": 평일 평균 계획액, "<aid>|we": 주말 평균 계획액, "<aid>": 전체 날 평균 계획액}.
엔진(consumption._plan_baseline)이 오늘이 주말인지에 따라 앞의 둘 중 하나를 고르고, 그 칸이 비면 사람 하나의
값("<aid>")을 쓴다. [2026-10-05] 예전에는 사람 하나의 값을 쓰지 않아, 정책 전 주말에 계획이 없던 사람은 주말마다
엔진이 조용히 옛 공식(평소 소비와 계획 중 큰 값)으로 돌아갔다. 요일종류를 섞지 않는 이유: 계획/앵커 비가 금 0.500 · 토 1.568 로
반대로 움직여, 섞인 기준선은 클램프에 걸려 신호를 누른다(recompute_plan_channel.py 기록).

지킬 것: 기준선 날짜는 **정책이 없는 날**이어야 한다(지원금이 있는 시뮬레이션의 지급 뒤 날을 넣으면 정책
반응이 기준선에 섞여 효과를 지운다). 그래서 날짜를 명시적으로 받고, 그 날의 cohort 에 정책이 있으면 멈춘다.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys as _sys
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "scripts" / "sim"))
from kr_holidays import is_day_off  # noqa: E402  공휴일 판정(엔진과 같은 표)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="metrics 폴더 (day_<날짜>.jsonl)")
    ap.add_argument("--days", required=True, help="쉼표로 나눈 정책 없는 날짜")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    days = [d.strip() for d in a.days.split(",") if d.strip()]
    vals = defaultdict(list)
    whole = defaultdict(list)     # 요일종류를 가리지 않은 사람별 값 — 한쪽 칸이 빌 때 쓴다
    for d in days:
        p = Path(a.metrics) / f"day_{d}.jsonl"
        if not p.is_file():
            raise SystemExit(f"없는 날: {p}")
        wk = "we" if is_day_off(date.fromisoformat(d)) else "wd"   # 공휴일은 주말 칸(엔진과 같은 판정)
        for line in io.open(p, encoding="utf-8"):
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            if r.get("experience_policy_ids") or (r.get("grant_applied_today") or 0) or (r.get("grant_remaining_total") or 0):
                raise SystemExit(f"정책이 있는 날이 섞였다: {r['aid']} {d}")
            v = r.get("cm_planned_total")
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                vals[f"{r['aid']}|{wk}"].append(float(v))
                whole[r["aid"]].append(float(v))
    out = {k: round(st.mean(v), 2) for k, v in vals.items()}
    out.update({aid: round(st.mean(v), 2) for aid, v in whole.items()})
    people = set(whole)
    missing = sum(1 for p in people for wk in ("wd", "we") if f"{p}|{wk}" not in out)
    io.open(a.out, "w", encoding="utf-8").write(json.dumps(out, ensure_ascii=False, sort_keys=True))
    print("기준선 %d명 · 평일/주말 칸 %d · 빈 칸 %d (빈 칸은 엔진이 사람 하나의 값을 쓴다)"
          % (len(people), len(vals), missing))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
