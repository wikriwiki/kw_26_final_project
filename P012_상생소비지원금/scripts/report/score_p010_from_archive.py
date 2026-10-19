"""P010(민생회복 소비쿠폰) 을 **보관된 원장에서** 채점한다 — 그래프 없이, GPU 없이.

    python scripts/report/score_p010_from_archive.py

## 왜 따로 만드나

P010 은 채점표에 결과 블록이 하나도 없었다. 런은 있는데(EXP-001, 28일 × 200명)
채점이 안 됐다. 표준 채점기(`score_policy.py`)는 **그래프**를 읽는데 그 그래프는
지워졌고, 지금은 다른 실험이 쓰고 있다.

그런데 **원장만으로 낼 수 있는 지표가 있다.** `cm_mpc_new_share` 가 그것이다.

## MPC 의 정의는 우리 코드에 있다 — 옮겨 적지 않는다

`consumption.py` 가 매 에이전트-일마다 계산해 원장에 적어 둔다.

    MPC = Σ(정책지갑 결제분 중 신규 소비) / Σ(정책지갑 결제액)

    건별로   _amt = 그 건의 계획 금액
            _c   = _amt × choice_share      (정책지갑으로 낸 몫)
            _ex  = extra_spent, 없으면 would_buy_anyway ? 0 : _amt
            _ex  = clamp(_ex, 0, _amt)      <- **상한**
            누적  _w_tot += _c ·  _w_new += _ex × (_c / _amt)   <- **안분**
    MPC = _w_new / _w_tot

**상한과 안분이 핵심이다.** 이벤트 파일에서 `Σex / Σsp` 를 그냥 나누면 **0.7163**
이 나온다 — 상한을 안 씌우고 안분을 안 한 값이다. 코드의 정의를 따르면 0.216 이다.
같은 자료에서 세 배 차이가 난다.

## 집계는 가중이다

`cm_mpc_new_share` 는 에이전트-일마다의 **비율**이다. 비율의 평균은 MPC 가 아니다 —
분모(정책결제액)가 칸마다 다르다. 그래서 `policy_spend_today` 로 가중한다.

    가중      0.2160   <- 이것이 MPC 다
    단순평균   0.2386
    0 제외    0.5173

## 낼 수 있는 것과 없는 것

    P010-1  MPC          **낸다.** 원장의 cm_mpc_new_share 만으로 된다
    P010-2  적격업종 지출  못 낸다 — OFF 창이 필요하다
    P010-3  총소비        못 낸다 — 같은 이유

P010 의 등록 창은 `off 2025-07-15:16 / on 2025-07-22:23` 인데 **런이 07-21 에
시작한다.** OFF 창이 자료에 없다. 그리고 `out_BASE` 는 이름이 nopolicy 인데
**정책이 들어 있다**(정책지출 3,300~4,200원/일 · 총소비도 FINAL 과 잡음 범위 안).
무정책 대조 팔이 없다. 둘은 새 런이 필요하다.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = "output/exp001_archive/out_FINAL/metrics"

# 한국은행 이슈노트 2026-13. 정의가 다른 역사적 규모 참고치일 뿐이다.
TRUTH_MPC = 0.21


def load_rows(mdir: Path):
    for f in sorted(glob.glob(str(mdir / "day_*.jsonl"))):
        day = os.path.basename(f)[4:14]
        for line in io.open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") == "ok":
                yield day, r


def weighted_mpc(rows):
    """Σ(share × 정책결제) / Σ정책결제 — 비율의 평균이 아니다."""
    num = den = 0.0
    vals, ndays, aids = [], set(), set()
    for day, r in rows:
        m, w = r.get("cm_mpc_new_share"), r.get("policy_spend_today") or 0
        if not isinstance(m, (int, float)):
            continue
        vals.append(m)
        ndays.add(day)
        aids.add(r.get("aid"))
        if w > 0:
            num += m * w
            den += w
    return {
        "mpc": (num / den) if den else None,
        "denom_won": den, "cells": len(vals), "agents": len(aids), "days": len(ndays),
        "plain_mean": st.mean(vals) if vals else None,
        "nonzero_mean": st.mean([v for v in vals if v > 0]) if any(v > 0 for v in vals) else None,
    }


def boot_ci(pairs, n=2000, seed=20260925):
    """시민 단위 재표집. 같은 시민의 여러 날은 항상 함께 뽑는다."""
    import random
    if not pairs:
        return (None, None)
    rnd = random.Random(seed)
    by_aid = defaultdict(lambda: [0.0, 0.0])
    for aid, m, w in pairs:
        by_aid[aid][0] += m * w
        by_aid[aid][1] += w
    units = list(by_aid.values())
    k, out = len(units), []
    for _ in range(n):
        num = den = 0.0
        for _ in range(k):
            numerator, w = units[rnd.randrange(k)]
            num += numerator
            den += w
        if den:
            out.append(num / den)
    out.sort()
    if not out:
        return (None, None)
    return (out[int(len(out) * 0.025)], out[int(len(out) * 0.975)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=DEFAULT)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    mdir = ROOT / a.metrics
    if not mdir.exists():
        print("원장이 없다: %s" % mdir)
        return 1

    rows = list(load_rows(mdir))
    r = weighted_mpc(rows)
    pairs = [(x.get("aid"), x.get("cm_mpc_new_share"), x.get("policy_spend_today") or 0)
             for _d, x in rows
             if isinstance(x.get("cm_mpc_new_share"), (int, float))
             and (x.get("policy_spend_today") or 0) > 0]
    lo, hi = boot_ci(pairs)

    print("# P010 — 보관 원장에서 채점 (그래프·GPU 없이)")
    print()
    print("  원장   %s" % a.metrics)
    print("  범위   %d일 · 시민 %d명 · MPC 값이 있는 시민-일 %d칸" %
          (r["days"], r["agents"], r["cells"]))
    print()
    print("## P010-1  역사적 원장 비율 (직접 실측 오차 아님)")
    print("   한국은행 조사 참고치    %.2f" % TRUTH_MPC)
    print("   시뮬 · 정책결제 가중    **%.4f**   시민별 재표집 95%% 구간 [%.4f, %.4f]"
          % (r["mpc"], lo, hi) if r["mpc"] is not None else "   시뮬  없음")
    if r["mpc"] is not None:
        print("   비교 상태              정의 불일치: 실현 거래 vs 조사 시점 사용·계획")
        print("   분모(정책결제 합)       %,d원".replace(",", "") % r["denom_won"])
        print()
        print("   참고 — 집계를 달리하면")
        print("     단순 평균            %.4f  (비율의 평균은 MPC 가 아니다)" % r["plain_mean"])
        print("     0 인 칸 제외 평균     %.4f  (쓰지 않은 사람을 빼면 위로 뜬다)"
              % r["nonzero_mean"])
    print()
    print("## 못 내는 것 — 새 런이 필요하다")
    print("   P010-2 적격업종 지출 · P010-3 총소비")
    print("     등록 창은 off 2025-07-15:16 / on 2025-07-22:23 인데 런은 07-21 시작이다")
    print("     그리고 out_BASE 는 이름이 nopolicy 인데 **정책이 들어 있다**")
    print("     (정책지출 3,300~4,200원/일) — 무정책 대조 팔이 없다")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps({
            "P010-1": {"metric": "mpc_amount", "mean": r["mpc"], "ci": [lo, hi],
                       "n": r["agents"], "n_cells": r["cells"], "bootstrap_unit": "aid", "실측": TRUTH_MPC,
                       "comparison": "different_estimand", "hit": None,
                       "note": "역사적 정책결제 가중 신규소비 비율. 계획 단계 cm_mpc_new_share를 최종 결제액으로 가중했다. 한국은행 조사 시점의 사용·계획 품목별 자기보고와 정의가 달라 직접 오차·적중을 산출하지 않는다. CI는 aid 단위 재표집"},
            "_source": a.metrics, "_days": r["days"],
        }, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
