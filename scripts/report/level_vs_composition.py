"""오차를 **수준**과 **구성**으로 가른다 — 어느 쪽이 모자란지 말할 수 있어야 한다.

    python scripts/report/level_vs_composition.py

## 왜 가르는가

P012-1 의 오차가 18.92%p 다(실측 +20.82% · 시뮬 +1.90%). 이 한 수를 보면
"시뮬이 정책 효과를 거의 못 낸다" 로 읽힌다. 그런데 정답지는 **두 수**를 준다.

    적립업종  +20.82%      제외업종  +2.85% (유의하지 않음)

적립업종이 지출의 몫 `s` 를 차지하면 두 수가 **총소비 변화를 함의한다.**

    총소비 = s x (1+적립) + (1-s) x (1+제외) - 1

s=0.268 이면 **+7.67%** 다. 우리 총소비 변화는 회계 고침에서 +5.41% 이므로
**총소비 기준 오차는 2.26%p** 다. 같은 런의 같은 값이 자에 따라 18.92 와 2.26 이
된다.

## 그러면 어느 쪽이 맞는 자인가 — **둘 다 적는다**

등록된 지표는 **적립업종**이고 그 오차는 줄지 않는다. 자를 바꿔 작은 수를 고르는
것은 진전이 아니다(`project_ruler_moves_are_not_progress`). 이 스크립트가 하는 일은
**오차를 두 조각으로 나누는 것**이다.

    수준 조각   총소비가 얼마나 움직였나          우리가 낼 수 있는 것
    구성 조각   적립과 제외가 얼마나 갈라졌나      **우리 판에 없는 것**
                (적립 몫이 상수 0.2535 라 우리 간격은 항상 0)

구성 조각의 크기는 실측 간격 그대로다(20.82 − 2.85 = **17.97%p**). 이것이 앞서
'산술 바닥' 이라고 적은 값이고, 여기서 같은 수가 다시 나온다 — 우연이 아니라
같은 것을 두 방향에서 본 것이다.

## 이 분해가 쓸모 있는 이유

"오차 18.92%p" 는 무엇을 고쳐야 하는지 말해 주지 않는다. 분해는 말해 준다 —
**수준은 2.26%p 안에 들어와 있으니 프롬프트·회계를 더 만질 일이 아니고, 남은
17.97%p 는 판에 제외업종 출발지를 넣지 않으면 한 푼도 줄지 않는다.**
`experiments/error_budget/p012_2_why_unproducible.md`
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (정책, 라벨, 적립 몫 s, 실측 적립, 실측 제외, 우리 총소비 변화 후보들)
# s 는 EXP_SANGSAENG_BASE_RATIO 와 같은 값을 쓴다 — 우리 회계가 쓰는 그 몫이다.
CASES = [
    {"policy": "P012", "name": "상생소비지원금", "share": 0.268,
     "truth_in": 20.82, "truth_out": 2.85,
     "ours": [("현행 max(앵커,계획)", 2.37),
              ("회계 고침", 5.41),
              ("채점표 기록값", 1.90)]},
]


def implied_total(share: float, t_in: float, t_out: float) -> float:
    """정답지의 두 수가 함의하는 총소비 변화(%)."""
    return 100 * ((share * (1 + t_in / 100)
                   + (1 - share) * (1 + t_out / 100)) - 1)


def split_error(share: float, t_in: float, t_out: float, ours_total: float) -> dict:
    """오차를 수준 조각과 구성 조각으로."""
    tot = implied_total(share, t_in, t_out)
    return {
        "implied_total": tot,
        "level_err": abs(ours_total - tot),          # 우리가 낼 수 있는 몫
        "composition_gap": t_in - t_out,             # 우리 판에 없는 몫
        "as_registered": abs(ours_total - t_in),     # 등록된 지표의 오차(줄지 않는다)
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    print("# 오차를 수준과 구성으로 가른다")
    print()
    out = []
    for c in CASES:
        tot = implied_total(c["share"], c["truth_in"], c["truth_out"])
        print("## %s (%s)" % (c["policy"], c["name"]))
        print("  정답지   적립 %+.2f%% · 제외 %+.2f%% · 적립 몫 %.3f"
              % (c["truth_in"], c["truth_out"], c["share"]))
        print("           -> 함의하는 **총소비 %+.2f%%**" % tot)
        print()
        print("  %-22s %8s  %10s  %12s" % ("우리 값", "총소비", "수준 오차", "등록지표 오차"))
        for lbl, v in c["ours"]:
            d = split_error(c["share"], c["truth_in"], c["truth_out"], v)
            print("  %-22s %+7.2f%%  %9.2f%%p  %11.2f%%p"
                  % (lbl, v, d["level_err"], d["as_registered"]))
            out.append({"policy": c["policy"], "label": lbl, "ours_total": v, **d})
        print()
        print("  구성 조각 = 실측 간격 %.2f − %.2f = **%.2f%%p**"
              % (c["truth_in"], c["truth_out"], c["truth_in"] - c["truth_out"]))
        print("             우리 간격은 항상 **0** 이다 — 적립 몫이 상수 %.4f 다" % c["share"])
        print()

    print("## 읽는 법")
    print("  · **등록지표 오차가 공식 성적이다.** 수준 오차를 대신 적지 않는다")
    print("  · 수준 오차는 '우리가 낼 수 있는 것이 얼마나 맞았나' 를 말한다")
    print("  · 구성 조각은 판을 고치지 않으면 한 푼도 줄지 않는다")
    print("    -> experiments/error_budget/p012_2_why_unproducible.md")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps(out, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
