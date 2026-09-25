"""정책마다 **핵심 검증지표 하나**를 골라 실측과 나란히 놓는다 — 빼지 않고.

    python scripts/report/core_indicator_table.py

## 왜 이 표가 따로 필요한가

`error_budget.py` 는 오차를 **더할 수 있는 것만** 센다. 그래서 값이 있어도 빠진다.

    P012-6   시뮬 0.000 · 실측 21.0%   -> "이틀 창으로는 구조적 불가" 로 제외돼 있었다
    P012-4   시뮬 1,311원 · 실측 47,880원/월 -> "단위 불일치" 로 제외돼 있었다
    P012-3   시뮬 0.168 (16.8%)        -> 정답지가 ">0" 이라 오차를 못 내지만 **값은 있다**

빼는 것 자체는 옳다 — 창이 다른 수를 더하면 총합이 거짓이 된다. 그러나 **정책마다
핵심 지표가 얼마인지는 알 수 있어야 한다.** 이 표가 그 자리다: 오차를 더하지 않고,
값과 실측을 나란히 놓고 왜 그냥 더할 수 없는지 한 줄로 적는다.

## 핵심 지표를 무엇으로 고르나

정답지가 **머리기사로 내세운 수**다. 정책이 들었는지를 그 하나로 말하는 지표.
아래 `CORE` 에 정책별로 못 박고 근거를 적었다. 임의로 바꾸지 않는다.

## 값은 어디서 오나

`data/experiments/scoring_table.json` 의 `result_*` 블록. 블록이 여러 개면
**가장 큰 표본(n)** 을 읽는다 — `project_selected_prompt_v5` 의 규칙이다.
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCORING = ROOT / "data/experiments/scoring_table.json"

DASH = str.maketrans({"−": "-", "–": "-", "—": "-"})

# 정책 -> (핵심 지표, 왜 그것이 핵심인가)
CORE = {
    "P010": ("P010-1", "한국은행이 내세운 수는 MPC 0.21 이다. 쿠폰이 소비를 얼마나 늘렸나가 이 정책의 전부다"),
    "P012": ("P012-1", "KDI 머리기사 — 적립업종 카드지출 +20.82%. 정책 목표가 골목상권 소비 진작이다"),
    "EMERGENCY_2020": ("EM-3", "KDI 2020-12-22 보도자료 — 5/11~6/21 전체 카드매출 전년비 +7.3%. 시뮬 이틀 전후 변화와 직접 비교할 수 없다"),
    "DISTANCING_2020": ("DS-1", "서울연구원 머리기사 — 음식점 지출 −14.1%. 영업제한이 직접 때린 업종이다"),
    "LOCAL_VOUCHER": ("LV-2", "조세재정연구원의 기전 — 구매처가 거주지 상권으로 옮겨가는가. LV-1(총소비)은 실측이 '영향 미미'로 수치가 없다"),
    "SECTOR_VOUCHER_2020": ("HO-1", "8대 쿠폰의 설계 목표 — 대상 업종(마트) 지출 몫 증가"),
    "P016": ("C1", "농식품부 보고서 머리기사 — 대상 품목 지출 +6.957%"),
    "GATHERING_2020": ("GA-3", "사적모임 제한의 직접 효과 — 재택 시간 증가. 총소비(GA-1)는 무반응이 기대값이다"),
}


def truth_of(desc: str, explicit=None):
    """실측 수치 — **`실측` 이라고 적힌 자리에서만** 읽는다.

    자유 서술에서 숫자를 주워 오면 **정책 파라미터를 결과로 읽는다.** 실제로
    HO-1 의 "20% 할인(최대 1만원) 대상" 에서 20% 를 실측으로 집어 +20.00% 라고
    적었다. 할인율은 정답지가 잰 수가 아니다. 그래서 두 자리만 본다 —
    블록의 `실측` 필드, 그리고 desc 안의 `(실측 ...)` 괄호.
    """
    srcs = []
    if explicit:
        srcs.append(str(explicit))
    if desc:
        m = re.search(r"\(실측\s*([^)]*)\)", str(desc))
        if m:
            srcs.append(m.group(1))
    for t in srcs:
        t = t.translate(DASH)
        m = re.search(r"([+-]?\d+(?:\.\d+)?)\s*(%p|%|원)", t)
        if m:
            return float(m.group(1)), m.group(2)
        # 단위 없는 수 — MPC 처럼 무차원 지표
        m = re.search(r"([+-]?\d*\.\d+)", t)
        if m:
            return float(m.group(1)), ""
    return None, None


# 기각된 후보의 수를 정답지와 맞대면 안 된다. 선택된 프롬프트는 v5 다
# (experiments/SELECTED_PROMPT.md · 아홉 후보가 열 번 맞대 못 이겼다).
REJECTED = ("v45", "v51", "v7", "v8", "v9", "v50_v45")


def best_block(pol: dict, iid: str):
    """그 지표가 담긴 블록 중 **표본이 가장 큰** 것. 기각된 후보 블록은 뺀다.

    '가장 큰 표본' 규칙은 같은 프롬프트의 런들 사이에서만 쓴다. 후보를 가로질러
    쓰면 n 이 하나 더 많다는 이유로 **기각된 후보의 수**를 읽게 된다 — 실제로
    P012-1 에서 result_r2_v45(n=499) 가 result_r2_v5(n=497) 를 이겼다.
    """
    best = None
    for bk, bv in pol.items():
        if not isinstance(bv, dict):
            continue
        if any(("_" + r) in bk or bk.endswith(r) for r in REJECTED):
            continue
        e = bv.get(iid)
        if not isinstance(e, dict):
            continue
        has = any(isinstance(e.get(k), (int, float)) for k in ("pct", "mean"))
        if not has:
            continue
        if best is None or (e.get("n") or 0) > (best[1].get("n") or 0):
            best = (bk, e)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    sc = json.loads(SCORING.read_text(encoding="utf-8"))

    print("# 정책별 핵심 검증지표 — 값과 실측을 나란히")
    print()
    print("%-18s %-7s %12s %12s %7s  %s"
          % ("정책", "지표", "실측", "시뮬", "n", "읽는 단서"))
    print("-" * 118)

    rows, missing = [], []
    for pk, (iid, why) in CORE.items():
        pol = sc.get(pk)
        if not isinstance(pol, dict):
            missing.append((pk, iid, "채점표에 정책 블록이 없다"))
            continue
        spec = next((i for i in (pol.get("indicators") or []) if i["id"] == iid), None)
        if spec is None:
            missing.append((pk, iid, "지표가 등록돼 있지 않다"))
            continue
        hit = best_block(pol, iid)
        tv, tu = truth_of(spec.get("desc"), (hit[1].get("실측") if hit else None))
        if hit is None:
            missing.append((pk, iid, "**시뮬 값이 하나도 없다 — 이 정책은 채점된 런이 없다**"))
            print("%-18s %-7s %12s %12s %7s  %s"
                  % (pk[:18], iid,
                     ("%+.2f%s" % (tv, tu)) if tv is not None else "없음",
                     "**없음**", "-", "채점된 런이 없다"))
            continue
        bk, e = hit
        pct, mean, n = e.get("pct"), e.get("mean"), e.get("n")
        shown = ("%+.2f%%" % pct) if isinstance(pct, (int, float)) else (
            "%.4g" % mean if isinstance(mean, (int, float)) else "-")
        note = ""
        if not isinstance(pct, (int, float)):
            note = "비율이 아니라 절대값(율·원)이다 — 실측과 단위를 맞춰야 한다"
        elif tv is None:
            note = "정답지에 수치가 없다 — 방향으로만 판정한다"
        print("%-18s %-7s %12s %12s %7s  %s"
              % (pk[:18], iid,
                 ("%+.2f%s" % (tv, tu)) if tv is not None else "없음",
                 shown, str(n or "-"), note or bk[:34]))
        rows.append({"policy": pk, "id": iid, "truth": tv, "truth_unit": tu,
                     "pct": pct, "mean": mean, "n": n, "block": bk, "why": why})

    print()
    print("## 핵심 지표를 왜 그것으로 골랐나")
    for pk, (iid, why) in CORE.items():
        print("  %-18s %-7s %s" % (pk[:18], iid, why))

    if missing:
        print()
        print("## 알 수 없는 것 — 여기가 메워야 할 자리다")
        for pk, iid, reason in missing:
            print("  %-18s %-7s %s" % (pk[:18], iid, reason))

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
            json.dumps({"rows": rows, "missing": missing},
                       ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
