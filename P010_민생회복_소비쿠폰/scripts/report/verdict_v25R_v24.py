# -*- coding: utf-8 -*-
"""v25R·v24 의 사전등록 판정을 기계로 내린다.

두 라운드의 성공선은 문서에 이렇게 적혀 있고, 둘 다 기계로 확인된다.

  v25R  대조 v23(프롬프트 v25 × 볼일) 대비 칸당 종류가 늘고
        칸을 짝지은 95% 구간이 0을 제외한다
  v24   비통근 26명에서 v39 − v25 의 짝지은 95% 구간이 0을 제외하고 양수
        (부 지표: 통근자에게서도 크게 움직이면 기전 설명이 틀린 것이다)

    python3 verdict_v25R_v24.py [v25R|v24]
"""
import json, os, sys
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows, measure, paired_bootstrap

P = json.load(open("/data/cohort60/personas.json", encoding="utf-8"))
ps = P["personas"] if isinstance(P, dict) else P
prof = {a["id"]: a for a in ps}
EMP = {a for a in prof if isinstance(prof[a].get("commute_min"), (int, float))
       and prof[a]["commute_min"] > 0}
NON = set(prof) - EMP


def arm(path, rep):
    return measure(load_rows("%s@%s" % (path, rep)))


def sub(m, who):
    return {k: v for k, v in m["by_cell"].items() if k[0] in who}


def line(name, a_cells, b_cells):
    r = paired_bootstrap(None, None, a_cells=a_cells, b_cells=b_cells)
    if r is None:
        print("  %-12s (칸이 모자라 견줄 수 없다)" % name); return None
    print("  %-12s 칸 %3d · 집밖 %3d → %3d · %+.4f [%+.4f, %+.4f] %s"
          % (name, r["cells_compared"],
             sum(1 for x in a_cells.values() if x), sum(1 for x in b_cells.values() if x),
             r["mean_difference"], r["ci95"][0], r["ci95"][1],
             "갈림" if r["resolved"] else "0을 지남"))
    return r


which = sys.argv[1] if len(sys.argv) > 1 else "v25R"

if which == "v25R":
    A = "/data/validation_v23_cohort60/v25e/plans_0/responses.jsonl"     # 대조: v25 × 볼일
    B = "/data/validation_v25R_cohort60/v39e/plans_0/responses.jsonl"    # 후보: v39 × 볼일
    if not os.path.exists(B):
        print("v25R 아직 안 끝났다"); raise SystemExit(2)
    a, b = arm(A, "70001"), arm(B, "70001")
    print("=== v25R · 대조 v23(프롬프트 v25 × 볼일) 대비 ===")
    main = line("전체(주지표)", a["by_cell"], b["by_cell"])
    line("비통근", sub(a, NON), sub(b, NON))
    line("통근", sub(a, EMP), sub(b, EMP))
    ok = bool(main and main["resolved"] and main["mean_difference"] > 0)
    print()
    print("사전등록 판정:", "통과" if ok else "기각")
    raise SystemExit(0 if ok else 1)

A = "/data/validation_v24_cohort60/v25/plans_0/responses.jsonl"
B = "/data/validation_v24_cohort60/v39/plans_0/responses.jsonl"
if not (os.path.exists(A) and os.path.exists(B)):
    print("v24 아직 안 끝났다"); raise SystemExit(2)
a, b = arm(A, "70002"), arm(B, "70002")
print("=== v24 · seed 70002 에서 v39 − v25 ===")
sub_r = line("비통근(주지표)", sub(a, NON), sub(b, NON))
com_r = line("통근(부지표)", sub(a, EMP), sub(b, EMP))
line("전체(참고)", a["by_cell"], b["by_cell"])
ok = bool(sub_r and sub_r["resolved"] and sub_r["mean_difference"] > 0)
print()
print("사전등록 판정:", "확증" if ok else "확증 실패")
if ok and com_r and com_r["resolved"]:
    print("  ** 주의: 통근자에게서도 갈렸다 — '의무 없는 사람의 틀을 푼다'는")
    print("     기전 설명이 틀렸을 수 있다. 확증이 아니라 새 물음으로 적는다. **")
raise SystemExit(0 if ok else 1)
