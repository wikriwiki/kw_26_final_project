# -*- coding: utf-8 -*-
"""v21 의 사전등록 판정을 기계로 내린다.

성공선은 이렇게 적혀 있다 — "칸당 종류가 늘고, 칸을 짝지은 95% 구간이 0을 제외한다".
둘 다 기계로 확인되므로 사람 판단이 끼어들 자리가 없다. 그래서 여기서 판정하고,
셋 다 기각이면 v22(v39)가 돌 조건이 충족된 것이다.

종료코드 0 = 셋 다 기각(v22 조건 충족) · 1 = 하나라도 통과 · 2 = 아직 못 냄
"""
import json, sys
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows, measure, paired_bootstrap

BASE = "/data/validation_v18_cohort60/v25/plans_0/responses.jsonl@70001"
CAND = {c: "/data/validation_v21_cohort60/%s/plans_0/responses.jsonl@70001" % c
        for c in ("v36", "v37", "v38")}

import os
missing = [c for c, p in CAND.items()
           if not os.path.exists("/data/validation_v21_cohort60/%s/plans_0/summary.json" % c)]
if missing:
    print("아직 안 끝난 후보:", missing)
    raise SystemExit(2)

b = measure(load_rows(BASE))
print("대조 v25  칸당 %.4f · 집 밖 칸 %d"
      % (sum(b["per_cell"]) / len(b["per_cell"]), sum(1 for v in b["by_cell"].values() if v)))
passed = []
for c, spec in CAND.items():
    m = measure(load_rows(spec))
    r = paired_bootstrap(b["per_cell"], m["per_cell"],
                         a_cells=b["by_cell"], b_cells=m["by_cell"])
    ok = bool(r and r["resolved"] and r["mean_difference"] > 0)
    print("  %-4s 칸당 %.4f · 집 밖 칸 %3d · 차이 %+.4f [%+.4f, %+.4f] → %s"
          % (c, sum(m["per_cell"]) / len(m["per_cell"]),
             sum(1 for v in m["by_cell"].values() if v),
             r["mean_difference"], r["ci95"][0], r["ci95"][1], "통과" if ok else "기각"))
    if ok:
        passed.append(c)
print()
if passed:
    print("통과한 후보:", passed, "→ v22 는 돌지 않는다 (조건 미충족)")
    raise SystemExit(1)
print("셋 다 기각 → v22(v39) 조건 충족")
raise SystemExit(0)
