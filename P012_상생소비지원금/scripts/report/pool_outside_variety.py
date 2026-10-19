# -*- coding: utf-8 -*-
"""세 라운드를 묶어 v34−v25 를 한 번 추정한다.

라운드를 먼저 재추출하고 그 안에서 칸을 재추출한다. 라운드가 칸보다 더 많이
다르기 때문이다 — v16·v17 은 열두 명이고 v18 은 예순 명이라 기저 수준 자체가
다섯 배 다르다. 그 차이를 칸 재추출만으로는 볼 수 없다.
"""
import json, random, sys, statistics
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows, measure

ROUNDS = {
    "v16": ("/data/validation_v16_final/v25/plans_0/responses.jsonl",
            "/data/validation_v16_final/v34/plans_0/responses.jsonl"),
    "v17": ("/data/validation_v17_power/v25/plans_0/responses.jsonl",
            "/data/validation_v17_power/v34/plans_0/responses.jsonl"),
    "v18": ("/data/validation_v18_cohort60/v25/plans_0/responses.jsonl",
            "/data/validation_v18_cohort60/v34/plans_0/responses.jsonl"),
}

blocks = {}
print("%-6s %8s %8s %9s %9s" % ("라운드", "칸(v25)", "칸(v34)", "v25", "v34"))
for name, (a, b) in ROUNDS.items():
    pa = measure(load_rows(a))["per_cell"]
    pb = measure(load_rows(b))["per_cell"]
    blocks[name] = (pa, pb)
    print("%-6s %8d %8d %9.4f %9.4f" % (name, len(pa), len(pb),
                                        statistics.mean(pa), statistics.mean(pb)))

names = list(blocks)
rng = random.Random(20260921)
draws = []
for _ in range(6000):
    chosen = [names[rng.randrange(len(names))] for _ in names]
    sa, sb, na, nb = 0.0, 0.0, 0, 0
    for nm in chosen:
        pa, pb = blocks[nm]
        for _ in range(len(pa)):
            sa += pa[rng.randrange(len(pa))]; na += 1
        for _ in range(len(pb)):
            sb += pb[rng.randrange(len(pb))]; nb += 1
    draws.append(sb / nb - sa / na)
draws.sort()
lo, hi = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
alla = [x for nm in names for x in blocks[nm][0]]
allb = [x for nm in names for x in blocks[nm][1]]
point = statistics.mean(allb) - statistics.mean(alla)
print()
print("합산 (칸 %d + %d)" % (len(alla), len(allb)))
print("  차이 (v34 − v25)  %+.4f  [%+.4f, %+.4f]  %s"
      % (point, lo, hi, "0 제외" if (lo > 0) == (hi > 0) else "← 0을 지난다"))
print("  라운드를 블록으로 재추출했다. 블록이 셋뿐이라 구간이 넓다 — 그것이 정직한 폭이다.")
