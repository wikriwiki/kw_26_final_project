# -*- coding: utf-8 -*-
"""조립에서 살아남은 칸이 무작위인가.

120명에서 960칸 중 487칸만 남았다. 남은 칸이 특정 사람에게 쏠려 있으면
지표는 그 사람들의 지표이지 코호트의 지표가 아니다. 그러면 정답지와 견주는
것 자체가 흔들린다.
"""
import json, collections, statistics, random

ASM = "/data/validation_v20_cohort120/v25/asm_71001.json"
SRC = "/data/cohort120/source_meal.json"
PER = "/data/cohort120/personas.json"

src = json.load(open(SRC, encoding="utf-8"))
asm = json.load(open(ASM, encoding="utf-8"))
P = json.load(open(PER, encoding="utf-8"))
ps = P["personas"] if isinstance(P, dict) else P
prof = {a["id"]: a for a in ps}

designed = collections.Counter(c["aid"] for c in src["cells"])
kept = collections.Counter(c["aid"] for c in asm["cells"])
print("설계 %d칸 · 조립 %d칸 · 시민 %d명" % (len(src["cells"]), len(asm["cells"]), len(designed)))
print()

rates = {a: kept[a] / designed[a] for a in designed}
v = sorted(rates.values())
print("사람별 생존율: 최소 %.2f · 1사분 %.2f · 중앙 %.2f · 3사분 %.2f · 최대 %.2f"
      % (v[0], v[len(v)//4], statistics.median(v), v[3*len(v)//4], v[-1]))
zero = [a for a in rates if rates[a] == 0]
full = [a for a in rates if rates[a] == 1]
print("한 칸도 안 남은 시민 %d명 · 전부 남은 시민 %d명 (전체 %d)" % (len(zero), len(full), len(rates)))
print()

print("생존율이 특징과 함께 움직이나 (상위 절반 대 하위 절반):")
hi = sorted(rates, key=lambda a: -rates[a])[: len(rates)//2]
lo = sorted(rates, key=lambda a: -rates[a])[len(rates)//2 :]
for key, name in (("daily_wd", "평일 소비액"), ("home_h_wd", "평일 집 체류"),
                  ("commute_min", "통근 분"), ("mobility", "이동성 분위"),
                  ("spend_decile", "소비 분위")):
    a = [prof[x].get(key) for x in hi if isinstance(prof[x].get(key), (int, float))]
    b = [prof[x].get(key) for x in lo if isinstance(prof[x].get(key), (int, float))]
    if len(a) < 5 or len(b) < 5:
        continue
    obs = statistics.mean(a) - statistics.mean(b)
    pool = a + b
    rng = random.Random(20260921); hits = 0
    for _ in range(20000):
        rng.shuffle(pool)
        if abs(statistics.mean(pool[:len(a)]) - statistics.mean(pool[len(a):])) >= abs(obs):
            hits += 1
    print("  %-10s 생존 상위 %.1f · 하위 %.1f · 차이 %+.1f · p=%.4f%s"
          % (name, statistics.mean(a), statistics.mean(b), obs, (hits+1)/20001,
             "   ← 쏠림" if (hits+1)/20001 < 0.05 else ""))

print()
print("기전·팔별 생존율 (대조가 깨지지 않았나):")
d2 = collections.Counter((c["case"], c["arm"]) for c in src["cells"])
k2 = collections.Counter((c["case"], c["arm"]) for c in asm["cells"])
for k in sorted(d2):
    print("  %-14s %-4s  %3d / %3d  (%.0f%%)" % (k[0], k[1], k2[k], d2[k], 100*k2[k]/d2[k]))
