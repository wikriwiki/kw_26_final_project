# -*- coding: utf-8 -*-
"""이미 나가는 사람만 집 밖에서 사는가 — 한 런 안에서, 재직자만으로 본다."""
import json, sys, statistics, random, collections
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows, measure

P = json.load(open("/data/cohort60/personas.json", encoding="utf-8"))
ps = P["personas"] if isinstance(P, dict) else P
prof = {a["id"]: a for a in ps}

for label, spec in (("v25", "/data/validation_v18_cohort60/v25/plans_0/responses.jsonl@70001"),
                    ("v36", "/data/validation_v21_cohort60/v36/plans_0/responses.jsonl@70001")):
    m = measure(load_rows(spec))
    out_cells = collections.Counter()
    for (aid, d, c, a), v in m["by_cell"].items():
        out_cells[aid] += (1 if v else 0)
    emp = [a for a in prof if isinstance(prof[a].get("commute_min"), (int, float))
           and prof[a]["commute_min"] > 0]
    went = [a for a in emp if out_cells[a]]
    stay = [a for a in emp if not out_cells[a]]
    if not went or not stay:
        print("%s: 갈라지지 않음" % label); continue
    cw = [prof[a]["commute_min"] for a in went]
    cs = [prof[a]["commute_min"] for a in stay]
    obs = statistics.mean(cw) - statistics.mean(cs)
    pool = cw + cs
    rng = random.Random(20260921)
    hits = 0
    for _ in range(20000):
        rng.shuffle(pool)
        if statistics.mean(pool[:len(cw)]) - statistics.mean(pool[len(cw):]) >= obs:
            hits += 1
    print("%s · 재직자 %d명 (나간 적 있음 %d · 없음 %d)" % (label, len(emp), len(went), len(stay)))
    print("   통근 %.1f분 대 %.1f분 · 차이 %+.1f분 · 뒤섞기 p=%.4f"
          % (statistics.mean(cw), statistics.mean(cs), obs, (hits + 1) / 20001))
    # 비재직자는 어떤가
    non = [a for a in prof if a not in emp]
    nw = sum(1 for a in non if out_cells[a])
    print("   비재직자 %d명 중 나간 적 있는 사람 %d명 (%.0f%%) · 재직자 %d/%d (%.0f%%)"
          % (len(non), nw, 100*nw/max(1,len(non)), len(went), len(emp), 100*len(went)/max(1,len(emp))))
