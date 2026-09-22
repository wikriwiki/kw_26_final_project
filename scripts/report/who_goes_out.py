# -*- coding: utf-8 -*-
"""집 밖 구매가 나오는 3.3% 는 누구에게 떨어지나.

특정 시민에게 몰리면 입력의 어떤 특징이 몰아주는 것이고, 고르게 흩어지면 우연이다.
전자면 그 특징이 다음 후보의 표적이 된다.
"""
import json, sys, collections, statistics
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows, measure

RUNS = {
    "v25": "/data/validation_v18_cohort60/v25/plans_0/responses.jsonl@70001",
    "v34": "/data/validation_v18_cohort60/v34/plans_0/responses.jsonl@70001",
    "v36": "/data/validation_v21_cohort60/v36/plans_0/responses.jsonl@70001",
}

P = json.load(open("/data/cohort60/personas.json", encoding="utf-8"))
ps = P["personas"] if isinstance(P, dict) else P
prof = {a["id"]: a for a in ps}

people = {}
for label, spec in RUNS.items():
    m = measure(load_rows(spec))
    who = collections.Counter()
    for (aid, date, case, arm), v in m["by_cell"].items():
        if v:
            who[aid] += 1
    people[label] = who
    print("%-5s 집 밖 칸 %d개 · 그것을 가진 시민 %d명 (전체 %d명)"
          % (label, sum(who.values()), len(who), len(prof)))

labels = list(people)
sets = {l: set(people[l]) for l in labels}
print()
print("시민 수준 겹침:")
for i, a in enumerate(labels):
    for b in labels[i + 1:]:
        inter = sets[a] & sets[b]
        print("  %s ∩ %s = %d명   (%s %d명 · %s %d명)"
              % (a, b, len(inter), a, len(sets[a]), b, len(sets[b])))

allp = set().union(*sets.values())
print()
print("한 번이라도 집 밖 구매를 계획한 시민 %d / %d명" % (len(allp), len(prof)))
print()
print("그 시민들의 특징 (전체 평균과 견줌):")
for key, name in (("home_h_wd", "평일 집 체류"), ("daily_wd", "평일 소비액"),
                  ("commute_min", "통근 분"), ("mobility", "이동성 분위")):
    inv = [prof[a].get(key) for a in allp if isinstance(prof[a].get(key), (int, float))]
    out = [prof[a].get(key) for a in prof if a not in allp and isinstance(prof[a].get(key), (int, float))]
    if inv and out:
        print("  %-10s 나간 적 있는 %d명 %.1f · 없는 %d명 %.1f"
              % (name, len(inv), statistics.mean(inv), len(out), statistics.mean(out)))
job_in = collections.Counter(prof[a].get("job", "?")[:6] for a in allp)
print("  직업(나간 적 있는 시민):", dict(job_in.most_common(6)))
