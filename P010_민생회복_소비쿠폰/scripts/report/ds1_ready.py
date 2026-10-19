# -*- coding: utf-8 -*-
"""DS-1 이 잴 것을 갖게 됐나 — 결제 전에 계획에서 미리 본다.

실측 -14.1% 는 매장 취식 붕괴가 배달 증가를 압도한 값이다. 그러려면
무정책 팔에 **줄어들 매장 취식**이 있어야 한다. 그게 계획에 있는지 본다.
"""
import json, sys, collections
sys.path.insert(0, "/data/jp")
from outside_activity_variety import load_rows

DINE = ("meal_dine_in", "meal_takeaway", "cafe_dine_in", "cafe_takeaway")
DELI = ("home_delivery", "office_delivery")

RUNS = {
    "v25 (볼일 없음)": "/data/validation_v18_cohort60/v25/plans_0/responses.jsonl@70001",
    "v39 (볼일 없음)": "/data/validation_v22_cohort60/v39/plans_0/responses.jsonl@70001",
    "v23 (볼일 있음)": "/data/validation_v23_cohort60/v25e/plans_0/responses.jsonl@70001",
}

for label, spec in RUNS.items():
    rows = load_rows(spec)
    cells = collections.Counter(); dine = collections.Counter(); deli = collections.Counter()
    dcells = collections.Counter()
    for (rep, aid, date, case, arm), r in rows.items():
        if case != "distancing":
            continue
        cells[arm] += 1
        evs = (r.get("execution_plan") or {}).get("events", [])
        d = sum(1 for e in evs if e["activity_id"] in DINE)
        v = sum(1 for e in evs if e["activity_id"] in DELI)
        dine[arm] += d; deli[arm] += v
        if d:
            dcells[arm] += 1
    print("=== %s ===" % label)
    print("  거리두기 칸  무정책 %d · 정책 %d" % (cells["off"], cells["on"]))
    print("  매장·포장 건수   %3d → %3d   (그런 칸 %d → %d)"
          % (dine["off"], dine["on"], dcells["off"], dcells["on"]))
    print("  배달 건수        %3d → %3d" % (deli["off"], deli["on"]))
    # 0건이냐 아니냐로 가르면 안 된다. 1건짜리 분모는 분모가 아니다 —
    # v18 의 DS-4 가 디저트 한 건으로 [-100, -100] 을 냈던 자리가 그 교훈이다.
    if dine["off"] >= 10:
        print("  >> 무정책 팔 매장 %d건 — DS-1 이 잴 것을 갖는다" % dine["off"])
    else:
        print("  >> 무정책 팔 매장 %d건뿐 — 분모가 아니다. DS-1 은 사실상 배달을 잰다"
              % dine["off"])
