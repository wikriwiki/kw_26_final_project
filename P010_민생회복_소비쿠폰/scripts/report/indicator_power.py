# -*- coding: utf-8 -*-
"""수렴할 수 있는 지표가 남아 있나 — 필요한 표본을 센다.

이 파일은 과거 탐색 단계의 표본 크기 계산이다. EM-3의 이틀 전후 시뮬 비율과
KDI 전년동기 대비 비율은 기간·대조군이 달라 외부 크기 검증으로 쓰지 않는다.
여기서 계산한 칸 수를 새 검증 런의 표본 크기로 전용할 수 없다.
"""
import json, math, sys

RUNS = [
    ("v18 · 60명 · 전체", "/data/validation_v18_cohort60/v25/pooled_cluster.json", 328),
    ("v20 · 120명 · 전체", "/data/validation_v20_cohort120/v25/pooled_cluster.json", 486),
    ("v20 · 120명 · 짝만", "/data/validation_v20_cohort120/v25/pooled_paired.json", 394),
]
TARGET = {"EM-3": 7.3, "DS-2": 4.2, "P012-1": 20.82, "LV-2": None}

print("%-22s %-8s %10s %26s %10s" % ("런", "지표", "점추정", "95% 구간", "반폭"))
rows = {}
for label, path, cells in RUNS:
    try:
        d = json.load(open(path, encoding="utf-8"))["pooled"]
    except Exception as e:
        print("  (%s 없음)" % label); continue
    for k in ("EM-3", "DS-2", "P012-1"):
        b = d.get(k)
        if not b or b.get("lo") is None:
            continue
        half = (b["hi"] - b["lo"]) / 2
        rows.setdefault(k, []).append((label, cells, b["mean"], b["lo"], b["hi"], half))
        print("%-22s %-8s %+10.1f  [%+9.1f, %+9.1f] %10.1f"
              % (label, k, b["mean"], b["lo"], b["hi"], half))

print()
print("필요한 칸 수 — 구간 반폭이 점추정보다 작아지려면 (반폭 ∝ 1/√n)")
print("%-8s %10s %10s %12s %14s" % ("지표", "실측", "현재 점추정", "현재 반폭", "필요 칸 수"))
for k, v in rows.items():
    label, cells, mean, lo, hi, half = v[-1]          # 가장 최근·가장 큰 런
    if mean == 0:
        continue
    need = cells * (half / abs(mean)) ** 2
    t = TARGET.get(k)
    print("%-8s %10s %+10.1f %12.1f %14s"
          % (k, ("%+.1f" % t) if t else "수치없음", mean, half,
             ("%,d" % int(need)) if need < 1e7 else "천만 칸 초과"))
print()
print("주의: 반폭 ∝ 1/√n 은 분모가 충분할 때의 근사다. 구매 건수가 수십 건인")
print("      지금 상태에서는 낙관적인 값이고, 실제로는 더 필요하다.")
