# -*- coding: utf-8 -*-
"""문법이 집 밖 구매 활동을 허용하는가. 금지돼 있으면 프롬프트로는 못 고친다."""
import json, glob, re, collections, os

OUTSIDE = ['meal_dine_in','meal_takeaway','cafe_dine_in','cafe_takeaway','dessert',
           'groceries','convenience','shopping','hair','health_goods',
           'leisure_service','education_service','other_service','bar']
AT_HOME = ['home_delivery','office_delivery','home_online_goods']

d = "/data/validation_v18_cohort60/v25/plans_0/attempts"
paths = sorted(glob.glob(os.path.join(d, "*_grammar.json")))[:60]
admits = collections.Counter()
slots = collections.Counter()
n = 0
for p in paths:
    try:
        eb = json.load(open(p, encoding="utf-8"))["ebnf"]
    except Exception:
        continue
    n += 1
    for a in OUTSIDE + AT_HOME:
        tok = '\\"%s\\"' % a
        if tok in eb:
            admits[a] += 1
            slots[a] += eb.count(tok)

print("문법 %d개를 봤다 (v18 v25)" % n)
print()
print("%-20s %10s %14s" % ("활동", "허용한 문법", "자리(등장 횟수)"))
for a in OUTSIDE:
    print("%-20s %8d/%d %12d" % (a, admits[a], n, slots[a]))
print("  ---- 집에서 끝나는 것 ----")
for a in AT_HOME:
    print("%-20s %8d/%d %12d" % (a, admits[a], n, slots[a]))
