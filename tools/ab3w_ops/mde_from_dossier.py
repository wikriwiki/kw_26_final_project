# 사람 안 하루 분산으로 D일 짝 비교 최소 차이(p016_power_gate.py 와 같은 식)를 기존 dossier 로 미리 잰다.
import json, sys, math, re, statistics as st
path, D = sys.argv[1], int(sys.argv[2])
pol = json.load(open("/data/repo_ab3w_20261011/data/experiments/P016_ab3w_policy_20261011.json"))
rx = re.compile(pol["eligibility"]["include"]["name_regex"]); subs = set(pol["eligibility"]["include"]["subs"])
M = {"참여 체인 장보기 결제": lambda i: i["sub_category"] in subs and rx.match(i.get("poi") or ""),
     "장보기 업종 결제(모든 가게)": lambda i: i["sub_category"] in subs}
per = {k: [] for k in M}
for l in open(path, encoding="utf-8"):
    r = json.loads(l)
    days = sorted({p["day"] for p in r.get("plans") or []})
    for k, f in M.items():
        v = {d: 0.0 for d in days}
        for p in r.get("plans") or []:
            for i in p.get("items") or []:
                if (i.get("actual_spent") or 0) > 0 and f(i):
                    v[p["day"]] += float(i["actual_spent"])
        per[k].append(list(v.values()))
n = len(per[next(iter(M))])
for k, xs in per.items():
    mean_day = st.mean(v for x in xs for v in x)
    within = st.mean(st.variance(x) for x in xs if len(x) > 1)
    mde = 2.8 * math.sqrt(2 * D * within) / math.sqrt(n)
    users = sum(1 for x in xs if sum(x) > 0)
    print(f"{k}: n={n} · {D}일 1인 기준 {mean_day*D:,.0f}원 · 한 번이라도 {users}/{n} · 최소 차이 {mde:,.0f}원 = {100*mde/(mean_day*D):.1f}%")
