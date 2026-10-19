"""업종을 5개 묶음으로 통합하고, 지출 비중으로 '주요 업종'과 '소액 업종'을 나눠 따로 잰다.

주요 업종 = 업종 합계 · 음식점·카페 · 소매·유통 (정책 없는 세계 지출의 95.4%, 시민 71~100%가 결제)
소액 업종 = 상품(의류·가전·가구) · 서비스(미용·의료·교육·여가) (지출의 4.5%, 결제 시민 4~27%)
"""
import io, os, csv, json, math
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(20261019)
rows = list(csv.DictReader(io.open(os.path.join(OUT, "pairs.csv"), encoding="utf-8-sig")))
GROUP = {"총지출": "업종 합계", "정책 대상 업종": "업종 합계", "정책 비대상 업종": "업종 합계",
         "음식점": "음식점·카페", "카페": "음식점·카페", "소매·유통": "소매·유통",
         "의류·준내구재": "상품(의류·가전·가구)", "가전·가구": "상품(의류·가전·가구)",
         "미용·대면서비스": "서비스(미용·의료·교육·여가)", "의료": "서비스(미용·의료·교육·여가)",
         "교육": "서비스(미용·의료·교육·여가)", "여가·오락": "서비스(미용·의료·교육·여가)"}
MAJOR = {"업종 합계", "음식점·카페", "소매·유통"}
for r in rows:
    r["group"] = GROUP[r["category"]]; r["tier"] = "주요" if r["group"] in MAJOR else "소액"
with io.open(os.path.join(OUT, "pairs_grouped.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
M = json.load(io.open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
TOT = M["baselines"]["B1_uniform_total"]


def arr(sel):
    x = np.array([float(r["sim"]) for r in sel]); y = np.array([float(r["truth"]) for r in sel])
    b1 = np.array([TOT[r["policy"]] for r in sel]); return x, y, b1


def met(x, y, b1):
    mse = lambda p: float(((p - y) ** 2).mean())
    return dict(n=len(x), pearson=float(np.corrcoef(x, y)[0, 1]), spearman=float(stats.spearmanr(x, y).statistic),
                ccc=float(2 * np.mean((x - x.mean()) * (y - y.mean())) / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)),
                slope=float(stats.linregress(y, x).slope), mae=float(np.abs(x - y).mean()),
                skill_null=1 - mse(x) / mse(np.zeros_like(y)), skill_uniform=1 - mse(x) / mse(b1),
                nse=1 - mse(x) / mse(np.full_like(y, y.mean())), sign=float((np.sign(x) == np.sign(y)).mean()))


res = {}
for tier in ("주요", "소액", "전체"):
    sel = [r for r in rows if tier == "전체" or r["tier"] == tier]
    x, y, b1 = arr(sel); pt = met(x, y, b1); n = len(x)
    bs = {k: [] for k in pt if k != "n"}
    for _ in range(10000):
        ix = rng.integers(0, n, n)
        if np.std(x[ix]) == 0 or np.std(y[ix]) == 0:
            continue
        m = met(x[ix], y[ix], b1[ix])
        for k in bs:
            bs[k].append(m[k])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    P = 20000; pr = np.empty(P); ps = np.empty(P); pk = np.empty(P)
    for i in range(P):
        yp = rng.permutation(y); pr[i] = np.corrcoef(x, yp)[0, 1]; ps[i] = stats.spearmanr(x, yp).statistic
        pk[i] = 1 - ((x - yp) ** 2).mean() / (yp ** 2).mean()
    pv = dict(pearson=float((pr >= pt["pearson"]).mean()), spearman=float((ps >= pt["spearman"]).mean()),
              skill_null=float((pk >= pt["skill_null"]).mean()),
              sign_binom=float(stats.binomtest(int(round(pt["sign"] * n)), n, 0.5, alternative="greater").pvalue))
    pols = sorted(set(r["policy"] for r in sel))
    lopo = {p: float(np.corrcoef(*[np.array([float(r[k]) for r in sel if r["policy"] != p]) for k in ("sim", "truth")])[0, 1])
            for p in pols if sum(r["policy"] != p for r in sel) >= 4}
    res[tier] = dict(point=pt, ci=ci, p=pv, lopo=lopo, policies=pols,
                     pairs=[f"{r['policy']}:{r['label']}" for r in sel])
    np.save(os.path.join(OUT, f"perm_tier_{tier}.npy"), pr)
# 묶음별
res["by_group"] = {}
for g in ["업종 합계", "음식점·카페", "소매·유통", "상품(의류·가전·가구)", "서비스(미용·의료·교육·여가)"]:
    sel = [r for r in rows if r["group"] == g]; x, y, b1 = arr(sel)
    res["by_group"][g] = dict(n=len(sel), mae=float(np.abs(x - y).mean()), sign=float((np.sign(x) == np.sign(y)).mean()),
                              pearson=float(np.corrcoef(x, y)[0, 1]) if len(sel) >= 4 else None)
res["spending_share_offarm"] = {"음식점·카페": 75.4, "소매·유통": 20.0, "서비스(미용·의료·교육·여가)": 3.4,
                               "상품(의류·가전·가구)": 1.1, "기타": 0.1}
res["payer_share_range_pct"] = {"음식점·카페": [99.2, 100.0], "소매·유통": [71.0, 97.0],
                                "서비스(미용·의료·교육·여가)": [20.9, 26.9], "상품(의류·가전·가구)": [4.2, 7.7]}
json.dump(res, io.open(os.path.join(OUT, "tiers.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
for t in ("주요", "소액", "전체"):
    print(t, {k: round(v, 3) for k, v in res[t]["point"].items()}, {k: [round(a, 3) for a in v] for k, v in res[t]["ci"].items()}, res[t]["p"], res[t]["lopo"])
print(res["by_group"])
