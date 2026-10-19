"""가중 방식이 결론을 좌우하지 않는지 확인한다.

1) 무작위효과 가중: w_i = 1 / (SE_i^2 + tau^2). tau^2 = 시뮬과 실측 사이 '모형 오차' 분산(Paule-Mandel).
   고정효과(1/SE^2)는 점 하나가 가중치 77%(유효 표본 1.7)라 쓰지 않는다.
2) 정밀도 문턱별 민감도: 시뮬 95% 신뢰구간 반폭이 h 이하인 쌍만 남겨 r 을 계산.
3) 희석 보정 상관: 시뮬 표본오차로 낮아진 r 을 신뢰도로 나눈다(Spearman 1904).
"""
import io, os, json, math, csv
import numpy as np
from scipy import stats, optimize

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
rng = np.random.default_rng(20261019)
rows = list(csv.DictReader(io.open(os.path.join(OUT, "pairs.csv"), encoding="utf-8-sig")))
x = np.array([float(r["sim"]) for r in rows]); y = np.array([float(r["truth"]) for r in rows])
pol = np.array([r["policy"] for r in rows])
se = np.array([((float(r["hi"]) - float(r["lo"])) / 3.919928) if r["lo"] else np.nan for r in rows])
has = ~np.isnan(se)
M = json.load(io.open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
TOT = M["baselines"]["B1_uniform_total"]; b1 = np.array([TOT[p] for p in pol])


def pm_tau2(e, v):
    """Paule-Mandel: sum e^2/(v+t) = k-1 (평균 0 이 아니라 가중 평균을 뺀 잔차)."""
    k = len(e)
    def f(t):
        w = 1 / (v + t); m = (w * e).sum() / w.sum()
        return (w * (e - m) ** 2).sum() - (k - 1)
    if f(0) <= 0:
        return 0.0
    return optimize.brentq(f, 0, 10)


def wstats(a, b, w, base):
    w = w / w.sum(); wm = lambda v: float((w * v).sum())
    ma, mb = wm(a), wm(b); va, vb = wm((a - ma) ** 2), wm((b - mb) ** 2); cov = wm((a - ma) * (b - mb))
    mse = lambda p: wm((p - b) ** 2)
    return dict(r=cov / math.sqrt(va * vb), ccc=2 * cov / (va + vb + (ma - mb) ** 2), slope=cov / vb,
                skill_null=1 - mse(a) / mse(np.zeros_like(b)), skill_uniform=1 - mse(a) / mse(base),
                nse=1 - mse(a) / mse(np.full_like(b, mb)), mae=wm(np.abs(a - b)), ess=1 / (w ** 2).sum())


ix = np.where(has)[0]
e = x[ix] - y[ix]; v = se[ix] ** 2
tau2 = pm_tau2(e, v)
w_re = 1 / (v + tau2)
point_re = wstats(x[ix], y[ix], w_re, b1[ix])

# 부트스트랩: 쌍 복원추출, 매번 tau^2 재추정
B = 10000; bs = {k: [] for k in point_re}
for _ in range(B):
    j = rng.integers(0, len(ix), len(ix)); k = ix[j]
    if len(set(pol[k])) < 2 or np.std(x[k]) == 0:
        continue
    t2 = pm_tau2(x[k] - y[k], se[k] ** 2)
    s = wstats(x[k], y[k], 1 / (se[k] ** 2 + t2), b1[k])
    for kk, vv in s.items():
        bs[kk].append(vv)
ci_re = {k: [float(np.percentile(vv, 2.5)), float(np.percentile(vv, 97.5))] for k, vv in bs.items()}
# 순열(실측을 섞는다; 가중치는 시뮬 쪽 SE 와 고정 tau^2)
P = 20000; pr = np.empty(P); psk = np.empty(P)
for i in range(P):
    yp = rng.permutation(y[ix]); s = wstats(x[ix], yp, w_re, b1[ix]); pr[i] = s["r"]; psk[i] = s["skill_null"]
p_re = dict(r=float((pr >= point_re["r"]).mean()), skill_null=float((psk >= point_re["skill_null"]).mean()))
np.save(os.path.join(OUT, "perm_re_r.npy"), pr)

# 정책 하나씩 빼기 (무작위효과 가중)
lopo_re = {}
for p in sorted(set(pol)):
    k = ix[pol[ix] != p]
    t2 = pm_tau2(x[k] - y[k], se[k] ** 2)
    lopo_re[p] = wstats(x[k], y[k], 1 / (se[k] ** 2 + t2), b1[k])["r"]

# 정밀도 문턱 민감도
half = se * 1.959964
curve = []
for h in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 5.0]:
    k = np.where(has & (half <= h))[0]
    if len(k) >= 5:
        curve.append(dict(h=h, n=int(len(k)), pearson=float(np.corrcoef(x[k], y[k])[0, 1]),
                          spearman=float(stats.spearmanr(x[k], y[k]).statistic),
                          mae=float(np.abs(x[k] - y[k]).mean()),
                          sign=float((np.sign(x[k]) == np.sign(y[k])).mean())))

# 희석 보정: 신뢰도 = 1 - 평균 표본분산 / 관측분산 (시뮬 쪽만; 실측 오차는 모름 -> 보수적)
rel_x = 1 - np.mean(se[ix] ** 2) / np.var(x[ix], ddof=1)
r_obs = float(np.corrcoef(x[ix], y[ix])[0, 1])
r_dis = r_obs / math.sqrt(rel_x) if rel_x > 0 else None

res = dict(tau2=tau2, tau=math.sqrt(tau2), n=int(len(ix)), random_effects=point_re, ci_random_effects=ci_re,
           permutation_p_random_effects=p_re, lopo_random_effects=lopo_re, precision_curve=curve,
           disattenuation=dict(r_obs_n25=r_obs, reliability_sim=float(rel_x), r_disattenuated=r_dis),
           fixed_effect_ess=float((1 / v).sum() ** 2 / ((1 / v) ** 2).sum()))
json.dump(res, io.open(os.path.join(OUT, "robust.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))

# ---- 추가: 정밀 부분집합의 기술점수 · 무작위효과 가중 정책 내 상관 ----
def subset_skill(h):
    k = np.where(has & (half <= h))[0]
    a, b = x[k], y[k]
    mse = lambda p: float(((p - b) ** 2).mean())
    return dict(h=h, n=int(len(k)), skill_null=1 - mse(a) / mse(np.zeros_like(b)),
                skill_uniform=1 - mse(a) / mse(b1[k]), nse=1 - mse(a) / mse(np.full_like(b, b.mean())),
                ccc=float(2 * np.mean((a - a.mean()) * (b - b.mean())) / (a.var() + b.var() + (a.mean() - b.mean()) ** 2)),
                policies=sorted(set(pol[k])), labels=[rows[i]["policy"] + ":" + rows[i]["label"] for i in k])
sub = [subset_skill(h) for h in (0.15, 0.2, 0.3)]
w = w_re / w_re.sum(); a2 = x[ix].copy(); b2 = y[ix].copy(); keep = np.zeros(len(ix), bool); g = pol[ix]
for p in set(g):
    m = g == p
    if m.sum() >= 2:
        ww = w[m] / w[m].sum(); a2[m] -= (ww * x[ix][m]).sum(); b2[m] -= (ww * y[ix][m]).sum(); keep |= m
wk = w[keep] / w[keep].sum()
within_re = float((wk * a2[keep] * b2[keep]).sum() / math.sqrt((wk * a2[keep] ** 2).sum() * (wk * b2[keep] ** 2).sum()))
res["precise_subset"] = sub; res["within_policy_r_random_effects"] = within_re
json.dump(res, io.open(os.path.join(OUT, "robust.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(dict(sub=sub, within=within_re), ensure_ascii=False, indent=1))
