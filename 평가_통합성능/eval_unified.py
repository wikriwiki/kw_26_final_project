"""정책 6개(본런 5 + 금연구역)의 실측-시뮬 효과를 공통 업종으로 묶어 하나의 자로 평가한다.

입력: 각 본런 채점 결과(서버에서 받은 score/*.json)와 실측 원문 값(채점 코드·기록 문서에 적힌 값).
단위: 모든 효과를 로그포인트 L = ln(1 + x) 로 바꾼다(P017 은 원래 로그 차).
"""
import io, json, math, random, csv, os
from collections import defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20261019)


def jl(p):
    return json.load(io.open(os.path.join(HERE, p), encoding="utf-8"))


def L(pct):  # % -> 로그포인트
    return math.log1p(pct / 100.0)


rows = []


def add(policy, cat, label, truth_L, sim_L, ci_L, truth_src, note=""):
    rows.append(dict(policy=policy, category=cat, label=label, truth=truth_L, sim=sim_L,
                     lo=ci_L[0] if ci_L else None, hi=ci_L[1] if ci_L else None,
                     truth_src=truth_src, note=note))


# ---------------- P012 상생소비지원금 (KDI 2022.9, 10월 한 달; 시뮬 7일, 2,000명) ----------------
p12 = {r["id"]: r for r in jl("KEEP_p012_main_20261010/score/p012_score.json")["rows"]}
P12MAP = [("K1", "총지출", "총소비"), ("K6", "음식점", "요식"), ("K5", "소매·유통", "유통(적립)"),
          ("K3", "가전·가구", "가전·가구"), ("K8", "미용·대면서비스", "이·미용"), ("K7", "교육", "학원"),
          ("K4", "여가·오락", "여행·레저"), ("K9", "정책 비대상 업종", "제외업종 총 결제")]
for k, cat, lab in P12MAP:
    r = p12[k]
    add("P012", cat, lab, L(r["truth"]), L(r["sim"]), [L(r["ci"][0]), L(r["ci"][1])],
        "KDI 상생소비지원금의 소비 진작 효과(2022.9) 삼중차분, 10월")

# ---------------- P013 긴급재난지원금 (KDI 업종별 매출 증가율 차, %p; 시뮬 지급 뒤 7일) ----------------
p13 = {r["id"]: r for r in jl("KEEP_p013_main_20261010/score/p013_score.json")["rows"]}
P13MAP = [("S4", "음식점", "음식업"), ("S2", "소매·유통", "필수재"), ("S1", "의류·준내구재", "준내구재"),
          ("S3", "미용·대면서비스", "대면서비스")]
for k, cat, lab in P13MAP:
    r = p13[k]
    t = r.get("truth")
    if t is None:
        t = r.get("truth_pp")
    add("P013", cat, lab, L(float(t)), L(r["sim"]), [L(r["ci"][0]), L(r["ci"][1])],
        "KDI 긴급재난지원금 업종별 매출 효과(2020)")

# ---------------- P017 서울시 재난긴급생활비 (IZA DP 14662, 로그; 시뮬 받은 날부터 3주) ----------------
p17 = jl("seoul_main/score/p017_score_21d.json")
T17 = p17["truth"]; M17 = p17["metrics"]
add("P017", "총지출", "총지출", T17["total_apr"], M17["total"]["log_diff_from_receipt"],
    M17["total"]["ci95_from_receipt"], "IZA DP 14662 표 3 (4월)")
for k, cat, lab in [("restaurants", "음식점", "음식점"), ("retail", "소매·유통", "소매"),
                    ("health", "의료", "의료"), ("beauty", "미용·대면서비스", "미용"),
                    ("education", "교육", "교육"), ("entertainment", "여가·오락", "오락여가")]:
    add("P017", cat, lab, T17[k], M17[k]["log_diff_from_receipt"], M17[k]["ci95_from_receipt"],
        "IZA DP 14662 표 4 (4~8월)")


def p017_extra(sub_set, boot=2000, seed=20261011):
    """score_p017.py 와 같은 추정량(받은 날부터 로그 차, 시민 복원추출)으로 하위 업종 하나를 잰다."""
    S = os.path.join(HERE, "seoul_main/score")
    rd = lambda p: [json.loads(l) for l in io.open(os.path.join(S, p), encoding="utf-8") if l.strip()]
    days = [r for r in p17["days"]]
    val = {a: defaultdict(float) for a in ("on", "off")}
    seen = {a: set() for a in ("on", "off")}
    for a in ("on", "off"):
        for r in rd(f"{a}.sector.ledger.jsonl"):
            if r["day"] not in days:
                continue
            seen[a].add(r["aid"])
            for s, amt in (r.get("by_sub") or {}).items():
                if s in sub_set:
                    val[a][(r["aid"], r["day"])] += float(amt or 0)
    recv = {}
    for r in sorted(rd("on.policy.ledger.jsonl"), key=lambda x: x["day"]):
        if r["day"] in days and r.get("grant_received_cumulative", 0) > 0 and r["aid"] not in recv:
            recv[r["aid"]] = r["day"]
    people = sorted(seen["on"] & seen["off"])
    per = lambda a, p: sum(val[a][(p, d)] for d in days if d >= recv[p])
    on_v = np.array([per("on", p) if p in recv else 0 for p in people])
    off_v = np.array([per("off", p) if p in recv else 0 for p in people])
    ok = np.array([p in recv for p in people])
    on_v, off_v = on_v[ok], off_v[ok]
    est = math.log(on_v.sum() / off_v.sum())
    rnd = random.Random(seed); n = len(on_v); bs = []
    for _ in range(boot):
        ix = [rnd.randrange(n) for _ in range(n)]
        a, b = on_v[ix].sum(), off_v[ix].sum()
        if a > 0 and b > 0:
            bs.append(math.log(a / b))
    bs.sort()
    return est, [bs[int(.025 * len(bs))], bs[int(.975 * len(bs)) - 1]], int(on_v.sum()), int(off_v.sum())


est, ci, on_w, off_w = p017_extra({"의류"})
add("P017", "의류·준내구재", "의류", 0.330, est, ci, "IZA DP 14662 표 4 (4~8월)",
    f"이 평가에서 새로 잰 값(score_p017.py 와 같은 추정량), 의류 결제 on {on_w:,}원 / off {off_w:,}원")

# ---------------- 거리두기 2단계 격상 (시뮬: 같은 사람 2단계 - 1.5단계, 7일) ----------------
# 시뮬 값은 data/experiments/NOTION_SHARE_INDICATORS_20261011.md (채점 원장으로 재계산한 값, 2,000회 시민 복원추출)
DS = [("음식점", "음식점업(카페·제과 포함)", -14.0, -7.05, (-9.4, -4.7), "서울연구원 정책리포트 322호 표 10 회귀"),
      ("소매·유통", "소매업", 5.4, 2.05, (-4.24, 8.91), "서울연구원 정책리포트 322호 표 12 회귀"),
      ("총지출", "소상공인 전체 매출", -9.3, -5.3, (-7.7, -2.9), "한국신용데이터 주간지수(동아일보 2020.12.2)"),
      ("카페", "커피전문점", -26.0, -39.8, (-43.86, -35.71), "A사 수도권 전주 대비(조선비즈 2020.12.1)")]
for cat, lab, t, s, c, src in DS:
    add("DIST", cat, lab, L(t), L(s), [L(c[0]), L(c[1])], src)

# ---------------- P010 민생회복 소비쿠폰 (KDI, 사용처 지출) ----------------
p10 = jl("p010_main/score/p010_score.json")
m4 = p10["metrics"]["M4_eligible_spend_pct"] if "metrics" in p10 else p10["M4_eligible_spend_pct"]
add("P010", "정책 대상 업종", "쿠폰 사용처 지출", L(4.93), L(m4["value"]), [L(m4["ci95"][0]), L(m4["ci95"][1])],
    "KDI 민생회복 소비쿠폰 효과(사용처 매출)")

# ---------------- 금연구역 (doinggyu; 시드 1개라 신뢰구간 없음) ----------------
add("NOSMOKE", "여가·오락", "당구장 매출", math.log(math.exp(0.127)), math.log(3433129 / 3668558), None,
    "보건복지부 제출 최종보고서(2018) 표 12, β=0.127")
add("NOSMOKE", "여가·오락", "실내골프 매출", 0.00487, math.log(4192503 / 3983594), None,
    "보건복지부 제출 최종보고서(2018) 표 14, β=0.00487")

# =================================================================================================
POL_NAME = {"P012": "P012 상생소비지원금", "P013": "P013 긴급재난지원금", "P017": "P017 서울시 재난긴급생활비",
            "DIST": "거리두기 2단계 격상", "P010": "P010 민생회복 소비쿠폰", "NOSMOKE": "실내체육시설 금연구역"}
with io.open(os.path.join(OUT, "pairs.csv"), "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]

y = np.array([r["truth"] for r in rows]); x = np.array([r["sim"] for r in rows])
pol = np.array([r["policy"] for r in rows]); n = len(rows)
se = np.array([((r["hi"] - r["lo"]) / (2 * 1.959964)) if r["lo"] is not None else np.nan for r in rows])


def pearson(a, b): return float(np.corrcoef(a, b)[0, 1])
def spearman(a, b): return float(stats.spearmanr(a, b).statistic)
def ccc(a, b):
    ma, mb = a.mean(), b.mean(); va, vb = a.var(), b.var(); cov = ((a - ma) * (b - mb)).mean()
    return float(2 * cov / (va + vb + (ma - mb) ** 2))
def wpearson(a, b, w):
    w = w / w.sum(); ma, mb = (w * a).sum(), (w * b).sum()
    cov = (w * (a - ma) * (b - mb)).sum()
    return float(cov / math.sqrt((w * (a - ma) ** 2).sum() * (w * (b - mb) ** 2).sum()))
def within(a, b, g):
    a2, b2 = a.copy(), b.copy()
    for p in set(g):
        m = g == p
        a2[m] -= a[m].mean(); b2[m] -= b[m].mean()
    keep = np.array([(g == p).sum() >= 2 for p in g])
    return pearson(a2[keep], b2[keep])
def mse(pred, obs): return float(((pred - obs) ** 2).mean())


# 기준 모형
# B0 무반응: 모든 효과 0
# B1 업종 무차별(균일 반응): 같은 정책의 모든 업종에 '시뮬의 그 정책 총지출 효과'를 그대로 준다
#    = 정책이 소비 총량만 바꾸고 업종 구성은 못 바꾸는 모형(MPC 하나로 전파하는 거시 승수식 가정)
TOT_SIM = {"P012": L(p12["K1"]["sim"]), "P013": L(next(r for r in jl("KEEP_p013_main_20261010/score/p013_score.json")["rows"] if r["id"] == "T-total")["sim"]),
           "P017": M17["total"]["log_diff_from_receipt"], "DIST": L(-5.3),
           "P010": L(p10["metrics"]["R_total_from_receipt_pct"]["value"] if "metrics" in p10 else p10["R_total_from_receipt_pct"]["value"]),
           "NOSMOKE": float(np.mean([r["sim"] for r in rows if r["policy"] == "NOSMOKE"]))}
b1 = np.array([TOT_SIM[p] for p in pol])
# B2 상수(실측 평균; 사후적으로만 알 수 있는 값이라 유리한 기준) -> NSE 의 분모


def metrics(ix):
    a, b, g = x[ix], y[ix], pol[ix]
    out = dict(n=len(ix), pearson=pearson(a, b), spearman=spearman(a, b), ccc=ccc(a, b),
               mae=float(np.abs(a - b).mean()), rmse=math.sqrt(mse(a, b)),
               nse=1 - mse(a, b) / mse(np.full_like(b, b.mean()), b),
               skill_vs_null=1 - mse(a, b) / mse(np.zeros_like(b), b),
               skill_vs_uniform=1 - mse(a, b) / mse(b1[ix], b),
               sign_acc=float((np.sign(a) == np.sign(b)).mean()))
    sl = stats.linregress(b, a)  # 시뮬 = α + β·실측 (반응 감도)
    out["slope_sim_on_truth"] = float(sl.slope); out["intercept"] = float(sl.intercept)
    try:
        out["within_policy_r"] = within(a, b, g)
    except Exception:
        out["within_policy_r"] = float("nan")
    return out


allix = np.arange(n)
point = metrics(allix)
wmask = ~np.isnan(se)
point["weighted_pearson"] = wpearson(x[wmask], y[wmask], 1 / se[wmask] ** 2)
point["n_weighted"] = int(wmask.sum())

# 쌍 부트스트랩 (B=10000)
B = 10000
boot = defaultdict(list)
for _ in range(B):
    ix = rng.integers(0, n, n)
    if len(set(pol[ix])) < 2 or np.std(x[ix]) == 0 or np.std(y[ix]) == 0:
        continue
    m = metrics(ix)
    for k, v in m.items():
        if isinstance(v, float) and not math.isnan(v):
            boot[k].append(v)
    wm = wmask[ix]
    if wm.sum() > 3:
        boot["weighted_pearson"].append(wpearson(x[ix][wm], y[ix][wm], 1 / se[ix][wm] ** 2))
ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in boot.items()}

# 정책 군집 부트스트랩 (정책 단위로 복원추출; 군집 6개라 보수적)
pols = sorted(set(pol)); cboot = defaultdict(list)
for _ in range(B):
    pick = rng.choice(pols, len(pols), replace=True)
    ix = np.concatenate([np.where(pol == p)[0] for p in pick])
    if len(set(pick)) < 2 or np.std(x[ix]) == 0:
        continue
    cboot["pearson"].append(pearson(x[ix], y[ix])); cboot["spearman"].append(spearman(x[ix], y[ix]))
    cboot["ccc"].append(ccc(x[ix], y[ix]))
cci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in cboot.items()}

# 순열 검정 (실측 값을 쌍 사이에서 섞는다, 20,000회) — 단측(양의 상관)
P = 20000; perm_r = np.empty(P); perm_rho = np.empty(P); perm_skill = np.empty(P)
for i in range(P):
    yp = rng.permutation(y)
    perm_r[i] = pearson(x, yp); perm_rho[i] = spearman(x, yp)
    perm_skill[i] = 1 - mse(x, yp) / mse(np.zeros_like(yp), yp)
pval = dict(pearson=float((perm_r >= point["pearson"]).mean()),
            spearman=float((perm_rho >= point["spearman"]).mean()),
            skill_vs_null=float((perm_skill >= point["skill_vs_null"]).mean()))
# 부호 정확도의 정확 이항검정 (귀무: 0.5)
k_sign = int(round(point["sign_acc"] * n))
pval["sign_acc_binom"] = float(stats.binomtest(k_sign, n, 0.5, alternative="greater").pvalue)

# 정책 하나씩 빼기
lopo = {p: pearson(x[pol != p], y[pol != p]) for p in pols}
# 정책별
per_pol = {}
for p in pols:
    ix = np.where(pol == p)[0]
    per_pol[p] = dict(n=len(ix), mae=float(np.abs(x[ix] - y[ix]).mean()),
                      sign_acc=float((np.sign(x[ix]) == np.sign(y[ix])).mean()),
                      pearson=pearson(x[ix], y[ix]) if len(ix) >= 4 else None)
# 업종별
per_cat = defaultdict(list)
for r in rows:
    per_cat[r["category"]].append(r)

# ---------------- 수준 지표(단위가 효과율이 아닌 양의 스칼라): 배율 오차 ----------------
LEVEL = [("P010", "쿠폰 1원당 추가 소비(MPC)", 0.21, 0.1449, "한국은행"),
         ("P010", "받은 날부터 하루 소진율(%/일)", 7.05, 24.08, "한국은행 6.6~7.5"),
         ("P013", "사용처 추가 매출 ÷ 지원금(%)", 5.0, 3.53, "KDI 4.2~5.8"),
         ("P013", "1주 누적 소진율(%)", 13.5, 12.0, "서울 실측"),
         ("P012", "캐시백 1원당 추가 카드소비(원)", 1.65, 4.20, "KDI"),
         ("P012", "1인 캐시백(7일 환산, 원)", 5406, 9925, "KDI 두 달 합 ÷2 ×7/31"),
         ("P012", "적립업종 − 제외업종 증가율 차(%p)", 8.40, 7.66, "KDI"),
         ("DIST", "상권 유동인구 감소(%)", 3.7, 4.8, "서울연구원 322호"),
         ("DIST", "주말 다른 구 방문 감소(%)", 13.9, 27.2, "중앙재난안전대책본부 이동량"),
         ("DIST", "소상공인 음식점 매출 감소(%)", 18.5, 7.0, "한국신용데이터")]
lv = [dict(policy=p, label=l, truth=t, sim=s, src=src, log_ratio=math.log(s / t)) for p, l, t, s, src in LEVEL]
lr = np.array([d["log_ratio"] for d in lv])
level_summary = dict(n=len(lv), fac2=float((np.abs(lr) <= math.log(2)).mean()),
                     fac15=float((np.abs(lr) <= math.log(1.5)).mean()),
                     gm_abs_ratio=float(math.exp(np.abs(lr).mean())),
                     median_ratio=float(math.exp(np.median(lr))))
# 구성 유사도(1 - 총변동거리): P010 M3 66.4% · P013 U2 58.9%(상한 95.0%)
composition = [dict(policy="P010", label="쿠폰 결제 업종 구성", similarity=66.4, ci=[65.49, 67.14], cap=100.0),
               dict(policy="P013", label="지원금 사용처 업종 구성", similarity=58.9, ci=[57.4, 60.4], cap=95.0)]

# ---------------- 정밀도 가중(역분산) 지표: 시뮬 추정치의 표본 오차가 알려져 있으므로 메타분석처럼 가중 ----------------
def wmetrics(ix):
    ix = ix[wmask[ix]]
    a, b, g = x[ix], y[ix], pol[ix]
    w = 1 / se[ix] ** 2; w = w / w.sum()
    wm = lambda v: float((w * v).sum())
    wmse = lambda p: wm((p - b) ** 2)
    ma, mb = wm(a), wm(b); va, vb = wm((a - ma) ** 2), wm((b - mb) ** 2); cov = wm((a - ma) * (b - mb))
    out = dict(n=len(ix), w_pearson=cov / math.sqrt(va * vb), w_ccc=2 * cov / (va + vb + (ma - mb) ** 2),
               w_mae=wm(np.abs(a - b)), w_rmse=math.sqrt(wmse(a)),
               w_nse=1 - wmse(a) / wmse(np.full_like(b, mb)),
               w_skill_vs_null=1 - wmse(a) / wmse(np.zeros_like(b)),
               w_skill_vs_uniform=1 - wmse(a) / wmse(b1[ix]),
               w_slope=cov / vb)
    a2, b2 = a.copy(), b.copy(); keep = np.zeros(len(ix), bool)
    for p in set(g):
        m = g == p
        if m.sum() >= 2:
            ww = w[m] / w[m].sum(); a2[m] -= (ww * a[m]).sum(); b2[m] -= (ww * b[m]).sum(); keep |= m
    wk = w[keep] / w[keep].sum()
    out["w_within_policy_r"] = float((wk * a2[keep] * b2[keep]).sum() / math.sqrt((wk * a2[keep] ** 2).sum() * (wk * b2[keep] ** 2).sum()))
    return out


wpoint = wmetrics(allix)
wboot = defaultdict(list)
for _ in range(B):
    ix = rng.integers(0, n, n)
    if wmask[ix].sum() < 5 or len(set(pol[ix][wmask[ix]])) < 2:
        continue
    try:
        m = wmetrics(ix)
    except Exception:
        continue
    for k, v in m.items():
        if isinstance(v, float) and not math.isnan(v):
            wboot[k].append(v)
wci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in wboot.items()}
# 가중 순열 검정: 실측을 섞고(가중치는 시뮬 쪽에 붙어 있다) w_pearson·w_skill 재계산
wix = np.where(wmask)[0]; xw, sw = x[wix], se[wix]; yw = y[wix]; ww = 1 / sw ** 2; ww = ww / ww.sum()
def _wp(a, b):
    ma, mb = (ww * a).sum(), (ww * b).sum()
    return ((ww * (a - ma) * (b - mb)).sum()) / math.sqrt((ww * (a - ma) ** 2).sum() * (ww * (b - mb) ** 2).sum())
perm_wr = np.array([_wp(xw, rng.permutation(yw)) for _ in range(P)])
perm_wsk = np.array([(lambda yp: 1 - (ww * (xw - yp) ** 2).sum() / (ww * yp ** 2).sum())(rng.permutation(yw)) for _ in range(P)])
pval["w_pearson"] = float((perm_wr >= wpoint["w_pearson"]).mean())
pval["w_skill_vs_null"] = float((perm_wsk >= wpoint["w_skill_vs_null"]).mean())
np.save(os.path.join(OUT, "perm_wr.npy"), perm_wr)
# 정책 하나씩 빼기(가중)
wlopo = {}
for p in pols:
    ix = np.where(pol != p)[0]
    try:
        wlopo[p] = wmetrics(ix)["w_pearson"]
    except Exception:
        wlopo[p] = None
# 민감도: 신뢰구간 없는 금연구역 두 점에 가중치 중앙값을 주고 포함
se_med = float(np.nanmedian(se)); se2 = np.where(np.isnan(se), se_med, se)
w2 = 1 / se2 ** 2
sens_all = wpearson(x, y, w2)

res = dict(n_pairs=n, point=point, weighted_point=wpoint, ci_weighted=wci, weighted_lopo=wlopo,
           sensitivity_weighted_with_nosmoke_median_weight=sens_all,
           point_unweighted_note="가중하지 않은 값", ci_pair_bootstrap=ci, ci_policy_cluster_bootstrap=cci, permutation_p=pval,
           leave_one_policy_out_pearson=lopo, per_policy=per_pol,
           per_category={c: [dict(policy=r["policy"], label=r["label"], truth=r["truth"], sim=r["sim"]) for r in v]
                         for c, v in per_cat.items()},
           baselines=dict(B1_uniform_total=TOT_SIM), level=lv, level_summary=level_summary,
           composition=composition, B=B, P=P, policy_names=POL_NAME)
json.dump(res, io.open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
np.save(os.path.join(OUT, "perm_r.npy"), perm_r)
print(json.dumps(dict(n=n, point=point, w=wpoint, wci=wci, cci=cci, p=pval, lopo=lopo, wlopo=wlopo, sens=sens_all, level=level_summary), ensure_ascii=False))
