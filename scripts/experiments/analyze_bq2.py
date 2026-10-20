"""BQ2 인터뷰 결과를 실측 이용객 설문(표 55)과 대조한다. 지표는 BQ2_PREREG.md 에 결과 보기 전 고정."""
import json, random, sys, io
from pathlib import Path

D = Path(sys.argv[1])
TRUTH = {"pre": {"찬성": 83.67, "반대": 13.33, "모름": 3.0}, "post": {"찬성": 88.83, "반대": 7.5, "모름": 3.67}}
TRUTH_SMOKE = {"smoker": {"pre": 63.0, "post": 75.4}, "non_smoker": {"pre": 88.9, "post": 94.1}}
DELTA = {"찬성": 5.17, "반대": -5.83, "모름": 0.67}
C3 = ["찬성", "반대", "모름"]
SURVEY_SMOKER_SHARE = {"pre": 138 / 600, "post": 167 / 600}


def load(ph):
    rows = [json.loads(l) for l in io.open(D / f"{ph}.jsonl", encoding="utf-8")]
    rows = [r for r in rows if "_10대_" not in r["aid"]]  # 실제 설문은 만 19세 이상
    ok = [r for r in rows if r.get("answer")]
    return rows, ok


def shares(ok, w=None):
    tot = sum((w(r) if w else 1) for r in ok)
    return {c: 100 * sum((w(r) if w else 1) for r in ok if r["answer"]["choice"] == c) / tot for c in C3}


def support(ok, grp):
    s = [r for r in ok if r["smoking_status"] == grp]
    return (100 * sum(r["answer"]["choice"] == "찬성" for r in s) / len(s)) if s else None, len(s)


def stats(pre, post, w_pre=None, w_post=None):
    sp, so = shares(pre, w_pre), shares(post, w_post)
    d = {c: so[c] - sp[c] for c in C3}
    mae = sum(abs(d[c] - DELTA[c]) for c in C3) / 3
    sim_pre = 1 - 0.5 * sum(abs(sp[c] - TRUTH["pre"][c]) for c in C3) / 100
    sim_post = 1 - 0.5 * sum(abs(so[c] - TRUTH["post"][c]) for c in C3) / 100
    gp = {ph: (support(x, "non_smoker")[0] or 0) - (support(x, "smoker")[0] or 0) for ph, x in (("pre", pre), ("post", post))}
    return dict(pre=sp, post=so, delta=d, mae=mae, sim_pre=sim_pre, sim_post=sim_post, gap_pre=gp["pre"], gap_post=gp["post"],
                smoker=(support(pre, "smoker")[0], support(post, "smoker")[0]),
                non_smoker=(support(pre, "non_smoker")[0], support(post, "non_smoker")[0]))


def reweight(ok, ph):
    n = len(ok); ns = sum(r["smoking_status"] == "smoker" for r in ok)
    t = SURVEY_SMOKER_SHARE[ph]
    ws, wn = t / (ns / n), (1 - t) / (1 - ns / n)
    return lambda r: ws if r["smoking_status"] == "smoker" else wn


rows_pre, pre = load("pre"); rows_post, post = load("post")
point = stats(pre, post)
rw = stats(pre, post, reweight(pre, "pre"), reweight(post, "post"))
rnd = random.Random(20261020); B = 4000; bs = {k: [] for k in ("mae", "sim_pre", "sim_post", "gap_pre", "gap_post", "d찬성", "d반대", "d모름", "smk_pre", "smk_post", "non_pre", "non_post")}
for _ in range(B):
    a = [pre[rnd.randrange(len(pre))] for _ in pre]; b = [post[rnd.randrange(len(post))] for _ in post]
    s = stats(a, b)
    bs["mae"].append(s["mae"]); bs["sim_pre"].append(s["sim_pre"]); bs["sim_post"].append(s["sim_post"])
    bs["gap_pre"].append(s["gap_pre"]); bs["gap_post"].append(s["gap_post"])
    for c in C3:
        bs["d" + c].append(s["delta"][c])
    bs["smk_pre"].append(s["smoker"][0] or 0); bs["smk_post"].append(s["smoker"][1] or 0)
    bs["non_pre"].append(s["non_smoker"][0] or 0); bs["non_post"].append(s["non_smoker"][1] or 0)
ci = {k: [sorted(v)[int(.025 * B)], sorted(v)[int(.975 * B) - 1]] for k, v in bs.items()}
res = dict(n_pre=len(rows_pre), ok_pre=len(pre), n_post=len(rows_post), ok_post=len(post),
           counts_pre={g: support(pre, g)[1] for g in ("smoker", "non_smoker", "unknown")},
           counts_post={g: support(post, g)[1] for g in ("smoker", "non_smoker", "unknown")},
           point=point, reweighted_to_survey_smoker_share=rw, ci=ci, truth=TRUTH, truth_smoke=TRUTH_SMOKE, truth_delta=DELTA)
(D / "analysis.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
f = lambda x: "-" if x is None else f"{x:.1f}"
print(f"응답 시행 전 {len(pre)}/{len(rows_pre)} · 시행 후 {len(post)}/{len(rows_post)}  흡연자 {res['counts_pre']['smoker']}/{res['counts_post']['smoker']}")
for ph in ("pre", "post"):
    print(ph, "시뮬", {c: f(point[ph][c]) for c in C3}, "실측", TRUTH[ph])
print("변화 시뮬", {c: f(point["delta"][c]) for c in C3}, "실측", DELTA, "MAE", f(point["mae"]), "CI", [f(x) for x in ci["mae"]])
print("수준 유사도 전/후", round(point["sim_pre"], 3), round(point["sim_post"], 3))
print("흡연자 찬성 전→후", f(point["smoker"][0]), "→", f(point["smoker"][1]), "(실측 63.0→75.4)", " 비흡연자", f(point["non_smoker"][0]), "→", f(point["non_smoker"][1]), "(실측 88.9→94.1)")
print("집단 차 전/후", f(point["gap_pre"]), f(point["gap_post"]), "(실측 25.9 / 18.7)", "CI", [f(x) for x in ci["gap_pre"]], [f(x) for x in ci["gap_post"]])
print("재가중 MAE", f(rw["mae"]), "전", {c: f(rw["pre"][c]) for c in C3}, "후", {c: f(rw["post"][c]) for c in C3})
