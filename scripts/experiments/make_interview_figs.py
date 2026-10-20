"""포스터 그림: 금연구역 — 시뮬 시민 인터뷰 대 실제 이용객 설문, 군집별 이유."""
import json, os, io, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "bq2_run")
OUT = os.path.join(HERE, "..", "eval", "out", "poster"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9ca3af"})
INK, MUTED, SIM, REAL = "#111827", "#6b7280", "#2563eb", "#9ca3af"
A = json.load(io.open(os.path.join(D, "analysis.json"), encoding="utf-8"))
P = A["point"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=220, bbox_inches="tight", facecolor="white"); plt.close(fig)


# ---------- 6. 찬성 · 반대 · 모름: 실측 설문 대 시뮬 인터뷰 (시행 전 · 후) ----------
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
for ax, ph, title in [(axs[0], "pre", "시행 전"), (axs[1], "post", "시행 후 2주")]:
    cats = ["찬성", "반대", "모름"]; x = np.arange(3); w = 0.36
    t = [A["truth"][ph][c] for c in cats]; s = [P[ph][c] for c in cats]
    ax.bar(x - w / 2, t, w, color=REAL); ax.bar(x + w / 2, s, w, color=SIM)
    for xi, tv, sv in zip(x, t, s):
        ax.text(xi - w / 2, tv + 1.5, f"{tv:.1f}", ha="center", fontsize=13, color="#4b5563", fontweight="bold")
        ax.text(xi + w / 2, sv + 1.5, f"{sv:.1f}", ha="center", fontsize=13, color=SIM, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(cats, fontsize=14)
    sim = P["sim_pre"] if ph == "pre" else P["sim_post"]
    ax.set_title(f"{title}  ·  분포 유사도 {sim * 100:.0f}%", fontsize=14, color=INK, loc="left")
    ax.set_ylim(0, 105)
axs[0].set_ylabel("응답 비율 (%)", fontsize=12)
axs[1].legend(handles=[Patch(color=REAL, label="실제 이용객 설문 (600명)"), Patch(color=SIM, label="시뮬 시민 인터뷰")], fontsize=12, frameon=False, loc="upper right")
fig.suptitle("금연구역 찬반: 시뮬 시민 인터뷰가 실제 이용객 설문을 92~93% 재현", x=0.06, y=1.06, ha="left", fontsize=18, fontweight="bold", color=INK)
d = P["delta"]
fig.text(0.06, 0.975, f"시행 전→후 변화  찬성 {d['찬성']:+.1f}%p (실측 +5.2) · 반대 {d['반대']:+.1f}%p (실측 -5.8) · 모름 {d['모름']:+.1f}%p (실측 +0.7)  →  변화폭 평균 오차 {P['mae']:.1f}%p",
         fontsize=12.5, color=SIM, fontweight="bold")
fig.text(0.06, -0.03, "질문은 실제 설문(보고서 표 55, BQ2)과 같은 3지선다. 시뮬 시민은 시뮬레이션 그래프에 쌓인 자기 기록(흡연 여부·이용 내역)을 보고 K-EXAONE으로 답했다.",
         fontsize=10, color=MUTED)
save(fig, "p6_interview_vs_survey.png")

# ---------- 7. 군집별로 말하는 이유가 다르다 ----------
TH = {"health": "간접흡연·건강", "clean": "쾌적함·냄새", "family": "가족·아이", "smoker_burden": "흡연자 불편", "rights": "흡연 권리·과한 규제"}
ans = {json.loads(l)["aid"]: json.loads(l) for l in io.open(os.path.join(D, "pre.jsonl"), encoding="utf-8") if l.strip()}
for ph in ("post",):
    p = os.path.join(D, f"{ph}.jsonl")
    if os.path.exists(p):
        pass
th = [json.loads(l) for l in io.open(os.path.join(D, "themes_pre.jsonl"), encoding="utf-8")]
groups = collections.defaultdict(list)
for t in th:
    r = ans.get(t["aid"])
    if not r or t["themes"] is None or not r.get("answer") or "_10대_" in r["aid"]:
        continue
    if r["smoking_status"] == "non_smoker" and r["answer"]["choice"] == "찬성":
        groups["비흡연자 · 찬성"].append(t["themes"])
    elif r["smoking_status"] == "smoker" and r["answer"]["choice"] == "찬성":
        groups["흡연자 · 찬성"].append(t["themes"])
    elif r["answer"]["choice"] == "반대":
        groups["반대한 시민"].append(t["themes"])
order = ["비흡연자 · 찬성", "흡연자 · 찬성", "반대한 시민"]
fig, ax = plt.subplots(figsize=(12, 5.4))
keys = list(TH); x = np.arange(len(keys)); w = 0.26
cols = {"비흡연자 · 찬성": "#2563eb", "흡연자 · 찬성": "#60a5fa", "반대한 시민": "#D55E00"}
for i, g in enumerate(order):
    v = groups[g]; n = len(v)
    vals = [100 * sum(k in s for s in v) / n for k in keys]
    ax.bar(x + (i - 1) * w, vals, w, color=cols[g], label=f"{g} (n={n})")
    for xi, val in zip(x + (i - 1) * w, vals):
        if val >= 5:
            ax.text(xi, val + 1.5, f"{val:.0f}", ha="center", fontsize=10.5, color=INK)
ax.set_xticks(x); ax.set_xticklabels([TH[k] for k in keys], fontsize=13)
ax.set_ylabel("그 이유를 말한 비율 (%)", fontsize=12); ax.set_ylim(0, 112)
ax.legend(fontsize=11.5, frameon=False, loc="upper right")
ax.set_title("누가 답하느냐에 따라 이유가 다르다", loc="left", fontsize=18, fontweight="bold", color=INK, pad=12)
fig.text(0.06, -0.04, f"시행 전 인터뷰 {sum(len(groups[g]) for g in order)}명(만 19세 이상)의 이유 문장을 주제로 분류. 분류기는 이유 문장만 보고 흡연 여부·찬반은 보지 않는다.\n실제 현장 발언(기사): \"놀면서 담배를 피우는 최후의 보루였는데 아쉽다\", \"이제는 어느 당구장이든 갈 수 있어서 좋다\".",
         fontsize=10, color=MUTED)
save(fig, "p7_reasons_by_cluster.png")
print("ok")

# ---------- 9. 흡연자도 찬성한다: 실측 대 시뮬 ----------
fig, ax = plt.subplots(figsize=(11, 5.0))
groups = [("흡연자 · 시행 전", 63.0, P["smoker"][0]), ("흡연자 · 시행 후", 75.4, P["smoker"][1]),
          ("비흡연자 · 시행 전", 88.9, P["non_smoker"][0]), ("비흡연자 · 시행 후", 94.1, P["non_smoker"][1])]
x = np.arange(len(groups)); w = 0.36
ax.bar(x - w / 2, [g[1] for g in groups], w, color=REAL); ax.bar(x + w / 2, [g[2] for g in groups], w, color=SIM)
for xi, g in zip(x, groups):
    ax.text(xi - w / 2, g[1] + 1.5, f"{g[1]:.1f}", ha="center", fontsize=13, fontweight="bold", color="#4b5563")
    ax.text(xi + w / 2, g[2] + 1.5, f"{g[2]:.1f}", ha="center", fontsize=13, fontweight="bold", color=SIM)
ax.axhline(50, color="#d1d5db", lw=1, ls=":")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=12.5)
ax.set_ylim(0, 118); ax.set_ylabel("금연구역 찬성 비율 (%)", fontsize=12)
ax.legend(handles=[Patch(color=REAL, label="실제 이용객 설문"), Patch(color=SIM, label="시뮬 시민 인터뷰")], fontsize=12, frameon=True, framealpha=1, edgecolor="#e5e7eb", loc="upper left", ncol=2)
ax.set_title("흡연자도 과반이 찬성한다 — 실측도, 시뮬도", loc="left", fontsize=18, fontweight="bold", color=INK, pad=12)
fig.text(0.06, -0.05, "실측: 보고서 표 55(현재흡연자 시행 전 138명 · 후 167명, 비흡연자 405명). 시뮬: 만 19세 이상 이용 시민. 시뮬은 흡연자의 찬성을 실측보다 높게 잡는다(시행 전 +23%p).",
         fontsize=10, color=MUTED)
save(fig, "p9_smoker_support.png")
print("p9 ok")
