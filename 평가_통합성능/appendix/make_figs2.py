import io, json, os, math, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#6b7280",
                     "axes.labelcolor": "#111827", "xtick.color": "#374151", "ytick.color": "#374151"})
T = json.load(io.open(os.path.join(OUT, "tiers.json"), encoding="utf-8"))
M = json.load(io.open(os.path.join(OUT, "metrics.json"), encoding="utf-8"))
rows = list(csv.DictReader(io.open(os.path.join(OUT, "pairs_grouped.csv"), encoding="utf-8-sig")))
for r in rows:
    for k in ("truth", "sim", "lo", "hi"):
        r[k] = float(r[k]) if r[k] not in ("", None) else None
NAME = M["policy_names"]
ORDER = ["P012", "P013", "P017", "P010", "DIST", "NOSMOKE"]
COL = {"P012": "#0072B2", "P013": "#E69F00", "P017": "#009E73", "P010": "#CC79A7", "DIST": "#D55E00", "NOSMOKE": "#56B4E9"}
MK = {"P012": "o", "P013": "s", "P017": "D", "P010": "^", "DIST": "v", "NOSMOKE": "X"}
GROUPS = ["업종 합계", "음식점·카페", "소매·유통", "상품(의류·가전·가구)", "서비스(미용·의료·교육·여가)"]
S = 100.0
MAJ, ALL, MIN = T["주요"], T["전체"], T["소액"]


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130, bbox_inches="tight", facecolor="white"); plt.close(fig)


def legend_pol(ax, loc):
    hd = [Line2D([0], [0], marker=MK[p], color="w", markerfacecolor=COL[p], markersize=9, label=NAME[p]) for p in ORDER]
    ax.legend(handles=hd, loc=loc, fontsize=8.4, frameon=True, framealpha=0.95, edgecolor="#e5e7eb")


# ---------- 그림 1: (a) 주요 업종 14쌍 (b) 전체 27쌍 ----------
fig, axs = plt.subplots(1, 2, figsize=(13.4, 6.2))
for ax, tier, lim in [(axs[0], "주요", (-55, 30)), (axs[1], "전체", (-60, 95))]:
    sel = [r for r in rows if tier == "전체" or r["tier"] == tier]
    ax.plot(lim, lim, color="#9ca3af", lw=1.2, ls="--", zorder=1)
    ax.axhline(0, color="#e5e7eb", lw=0.8); ax.axvline(0, color="#e5e7eb", lw=0.8)
    for r in sel:
        t, s = r["truth"] * S, r["sim"] * S; p = r["policy"]
        if r["lo"] is not None:
            ax.plot([t, t], [max(r["lo"] * S, lim[0]), min(r["hi"] * S, lim[1])], color=COL[p], lw=1.1, alpha=0.5, zorder=2)
        hollow = tier == "전체" and r["tier"] == "소액"
        ax.scatter([t], [min(max(s, lim[0]), lim[1])], s=70, marker=MK[p], zorder=3,
                   color="white" if hollow else COL[p], edgecolor=COL[p] if hollow else "white", linewidth=1.6 if hollow else 1.0)
    xs = np.array([r["truth"] for r in sel]) * S; ys = np.array([r["sim"] for r in sel]) * S
    b, a = np.polyfit(xs, ys, 1); ax.plot(np.array(lim), a + b * np.array(lim), color="#111827", lw=1.5)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("실측 효과 (로그포인트 × 100)"); ax.set_ylabel("시뮬레이션 효과 (로그포인트 × 100)")
    d = T[tier]
    txt = (f"피어슨 r = {d['point']['pearson']:.2f}  [{d['ci']['pearson'][0]:.2f}, {d['ci']['pearson'][1]:.2f}]\n"
           f"일치상관 CCC = {d['point']['ccc']:.2f}   기울기 = {d['point']['slope']:.2f}\n"
           f"부호 정확도 = {round(d['point']['sign'] * d['point']['n'])}/{d['point']['n']}   n = {d['point']['n']}")
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, va="top", fontsize=9.6,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#e5e7eb"))
axs[0].set_title("(a) 주요 업종 14쌍 — 업종 합계 · 음식점·카페 · 소매·유통 (지출의 95%)", loc="left", fontsize=11)
axs[1].set_title("(b) 전체 27쌍 — 빈 표시는 소액 업종(상품 · 서비스, 지출의 5%)", loc="left", fontsize=11)
legend_pol(axs[0], "lower right")
fig.suptitle("그림 1. 실측 대 시뮬레이션 — 정책 6개의 업종 효과", x=0.05, ha="left", fontsize=13, fontweight="bold")
fig.text(0.05, -0.02, "점선 = 완전 일치(y = x), 실선 = 최소제곱 회귀, 세로 막대 = 시뮬 95% 신뢰구간(시민 단위 부트스트랩). 금연구역(×)은 시드 1개라 막대 없음.",
         fontsize=8.6, color="#4b5563")
save(fig, "fig1_scatter.png")

# ---------- 그림 2: 지표별 점추정과 95% 신뢰구간 ----------
items = [("피어슨 r", "pearson"), ("스피어만 ρ", "spearman"), ("일치상관 CCC", "ccc"), ("기울기 (1 = 크기 같음)", "slope"),
         ("기술점수: 무반응 기준", "skill_null"), ("기술점수: 업종 균일 반응 기준", "skill_uniform"), ("내시-서트클리프 효율", "nse")]
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for i, (lab, k) in enumerate(items[::-1]):
    for off, d, col in [(0.17, MAJ, "#1d4ed8"), (-0.17, ALL, "#9ca3af")]:
        v, c = d["point"][k], d["ci"][k]
        lo = max(c[0], -0.99)
        ax.plot([lo, c[1]], [i + off, i + off], color=col, lw=2.2)
        if c[0] < -0.99:
            ax.annotate("", xy=(-1.0, i + off), xytext=(-0.88, i + off), arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        ax.scatter([max(v, -0.99)], [i + off], color=col, s=40, zorder=3)
        if d is MAJ:
            ax.text(c[1] + 0.04, i + off, f"{v:.2f} [{c[0]:.2f}, {c[1]:.2f}]", va="center", fontsize=8.6, color="#1d4ed8")
ax.axvline(0, color="#374151", lw=1)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]])
ax.set_xlim(-1.0, 1.85); ax.set_xlabel("값 (상관 · 기술점수: 1 = 완전 일치, 0 = 정보 없음)")
ax.legend(handles=[Line2D([0], [0], color="#1d4ed8", lw=2.2, marker="o", label="주요 업종 14쌍"),
                   Line2D([0], [0], color="#9ca3af", lw=2.2, marker="o", label="전체 27쌍")], loc="upper left", fontsize=9, frameon=True, edgecolor="#e5e7eb")
ax.set_title("그림 2. 성능 지표와 95% 부트스트랩 신뢰구간", loc="left", fontsize=12, fontweight="bold")
ax.text(0.0, -0.17, "쌍 단위 퍼센타일 부트스트랩 B = 10,000. -1 아래로 내려가는 구간은 화살표로 잘라 표시(원값은 표 2).",
        transform=ax.transAxes, fontsize=8.3, color="#4b5563")
save(fig, "fig2_forest.png")

# ---------- 그림 3: 순열 귀무분포 (주요 업종) ----------
pr = np.load(os.path.join(OUT, "perm_tier_주요.npy"))
fig, ax = plt.subplots(figsize=(7.4, 3.6))
ax.hist(pr, bins=60, color="#cbd5e1", edgecolor="white")
obs = MAJ["point"]["pearson"]; ax.axvline(obs, color="#1d4ed8", lw=2.4)
ax.text(obs - 0.02, ax.get_ylim()[1] * 0.92, f"관측값 {obs:.2f}\n20,000회 중 이보다\n큰 값 0회 (p < 0.0001)", ha="right", va="top", color="#1d4ed8", fontsize=9.5)
ax.set_xlim(-1, 1.02); ax.set_xlabel("실측을 쌍 사이에서 무작위로 섞었을 때의 피어슨 r (주요 업종 14쌍, 20,000회)"); ax.set_ylabel("횟수")
ax.set_title("그림 3. 귀무가설(실측과 시뮬이 무관) 아래 상관의 분포", loc="left", fontsize=12, fontweight="bold")
save(fig, "fig3_permutation.png")

# ---------- 그림 4: 업종 묶음별 쌍 비교 (빈칸 없는 덤벨) ----------
order_rows = []
for g in GROUPS:
    for p in ORDER:
        for r in rows:
            if r["group"] == g and r["policy"] == p:
                order_rows.append(r)
n = len(order_rows)
fig, ax = plt.subplots(figsize=(9.8, 0.36 * n + 1.6))
yy = 0; yt, yl, gsep = [], [], []
lim = (-60, 95)
for g in GROUPS:
    rr = [r for r in order_rows if r["group"] == g]
    gsep.append((yy, g, len(rr)))
    for r in rr:
        t, s = r["truth"] * S, r["sim"] * S
        ax.plot([t, min(max(s, lim[0]), lim[1])], [yy, yy], color="#d1d5db", lw=2, zorder=1)
        ax.scatter([t], [yy], s=46, color="#6b7280", zorder=3, marker="o")
        ax.scatter([min(max(s, lim[0]), lim[1])], [yy], s=58, color=COL[r["policy"]], marker=MK[r["policy"]], edgecolor="white", zorder=4)
        if abs(s) > 60 or s < lim[0] or s > lim[1]:
            ax.text(min(max(s, lim[0]), lim[1]) + 2, yy, f"{s:+.0f}", va="center", fontsize=7.8, color=COL[r["policy"]])
        yt.append(yy); yl.append(f"{NAME[r['policy']].split(' ')[0] if r['policy'] not in ('DIST', 'NOSMOKE') else ('거리두기' if r['policy'] == 'DIST' else '금연구역')} · {r['label']}")
        yy += 1
    yy += 0.9
for y0, g, k in gsep:
    tier = "주요" if g in ("업종 합계", "음식점·카페", "소매·유통") else "소액"
    d = T["by_group"][g]
    ax.text(lim[0] + 1, y0 - 0.75, f"{g}  ·  {tier} 업종  ·  평균 절대 오차 {d['mae'] * 100:.1f}  ·  부호 정확도 {round(d['sign'] * d['n'])}/{d['n']}",
            fontsize=9.2, fontweight="bold", color="#111827", va="center", bbox=dict(boxstyle="round,pad=0.25", fc="#f3f4f6", ec="none"), zorder=5)
ax.axvline(0, color="#374151", lw=1)
ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8.6); ax.invert_yaxis(); ax.set_xlim(lim)
ax.set_xlabel("효과 (로그포인트 × 100)")
ax.legend(handles=[Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b7280", markersize=8, label="실측"),
                   Line2D([0], [0], marker="o", color="w", markerfacecolor="#1d4ed8", markersize=8, label="시뮬 (색 = 정책)")],
          loc="lower right", fontsize=9, frameon=True, edgecolor="#e5e7eb")
ax.set_title("그림 4. 업종 묶음별 실측과 시뮬 — 27쌍 전부", loc="left", fontsize=12, fontweight="bold")
save(fig, "fig4_dumbbell.png")

# ---------- 그림 5: 수준 지표 배율 ----------
lv = M["level"]
fig, ax = plt.subplots(figsize=(7.8, 4.6))
ax.axvspan(math.log(0.5), math.log(2), color="#dbeafe", zorder=0); ax.axvline(0, color="#374151", lw=1)
for i, d in enumerate(lv[::-1]):
    ax.scatter([d["log_ratio"]], [i], color=COL[d["policy"]], marker=MK[d["policy"]], s=60, edgecolor="white", zorder=3)
    ax.text(d["log_ratio"] + (0.06 if d["log_ratio"] >= 0 else -0.06), i, f"×{math.exp(d['log_ratio']):.2f}",
            va="center", ha="left" if d["log_ratio"] >= 0 else "right", fontsize=8.6)
lab = lambda d: (NAME[d["policy"]].split(" ")[0] if d["policy"] not in ("DIST",) else "거리두기") + " · " + d["label"].replace(chr(8722), "-")
ax.set_yticks(range(len(lv))); ax.set_yticklabels([lab(d) for d in lv[::-1]], fontsize=8.8)
tk = [0.25, 0.5, 1, 2, 4]; ax.set_xticks([math.log(t) for t in tk]); ax.set_xticklabels([f"×{t:g}" for t in tk])
ax.set_xlim(math.log(0.2), math.log(5)); ax.set_xlabel("시뮬 / 실측 (로그 축)")
ls = M["level_summary"]
ax.set_title(f"그림 5. 수준 지표 {ls['n']}개 — 2배 이내 {ls['fac2'] * 100:.0f}% · 배율 중앙값 ×{ls['median_ratio']:.2f}",
             loc="left", fontsize=11.5, fontweight="bold")
ax.text(0, -0.17, "파란 띠 = 2배 이내(FAC2). 감소율 지표는 감소 폭끼리 비교.", transform=ax.transAxes, fontsize=8.3, color="#4b5563")
save(fig, "fig5_levels.png")

# ---------- 그림 6: 정책 하나씩 빼기 (주요 업종) ----------
lo = MAJ["lopo"]; ks = [k for k in ORDER if k in lo]
fig, ax = plt.subplots(figsize=(7.4, 3.0))
ax.barh(range(len(ks)), [lo[k] for k in ks], color=[COL[k] for k in ks], height=0.6)
ax.axvline(MAJ["point"]["pearson"], color="#111827", ls="--", lw=1.2)
for i, k in enumerate(ks):
    ax.text(lo[k] + 0.01, i, f"{lo[k]:.2f}", va="center", fontsize=9)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([f"{NAME[k]} 제외" for k in ks]); ax.invert_yaxis()
ax.set_xlim(0, 1.05); ax.set_xlabel(f"남은 정책의 주요 업종 쌍으로 계산한 피어슨 r (점선 = 전체 {MAJ['point']['pearson']:.2f})")
ax.set_title("그림 6. 정책 하나를 뺐을 때의 상관 — 한 정책에 기대지 않는다", loc="left", fontsize=12, fontweight="bold")
save(fig, "fig6_lopo.png")

# ---------- 그림 7: 지출 비중과 오차 ----------
share = T["spending_share_offarm"]; pay = T["payer_share_range_pct"]
G4 = ["음식점·카페", "소매·유통", "서비스(미용·의료·교육·여가)", "상품(의류·가전·가구)"]
fig, axs = plt.subplots(1, 2, figsize=(14.0, 3.9)); fig.subplots_adjust(wspace=0.62)
ax = axs[0]
ax.barh(range(4), [share[g] for g in G4], color=["#1d4ed8", "#1d4ed8", "#9ca3af", "#9ca3af"], height=0.6)
for i, g in enumerate(G4):
    ax.text(share[g] + 1, i, f"{share[g]:.1f}%  (결제 시민 {pay[g][0]:.0f}~{pay[g][1]:.0f}%)", va="center", fontsize=8.8)
ax.set_yticks(range(4)); ax.set_yticklabels(G4); ax.invert_yaxis(); ax.set_xlim(0, 135)
ax.set_xlabel("정책 없는 세계 지출에서 차지하는 비중 (%)"); ax.set_title("(a) 지출 비중", loc="left", fontsize=10.5)
ax = axs[1]
bg = T["by_group"]; G5 = ["업종 합계"] + G4
ax.barh(range(5), [bg[g]["mae"] * 100 for g in G5], color=["#1d4ed8"] * 3 + ["#9ca3af"] * 2, height=0.6)
for i, g in enumerate(G5):
    ax.text(bg[g]["mae"] * 100 + 0.8, i, f"{bg[g]['mae'] * 100:.1f}  (n = {bg[g]['n']})", va="center", fontsize=8.8)
ax.set_yticks(range(5)); ax.set_yticklabels(G5); ax.invert_yaxis(); ax.set_xlim(0, 48)
ax.set_xlabel("평균 절대 오차 (로그포인트 × 100)"); ax.set_title("(b) 업종 묶음별 오차", loc="left", fontsize=10.5)
fig.suptitle("그림 7. 지출의 95%가 몰린 업종에서 오차가 작다", x=0.05, y=1.06, ha="left", fontsize=12, fontweight="bold")
save(fig, "fig7_share_error.png")
print("ok")
