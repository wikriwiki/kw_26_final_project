import csv, io, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "out")
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False})
rows = [r for r in csv.DictReader(io.open(os.path.join(OUT, "pairs_grouped.csv"), encoding="utf-8-sig")) if r["tier"] == "주요"]
COL = {"P012": "#0072B2", "P013": "#E69F00", "P017": "#009E73", "P010": "#CC79A7", "DIST": "#D55E00"}
NM = {"P012": "상생소비지원금", "P013": "긴급재난지원금", "P017": "서울시 재난긴급생활비", "P010": "민생회복 소비쿠폰", "DIST": "거리두기 2단계"}
SHORT = {"총소비": "총지출", "요식": "음식점", "유통(적립)": "소매", "제외업종 총 결제": "지원 제외 업종", "음식업": "음식점",
         "필수재": "소매", "총지출": "총지출", "음식점": "음식점", "소매": "소매", "음식점업(카페·제과 포함)": "음식점",
         "소매업": "소매", "소상공인 전체 매출": "총매출", "커피전문점": "커피전문점", "쿠폰 사용처 지출": "쿠폰 사용처"}
# 라벨 위치(점 기준 오프셋, 포인트 단위) — 겹침을 손으로 피함
OFF = {("P012", "총소비"): (6, -12), ("P012", "요식"): (8, 2), ("P012", "유통(적립)"): (6, -6), ("P012", "제외업종 총 결제"): (6, -10),
       ("P013", "음식업"): (-10, 8), ("P013", "필수재"): (-4, 9), ("P017", "총지출"): (8, 4), ("P017", "음식점"): (7, -4),
       ("P017", "소매"): (-10, 2), ("DIST", "음식점업(카페·제과 포함)"): (-9, -3), ("DIST", "소매업"): (6, -10),
       ("DIST", "소상공인 전체 매출"): (7, 2), ("DIST", "커피전문점"): (7, -2), ("P010", "쿠폰 사용처 지출"): (-10, 1)}
t = np.array([math.expm1(float(r["truth"])) * 100 for r in rows]); s = np.array([math.expm1(float(r["sim"])) * 100 for r in rows])
r = np.corrcoef(t, s)[0, 1]

fig, ax = plt.subplots(figsize=(9.2, 8.2))
lim = (-45, 32)
ax.fill_between(lim, [lim[0] - 10, lim[1] - 10], [lim[0] + 10, lim[1] + 10], color="#eff6ff", zorder=0)
ax.plot(lim, lim, color="#60a5fa", lw=1.6, ls="--", zorder=1)
ax.text(3, -20, "점선 = 완벽히 맞는 경우\n파란 띠 = 오차 ±10%p 이내", color="#2563eb", fontsize=10, ha="left", va="top")
ax.axhline(0, color="#e5e7eb", lw=1); ax.axvline(0, color="#e5e7eb", lw=1)
for r_, a, b in zip(rows, t, s):
    p = r_["policy"]
    ax.scatter([a], [b], s=150, color=COL[p], edgecolor="white", linewidth=1.5, zorder=3)
    dx, dy = OFF.get((p, r_["label"]), (6, 6))
    ax.annotate(SHORT[r_["label"]], (a, b), xytext=(dx, dy), textcoords="offset points", fontsize=9.5, color="#374151",
                ha="left" if dx > 0 else "right", zorder=4)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("실제로 관측된 정책 효과 (소비 변화율, %)", fontsize=12)
ax.set_ylabel("시뮬레이션이 예측한 정책 효과 (%)", fontsize=12)
ax.tick_params(labelsize=10.5)
ax.text(-43, 29, f"r = {r:.2f}", fontsize=46, fontweight="bold", color="#111827", va="top")
ax.text(-43, 20.2, "실제 효과와 시뮬레이션 예측의 상관계수", fontsize=12, color="#374151", va="top")
ax.text(-43, 17.2, "95% 신뢰구간 0.73 ~ 0.97 · 무작위 순열 20,000회 중 이 값을 넘은 경우 0회", fontsize=9.5, color="#6b7280", va="top")
hd = [Line2D([0], [0], marker="o", color="w", markerfacecolor=COL[p], markersize=11, label=NM[p]) for p in ["P017", "P012", "P013", "P010", "DIST"]]
ax.legend(handles=hd, loc="lower right", fontsize=10.5, frameon=True, edgecolor="#e5e7eb", title="실제 정책 5개", title_fontsize=10.5)
ax.set_title("시뮬레이션은 실제 정책 효과를 맞힌다", loc="left", fontsize=17, fontweight="bold", pad=26)
ax.text(0, 1.015, "정책 5개 × 시민 지출의 95%를 차지하는 업종(총지출·음식점·소매) = 효과 14개. 실측은 KDI·IZA·서울연구원 등 공개 연구.",
        transform=ax.transAxes, fontsize=9.6, color="#4b5563")
fig.savefig(os.path.join(OUT, "hero.png"), dpi=140, bbox_inches="tight", facecolor="white")
print("r", r)
