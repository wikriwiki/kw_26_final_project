# -*- coding: utf-8 -*-
"""서울사랑상품권 가맹점 데이터 — 분포 통계·사용성 평가·시각화.

입력: data/coupon/seoul_love_merchants.csv (convert_merchant_xlsx.py 산출)
      output/stats/agent_profiles.json (동별 인구 → 자치구 인구)
출력: output/sim/report/coupon_merchant_stats.png (3패널)
      + 콘솔 통계 리포트 (자치구별 가맹점 수·인구 1천명당 밀도·업종 분포·사용성 평가)

사용성 평가 관점 (백테스트 T2/T3 전제):
  - 자치구 간 가맹점 밀도 편차 → 쿠폰 사용처 접근성의 공간 이질성 (T3 해석 변수)
  - 업종 편중 → 사용가능/불가 업종 대비의 실측 구조 (T2 설계 정합)
  - 기준일(2024-03) vs 정책(2025-07) 시차 → False Negative 원인 → 룰 fallback 필요성
"""
from __future__ import annotations

import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "coupon" / "seoul_love_merchants.csv"
PROFILES = ROOT / "output" / "stats" / "agent_profiles.json"
OUT_PNG = ROOT / "output" / "sim" / "report" / "coupon_merchant_stats.png"

# 서울 자치구 코드(앞 5자리, KOSIS/NSO) → 이름
GU_NAME = {
    "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
    "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
    "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
    "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
    "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
    "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
    "11740": "강동구",
}


def main() -> None:
    # ── 가맹점 분포 ──
    gu_cnt: Counter = Counter()
    cat_cnt: Counter = Counter()
    n = 0
    with io.open(CSV, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gu_cnt[row["자치구명"].strip()] += 1
            cat_cnt[(row["서울페이앱분류업종"] or "?").strip()] += 1
            n += 1

    # ── 자치구 인구 (agent_profiles의 동별 population 집계) ──
    pop_gu: dict[str, float] = defaultdict(float)
    try:
        P = json.loads(PROFILES.read_text(encoding="utf-8"))
        for v in P.values():
            cd = (v.get("location") or {}).get("adm_cd_8") or ""
            pop = (v.get("demographics") or {}).get("population") or 0
            name = GU_NAME.get(cd[:5])
            if name:
                pop_gu[name] += pop
    except Exception as e:
        print("(인구 집계 실패 — 밀도 생략)", e)

    density = {g: gu_cnt[g] / (pop_gu[g] / 1000.0)
               for g in gu_cnt if pop_gu.get(g, 0) > 0}

    # ── 리포트 ──
    print(f"가맹점 총 {n:,}건 / 자치구 {len(gu_cnt)}개 / 업종 {len(cat_cnt)}종")
    vals = sorted(gu_cnt.values())
    print(f"자치구별 가맹점 수: min={vals[0]:,}({min(gu_cnt, key=gu_cnt.get)}) "
          f"max={vals[-1]:,}({max(gu_cnt, key=gu_cnt.get)}) "
          f"max/min={vals[-1]/vals[0]:.2f}배")
    if density:
        dv = sorted(density.items(), key=lambda x: -x[1])
        print(f"인구 1천명당 가맹점: max={dv[0][1]:.1f}({dv[0][0]}) min={dv[-1][1]:.1f}({dv[-1][0]}) "
              f"— 접근성 공간 이질성 {dv[0][1]/dv[-1][1]:.2f}배")
    top_cat = cat_cnt.most_common(10)
    share_food = 100 * cat_cnt.get("음식점", 0) / max(n, 1)
    print(f"업종: 음식점 {share_food:.1f}% 최다, top10 = {[(c, f'{100*v/n:.1f}%') for c, v in top_cat]}")
    print("\n[사용성 평가]")
    print(f"  ① 커버리지 상한: 가맹점 {n/1000:.0f}K vs 상가 POI ~540K → 최대 매칭률 ~26%"
          f" — 미매칭 POI는 룰 fallback 필수 (설계 반영됨)")
    print(f"  ② 시차: 기준일 2024-03 vs 정책 2025-07 — 이후 신규 가맹 누락(False Negative)"
          f" → True 확정 전용, False 단정 금지 (설계 반영됨)")
    print(f"  ③ 공간: 자치구 간 밀도 편차 존재 — T3(공간 벡터) 해석 시 접근성 통제 변수로 사용 가능")

    # ── 시각화 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("(matplotlib 없음 — 시각화 생략)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    fig.suptitle("서울사랑상품권 가맹점(=소비쿠폰 사용처 실측) 분포 — 2024-03 기준 140,140건", fontsize=13)

    gu_sorted = sorted(gu_cnt.items(), key=lambda x: -x[1])
    axes[0].barh([g for g, _ in gu_sorted][::-1], [v for _, v in gu_sorted][::-1], color="#4C78A8")
    axes[0].set_title("자치구별 가맹점 수")
    axes[0].tick_params(labelsize=8)

    if density:
        d_sorted = sorted(density.items(), key=lambda x: -x[1])
        axes[1].barh([g for g, _ in d_sorted][::-1], [v for _, v in d_sorted][::-1], color="#F58518")
        axes[1].set_title("인구 1천명당 가맹점 (접근성)")
        axes[1].tick_params(labelsize=8)
    else:
        axes[1].axis("off")

    top12 = cat_cnt.most_common(12)
    axes[2].barh([c for c, _ in top12][::-1], [v for _, v in top12][::-1], color="#54A24B")
    axes[2].set_title("업종 분포 (서울페이 분류, top12)")
    axes[2].tick_params(labelsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\n→ 시각화 저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
