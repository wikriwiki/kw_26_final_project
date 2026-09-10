#!/usr/bin/env python3
"""BOK 대조 스코어카드 — 한 번에 전 지표 산출.

사용: python3 scorecard.py <metrics_dir>
주의: 이 파일의 실측값은 **판정 표시용**이며 시뮬 입력이 아니다.
"""
import json, glob, os, sys, math
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else "/data/exp001/out_R88/metrics"

# ── 판정용 실측 참조 (표시 전용, 모델 입력 아님) ──────────────────────────
REF_TABLE1_1CH = {"4주": 76.4, "8주": 92.7, "12주": 97.4}          # 표1 1차 누적
REF_TABLE3 = {"음식점": 46.0, "마트식료품": 21.6, "의료": 6.2,
              "미용": 5.1, "학원": 4.0, "약국": 3.9}                # 표3 1차
REF_FIG18_1CH = {1: 0.27, 2: 0.245, 3: 0.185, 4: 0.21, 5: 0.165}   # 그림18 ◆ 판독
REF_FIG19_MPC = {"내구재": .48, "준내구재": .43, "여가": .41, "개인미용": .35,
                 "숙박음식": .29, "비내구재": .13, "병원": .09, "학원": .04}
REF_FIG19_SHARE = {"비내구재": .445, "숙박음식": .27, "병원": .09, "준내구재": .04,
                   "내구재": .035, "개인미용": .035, "학원": .035, "여가": .02}
REF_MPC_TOTAL = 0.21   # 1차

# 우리 L1 → BOK 그림19 품목 (내구재=가전·가구는 우리에 전용 카테고리 없음)
L1_TO_ITEM = {"식사": "숙박음식", "카페": "숙박음식", "디저트": "숙박음식", "주점": "숙박음식",
              "마트": "비내구재", "편의점": "비내구재",
              "건강": "병원", "미용": "개인미용", "교육": "학원",
              "여가": "여가", "쇼핑": "준내구재"}
# 우리 L1 → 표3 업종
L1_TO_T3 = {"식사": "음식점", "카페": "음식점", "디저트": "음식점", "주점": "음식점",
            "마트": "마트식료품", "편의점": "마트식료품",
            "건강": "의료", "미용": "미용", "교육": "학원"}


def load(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "day_*.jsonl"))):
        day = os.path.basename(f)[4:-6]
        rows = []
        for ln in open(f, encoding="utf-8"):
            try: rows.append(json.loads(ln))
            except Exception: pass
        out.append((day, [r for r in rows if r.get("status") == "ok"]))
    return out


def ms(xs):
    n = len(xs)
    if not n: return 0.0, 0.0, 0
    m = sum(xs) / n
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1) / n) if n > 1 else 0.0
    return m, se, n


def verdict(ours, ref, tol_rel=0.15):
    if ref == 0: return "—"
    d = abs(ours - ref) / abs(ref)
    return "[OK]" if d <= tol_rel else ("[근접]" if d <= tol_rel * 2 else "[NG]")


days = load(D)
if not days:
    print(f"결과 없음: {D}"); sys.exit(1)
print(f"═══ BOK 대조 스코어카드 — {D}  ({len(days)}일) ═══\n")

# ── A1. 소진 궤적 (지급액 대비 누적, 실측 방식과 동일하게 '지급액' 분모) ──
print("── A1. 소진 궤적 (표1 1차: 4주 76.4 / 8주 92.7 / 12주 97.4) ──")
grant0, used_cum = {}, defaultdict(float)
for i, (day, rows) in enumerate(days):
    for r in rows:
        aid = r.get("agent_id") or r.get("aid")
        if not aid: continue
        if i == 0:
            w = int(r.get("cm_policy_wallet_available") or 0) or int(r.get("cm_intended_grant_today") or 0)
            if w > 0: grant0[aid] = w
        if aid in grant0:
            used_cum[aid] += int(r.get("cm_policy_allocated_total") or 0)
    if grant0:
        tot_g = sum(grant0.values())
        tot_u = sum(min(used_cum[a], grant0[a]) for a in grant0)
        pct = 100 * tot_u / tot_g
        wk = (i + 1) / 7
        mark = ""
        if abs(wk - 4) < 0.08: mark = f"  ← 4주 (실측 {REF_TABLE1_1CH['4주']}) {verdict(pct, REF_TABLE1_1CH['4주'], .08)}"
        print(f"  D{i+1:>2} ({day})  누적 {pct:>5.1f}%{mark}")
print(f"  * n={len(grant0)}명, 지급총액 {sum(grant0.values()):,}원\n")

# ── 계층별 소진율 (결과 보고용. BOK에 대조 데이터 없음) ──
print("── 계층별 소진율 (BOK에 분위별 소진 데이터 없음 → 결과로만 보고) ──")
tier_of = {a: ("40만" if g >= 350000 else "30만" if g >= 250000 else "15만") for a, g in grant0.items()}
by_t = defaultdict(list)
for a, g in grant0.items():
    by_t[tier_of[a]].append(100 * min(used_cum[a], g) / g)
for t in ("40만", "30만", "15만"):
    if by_t.get(t):
        m, se, n = ms(by_t[t]); print(f"  {t}: 누적 {m:>5.1f}% ±{se:.1f} (n={n})")
print()

# ── C1/C2. MPC ──
print("── C1. 전체 MPC (실측 1차 0.21) / C2. 분위별 (그림18 ◆) ──")
mpc_all, mpc_q = [], defaultdict(list)
for day, rows in days:
    for r in rows:
        v = r.get("cm_mpc_new_share")
        if v is None: continue
        w = int(r.get("cm_policy_allocated_total") or 0)
        if w <= 0: continue
        mpc_all.append((float(v), w))
        d = r.get("spend_decile") or r.get("cm_spend_decile")
        if d: mpc_q[min(5, max(1, (int(d) + 1) // 2))].append((float(v), w))
if mpc_all:
    tot_w = sum(w for _, w in mpc_all)
    m = sum(v * w for v, w in mpc_all) / tot_w
    print(f"  전체 MPC {m:.3f}  (실측 {REF_MPC_TOTAL}) {verdict(m, REF_MPC_TOTAL)}")
    for q in sorted(mpc_q):
        tw = sum(w for _, w in mpc_q[q])
        mq = sum(v * w for v, w in mpc_q[q]) / tw if tw else 0
        ref = REF_FIG18_1CH.get(q, 0)
        print(f"   {q}분위 {mq:.3f} (실측 {ref}) {verdict(mq, ref, .20)}  n={len(mpc_q[q])}")
else:
    print("  MPC 측정값 없음 (would_buy_anyway 미기록)")
print()

# ── A3/C3/C4. 업종·품목 (그래프에서 집계) ──
print("── A3(표3) / C3·C4(그림19) 는 Neo4j 집계 필요 → cat_scorecard.py 참조 ──")
print("\n※ 이 파일의 실측 수치는 판정 표시용이며 시뮬 입력이 아니다.")
