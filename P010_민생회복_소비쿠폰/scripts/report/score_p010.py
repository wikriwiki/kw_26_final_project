"""P010 민생회복 소비쿠폰 1차(2025-07) 채점 — 같은 사람 정책 있음 − 없음, 보존된 원장만 읽는다(2026-10-12).

    python scripts/report/score_p010.py --dir <BASE/score> --start 2025-07-21 --end 2025-07-27 \
        --policy <정책 사본 json> --json-out <score/p010_score.json>

실측 값은 판정에만 쓰고 모델 입력에는 넣지 않는다(한국은행 이슈노트 2026-13 은 채점용만 — 2026-10-11 사용자 결정).
지표와 판정 규칙은 시뮬레이션 결과를 보기 전에 고정했다(data/experiments/P010_RERUN_DESIGN_20261012.md).

판정 지표
- M1 쿠폰 1원당 추가 소비(같은 사람 짝 MPC) = (있음 총지출 − 없음 총지출) ÷ 쿠폰 사용액, 사람마다 받은 날부터.
     정답지 한국은행 1차 MPC 0.21(설문: 쿠폰 사용액 중 새로 생긴 소비의 비중). 판정: 방향(> 0), 구간이 0.21 을 포함하는가.
- M2 받은 날부터 하루 소진율(남은 쿠폰 대비 그날 쓴 비율, 사람-날 합산). 정답지 한국은행 1차 6.6~7.5%/일
     (대조표 A-4: 정부 지급 개시일 시계의 곡선과는 직접 비교할 수 없고, 하루 소진율만 시계가 맞는다).
- M3 쿠폰 결제의 업종 구성 = 1 − ½Σ|시뮬 몫 − 실측 몫|(7개 묶음, 기타 포함). 정답지 행정안전부 최종 집계
     2025-12-05(신용·체크카드 1·2차 합산): 음식점 40.3·마트식료품 16.0·편의점 10.8·병원약국 8.8·학원 3.7·의류잡화 3.6·기타 16.8.
     참고: 한국은행 표 3 1차(음식점 46.0·마트식료품 21.6 …). 판정: 1·2위 순서, 유사도.
- M4 쿠폰 사용처(사용 가능 가게) 지출 효과, 받은 날부터. 정답지 KDI(사용 가능 업종 매출 +4.93%, 전국, 지급 후 6주 대
     직전 2주)와 한국은행 사용처 매출 효과의 수도권 값 −0.04%(유의하지 않음). 두 정답지가 다르다 — 방향만 적는다.
참고(판정 안 함)
- 총지출 효과, 받기 전 위약(placebo) 비교(0 이어야 한다), 계층(40만·30만·15만)별 7일 사용 비율,
  정책 전 소비 수준 5분위별 MPC(검출 최소 차이 0.45 이상 — 정답지 차이 0.08 을 가를 수 없어 판정 지표에서 뺐다).
무결성(실패 시 종료 2): 정책 없는 쪽 쿠폰 결제, 받은 금액 ≠ 계층 표, 창 안에 받지 않은 사람.
신뢰구간: 사람 단위 짝 붓스트랩(다항분포 가중치) 2,000번, 2.5~97.5 백분위.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict
from datetime import date, timedelta

import numpy as np

TRUTH = {
    "mpc_bok_1st": 0.21, "mpc_bok_q1": 0.25, "mpc_bok_q5": 0.17,
    "hazard_bok_1st": [6.6, 7.5],
    "mix_mois_202512": {"음식점": 40.3, "마트·식료품": 16.0, "편의점": 10.8, "병원·약국": 8.8, "학원": 3.7,
                        "의류·잡화": 3.6, "기타": 16.8},
    "mix_bok_1st_top": {"음식점": 46.0, "마트·식료품": 21.6, "의료": 6.2, "미용": 5.1, "학원": 4.0, "약국": 3.9},
    "eligible_sales_kdi": 4.93, "eligible_sales_bok_capital": -0.04,
}
MIX_MAP = {
    "음식점": ["한식", "일식", "중식", "양식", "아시안", "기타식사", "분식", "치킨", "피자", "구내식당·뷔페", "카페",
             "아이스크림", "일반주점", "호프", "베이커리"],
    "마트·식료품": ["슈퍼마켓", "식료품", "정육", "청과", "수산", "음료소매", "종합소매", "건강보조식품"],
    "편의점": ["편의점"],
    "병원·약국": ["병원", "의원", "치과", "한의원", "약국", "기타보건"],
    "학원": ["학원", "기타교육", "교육지원"],
    "의류·잡화": ["의류", "화장품", "생활용품"],
}
REST = "기타"


def rows(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def pct(lo_hi):
    return None if lo_hi is None else [round(100 * lo_hi[0], 2), round(100 * lo_hi[1], 2)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--policy", required=True, help="정책 사본(aid_grants·decile_grants 로 기대 지급액을 계산)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [str(d0 + timedelta(i)) for i in range((d1 - d0).days + 1)]
    S = a.dir.rstrip("/")
    pol = json.load(io.open(a.policy, encoding="utf-8"))
    aid_grants = pol.get("aid_grants") or {}
    default_grant = sorted({int(v) for v in (pol.get("decile_grants") or {}).values()})
    problems, hard = [], []
    if len(default_grant) != 1:
        problems.append(f"기본 지급액이 하나가 아니다: {default_grant}")
    default_grant = default_grant[0] if default_grant else 0

    sec = {arm: defaultdict(dict) for arm in ("on", "off")}
    for arm in ("on", "off"):
        for r in rows(f"{S}/{arm}.sector.ledger.jsonl"):
            if r["day"] in days:
                sec[arm][r["aid"]][r["day"]] = r
    pol_on = defaultdict(dict)
    for r in rows(f"{S}/on.policy.ledger.jsonl"):
        if r["day"] in days:
            pol_on[r["aid"]][r["day"]] = r
    people = sorted(set(sec["on"]) & set(sec["off"]))
    n = len(people)
    miss = [p for p in people if any(d not in sec["on"][p] or d not in sec["off"][p] or d not in pol_on[p] for d in days)]
    if miss:
        problems.append(f"날이 빠진 사람 {len(miss)}명")
        hard.append("날 빠짐")
    off_funded = sum(float(r.get("policy_funded_won") or 0) for p in people for r in sec["off"][p].values())
    if off_funded > 0:
        hard.append(f"정책 없는 쪽 쿠폰 결제 {off_funded:,.0f}원")

    # 받은 날·금액·쓴 돈
    recv, amt = {}, {}
    for p in people:
        for d in days:
            r = pol_on[p].get(d) or {}
            if r.get("grant_received_cumulative", 0) > 0 and p not in recv:
                recv[p] = d
                amt[p] = int(r["grant_received_cumulative"])
    no_recv = [p for p in people if p not in recv]
    if no_recv:
        hard.append(f"창 안에 받지 않은 사람 {len(no_recv)}명")
    wrong = [p for p in recv if amt[p] != int(aid_grants.get(p, default_grant))]
    if wrong:
        hard.append(f"받은 금액이 계층 표와 다른 사람 {len(wrong)}명 (예: {wrong[:3]})")
    tier = {p: int(aid_grants.get(p, default_grant)) for p in people}

    def v(arm, p, d, key):
        r = sec[arm][p].get(d) or {}
        if key == "total":
            return float(r.get("total_spent") or 0)
        if key == "eligible":
            return float(sum((r.get("eligible_by_sub") or {}).values()))
        raise KeyError(key)

    post = {p: [d for d in days if p in recv and d >= recv[p]] for p in people}
    pre = {p: [d for d in days if p in recv and d < recv[p]] for p in people}
    T1 = np.array([sum(v("on", p, d, "total") for d in post[p]) for p in people])
    T0 = np.array([sum(v("off", p, d, "total") for d in post[p]) for p in people])
    E1 = np.array([sum(v("on", p, d, "eligible") for d in post[p]) for p in people])
    E0 = np.array([sum(v("off", p, d, "eligible") for d in post[p]) for p in people])
    W1 = np.array([sum(v("on", p, d, "total") for d in days) for p in people])
    W0 = np.array([sum(v("off", p, d, "total") for d in days) for p in people])
    P1 = np.array([sum(v("on", p, d, "total") for d in pre[p]) for p in people])
    P0 = np.array([sum(v("off", p, d, "total") for d in pre[p]) for p in people])
    G = np.array([sum(float((pol_on[p].get(d) or {}).get("grant_spent_today") or 0) for d in days) for p in people])
    # 하루 소진율: 그날 아침 남은 쿠폰(받은 날은 받은 금액) 대비 그날 쓴 돈
    rem_sum = np.zeros(n); use_sum = np.zeros(n)
    for i, p in enumerate(people):
        left = float(amt.get(p, 0))
        for d in post[p]:
            spent = float((pol_on[p].get(d) or {}).get("grant_spent_today") or 0)
            rem_sum[i] += left
            use_sum[i] += spent
            left = max(0.0, left - spent)
    # 업종 구성(쿠폰 결제, funded_by_sub)
    cats = list(MIX_MAP) + [REST]
    sub2cat = {s: k for k, ss in MIX_MAP.items() for s in ss}
    M = np.zeros((n, len(cats)))
    unmapped = defaultdict(float)
    for i, p in enumerate(people):
        for d in days:
            for s, w in ((sec["on"][p].get(d) or {}).get("funded_by_sub") or {}).items():
                if not w:
                    continue
                k = sub2cat.get(s, REST)
                if k == REST:
                    unmapped[s] += w
                M[i, cats.index(k)] += w
    truth_mix = np.array([TRUTH["mix_mois_202512"][c] for c in cats]) / 100.0
    fund_total = M.sum()
    if fund_total > 0 and abs(fund_total - G.sum()) > 0.01 * G.sum():
        problems.append(f"업종별 쿠폰 결제 합 {fund_total:,.0f} ≠ 정책 원장 쿠폰 사용 {G.sum():,.0f}")

    rng = np.random.default_rng(a.seed)
    Wt = rng.multinomial(n, [1.0 / n] * n, size=a.boot).astype(float) if n else np.zeros((0, 0))

    def stat(f):
        base = f(np.ones(n))
        vals = np.array([f(w) for w in Wt])
        vals = vals[np.isfinite(vals)]
        if len(vals) < 0.9 * a.boot:
            return base, None
        vals.sort()
        return base, [float(vals[int(.025 * len(vals))]), float(vals[int(.975 * len(vals)) - 1])]

    def ratio(x, y):
        return lambda w: (w @ x) / (w @ y) - 1 if (w @ y) > 0 else np.nan

    def mpc(w):
        g = w @ G
        return (w @ T1 - w @ T0) / g if g > 0 else np.nan

    def sim_mix(w):
        s = w @ M
        return s / s.sum() if s.sum() > 0 else None

    def similarity(w):
        s = sim_mix(w)
        return np.nan if s is None else float(np.minimum(s, truth_mix).sum())

    res = {"people": n, "days": days, "received_in_window": len(recv), "not_received": len(no_recv),
           "tiers": {str(k): sum(1 for p in people if tier[p] == k) for k in sorted(set(tier.values()))},
           "truth": TRUTH, "problems": problems, "hard_failures": hard, "metrics": {}}
    m = res["metrics"]
    b, c = stat(mpc)
    m["M1_mpc"] = {"value": round(float(b), 4) if np.isfinite(b) else None, "ci95": None if c is None else [round(x, 4) for x in c],
                   "coupon_used_won": round(float(G.sum())), "extra_spend_won": round(float(T1.sum() - T0.sum()))}
    b, c = stat(lambda w: (w @ use_sum) / (w @ rem_sum) if (w @ rem_sum) > 0 else np.nan)
    m["M2_daily_use_rate_pct"] = {"value": round(100 * float(b), 2), "ci95": pct(c),
                                  "share_used_in_window_pct": round(100 * float(G.sum() / max(1.0, sum(amt.values()))), 2)}
    s0 = sim_mix(np.ones(n))
    b, c = stat(similarity)
    m["M3_mix"] = {"similarity_pct": round(100 * float(b), 1) if np.isfinite(b) else None, "ci95": pct(c),
                   "sim_pct": None if s0 is None else {k: round(100 * float(x), 1) for k, x in zip(cats, s0)},
                   "truth_pct": TRUTH["mix_mois_202512"],
                   "rest_subs_top": sorted(((k, round(w)) for k, w in unmapped.items()), key=lambda x: -x[1])[:8]}
    b, c = stat(ratio(E1, E0))
    m["M4_eligible_spend_pct"] = {"value": round(100 * float(b), 2), "ci95": pct(c)}
    b, c = stat(ratio(T1, T0))
    m["R_total_from_receipt_pct"] = {"value": round(100 * float(b), 2), "ci95": pct(c)}
    b, c = stat(ratio(W1, W0))
    m["R_total_window_pct"] = {"value": round(100 * float(b), 2), "ci95": pct(c)}
    b, c = stat(lambda w: np.log((w @ P1) / (w @ P0)) if (w @ P0) > 0 and (w @ P1) > 0 else np.nan)
    m["R_placebo_pre_receipt_log"] = {"value": round(float(b), 4) if np.isfinite(b) else None,
                                      "ci95": None if c is None else [round(x, 4) for x in c],
                                      "people": int(sum(1 for p in people if pre[p]))}
    by_tier = {}
    for k in sorted(set(tier.values())):
        ix = [i for i, p in enumerate(people) if tier[p] == k and p in amt]
        got = sum(amt[people[i]] for i in ix)
        by_tier[str(k)] = {"n": len(ix), "used_pct": round(100 * float(G[ix].sum()) / got, 1) if got else None}
    m["R_use_by_tier"] = by_tier
    base0 = np.array([sum(v("off", p, d, "total") for d in days) for p in people])
    qs = np.quantile(base0, [0.2, 0.8]) if n else [0, 0]
    for name, mask in (("bottom20", base0 <= qs[0]), ("top20", base0 >= qs[1])):
        g = G[mask].sum()
        m.setdefault("R_mpc_by_spend_level", {})[name] = {
            "n": int(mask.sum()), "mpc": round(float((T1[mask].sum() - T0[mask].sum()) / g), 3) if g > 0 else None}

    j = {}
    v1, c1 = m["M1_mpc"]["value"], m["M1_mpc"]["ci95"]
    j["M1"] = "판정 불가" if c1 is None else (("방향 맞음(> 0)" if c1[0] > 0 else ("0 과 구분 안 됨" if c1[1] >= 0 else "방향 틀림(< 0)"))
                                          + (" · 구간이 0.21 을 포함" if c1[0] <= TRUTH["mpc_bok_1st"] <= c1[1] else
                                             (" · 0.21 보다 크다" if c1[0] > TRUTH["mpc_bok_1st"] else " · 0.21 보다 작다")))
    c2 = m["M2_daily_use_rate_pct"]["ci95"]; lo, hi = TRUTH["hazard_bok_1st"]
    j["M2"] = "판정 불가" if c2 is None else ("실측 범위와 겹침" if c2[1] >= lo and c2[0] <= hi else
                                           ("시뮬이 빠르다" if c2[0] > hi else "시뮬이 느리다"))
    if s0 is not None:
        order_sim = [cats[i] for i in np.argsort(-s0)[:2]]
        j["M3"] = f"1·2위 시뮬 {order_sim} / 실측 ['음식점', '마트·식료품'] · " + (
            "같음" if order_sim == ["음식점", "마트·식료품"] else "다름")
    else:
        j["M3"] = "판정 불가(쿠폰 결제 없음)"
    c4 = m["M4_eligible_spend_pct"]["ci95"]
    j["M4"] = "판정 불가" if c4 is None else (("+ (KDI 방향과 같음)" if c4[0] > 0 else ("0 과 구분 안 됨 (한국은행 수도권 −0.04% 와 같은 결론)" if c4[1] >= 0 else "− (두 정답지와 다름)")))
    cp = m["R_placebo_pre_receipt_log"]["ci95"]
    j["placebo"] = "판정 불가" if cp is None else ("맞음(받기 전 차이 구간이 0 을 포함)" if cp[0] <= 0 <= cp[1] else "실패 — 받기 전부터 차이가 있다")
    res["judgement"] = j
    io.open(a.json_out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"# P010 채점 · {n}명 · {days[0]}~{days[-1]} · 창 안에 받은 사람 {len(recv)} · 계층 {res['tiers']}")
    for k, x in m.items():
        print(f"  {k}: {json.dumps(x, ensure_ascii=False)}")
    print("  문제:", problems or "없음", "| 무결성 실패:", hard or "없음")
    print("  판정:", json.dumps(j, ensure_ascii=False))
    return 2 if hard or n == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
