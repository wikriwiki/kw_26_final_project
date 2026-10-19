"""P016(농할 1차) 채점 — 같은 사람 정책 있음 − 없음, 보존된 기억 모음(dossier)만 읽는다(2026-10-11).

    python scripts/report/score_p016.py --on-dossier <on/dossier.jsonl> --off-dossier <off/dossier.jsonl> \
        --policy data/experiments/P016_ab3w_policy_20261011.json --start 2020-07-30 --end 2020-08-05 \
        --json-out <score/p016_score.json>

판정은 실행 전에 등록한 것(data/experiments/P016_DESIGN_20261011.md 3절) 그대로다.
- 참여 매장 = 가게 자신의 업종(poi_sub)이 정책 업종이고 이름이 정책 규칙에 맞는 곳 — 엔진 할인 판정과 같은 규칙.
  계획 업종(sub_category, '장보기'·'대형마트' 등)은 쓰지 않는다.
- C1 참여 매장 농축산물(produce_spent) · C2 참여 매장 결제 전체 · C3 참여 매장 결제 중 농축산물 비중
- E1(참고) (C1 있음 − 없음) ÷ 총 할인 · M1 비참여 장보기 가게(대체) · M2 할인 분포·체인별 한도 도달
- 신뢰구간: 사람 단위 짝 붓스트랩(같은 사람의 있음·없음을 함께 뽑는다), 기본 2,000번, 2.5~97.5 백분위.
정답지: KREI 「2022 농식품 소비정보 분석사업 결과보고서 Ⅲ」 — C1 +6.957%(보고서 57쪽·PDF 67쪽), C2 농축산물 +11.6% vs 전체 +4.6%(보고서 10쪽·PDF 20쪽),
C3 +1.9%p(보고서 10쪽·PDF 20쪽), M3 방문 고객 수 감소·객단가 증가(보고서 10쪽·PDF 20쪽), E1 약 4배(2021). 실측 값은 판정에만 쓰고 모델 입력에는 넣지 않는다.

종료 코드: 0 정상 · 2 무결성 실패(비참여 매장 할인·한도 초과·20% 초과·정책 없는 쪽 할인 등 — 엔진 회계가 틀렸다는 뜻).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import re
import statistics as st
import sys
from collections import defaultdict
from datetime import date, timedelta

TRUTH = {"C1_pct": 6.957, "C2_produce_pct": 11.6, "C2_total_pct": 4.6, "C3_pp": 1.9, "E1": 4.0, "M3_trips": "-", "M3_basket": "+"}
HARD = ("참여매장_아닌데_할인", "체인한도_초과", "20%초과", "농축산물_0인데_할인", "결제보다_큼")


def load(path):
    out = {}
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            out[r["aid"]] = r
    return out


def money(v):
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


class Scorer:
    def __init__(self, pol: dict, days: list[str]):
        sec = pol["sectors"]["농축산물"]
        self.subs = set(sec["eligibility"]["include"]["subs"])
        self.rx = re.compile(sec["eligibility"]["include"]["name_regex"])
        self.chains = [(c, re.compile(p)) for c, p in sec["chains"].items()]
        self.cap = int(sec.get("cap") or 0)
        self.days = days
        self.integ = defaultdict(int)
        self.problems: list[str] = []

    def eligible(self, it):
        return (it.get("poi_sub") in self.subs) and bool(self.rx.match(it.get("poi") or ""))

    def chain_of(self, it):
        if not self.eligible(it):
            return None
        return next((c for c, p in self.chains if p.search(it.get("poi") or "")), None)

    def person(self, rec, arm):
        m = defaultdict(float)
        got = {p["day"] for p in rec.get("plans") or []}
        miss = [d for d in self.days if d not in got]
        if miss:
            self.problems.append(f"{arm} {rec['aid']} 날 빠짐 {miss}")
        by_chain = defaultdict(float)
        I = self.integ
        for p in rec.get("plans") or []:
            if p["day"] not in self.days:
                continue
            for it in p.get("items") or []:
                amt = money(it.get("actual_spent"))
                if amt <= 0:
                    continue
                m["total"] += amt
                disc = money(it.get("discount_total"))
                if it.get("poi_sub") is None:
                    I[f"{arm}_가게업종_없음"] += 1
                prod = money(it.get("produce_spent"))
                if it.get("poi_sub") in self.subs:
                    m["gro_trips"] += 1
                    m["gro_total"] += amt
                    I[f"{arm}_장보기_결제"] += 1
                    if it.get("produce_spent") is None:
                        I[f"{arm}_장보기_농축산물_답없음"] += 1
                    if prod > amt + 0.5:
                        I[f"{arm}_농축산물이_결제보다_큼"] += 1
                    m["gro_produce"] += min(prod, amt)
                    if self.eligible(it):
                        m["C2"] += amt
                        m["C1"] += min(prod, amt)
                        m["part_trips"] += 1
                    else:
                        m["M1_nonpart_total"] += amt
                        m["M1_nonpart_produce"] += min(prod, amt)
                elif it.get("produce_spent") is not None:
                    I[f"{arm}_장보기_아닌데_농축산물_답"] += 1
                if disc > 0:
                    m["discount"] += disc
                    c = self.chain_of(it)
                    if c is None:
                        I[f"{arm}_참여매장_아닌데_할인"] += 1
                    else:
                        by_chain[c] += disc
                    if prod <= 0:
                        I[f"{arm}_농축산물_0인데_할인"] += 1
                    if disc > 0.2 * prod + 1:
                        I[f"{arm}_할인이_농축산물의_20%초과"] += 1
        for v in by_chain.values():
            if v > self.cap + 0.5:
                I[f"{arm}_체인한도_초과"] += 1
        m["chains_at_cap"] = sum(1 for v in by_chain.values() if v >= self.cap - 0.5)
        m["chains_used"] = len(by_chain)
        return m


def score(on: dict, off: dict, pol: dict, days: list[str], boot: int = 2000, seed: int = 20261011) -> dict:
    S = Scorer(pol, days)
    people = sorted(set(on) & set(off))
    if set(on) ^ set(off):
        S.problems.append(f"한쪽에만 있는 사람 {len(set(on) ^ set(off))}명")
    P_on = {i: S.person(on[i], "on") for i in people}
    P_off = {i: S.person(off[i], "off") for i in people}
    if sum(P_off[i]["discount"] for i in people) > 0:
        S.problems.append("정책 없는 쪽에 할인이 있다")
    n, D = len(people), len(days)
    rnd = random.Random(seed)
    idx_boot = [[rnd.randrange(n) for _ in range(n)] for _ in range(boot)] if n else []

    def tot(P, ids, k):
        return sum(P[people[j]][k] for j in ids)

    def ratio_pct(ids, k):
        so, sf = tot(P_on, ids, k), tot(P_off, ids, k)
        return 100 * (so - sf) / sf if sf > 0 else None

    def share_pp(ids):
        c2o, c2f = tot(P_on, ids, "C2"), tot(P_off, ids, "C2")
        if c2o <= 0 or c2f <= 0:
            return None
        return 100 * tot(P_on, ids, "C1") / c2o - 100 * tot(P_off, ids, "C1") / c2f

    def e1(ids):
        dsum = tot(P_on, ids, "discount")
        return (tot(P_on, ids, "C1") - tot(P_off, ids, "C1")) / dsum if dsum > 0 else None

    def c2gap(ids):
        a_, b_ = ratio_pct(ids, "C1"), ratio_pct(ids, "C2")
        return a_ - b_ if a_ is not None and b_ is not None else None

    def ci(f):
        if not idx_boot:
            return None
        vals = sorted(v for v in (f(ids) for ids in idx_boot) if v is not None)
        if len(vals) < 0.9 * boot:
            return None  # 붓스트랩 표본의 10% 넘게 정의되지 않으면(분모 0) 구간을 내지 않는다
        return [round(vals[int(0.025 * len(vals))], 2), round(vals[int(0.975 * len(vals)) - 1], 2)]

    def r2(v):
        return round(v, 2) if v is not None else None

    allidx = list(range(n))
    res = {"people": n, "days": days,
           "definition": "같은 사람 정책 있음 − 없음, 행사 기간 합. % 는 정책 없는 쪽 합 대비. 참여 매장 = 가게 업종 + 이름 규칙(엔진과 같음)",
           "truth": TRUTH, "problems": S.problems, "metrics": {}}
    for k, label in [("C1", "참여 매장 국산 신선 농축산물"), ("C2", "참여 매장 결제 전체"),
                     ("gro_produce", "장보기 농축산물(모든 가게)"), ("gro_total", "장보기 결제(모든 가게)"),
                     ("M1_nonpart_total", "비참여 장보기 가게 결제(대체)"), ("M1_nonpart_produce", "비참여 장보기 가게 농축산물"),
                     ("total", "총지출(방어선)"), ("gro_trips", "장보기 횟수"), ("part_trips", "참여 매장 장보기 횟수")]:
        so, sf = tot(P_on, allidx, k), tot(P_off, allidx, k)
        res["metrics"][k] = {"label": label, "on_sum": round(so), "off_sum": round(sf),
                             "on_per_person_week": round(7 * so / max(1, n) / max(1, D), 1),
                             "off_per_person_week": round(7 * sf / max(1, n) / max(1, D), 1),
                             "pct": r2(ratio_pct(allidx, k)), "ci95": ci(lambda ids, k=k: ratio_pct(ids, k)),
                             "people_with_any_on": sum(1 for i in people if P_on[i][k] > 0),
                             "people_with_any_off": sum(1 for i in people if P_off[i][k] > 0)}
    res["metrics"]["C3"] = {"label": "참여 매장 결제 중 농축산물 비중 차(%p)", "pp": r2(share_pp(allidx)), "ci95": ci(share_pp)}
    res["metrics"]["E1"] = {"label": "(C1 있음 − 없음) ÷ 총 할인", "value": r2(e1(allidx)), "ci95": ci(e1),
                            "discount_sum": round(tot(P_on, allidx, "discount"))}
    res["metrics"]["C2_order"] = {"label": "C1 증가율 − C2 증가율(%p) — 정답지는 농축산물이 전체보다 더 늘었다",
                                  "gap_pp": r2(c2gap(allidx)), "ci95": ci(c2gap)}
    def basket_pct(ids):
        to, tf = tot(P_on, ids, "part_trips"), tot(P_off, ids, "part_trips")
        if to <= 0 or tf <= 0:
            return None
        bo, bf = tot(P_on, ids, "C2") / to, tot(P_off, ids, "C2") / tf
        return 100 * (bo - bf) / bf if bf > 0 else None
    res["metrics"]["M3"] = {"label": "참여 매장 방문 횟수와 1회 결제액(정답지 보고서 10쪽: 방문 고객 수 감소·객단가 증가)",
                            "trips_pct": res["metrics"]["part_trips"]["pct"], "trips_ci95": res["metrics"]["part_trips"]["ci95"],
                            "basket_pct": r2(basket_pct(allidx)), "basket_ci95": ci(basket_pct)}
    rec = [P_on[i]["discount"] for i in people if P_on[i]["discount"] > 0]
    res["metrics"]["M2"] = {"label": "할인 분포(정책 있는 쪽)", "people_with_discount": len(rec),
                            "discount_per_person_mean": round(tot(P_on, allidx, "discount") / max(1, n), 1),
                            "discount_per_receiver_median": st.median(rec) if rec else None,
                            "people_at_cap_any_chain": sum(1 for i in people if P_on[i]["chains_at_cap"] > 0),
                            "people_multi_chain": sum(1 for i in people if P_on[i]["chains_used"] > 1)}
    res["integrity"] = dict(S.integ)
    on_g = S.integ.get("on_장보기_결제", 0) + S.integ.get("off_장보기_결제", 0)
    na = S.integ.get("on_장보기_농축산물_답없음", 0) + S.integ.get("off_장보기_농축산물_답없음", 0)
    res["integrity"]["농축산물_답_비율"] = round(1 - na / on_g, 4) if on_g else None
    # 실행 전에 등록한 판정
    j = {}
    c1 = res["metrics"]["C1"]
    if c1["pct"] is None or c1["ci95"] is None:
        j["C1"] = "판정 불가(정책 없는 쪽 참여 매장 농축산물이 0 이거나 표본 부족)"
    else:
        lo, hi = c1["ci95"]
        j["C1_direction"] = "맞음(+)" if c1["pct"] > 0 else "틀림(−)"
        j["C1_vs_truth"] = ("구간이 +6.957% 를 포함" if lo <= TRUTH["C1_pct"] <= hi else
                            ("크게 빗나감 — 실측보다 크다" if lo > TRUTH["C1_pct"] else "크게 빗나감 — 실측보다 작다"))
        j["C1_ci_halfwidth_pct"] = round((hi - lo) / 2, 1)
    g = res["metrics"]["C2_order"]
    j["C2_order"] = "판정 불가" if g["gap_pp"] is None else ("맞음(농축산물 > 전체)" if g["gap_pp"] > 0 else "틀림")
    c3 = res["metrics"]["C3"]
    j["C3_direction"] = "판정 불가" if c3["pp"] is None else ("맞음(+)" if c3["pp"] > 0 else "틀림(−)")
    m3 = res["metrics"]["M3"]
    j["M3_trips"] = "판정 불가" if m3["trips_pct"] is None else ("정답지와 같은 방향(−)" if m3["trips_pct"] < 0 else "정답지와 반대(+) — 방문을 늘렸다")
    j["M3_basket"] = "판정 불가" if m3["basket_pct"] is None else ("정답지와 같은 방향(+)" if m3["basket_pct"] > 0 else "정답지와 반대(−)")
    res["judgement"] = j
    res["hard_failures"] = [k for k in S.integ if any(w in k for w in HARD)] + \
        [p for p in S.problems if "정책 없는 쪽에 할인" in p]
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on-dossier", required=True)
    ap.add_argument("--off-dossier", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    pol = json.load(open(a.policy, encoding="utf-8"))
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [str(d0 + timedelta(i)) for i in range((d1 - d0).days + 1)]
    res = score(load(a.on_dossier), load(a.off_dossier), pol, days, a.boot, a.seed)
    io.open(a.json_out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"# P016 채점 · {res['people']}명 · {days[0]}~{days[-1]} ({len(days)}일)")
    for k, v in res["metrics"].items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    print("  무결성:", json.dumps(res["integrity"], ensure_ascii=False))
    print("  문제:", res["problems"] or "없음")
    print("  판정:", json.dumps(res["judgement"], ensure_ascii=False))
    if res["hard_failures"] or res["people"] == 0:
        print("무결성 실패:", res["hard_failures"], file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
