"""P017 서울시 재난긴급생활비(2020-04) 채점 — 같은 사람 정책 있음 − 없음, 보존된 원장만 읽는다(2026-10-11).

    python scripts/report/score_p017.py --dir <BASE/score> --start 2020-04-20 --end 2020-04-26 \
        --dossier-on <on/dossier.jsonl> --json-out <score/p017_score.json>

정답지: IZA DP 14662(= The Korean Economic Review 38(2), 2022) 표 3~5 — 로그 1인 월 지출의 이중차분,
처치 = 선불카드를 신한 플랫폼에 등록해 쓴 수급자. 저자들이 상한으로 읽으라고 했다(13쪽).
실측 값은 판정에만 쓰고 모델 입력에는 넣지 않는다.

지표(같은 사람의 두 갈래, 로그 차 = ln(있음 합 / 없음 합)):
- T1 총지출: 두 갈래 창 전체 / 받은 날부터(사람마다 받은 날 이후 날만, 같은 날을 없음 쪽에서도)
- T2 품목(실행 전 고정 대응): 음식점=식사+카페+디저트, 소매=마트+편의점+쇼핑, 의료=건강, 미용=미용, 오락여가=여가, 교육=교육.
  논문 '식음료(Food and Beverage)'는 소매와의 경계가 원문에 없어 채점하지 않는다.
- T3 소득: 페르소나 소득 하 대 중하(정답지 저소득 +0.294 > 중소득 +0.162 의 순서만 본다 — 페르소나 소득은 생성값).
- 참고: 창 안 지원금 사용 비율, 사용처 밖 결제.
신뢰구간: 사람 단위 짝 붓스트랩 2,000번, 2.5~97.5 백분위.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from collections import defaultdict
from datetime import date, timedelta

TRUTH = {"total_apr_aug": 0.178, "total_apr_aug_low_bound": 0.124, "total_apr": 0.130, "total_may": 0.352,
         "income_low": 0.294, "income_mid": 0.162,
         "restaurants": 0.241, "retail": 0.221, "health": 0.164, "beauty": 0.307, "entertainment": 0.027,
         "education": 0.153}
CATS = {"restaurants": {"식사", "카페", "디저트"}, "retail": {"마트", "편의점", "쇼핑"}, "health": {"건강"},
        "beauty": {"미용"}, "entertainment": {"여가"}, "education": {"교육"}}


def rows(path):
    return [json.loads(l) for l in io.open(path, encoding="utf-8") if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="채점 폴더(on/off.sector.ledger.jsonl, on.policy.ledger.jsonl)")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dossier-on", required=True, help="소득 등급(profile.income_level)을 읽는다")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    days = [str(d0 + timedelta(i)) for i in range((d1 - d0).days + 1)]
    S = a.dir.rstrip("/")
    sec = {arm: rows(f"{S}/{arm}.sector.ledger.jsonl") for arm in ("on", "off")}
    pol_on = rows(f"{S}/on.policy.ledger.jsonl")
    inc = {}
    for l in io.open(a.dossier_on, encoding="utf-8"):
        r = json.loads(l)
        inc[r["aid"]] = (r.get("profile") or {}).get("income_level")
    # 사람×날 값
    val = {arm: defaultdict(lambda: defaultdict(float)) for arm in ("on", "off")}
    problems = []
    for arm in ("on", "off"):
        for r in sec[arm]:
            if r["day"] not in days:
                continue
            v = val[arm][(r["aid"], r["day"])]
            v["total"] += float(r.get("total_spent") or 0)
            for l1, amt in (r.get("by_l1") or {}).items():
                for c, ls in CATS.items():
                    if l1 in ls:
                        v[c] += float(amt or 0)
    if sum(float(r.get("policy_funded_won") or 0) for r in sec["off"] if r["day"] in days) > 0:
        problems.append("정책 없는 쪽에 지원금 결제가 있다")
    recv = {}
    grant_amt, spent_grant = {}, defaultdict(float)
    for r in sorted(pol_on, key=lambda x: x["day"]):
        if r["day"] not in days:
            continue
        if r.get("grant_received_cumulative", 0) > 0 and r["aid"] not in recv:
            recv[r["aid"]] = r["day"]
            grant_amt[r["aid"]] = float(r["grant_received_cumulative"])
        spent_grant[r["aid"]] += float(r.get("grant_spent_today") or 0)
    people = sorted({k[0] for k in val["on"]} & {k[0] for k in val["off"]})
    missing = [p for p in people if any((p, d) not in val["on"] or (p, d) not in val["off"] for d in days)]
    if missing:
        problems.append(f"날이 빠진 사람 {len(missing)}명")
    no_recv = [p for p in people if p not in recv]
    n = len(people)

    def per(arm, p, k, from_day=None):
        return sum(val[arm][(p, d)].get(k, 0.0) for d in days if from_day is None or d >= from_day)

    def logdiff(ids, k, post=False):
        if post:  # 받은 날부터 — 창 안에 받지 않은 사람은 뺀다(받은 날이 없으면 비교할 날이 없다)
            ids = [j for j in ids if people[j] in recv]
        on = sum(per("on", people[j], k, recv.get(people[j]) if post else None) for j in ids)
        off = sum(per("off", people[j], k, recv.get(people[j]) if post else None) for j in ids)
        return math.log(on / off) if on > 0 and off > 0 else None

    rnd = random.Random(a.seed)
    boots = [[rnd.randrange(n) for _ in range(n)] for _ in range(a.boot)] if n else []

    def ci(f):
        vals = sorted(v for v in (f(b) for b in boots) if v is not None)
        if len(vals) < 0.9 * max(1, len(boots)):
            return None
        return [round(vals[int(.025 * len(vals))], 4), round(vals[int(.975 * len(vals)) - 1], 4)]

    allidx = list(range(n))
    res = {"people": n, "days": days, "received_in_window": len(recv), "not_received": len(no_recv),
           "truth": TRUTH, "problems": problems, "metrics": {}}
    for k in ["total"] + list(CATS):
        res["metrics"][k] = {"log_diff_window": logdiff(allidx, k), "ci95_window": ci(lambda b, k=k: logdiff(b, k)),
                             "log_diff_from_receipt": logdiff(allidx, k, True), "ci95_from_receipt": ci(lambda b, k=k: logdiff(b, k, True)),
                             "on_per_person": round(sum(per("on", p, k) for p in people) / max(1, n)),
                             "off_per_person": round(sum(per("off", p, k) for p in people) / max(1, n))}
    # 위약 대조: 받기 전 날의 있음−없음(0 이어야 한다 — P013 300c 에서 받기 전 사람의 지출이 +11.4% 였던 사고)
    def placebo(ids):
        on_ = off_ = 0.0
        for j in ids:
            p = people[j]
            if p not in recv:
                continue
            on_ += sum(val["on"][(p, d)].get("total", 0.0) for d in days if d < recv[p])
            off_ += sum(val["off"][(p, d)].get("total", 0.0) for d in days if d < recv[p])
        return math.log(on_ / off_) if on_ > 0 and off_ > 0 else None
    res["metrics"]["placebo_pre_receipt"] = {"log_diff": placebo(allidx), "ci95": ci(placebo),
                                             "people": sum(1 for p in people if p in recv and recv[p] > days[0])}
    groups = {g: [i for i, p in enumerate(people) if inc.get(p) == g] for g in ("하", "중하")}
    res["metrics"]["income"] = {g: {"n": len(ix), "log_diff_window": logdiff(ix, "total"),
                                    "log_diff_from_receipt": logdiff(ix, "total", True)} for g, ix in groups.items()}

    def gap(b):
        bl = [j for j in b if inc.get(people[j]) == "하"]; bm = [j for j in b if inc.get(people[j]) == "중하"]
        x, y = logdiff(bl, "total", True), logdiff(bm, "total", True)
        return x - y if x is not None and y is not None else None
    res["metrics"]["income"]["low_minus_mid_from_receipt"] = {"value": gap(allidx), "ci95": ci(gap)}
    tot_grant = sum(grant_amt.values())
    res["metrics"]["grant_use"] = {"grant_total": round(tot_grant), "spent_in_window": round(sum(spent_grant.values())),
                                   "share_spent": round(sum(spent_grant.values()) / tot_grant, 4) if tot_grant else None,
                                   "extra_spend_per_grant_won": (round((sum(per("on", p, "total") for p in people)
                                                                        - sum(per("off", p, "total") for p in people)) / tot_grant, 4)
                                                                 if tot_grant else None)}
    # 등록한 판정
    j = {}
    t = res["metrics"]["total"]
    for key in ("window", "from_receipt"):
        v, c = t[f"log_diff_{key}"], t[f"ci95_{key}"]
        if v is None or c is None:
            j[f"total_{key}"] = "판정 불가"
            continue
        j[f"total_{key}"] = ("방향 맞음(+)" if v > 0 else "방향 틀림") + (
            " · 구간이 4월 +0.130 을 포함" if c[0] <= TRUTH["total_apr"] <= c[1] else
            (" · 4월 +0.130 보다 크다" if c[0] > TRUTH["total_apr"] else " · 4월 +0.130 보다 작다")) + (
            " · 구간이 4~8월 범위 [0.124, 0.178] 과 겹침" if c[1] >= TRUTH["total_apr_aug_low_bound"] and c[0] <= TRUTH["total_apr_aug"] else "")
    ent = res["metrics"]["entertainment"]["log_diff_window"]
    others = [res["metrics"][c]["log_diff_window"] for c in ("restaurants", "retail", "health", "beauty")]
    if ent is not None and all(o is not None for o in others):
        j["entertainment_smallest"] = "맞음(오락여가가 가장 작다)" if ent <= min(others) else "틀림"
    g = res["metrics"]["income"]["low_minus_mid_from_receipt"]["value"]
    j["income_order"] = "판정 불가" if g is None else ("맞음(하 > 중하)" if g > 0 else "틀림(하 ≤ 중하)")
    pl = res["metrics"]["placebo_pre_receipt"]
    j["placebo_pre_receipt"] = ("판정 불가" if pl["ci95"] is None else
                                ("맞음(받기 전 차이 구간이 0 을 포함)" if pl["ci95"][0] <= 0 <= pl["ci95"][1] else "실패 — 받기 전부터 차이가 있다"))
    res["judgement"] = j
    res["hard_failures"] = [p for p in problems if "정책 없는 쪽" in p]
    io.open(a.json_out, "w", encoding="utf-8").write(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"# P017 채점 · {n}명 · {days[0]}~{days[-1]} · 창 안에 받은 사람 {len(recv)}")
    for k, v in res["metrics"].items():
        print(f"  {k}: {json.dumps(v, ensure_ascii=False)}")
    print("  문제:", problems or "없음")
    print("  판정:", json.dumps(j, ensure_ascii=False))
    return 2 if res["hard_failures"] or n == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
