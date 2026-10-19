"""3주 A/B(정책 전 주 공통 → 같은 사람 정책 있음/없음) 채점 — 같은 사람·같은 날의 차이만 본다 (2026-10-05).

    python scripts/report/score_ab3w.py --on <on/sector.ledger.jsonl> --off <off/sector.ledger.jsonl> \
        [--on-policy <on/policy.ledger.jsonl> --off-policy <off/policy.ledger.jsonl>] --out <score.json>

두 갈래는 정책 시작일 아침에 같은 그래프에서 출발하므로 정책 전 격차가 구조적으로 0 이다 — 차이는 정책
(또는 사회 배경)과 생성의 흔들림뿐이다. 흔들림은 사람 단위 재표집 구간과 부호 검정(동점 수 공개)으로 드러낸다.

출력하는 양(1인 1일 평균, 원):
  total / offline / online, 상위 업종별(by_l1), 정책 사용처 지출(정책 원장이 있으면 정책 자체 규칙으로 판정),
  정책으로 낸 돈(policy_funded_won), 받은 지원금(누적 마지막 날).
  비율: 늘어난 총지출 / 정책으로 낸 돈, 늘어난 총지출 / 받은 돈 — 둘 다 정의를 함께 적는다.
실측과의 대조(어느 정답지 지표에 어느 양을 대는지)는 정책별 평가항목에 따른다 — 여기서는 숫자만 낸다.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def read(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"빈 원장: {path}")
    return rows


def per_person(rows, field):
    out = defaultdict(int)
    for r in rows:
        v = r.get(field) or 0
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{field} 가 정수가 아니다: {r.get('aid')} {r.get('day')}")
        out[r["aid"]] += v
    return out


def per_person_map(rows, field):
    out = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for k, v in (r.get(field) or {}).items():
            out[r["aid"]][k] += int(v)
    return out


def check_pair(on, off):
    key_on = {(r["aid"], r["day"]) for r in on}
    key_off = {(r["aid"], r["day"]) for r in off}
    if key_on != key_off or len(key_on) != len(on) or len(key_off) != len(off):
        raise ValueError("두 갈래의 사람·날이 같지 않다(또는 중복 행)")
    people = sorted({a for a, _ in key_on})
    days = sorted({d for _, d in key_on})
    return people, days


def contrast(on_by, off_by, people, n_days, draws, seed):
    diffs = [(on_by.get(a, 0) - off_by.get(a, 0)) / n_days for a in people]
    on_mean = sum(on_by.get(a, 0) for a in people) / len(people) / n_days
    off_mean = sum(off_by.get(a, 0) for a in people) / len(people) / n_days
    d = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = []
    for _ in range(draws):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        boots.append(sum(sample) / len(sample))
    boots.sort()
    lo, hi = boots[int(0.025 * draws)], boots[int(0.975 * draws) - 1]
    up = sum(1 for x in diffs if x > 0)
    down = sum(1 for x in diffs if x < 0)
    ties = len(diffs) - up - down
    return {"on_per_person_day": round(on_mean, 1), "off_per_person_day": round(off_mean, 1),
            "diff": round(d, 1), "diff_pct": (round(100 * d / off_mean, 2) if off_mean else None),
            "ci95": [round(lo, 1), round(hi, 1)],
            "ci95_pct": ([round(100 * lo / off_mean, 2), round(100 * hi / off_mean, 2)] if off_mean else None),
            "people_up": up, "people_down": down, "people_tied": ties,
            "direction": ("+" if lo > 0 else "-" if hi < 0 else "0(구간이 0을 포함)")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    ap.add_argument("--on-policy")
    ap.add_argument("--off-policy")
    ap.add_argument("--subs", help="소분류별 차이를 낼 업종, 쉼표로(예: 슈퍼마켓,종합소매,한식)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261005)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    on, off = read(a.on), read(a.off)
    people, days = check_pair(on, off)
    n = len(days)
    res = {"people": len(people), "days": days,
           "definition": "1인 1일 평균, 정책 있음 - 없음(같은 사람·같은 날)", "measures": {}}
    for field in ("total_spent", "offline_spent", "online_spent", "policy_funded_won",
                  "instant_discount_won", "policy_rebate_won"):
        if not any(field in r for r in on):
            continue        # 옛 원장에는 할인·환급 칸이 없다
        res["measures"][field] = contrast(per_person(on, field), per_person(off, field),
                                          people, n, a.draws, a.seed)
    if a.subs:
        # "슈퍼마켓" 처럼 하나, 또는 "식품전문=식료품+정육+청과" 처럼 묶음(이름=소분류+소분류)
        on_sub, off_sub = per_person_map(on, "by_sub"), per_person_map(off, "by_sub")
        res["by_sub"] = {}
        for item in [x.strip() for x in a.subs.split(",") if x.strip()]:
            label, _, members = item.partition("=")
            parts = [m.strip() for m in (members or label).split("+") if m.strip()]
            res["by_sub"][label.strip()] = dict(
                contrast({p: sum(on_sub[p].get(k, 0) for k in parts) for p in people},
                         {p: sum(off_sub[p].get(k, 0) for k in parts) for p in people},
                         people, n, a.draws, a.seed), members=parts)
    on_l1, off_l1 = per_person_map(on, "by_l1"), per_person_map(off, "by_l1")
    sectors = sorted({k for m in list(on_l1.values()) + list(off_l1.values()) for k in m})
    res["by_l1"] = {k: contrast({p: on_l1[p].get(k, 0) for p in people},
                                {p: off_l1[p].get(k, 0) for p in people},
                                people, n, a.draws, a.seed) for k in sectors}
    if a.on_policy and a.off_policy:
        pon, poff = read(a.on_policy), read(a.off_policy)
        if check_pair(pon, poff) != (people, days):
            raise ValueError("정책 원장과 업종 원장의 사람·날이 다르다")
        res["measures"]["policy_eligible_offline_spent"] = contrast(
            per_person(pon, "eligible_offline_spent"), per_person(poff, "eligible_offline_spent"),
            people, n, a.draws, a.seed)
        last = max(days)
        received = sum(r.get("grant_received_cumulative") or 0 for r in pon if r["day"] == last)
        spent = sum(r.get("grant_spent_today") or 0 for r in pon)
        d_total = (sum(per_person(on, "total_spent").values())
                   - sum(per_person(off, "total_spent").values()))
        res["grant"] = {"received_won": received, "spent_won": spent,
                        "spent_share_of_received": (round(spent / received, 4) if received else None),
                        "extra_total_over_spent": (round(d_total / spent, 4) if spent else None),
                        "extra_total_over_received": (round(d_total / received, 4) if received else None),
                        "note": "늘어난 총지출(정책 있음-없음, 창 전체) / 정책으로 낸 돈, / 받은 돈"}
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    t = res["measures"]["total_spent"]
    print(f"총지출 {t['diff']:+.1f}원/인·일 ({t['diff_pct']}%, 95% {t['ci95']}) "
          f"사람 +{t['people_up']} / -{t['people_down']} / 동점 {t['people_tied']} -> {t['direction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
