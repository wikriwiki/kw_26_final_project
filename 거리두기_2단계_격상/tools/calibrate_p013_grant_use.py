"""P013 지원금 카드로 낸 몫(EXP_GRANT_USE)을 전국 소진 속도에 맞춘다 — GPU 없이 원장 위에서.

    python tools/calibrate_p013_grant_use.py --ledger <on.policy.ledger.jsonl> --out <calib.json> \
        [--target-date 2020-05-24 --target 38.6 --grant 280000]

규칙은 지표 계약서 _calibration 에 먼저 적었다(2026-10-03). 지원금 있는 시뮬레이션(몫 1.0)의 사람·날짜별
사용처 오프라인 지출 E 는 그대로 두고, 받은 날부터 하루 사용 = min(잔액, f x E) 로 다시 센다.
target-date 까지 누적 사용 / 배정액이 target(%) 이 되는 f 를 이분법으로 찾는다.

되먹임은 없다 — 잔액이 더 남으면 에이전트가 계획을 바꿀 수 있다. 그래서 찾은 f 로 지원금 있는 쪽을 다시
돌려 확인한다. 받은 날은 원장의 grant_received_cumulative 가 처음 0 보다 커진 날이다.
"""
from __future__ import annotations

import argparse
import io
import json
from collections import defaultdict


def load(path):
    rows = defaultdict(dict)
    with io.open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["aid"]][r["day"]] = r
    return rows


def replay(rows, f, grant, until):
    """f 로 다시 센 날짜별 누적 사용액 {day: 원}. 받은 날 전에는 쓰지 않는다."""
    days = sorted({d for by in rows.values() for d in by})
    cum = {d: 0.0 for d in days}
    for by in rows.values():
        left, got = None, False
        for d in days:
            r = by.get(d)
            if r is None:
                continue
            if not got and (r.get("grant_received_cumulative") or 0) > 0:
                got, left = True, float(grant)
            use = min(left, f * float(r.get("eligible_offline_spent") or 0)) if got else 0.0
            if got:
                left -= use
            cum[d] += use
    out, acc = {}, 0.0
    for d in days:
        acc += cum[d]
        if d <= until:
            out[d] = acc
    return out


def solve(rows, grant, until, target_pct, tol=1e-4):
    alloc = grant * len(rows)
    share = lambda f: 100.0 * replay(rows, f, grant, until)[until] / alloc
    hi_share = share(1.0)
    if hi_share < target_pct:
        return 1.0, hi_share, "몫 1.0 에서도 목표보다 느리다 — 줄일 근거가 없다"
    lo, hi = 0.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if share(mid) < target_pct:
            lo = mid
        else:
            hi = mid
    f = round((lo + hi) / 2, 4)
    return f, share(f), "ok"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-date", default="2020-05-24")
    ap.add_argument("--target", type=float, default=38.6, help="전국 누적 소진율(%) — 계약서 _calibration")
    ap.add_argument("--grant", type=int, default=280000)
    a = ap.parse_args()
    rows = load(a.ledger)
    if any(a.target_date not in by for by in rows.values()):
        raise SystemExit("원장에 %s 가 없는 사람이 있다 — 그 날까지 돈 런이어야 한다" % a.target_date)
    f, got, note = solve(rows, a.grant, a.target_date, a.target)
    alloc = a.grant * len(rows)
    path_f = {d: round(100 * v / alloc, 2) for d, v in replay(rows, f, a.grant, a.target_date).items()}
    path_1 = {d: round(100 * v / alloc, 2) for d, v in replay(rows, 1.0, a.grant, a.target_date).items()}
    res = {"grant_use": f, "share_at_target_date": round(got, 2), "target": a.target, "target_date": a.target_date,
           "note": note, "people": len(rows), "grant": a.grant, "ledger": a.ledger,
           "cum_share_pct_with_f": path_f, "cum_share_pct_with_1": path_1,
           "check": "서울 실측(U1) 1주(5/17) 13.5% · 2주(5/24) 38.7%, 전국 5/17 12.2% — 2주차는 보정과 같은 양"}
    io.open(a.out, "w", encoding="utf-8", newline="\n").write(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print("몫 f = %.4f · %s 누적 %.2f%% (목표 %.1f%%) · %s" % (f, a.target_date, got, a.target, note))
    for d in sorted(path_f):
        print("  %s  f=%.3f: %5.1f%%   f=1: %5.1f%%" % (d, f, path_f[d], path_1[d]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
