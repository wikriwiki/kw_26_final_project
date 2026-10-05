"""파일럿 채점 결과로 본런 규모를 정한다 — 지표마다 부호가 확실해지려면(95% 구간이 0 을 벗어나려면) 몇 명이 필요한가.

    python tools/project_p013_sample_size.py --score <score_p013.json> [--days 18] [--rate-lo 460 --rate-hi 550]

가정(그대로 적는다):
  · 구간 폭은 1/sqrt(사람 수) 로 준다 — 사람 단위 짝 부트스트랩이므로. 날 수가 바뀌면 이 가정이 깨진다
    (같은 날 수로 돌릴 때만 쓴다).
  · '시뮬 값이 그대로일 때' 와 '시뮬 값이 실측 값만큼일 때' 두 가지를 낸다. 앞은 지금 방향을 굳히는 데,
    뒤는 실측 크기의 효과를 잡아낼 수 있는지에 쓴다.
  · 시간 = 사람 x 날 x 2(지원금 있음·없음) / 시간당 처리 명·일. 처리량 460~550 은 P012 본런 실측
    (3,000명 x 7일, 45.6시간 / 38.2시간). 범위로만 말한다.
"""
from __future__ import annotations

import argparse
import io
import json
import math

Z = 1.959964


def need(n0, half, eff):
    if not eff or half is None or half <= 0:
        return None
    se0 = half / Z
    return math.ceil(n0 * (Z * se0 / abs(eff)) ** 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", required=True)
    ap.add_argument("--days", type=int, default=None, help="본런 날 수(기본: 파일럿과 같게)")
    ap.add_argument("--rate-lo", type=float, default=460.0)
    ap.add_argument("--rate-hi", type=float, default=550.0)
    a = ap.parse_args()
    s = json.load(io.open(a.score, encoding="utf-8"))
    n0 = s["n_agents"]
    days = a.days or len(s["days"])
    truth = {}
    try:
        c = json.load(io.open("data/experiments/P013_indicator_contract.json", encoding="utf-8"))
        truth = {i["id"]: i.get("truth") for i in c["indicators"] if isinstance(i.get("truth"), (int, float))}
    except OSError:
        pass
    print("파일럿 %d명 · %d일 기준. 시간은 같은 날 수(%d일) x 두 시뮬레이션, 처리량 %.0f~%.0f 명·일/시간"
          % (n0, len(s["days"]), days, a.rate_lo, a.rate_hi))
    print("%-10s %-34s %9s %9s %10s %12s" % ("지표", "이름", "시뮬", "±구간", "N(시뮬값)", "N(실측값)"))
    for r in s["rows"]:
        # 효과(있음 − 없음)만 — 소진율·구성 유사도·근접 몫(U1·U2·U4)은 수준 값이라 부호 검출력이 뜻이 없다
        if r.get("sim") is None or not r.get("ci") or r["ci"][0] is None or r["id"] in ("D3", "U1", "U2", "U4"):
            continue
        half = (r["ci"][1] - r["ci"][0]) / 2
        t = truth.get(r["id"])
        if t is None and r.get("truth_range"):
            t = sum(r["truth_range"]) / 2
        if t is None and r.get("truth_gap"):
            t = r["truth_gap"]
        n_sim, n_tr = need(n0, half, r["sim"]), need(n0, half, t)
        print("%-10s %-34s %+8.2f %9.2f %10s %12s" % (
            r["id"] + ("·%d주" % r["weeks"] if r.get("weeks") else ""), (r.get("name") or "")[:34], r["sim"], half,
            "{:,}".format(max(n_sim, 1)) if n_sim else "–", "{:,}".format(max(n_tr, 1)) if n_tr else "–"))
    print()
    for n in (1000, 1500, 2000, 3000):
        pd_ = n * days * 2
        print("  %5d명 x %d일 x 2 = %s 명·일 → %.0f~%.0f 시간 (%.1f~%.1f 일)"
              % (n, days, "{:,}".format(pd_), pd_ / a.rate_hi, pd_ / a.rate_lo, pd_ / a.rate_hi / 24, pd_ / a.rate_lo / 24))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
