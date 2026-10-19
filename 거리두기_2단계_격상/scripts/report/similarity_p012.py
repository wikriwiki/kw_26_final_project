"""P012 시뮬레이션이 KDI 실측과 얼마나 비슷한가 — 숫자로.

    python scripts/report/similarity_p012.py \
        --dir <채점 디렉터리(on/off 원장)> --score <score_full.json> --json-out <out.json>

## 무엇을 재는가

효과 지표 9개(K1, K3~K10)는 실측과 시뮬이 같은 단위(지원금이 없을 때보다 몇 % 더 썼나)라
바로 맞댈 수 있다. 여기에 네 가지 자를 쓴다.

1. **크기 유사도** — 지표마다 `1 - |시뮬 - 실측| / (|시뮬| + |실측|)` 의 평균.
   둘이 같으면 1, 한쪽이 다른 쪽의 절반이면 0.67, 부호가 반대면 0 이다. 단위가 없고
   0~1 로 닫혀 있어 지표끼리 평균낼 수 있다(대칭 평균 백분율 오차 sMAPE 의 보수 = 1 - sMAPE/2).
2. **방향 일치율** — 시뮬의 부호가 실측과 같은 지표의 비율. 순위 지표 K11·K12
   (어느 쪽이 더 올랐나)를 더해 11개로 센다.
3. **순위 상관** — 9개 지표를 실측 크기 순과 시뮬 크기 순으로 줄 세웠을 때의
   스피어만 상관. '어느 업종이 더 크게 반응하나' 의 모양이 닮았는지를 본다.
   p 는 9! 가지 줄 세우기를 모두 세어 낸 정확 순열검정(단측: 우연히 이만큼 닮을 확률).
4. **평균 절대 오차** — %p.

덧붙여 실측이 시뮬 95% 구간 안에 든 지표의 비율을 적는다. 구간이 넓으면 쉽게
높아지므로 이것 하나로 판단하지 않는다.

## 잡음

같은 사람을 두 번 돌렸으므로 사람 단위로 (지원금 있음, 없음) 짝을 함께 재표집한다 —
채점(score_p012_two_arm.py)과 같은 방식이다. 재표집마다 1~4 를 다시 계산해 95% 구간을
낸다. 한두 지표의 우연이 유사도를 얼마나 흔드는지가 이 구간에 드러난다.

## 지키는 것

- 실측 수치는 계약서(P012_indicator_contract.json)에서만 읽는다. 여기 적지 않는다.
- 시뮬의 점추정은 채점 결과(score_full.json)와 소수 6자리까지 같아야 한다 — 다르면
  다른 원장을 읽은 것이므로 멈춘다.
- 보정이나 가중치를 넣지 않는다. 7일 시뮬과 한 달 실측의 차이는 고치지 않고 적는다.
"""
from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_p012_two_arm as S  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

CERTAIN = 0.975      # 채점의 부호확실성 기준과 같다


def sym_similarity(s: float, t: float) -> float:
    """1 - |s-t|/(|s|+|t|). 같으면 1, 부호가 반대면 0, 둘 다 0 이면 1."""
    den = abs(s) + abs(t)
    if den == 0:
        return 1.0
    return 1.0 - abs(s - t) / den


def rankdata(x) -> list[float]:
    """평균 순위(동점은 평균)."""
    order = sorted(range(len(x)), key=lambda i: x[i])
    r = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(a, b) -> float | None:
    ra, rb = rankdata(list(a)), rankdata(list(b))
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return (num / den) if den > 0 else None


def spearman_perm_p(truth, sim) -> float | None:
    """정확 순열검정 단측 p — 실측 순위를 고정하고 시뮬 순위를 모든 순서로 섞는다."""
    n = len(truth)
    if n < 3 or n > 10:
        return None
    rt, rs = rankdata(list(truth)), rankdata(list(sim))
    m = sum(rt) / n
    ct = [x - m for x in rt]
    obs = sum(c * y for c, y in zip(ct, rs))
    ge = tot = 0
    for perm in itertools.permutations(rs):
        tot += 1
        if sum(c * y for c, y in zip(ct, perm)) >= obs - 1e-9:
            ge += 1
    return ge / tot


def metrics(eff_sim, eff_truth, rank_sim, rank_truth) -> dict:
    sims = [sym_similarity(s, t) for s, t in zip(eff_sim, eff_truth)]
    dirs = [(s > 0) == (t > 0) for s, t in zip(list(eff_sim) + list(rank_sim),
                                                 list(eff_truth) + list(rank_truth))]
    return {
        "magnitude_similarity": sum(sims) / len(sims),
        "direction_agreement": sum(dirs) / len(dirs),
        "rank_correlation": spearman(eff_truth, eff_sim),
        "mae_pp": sum(abs(s - t) for s, t in zip(eff_sim, eff_truth)) / len(eff_sim),
        "per_indicator_similarity": sims,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="on/off 원장이 든 채점 디렉터리")
    ap.add_argument("--score", required=True, help="score_p012_two_arm.py 의 json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--json-out", required=True)
    a = ap.parse_args()

    contract = json.loads(S.CONTRACT.read_text(encoding="utf-8"))
    score = json.loads(Path(a.score).read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in score["rows"]}
    d = Path(a.dir)
    sec_off, cash_off = S.load_arm(d, "off")
    sec_on, cash_on = S.load_arm(d, "on")
    aids = sorted(set(sec_off) & set(sec_on))
    if len(aids) != score["n_agents"]:
        raise SystemExit("원장 인원 %d 가 채점의 %d 와 다르다" % (len(aids), score["n_agents"]))

    # 효과 지표(증가율) — 계약서 순서 그대로
    eff = [i for i in contract["indicators"] if i.get("sim_metric") in S.METRIC_KEY]
    rnk = [i for i in contract["indicators"] if i.get("sim_metric") in S.RANK_METRIC]
    eff = [i for i in eff if rows.get(i["id"], {}).get("sim") is not None]
    rnk = [i for i in rnk if rows.get(i["id"], {}).get("sim") is not None]
    if len(eff) < 3:
        raise SystemExit("맞댈 효과 지표가 %d개뿐이다" % len(eff))

    # 사람 x 열 행렬: 열마다 (off, on) 합계를 재표집으로 다시 낸다
    cols: list[list[float]] = []
    def add(key):
        cols.append(S.series(sec_off, cash_off, aids, key))
        cols.append(S.series(sec_on, cash_on, aids, key))
        return len(cols) - 2
    eff_idx = [add(S.METRIC_KEY[i["sim_metric"]]) for i in eff]
    rnk_idx = [(add(S.RANK_METRIC[i["sim_metric"]][0]), add(S.RANK_METRIC[i["sim_metric"]][1]))
               for i in rnk]
    M = np.array(cols, dtype=np.float64).T          # (사람, 열)

    def effects(sums) -> tuple[list[float], list[float]] | None:
        sums = [float(x) for x in sums]
        e = []
        for j in eff_idx:
            if sums[j] <= 0:
                return None
            e.append(100.0 * (sums[j + 1] / sums[j] - 1.0))
        g = []
        for ja, jb in rnk_idx:
            if sums[ja] <= 0 or sums[jb] <= 0:
                return None
            g.append(100.0 * (sums[ja + 1] / sums[ja] - 1.0)
                     - 100.0 * (sums[jb + 1] / sums[jb] - 1.0))
        return e, g

    e0, g0 = effects(M.sum(axis=0))
    # 같은 원장을 읽었는지 — 채점의 점추정과 맞아야 한다
    for ind, v in zip(eff + rnk, e0 + g0):
        want = rows[ind["id"]]["sim"]
        if abs(want - v) > 1e-6 * max(1.0, abs(want)):
            raise SystemExit("%s: 채점 %.6f 과 여기 %.6f 가 다르다 — 다른 원장이다"
                             % (ind["id"], want, v))

    t_eff = [float(i["truth"]) for i in eff]
    t_rnk = [float(i.get("truth_gap")) for i in rnk]
    point = metrics(e0, t_eff, g0, t_rnk)
    point["rank_correlation_p"] = spearman_perm_p(t_eff, e0)

    rng = np.random.default_rng(a.seed)
    k = len(aids)
    boots = {"magnitude_similarity": [], "direction_agreement": [],
             "rank_correlation": [], "mae_pp": []}
    per_ind = [[] for _ in eff]
    done = 0
    while done < a.boot:
        b = min(250, a.boot - done)
        W = rng.multinomial(k, np.full(k, 1.0 / k), size=b).astype(np.float64)
        sums = W @ M
        for row in sums:
            r = effects(row)
            if r is None:
                continue
            m = metrics(r[0], t_eff, r[1], t_rnk)
            for key in boots:
                if m[key] is not None:
                    boots[key].append(m[key])
            for j, v in enumerate(m["per_indicator_similarity"]):
                per_ind[j].append(v)
        done += b

    def ci(v):
        if not v:
            return [None, None]
        v = sorted(v)
        return [v[int(len(v) * 0.025)], v[min(len(v) - 1, int(len(v) * 0.975))]]

    indicators = []
    for j, ind in enumerate(eff):
        r = rows[ind["id"]]
        lo, hi = (r.get("ci") or [None, None])
        indicators.append({
            "id": ind["id"], "name": ind["name"], "group": ind.get("group"),
            "truth": t_eff[j], "truth_significant": ind.get("sig"),
            "sim": e0[j], "ci": [lo, hi], "sign_conf": r.get("sign_conf"),
            "sign_test_p": r.get("sign_test_p"), "status": r.get("status"),
            "similarity": point["per_indicator_similarity"][j],
            "similarity_ci": ci(per_ind[j]),
            "abs_error_pp": abs(e0[j] - t_eff[j]),
            "truth_in_ci": (lo is not None and hi is not None and lo <= t_eff[j] <= hi),
            "direction_match": (e0[j] > 0) == (t_eff[j] > 0),
            "direction_certain": (r.get("sign_conf") or 0) >= CERTAIN,
        })
    ranks = []
    for j, ind in enumerate(rnk):
        r = rows[ind["id"]]
        ranks.append({
            "id": ind["id"], "name": ind["name"], "truth_gap": t_rnk[j], "sim_gap": g0[j],
            "ci": r.get("ci"), "sign_conf": r.get("sign_conf"), "status": r.get("status"),
            "direction_match": (g0[j] > 0) == (t_rnk[j] > 0),
            "direction_certain": (r.get("sign_conf") or 0) >= CERTAIN,
        })

    with_ci = [x for x in indicators if x["ci"][0] is not None]
    allrows = indicators + ranks
    out = {
        "n_agents": len(aids), "window": score.get("window"),
        "window_days": score.get("window_days"),
        "n_effect": len(indicators), "n_direction": len(allrows), "boot": a.boot,
        "headline": {
            "magnitude_similarity": point["magnitude_similarity"],
            "magnitude_similarity_ci": ci(boots["magnitude_similarity"]),
            "direction_agreement": point["direction_agreement"],
            "direction_agreement_ci": ci(boots["direction_agreement"]),
            "direction_matched": sum(1 for x in allrows if x["direction_match"]),
            "direction_certain_matched": sum(1 for x in allrows
                                             if x["direction_match"] and x["direction_certain"]),
            "direction_certain_mismatched": sum(1 for x in allrows
                                                if not x["direction_match"] and x["direction_certain"]),
            "rank_correlation": point["rank_correlation"],
            "rank_correlation_ci": ci(boots["rank_correlation"]),
            "rank_correlation_p": point["rank_correlation_p"],
            "mae_pp": point["mae_pp"], "mae_pp_ci": ci(boots["mae_pp"]),
            "truth_in_ci": sum(1 for x in with_ci if x["truth_in_ci"]),
            "truth_in_ci_of": len(with_ci),
        },
        "indicators": indicators, "rank_indicators": ranks,
        "score_rows": score["rows"],
    }
    Path(a.json_out).parent.mkdir(parents=True, exist_ok=True)
    io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(
        json.dumps(out, ensure_ascii=False, indent=1))

    h = out["headline"]
    f = lambda v: "-" if v is None else "%.3f" % v
    print("# P012 실측 유사도 — 시민 %d명 · %s (%s일)" % (len(aids), out["window"], out["window_days"]))
    print("  크기 유사도   %s  (95%% %s ~ %s)" % (f(h["magnitude_similarity"]), *map(f, h["magnitude_similarity_ci"])))
    print("  방향 일치율   %s  (%d/%d · 확실한 일치 %d · 확실한 불일치 %d)"
          % (f(h["direction_agreement"]), h["direction_matched"], len(allrows),
             h["direction_certain_matched"], h["direction_certain_mismatched"]))
    print("  순위 상관     %s  (95%% %s ~ %s · 순열 p=%s)"
          % (f(h["rank_correlation"]), *map(f, h["rank_correlation_ci"]), f(h["rank_correlation_p"])))
    print("  평균 절대오차 %s %%p" % f(h["mae_pp"]))
    print("  실측이 구간 안 %d/%d" % (h["truth_in_ci"], h["truth_in_ci_of"]))
    for x in indicators:
        print("   %-4s 실측 %+7.2f  시뮬 %+8.2f  유사도 %.2f  %s"
              % (x["id"], x["truth"], x["sim"], x["similarity"], x["status"]))
    print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
