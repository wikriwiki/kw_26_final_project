"""사전등록 채점표로 정책 런을 채점한다 — 부호만 본다.

`data/experiments/scoring_table.json` 에 등록된 지표만 계산한다. 여기 없는 지표로
후보 프롬프트를 고르면 실효 자유도가 늘어난다(docs/GENERALIZATION_METHOD.md §2).

**크기가 아니라 부호를 맞춘다.** 하루 총액이 페르소나 앵커에 묶여 있어 실측의
총소비 −14% 같은 크기는 구조적으로 못 낸다. 크기를 목표로 주면 프롬프트가 그걸
억지로 짜내고 홀드아웃에서 무너진다.

**불확실성을 함께 낸다.** 런을 여러 번 돌리는 대신(예산상 불가) 한 런 안에서
에이전트를 복원추출해 부트스트랩 신뢰구간을 내고, 표본을 반으로 갈라 양쪽에서
부호가 같은지 본다. 비용 0 이다(§8.1).

    python scripts/sim/score_policy.py --policy P012 \
        --off 2021-10-22:2021-10-24 --on 2021-10-25:2021-10-27
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
from _common import driver_session  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NULL_BAND = 0.10

TABLE = (Path(__file__).resolve().parents[2] / "data" / "experiments"
         / "scoring_table.json")

# 업종 묶음 — 채점표의 metric 이름에서 참조한다.
GROUP = {
    "식사": {"식사", "카페", "디저트", "주점"},
    "쇼핑+마트": {"쇼핑", "마트", "편의점"},
}


def daterange(spec: str) -> list[str]:
    a, b = spec.split(":")
    y0, m0, d0 = map(int, a.split("-"))
    y1, m1, d1 = map(int, b.split("-"))
    s, e = date(y0, m0, d0), date(y1, m1, d1)
    return [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


# =========================================================
# 원장 적재
# =========================================================
def fetch(days: list[str]) -> list[dict]:
    q = """
    MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
    WHERE coalesce(i.actual_spent,0) > 0 AND toString(pl.day) IN $days
    OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
    OPTIONAL MATCH (a)-[:LIVES_AT]->(h:POI)
    RETURN a.id AS aid, toString(pl.day) AS d, i.actual_spent AS amt,
           coalesce(i.time,'') AS t,
           coalesce(p.sangsaeng_eligible,false) AS elig,
           p.sangsaeng_kdi AS kdi, c.parent AS l1, c.name AS sub,
           p.dong_code AS pdong, h.dong_code AS hdong
    """
    with driver_session() as s:
        return [dict(r) for r in s.run(q, days=days)]


def fetch_cashback(last_day: str, rate: float, cap: int, ratio: float) -> dict:
    """정책 구간 마지막 날 State 로 캐시백 실적을 계산한다.

    문턱은 dawn_context._sangsaeng_monthly_anchor 와 같은 회계 단위로 잡는다 —
    적립업종 기준 2분기 월평균 x threshold_ratio. 분모를 총지출로 잡으면 문턱이
    4배 높아져 도달이 원천 불가능해진다.
    """
    q = """
    MATCH (a:Agent)-[:HAS_STATE {day: date($d)}]->(st:State)
    RETURN a.id AS aid, coalesce(st.sangsaeng_month_spent,0) AS spent,
           a.s_daily_wd AS wd, a.s_daily_we AS we
    """
    out = {}
    with driver_session() as s:
        for r in s.run(q, d=last_day):
            wd = float(r["wd"] or 0)
            we = float(r["we"] or wd)
            if wd <= 0:
                wd = we
            if wd <= 0:
                continue
            anchor = (wd * 5 + we * 2) / 7 * 30 * ratio
            thr = anchor * 1.03
            spent = float(r["spent"] or 0)
            excess = max(0.0, spent - thr)
            out[r["aid"]] = {
                "spent": spent, "thr": thr,
                "reached": 1.0 if spent >= thr else 0.0,
                "cashback": min(cap, excess * rate),
                "capped": 1.0 if excess * rate >= cap else 0.0,
            }
    return out


# =========================================================
# 지표 — 모두 (에이전트 → 값) 형태로 내서 쌍체차·부트스트랩에 쓴다
# =========================================================
def per_agent_daily(rows: list[dict], days: list[str], keep) -> dict[str, float]:
    """에이전트별 '해당 조건 지출의 하루 평균'."""
    acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if keep(r):
            acc[r["aid"]][r["d"]] += r["amt"]
    out = {}
    for aid, byday in acc.items():
        out[aid] = sum(byday.get(d, 0) for d in days) / len(days)
    return out


def per_agent_share(rows: list[dict], keep) -> dict[str, float]:
    """에이전트별 '해당 조건 지출이 총지출에서 차지하는 몫'."""
    tot: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for r in rows:
        tot[r["aid"]] += r["amt"]
        if keep(r):
            hit[r["aid"]] += r["amt"]
    return {a: hit[a] / t for a, t in tot.items() if t > 0}


def _hour(t: str) -> int | None:
    try:
        return int(str(t).split(":")[0])
    except (ValueError, IndexError):
        return None


def _sector_filter(key: str):
    """업종 이름 하나를 원장 한 줄에 대한 판정 함수로 바꾼다.

    KDI 8분류·L1 12종·세분류(Category.name) 어느 쪽 이름으로도 지정할 수 있게
    한다. 세분류를 빠뜨리면 위약의 대상 업종(의류 등)이 매칭되지 않아 반응이
    있어도 0 으로 읽힌다. `a|b|c` 로 여러 업종을 묶을 수 있다 — 관측 수가 적은
    업종(여행사·숙박·헬스장)을 합쳐야 채점이 가능해진다.
    """
    keys = [k.strip() for k in str(key).split("|") if k.strip()]
    subs: set[str] = set()
    names: set[str] = set()
    for k in keys:
        g = GROUP.get(k)
        if g:
            subs |= g
        else:
            names.add(k)

    def f(x):
        if subs and x["l1"] in subs:
            return True
        return bool(names) and (x["kdi"] in names or x["l1"] in names
                                or x.get("sub") in names)
    return f


def metric_values(name: str, off_rows, on_rows, off_days, on_days):
    """(off 값 dict, on 값 dict). 에이전트 id 를 키로 맞춰 쌍체차를 만든다."""
    def both(fn):
        return fn(off_rows, off_days), fn(on_rows, on_days)

    if name in ("total_spend_paired", "mpc_amount"):
        return both(lambda r, d: per_agent_daily(r, d, lambda x: True))
    if name in ("elig_spend_paired", "coupon_elig_spend_paired",
                "elig_spend_pre", "elig_spend_post"):
        return both(lambda r, d: per_agent_daily(r, d, lambda x: x["elig"]))
    if name == "excl_spend_paired":
        return both(lambda r, d: per_agent_daily(r, d, lambda x: not x["elig"]))
    if name.startswith("sector_spend:"):
        f = _sector_filter(name.split(":", 1)[1])
        return both(lambda r, d, f=f: per_agent_daily(r, d, f))
    if name.startswith("sector_share:"):
        # 업종 지출이 **총지출에서 차지하는 몫**. 하루 총액이 페르소나 앵커에
        # 묶여 있어 한 업종이 오르면 다른 업종이 빠진다(위약 PL-2 가 이 구조로
        # 실패했다). 몫으로 보면 총액 제약이 약분되어 "어디에 쓰는가" 만 남는다.
        f = _sector_filter(name.split(":", 1)[1])
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name == "elig_spend_share":
        f = lambda x: bool(x["elig"])                # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name == "late_night_share":
        f = lambda x: ((_hour(x["t"]) or 0) >= 21)   # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name == "out_district_ratio":
        f = lambda x: (str(x["pdong"] or "")[:5] != str(x["hdong"] or "")[:5])  # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name in ("home_dong_spend_share",):
        f = lambda x: (x["pdong"] and x["pdong"] == x["hdong"])   # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name == "out_district_spend_share":
        f = lambda x: (str(x["pdong"] or "")[:5] != str(x["hdong"] or "")[:5])  # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    return None, None


# =========================================================
# 쌍체차 + 부트스트랩 + 반분
# =========================================================
def paired(off: dict, on: dict) -> list[float]:
    return [on[a] - off[a] for a in on if a in off]


def boot_ci(d: list[float], n: int = 2000, seed: int = 7) -> tuple[float, float]:
    if len(d) < 3:
        return (0.0, 0.0)
    rnd = random.Random(seed)
    means = []
    k = len(d)
    for _ in range(n):
        means.append(sum(d[rnd.randrange(k)] for _ in range(k)) / k)
    means.sort()
    return (means[int(0.025 * n)], means[int(0.975 * n)])


def sign_of(lo: float, hi: float) -> str:
    if lo > 0:
        return "+"
    if hi < 0:
        return "-"
    return "0"


def split_half_agree(off: dict, on: dict, seed: int = 7) -> bool:
    ids = sorted(set(on) & set(off))
    if len(ids) < 8:
        return False
    random.Random(seed).shuffle(ids)
    mid = len(ids) // 2
    s = []
    for part in (ids[:mid], ids[mid:]):
        d = [on[a] - off[a] for a in part]
        m = sum(d) / len(d)
        s.append("+" if m > 0 else ("-" if m < 0 else "0"))
    return s[0] == s[1]


# =========================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True, help="채점표의 정책 키")
    ap.add_argument("--off", required=True, help="무정책 구간 YYYY-MM-DD:YYYY-MM-DD")
    ap.add_argument("--on", required=True, help="정책 구간 YYYY-MM-DD:YYYY-MM-DD")
    ap.add_argument("--label", default="", help="후보 프롬프트 이름 등")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    table = json.loads(TABLE.read_text(encoding="utf-8"))
    spec = table.get(a.policy)
    if not spec:
        print(f"채점표에 {a.policy} 없음. 가능: "
              f"{[k for k in table if not k.startswith('_')]}", file=sys.stderr)
        return 2

    off_days, on_days = daterange(a.off), daterange(a.on)
    off_rows, on_rows = fetch(off_days), fetch(on_days)
    if not off_rows or not on_rows:
        print("결제 데이터 없음 — 구간을 확인할 것", file=sys.stderr)
        return 2

    print(f"[{a.policy}] {a.label or '(무명)'}  기전={spec['mechanism']}")
    print(f"  무정책 {off_days[0]}~{off_days[-1]} ({len(off_rows):,}건) / "
          f"정책 {on_days[0]}~{on_days[-1]} ({len(on_rows):,}건)")
    print("=" * 78)

    results = []
    ranks: dict[str, float] = {}
    for ind in spec["indicators"]:
        name, expect = ind["metric"], ind["expect"]
        if expect == "rank":
            continue
        if name in ("threshold_reach_rate", "cashback_per_capita", "cap_reach_rate"):
            cb = fetch_cashback(on_days[-1], 0.10, 100_000, 0.268)
            fld = {"threshold_reach_rate": "reached",
                   "cashback_per_capita": "cashback",
                   "cap_reach_rate": "capped"}[name]
            vals = [v[fld] for v in cb.values()]
            if len(vals) < 3:
                results.append({**ind, "got": "관측부족", "hit": None})
                print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} 관측부족")
                continue
            lo, hi = boot_ci(vals)
            got = sign_of(lo, hi)
            hit = (got == expect)
            m = sum(vals) / len(vals)
            results.append({**ind, "got": got, "hit": hit, "mean": m,
                           "ci": [lo, hi], "n": len(vals)})
            print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} "
                  f"기대 {expect} / 실측 {got} {'O' if hit else 'X'}  "
                  f"평균 {m:,.3f} CI[{lo:,.3f},{hi:,.3f}] n={len(vals)}")
            continue

        off, on = metric_values(name, off_rows, on_rows, off_days, on_days)
        if off is None:
            results.append({**ind, "got": "미구현", "hit": None})
            print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} 미구현")
            continue
        d = paired(off, on)
        if len(d) < 3:
            results.append({**ind, "got": "관측부족", "hit": None})
            print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} 관측부족(n={len(d)})")
            continue
        m = sum(d) / len(d)
        lo, hi = boot_ci(d)
        base = sum(off[x] for x in on if x in off) / max(1, len(d))
        if expect == "0":
            # 동등성 검정 — 구간이 기준선 ±10% 밴드 **안에** 들어야 적중.
            # "구간이 0 을 포함하는가"로 보면 잡음이 큰 후보가 저절로 통과한다.
            band = NULL_BAND * abs(base) if base else 0.0
            got = "0" if (band > 0 and -band <= lo and hi <= band) else "≠0"
            hit = (got == "0")
        else:
            got = sign_of(lo, hi)
            hit = (got == expect)
        agree = split_half_agree(off, on)
        # 순위 지표용으로 평균 변화율을 남긴다
        ranks[name] = (m / base * 100) if base else 0.0
        results.append({**ind, "got": got, "hit": hit, "mean": m, "base": base,
                       "ci": [lo, hi], "n": len(d), "split_agree": agree})
        mark = "O" if hit else "X"
        print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} "
              f"기대 {expect} / 실측 {got} {mark}  "
              f"평균 {m:+,.0f} CI[{lo:+,.0f},{hi:+,.0f}] n={len(d)} "
              f"반분{'일치' if agree else '불일치'}")

    # 순위 지표
    for ind in spec["indicators"]:
        if ind["expect"] != "rank":
            continue
        hi_k, lo_k = ind["rank"]
        vals = {}
        for k in (hi_k, lo_k):
            mname = f"sector_spend:{k}"
            off, on = metric_values(mname, off_rows, on_rows, off_days, on_days)
            if off is None:
                continue
            d = paired(off, on)
            if len(d) < 3:
                continue
            base = sum(off[x] for x in on if x in off) / max(1, len(d))
            vals[k] = (sum(d) / len(d) / base * 100) if base else None
        if len(vals) < 2 or any(v is None for v in vals.values()):
            results.append({**ind, "got": "관측부족", "hit": None})
            print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} 관측부족")
            continue
        hit = vals[hi_k] > vals[lo_k]
        results.append({**ind, "got": f"{hi_k} {vals[hi_k]:+.1f}% vs "
                                     f"{lo_k} {vals[lo_k]:+.1f}%", "hit": hit})
        print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} "
              f"{hi_k} {vals[hi_k]:+.1f}% vs {lo_k} {vals[lo_k]:+.1f}% "
              f"{'O' if hit else 'X'}")

    scored = [r for r in results if r["hit"] is not None]
    hits = sum(1 for r in scored if r["hit"])
    print("=" * 78)
    if scored:
        print(f"  부호 일치 {hits}/{len(scored)} = {100*hits/len(scored):.0f}%"
              f"  (미채점 {len(results)-len(scored)})")
    else:
        print("  채점 가능한 지표가 없다")

    if a.json_out:
        Path(a.json_out).write_text(json.dumps(
            {"policy": a.policy, "label": a.label, "off": a.off, "on": a.on,
             "hits": hits, "scored": len(scored), "results": results},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  → {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
