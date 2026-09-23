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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neo4j_load._common import driver_session  # noqa: E402

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
    if e < s:
        raise ValueError("date range must be ascending")
    return [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


# =========================================================
# 원장 적재
# =========================================================
def fetch(days: list[str]) -> list[dict]:
    q = """
    MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
    WHERE toString(pl.day) IN $days
    OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
    OPTIONAL MATCH (a)-[:LIVES_AT]->(h:POI)
    RETURN a.id AS aid, toString(pl.day) AS d, coalesce(i.actual_spent, 0) AS amt,
           coalesce(i.time,'') AS t, coalesce(i.anchor,'') AS anchor,
           coalesce(p.sangsaeng_eligible,false) AS elig,
           p.sangsaeng_kdi AS kdi, c.parent AS l1, c.name AS sub,
           p.name AS pname, p.upjong_l3 AS upjong_l3,
           p.dong_code AS pdong, h.dong_code AS hdong
    """
    with driver_session() as s:
        return [dict(r) for r in s.run(q, days=days)]


def apply_policy_eligibility(rows: list[dict], policy_file: str | None) -> str:
    """`elig` 을 **해당 정책의 적격 규칙**으로 다시 계산한다.

    원장 조회의 `p.sangsaeng_eligible` 은 `11_sangsaeng_eligibility.py` 가 P012
    기준(대형마트·백화점·면세점 등만 제외)으로 백필한 값이다. 다른 정책을
    채점하면서 이 값을 그대로 쓰면 엉뚱한 업종 집합을 재게 된다 — 8대 소비쿠폰
    홀드아웃에서 HO-3 이 base 0.9872 로 천장에 붙어 무효가 된 원인이 이것이다.

    명세가 없는 정책은 둘로 갈린다.

      사용처 제한이 **있는** 정책(P010·P013 같은 쿠폰형 지갑)
          런타임이 실제로 쓴 함수(`is_coupon_eligible`)로 다시 계산한다.
          그래프에 `p.coupon_eligible` 이 **한 개도 없어서**(확인: 0/543,924)
          stage2 는 매번 이 함수로 즉석 판정했다. 그런데 채점은 상생 백필값을
          그대로 썼다 — **모델이 본 자와 채점하는 자가 달랐다.**

      사용처 제한이 **없는** 정책(P012 상생소비지원금)
          DB 백필값이 곧 그 정책의 기준이므로 그대로 둔다.
    """
    if not policy_file:
        return "DB 백필값(상생 기준)"
    fp = Path(__file__).resolve().parents[2] / policy_file
    if not fp.exists():
        return f"정책 파일 없음: {policy_file}"
    pol = json.loads(fp.read_text(encoding="utf-8"))
    spec = pol.get("eligibility")
    if not spec:
        if pol.get("poi_restricted"):
            # 런타임과 **같은 함수**를 쓴다. 규칙을 정규식으로 옮겨 적는 방법도
            # 있었지만 범용 평가기는 업종코드가 있으면 세분류 제외를 건너뛰므로
            # (POI 의 95.5% 가 코드를 갖고 있다) 옮겨 적은 것이 원본과 달라진다.
            # 같은 함수를 부르면 어긋날 자리가 없다.
            import sys as _sys
            _here = str(Path(__file__).resolve().parent)
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from coupon_eligibility import is_coupon_eligible
            for r in rows:
                r["elig"] = bool(is_coupon_eligible(
                    r.get("pname"), r.get("sub"), r.get("l1"))[0])
            return (f"{pol.get('id')} 쿠폰 사용처 룰(is_coupon_eligible)"
                    " — 런타임과 같은 함수")
        return "DB 백필값(상생 기준) — 사용처 제한이 없는 정책"
    from eligibility import Rules
    rules = Rules(spec)
    for r in rows:
        # 장소 조건이 있는 정책은 결제처와 사는 곳의 자치구를 대조한다.
        _pg = str(r.get("pdong") or "")[:5]
        _hg = str(r.get("hdong") or "")[:5]
        _same = (bool(_pg) and bool(_hg) and _pg == _hg) if (_pg or _hg) else None
        r["elig"] = bool(rules.eligible(r.get("pname"), r.get("sub"),
                                        r.get("l1"), r.get("upjong_l3"), _same)[0])
    return f"{pol.get('id')} 적격 규칙({spec.get('mode')})"


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
    # 해당 업종을 0원 쓴 것도 관측이다. 먼저 전체 활동 에이전트를 넣지 않으면
    # 정책 전후에 업종을 새로 이용하거나 끊은 사람이 쌍체 표본에서 빠진다.
    for row in rows:
        acc[row["aid"]]
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


def per_agent_share_within(rows: list[dict], keep, within) -> dict[str, float]:
    """에이전트별 '해당 조건 지출이 **지정한 분모 안에서** 차지하는 몫'.

    `per_agent_share` 는 분모가 총지출이다. 그런데 "전체 마트 매출 중 농축산물
    매출 비중" 처럼 **더 좁은 분모**를 쓰는 지표가 있다. 총지출을 분모로 쓰면
    다른 업종이 움직이기만 해도 값이 흔들려 정책 효과와 섞인다.
    """
    tot: dict[str, int] = defaultdict(int)
    hit: dict[str, int] = defaultdict(int)
    for r in rows:
        if not within(r):
            continue
        tot[r["aid"]] += r["amt"]
        if keep(r):
            hit[r["aid"]] += r["amt"]
    return {a: hit[a] / t for a, t in tot.items() if t > 0}


def per_agent_home_hours(rows: list[dict], days: list[str]) -> dict[str, float]:
    """에이전트별 '하루 평균 거주지 체류 시간(시간)'.

    계획은 시각이 붙은 이벤트 열이므로, 한 이벤트의 체류는 **다음 이벤트까지의
    간격**으로 본다. 마지막 이벤트는 자정까지로 둔다. 앵커가 residence 인
    구간만 더한다.

    구글 이동성의 residential 과 같은 양은 아니다 — 그쪽은 기기 위치 기반이고
    이쪽은 계획된 일정이다. 방향(늘었나 줄었나)만 견준다.
    """
    byday: dict[str, dict[str, list[tuple[int, str]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        mm = _minute(r.get("t"))
        if mm is None:
            continue
        byday[r["aid"]][r["d"]].append((mm, str(r.get("anchor") or "")))
    out = {}
    for aid, days_map in byday.items():
        tot = 0.0
        for d in days:
            ev = sorted(days_map.get(d, []))
            for k, (mm, a) in enumerate(ev):
                end = ev[k + 1][0] if k + 1 < len(ev) else 24 * 60
                if a == "residence":
                    tot += max(0, end - mm)
        out[aid] = tot / 60.0 / len(days)
    return out


def _minute(t) -> int | None:
    try:
        h, m = str(t).split(":")[:2]
        return int(h) * 60 + int(m)
    except (ValueError, IndexError):
        return None


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
    if name.startswith("sector_share_within:"):
        # `sector_share_within:<대상>/<분모>` — 분모를 좁힌 몫.
        # 예: sector_share_within:청과|정육|슈퍼마켓|식료품/마트
        arg = name.split(":", 1)[1]
        tgt, _, den = arg.partition("/")
        f = _sector_filter(tgt)
        w = _sector_filter(den) if den else (lambda x: True)
        return (per_agent_share_within(off_rows, f, w),
                per_agent_share_within(on_rows, f, w))
    if name == "elig_spend_share":
        f = lambda x: bool(x["elig"])                # noqa: E731
        return (per_agent_share(off_rows, f), per_agent_share(on_rows, f))
    if name == "home_hours":
        return both(per_agent_home_hours)
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
    # 부트스트랩 시드가 같으면 입력 순서도 같아야 CI가 재현된다.
    return [on[a] - off[a] for a in sorted(set(on) & set(off))]


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
    ap.add_argument("--per-agent", action="store_true",
                    help="지표별 에이전트 단위 off/on 값을 <json-out>.agents.jsonl 로 남긴다. "
                         "런 간 이동이 어디서 오는지는 총합만 보면 알 수 없다 — "
                         "같은 프롬프트·같은 에이전트인데 값이 움직이면 그 자국이 여기 남는다.")
    ap.add_argument("--elig-policy", default="",
                    help="적격 판정에 쓸 정책 JSON 경로. 생략하면 채점표의 policy_file"
                         " 을 쓰고, 그것도 없으면 DB 백필값(상생 기준)을 그대로 둔다.")
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

    # 적격 판정을 채점하는 정책의 규칙으로 맞춘다(위 함수 주석 참조).
    _pf = a.elig_policy or spec.get("policy_file") or ""
    _how = apply_policy_eligibility(off_rows, _pf) if _pf else "DB 백필값(상생 기준)"
    if _pf:
        apply_policy_eligibility(on_rows, _pf)

    print(f"[{a.policy}] {a.label or '(무명)'}  기전={spec['mechanism']}  적격={_how}")
    print(f"  무정책 {off_days[0]}~{off_days[-1]} ({len(off_rows):,}건) / "
          f"정책 {on_days[0]}~{on_days[-1]} ({len(on_rows):,}건)")
    print("=" * 78)

    results = []
    per_agent: list[dict] = []
    ranks: dict[str, float] = {}
    for ind in spec["indicators"]:
        name, expect = ind["metric"], ind["expect"]
        if expect == "rank":
            continue
        if expect == "info":
            # 채점에 쓰지 않기로 등록된 지표다. '미구현' 으로 찍으면 결함처럼 보인다.
            results.append({**ind, "got": "채점 대상 아님", "hit": None})
            print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} 채점 대상 아님(등록)")
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
        if a.per_agent:
            # 쌍이 맞는 에이전트만 — 채점이 쓰는 것과 같은 집합이다
            for _aid in sorted(set(off) & set(on)):
                per_agent.append({"id": ind["id"], "metric": name, "aid": _aid,
                                  "off": off[_aid], "on": on[_aid]})
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
        # 몷(share) 지표는 0~1 이라 정수로 반올림하면 화면에 전부 0 으로 찍힌다.
        # JSON 은 원값을 담지만 로그만 보고 판단하는 순간이 있어 자릿수를 갈라 쓴다.
        _sh = abs(base) <= 1.5
        _f = "{:+.4f}" if _sh else "{:+,.0f}"
        _pct = f" ({100*m/base:+.1f}%)" if base else ""
        print(f"  {ind['id']:<8} {ind['desc'][:44]:<46} "
              f"기대 {expect} / 실측 {got} {mark}  "
              f"평균 {_f.format(m)}{_pct} "
              f"CI[{_f.format(lo)},{_f.format(hi)}] n={len(d)} "
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

    # ------------------------------------------------------------------
    # 진단 지표 — **채점하지 않는다.**
    # 금액 지표는 하루 총액이 페르소나 앵커에 묶여 있어 한 업종이 오르면 다른
    # 업종이 빠진다. 같은 업종을 '총지출에서 차지하는 몫' 으로도 같이 남겨 두면
    # 나중에 원장 없이도 그 구조를 확인할 수 있다. 런이 끝나면 원장은 다음 런의
    # 97_reset 으로 지워지므로 이때 안 남기면 영영 못 본다.
    # 사전등록 지표가 아니므로 hits 계산에 들어가지 않는다.
    diagnostics = []
    for ind in spec["indicators"]:
        nm = ind.get("metric") or ""
        if not nm.startswith("sector_spend:"):
            continue
        share = "sector_share:" + nm.split(":", 1)[1]
        off, on = metric_values(share, off_rows, on_rows, off_days, on_days)
        if off is None:
            continue
        d = paired(off, on)
        if len(d) < 3:
            continue
        m = sum(d) / len(d)
        lo, hi = boot_ci(d)
        base = sum(off[x] for x in on if x in off) / max(1, len(d))
        diagnostics.append({"of": ind["id"], "metric": share, "scored": False,
                            "mean": m, "base": base, "ci": [lo, hi], "n": len(d)})
    if diagnostics:
        print("-" * 78)
        print("  [진단 · 채점 아님] 같은 업종을 몫으로 보면")
        for g in diagnostics:
            b = g["base"] or 1e-9
            print(f"  {g['of']:<8} {g['metric']:<28} "
                  f"몫 {g['base']:.3f} → {g['base']+g['mean']:.3f} "
                  f"({100*g['mean']/b:+.1f}%) CI[{g['ci'][0]:+.4f},{g['ci'][1]:+.4f}] n={g['n']}")

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
             "hits": hits, "scored": len(scored), "results": results,
             "diagnostics": diagnostics},
            ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  → {a.json_out}")
        if a.per_agent:
            side = Path(a.json_out).with_suffix("").as_posix() + ".agents.jsonl"
            with open(side, "w", encoding="utf-8") as fh:
                for row in per_agent:
                    fh.write(json.dumps(row, ensure_ascii=False) + chr(10))
            print(f"  → {side}  ({len(per_agent)}행 · 지표 {len({r['id'] for r in per_agent})}개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
