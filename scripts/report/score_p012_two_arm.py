"""P012 두 팔(정책 없음 / 있음) 원장을 KDI 실측과 맞댄다 — **방향을 잡음과 갈라서.**

    python scripts/report/score_p012_two_arm.py --dir data/experiments/p012_v53_pilot

## 무엇을 재는가

계약서 `data/experiments/P012_indicator_contract.json` 에 원문 인용과 함께 못 박은
지표 전부. 수치는 그 파일에서 읽는다 — 이 스크립트에 실측을 적어 두지 않는다.

## 왜 두 팔인가

정책 시행일이 10-01 이라 **10월 안에 정책 전 날이 없다.** 9월을 붙이면 월 누적
문턱에 9월 지출이 섞인다. 같은 사람을 정책 없음/있음으로 두 번 돌리면 그 둘을 다
피하고 **진짜 반사실**이 된다. 원문(KDI)은 수급/미수급 그룹 대조라 평행추세가정이
필요했는데, 우리 설계는 그 가정이 필요 없다.

## 방향을 잡음과 가른다 — 이것이 판정의 핵심

같은 방향이 나왔다고 맞춘 것이 아니다. 12명이면 부호는 동전던지기로도 맞는다.
그래서 지표마다 넷을 함께 낸다.

    방향      실측의 부호와 시뮬의 부호가 같은가
    구간      시뮬의 95% 구간이 0 을 지나는가 (지나면 **방향을 주장할 수 없다**)
    쌍체부호  사람별로 몇 명이 올랐나 (동점을 숨기지 않는다)
    크기      실측과의 차이 (%p)

그리고 합계에서 **'방향 적중'은 구간이 0 을 지나지 않는 지표만** 센다. 지나는 것은
'판정 불가' 로 따로 센다. 구간이 0 을 지나는데 부호가 맞았다고 세면 잡음을 성적으로
바꾸는 것이다.
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import statistics as st
from collections import defaultdict
from pathlib import Path

try:                              # 윈도 콘솔 기본이 cp949 라 한글 표가 깨진다
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "data/experiments/P012_indicator_contract.json"
# 우리 L1/L2 -> KDI 8분류. 이미 있는 파일을 쓴다(새로 적지 않는다).
IMAP = ROOT / "data/sangsaeng/industry_map.json"


def load_kdi_map():
    m = json.loads(IMAP.read_text(encoding="utf-8"))
    l1 = dict(m.get("l1_default") or {})
    sub = dict(m.get("sub_override") or {})
    return l1, sub


def arm_window(d: Path, arm: str) -> tuple[str, str, int]:
    """원장이 실제로 담은 (첫날, 마지막날, 날 수). **계약서의 창을 믿지 않는다.**"""
    days = set()
    for line in io.open(d / ("%s.sector.ledger.jsonl" % arm), encoding="utf-8"):
        line = line.strip()
        if line:
            days.add(json.loads(line)["day"])
    if not days:
        return ("", "", 0)
    return (min(days), max(days), len(days))


def load_arm(d: Path, arm: str):
    """(에이전트별 합계, 에이전트별 캐시백 합계) — 원장에 있는 날 전부."""
    l1map, submap = load_kdi_map()
    sec = defaultdict(lambda: defaultdict(float))
    cash = defaultdict(lambda: defaultdict(float))

    for line in io.open(d / ("%s.sector.ledger.jsonl" % arm), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        a = r["aid"]
        s = sec[a]
        s["total"] += float(r.get("total_spent") or 0)
        s["eligible"] += float(r.get("sangsaeng_eligible_offline_spent") or 0)
        s["online"] += float(r.get("online_spent") or 0)
        s["offline"] += float(r.get("offline_spent") or 0)
        # KDI 업종 — sub 우선, 없으면 l1. 접두 kdi: 는 업종 총액,
        # kdiE: 는 그 중 적립대상분(제외분은 둘의 차로 낸다).
        # 제외분에 BDC 구성비를 입힌 안분값 — K10 이 여기 걸려 있다.
        for k, v in (r.get("online_by_kdi_attributed") or {}).items():
            s["attr:" + k] += float(v or 0)
            s["_has_attr"] = 1.0
        for pre, fs, fl in (("kdi:", "by_sub", "by_l1"),
                            ("kdiE:", "eligible_by_sub", "eligible_by_l1")):
            if pre == "kdiE:":
                if fs not in r:
                    continue      # 구버전 원장 — 없으면 '출력부족' 으로 표에 적힌다
                s["_has_elig_sub"] = 1.0
            for k, v in (r.get(fs) or {}).items():
                kdi = submap.get(k)
                if kdi:
                    s[pre + kdi] += float(v or 0)
            for k, v in (r.get(fl) or {}).items():
                if k in submap:
                    continue
                kdi = l1map.get(k)
                if kdi:
                    s[pre + kdi] += float(v or 0)

    for line in io.open(d / ("%s.cashback.ledger.jsonl" % arm), encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        a = r["aid"]
        c = cash[a]
        c["cashback"] += float(r.get("cashback_accrued_won") or 0)
        c["cap"] = max(c.get("cap", 0.0), 1.0 if r.get("cashback_cap_reached") else 0.0)
        c["thr"] = float(r.get("threshold_won") or 0)
        c["anchor"] = max(c.get("anchor", 0.0), float(r.get("anchor_won") or 0))
        c["elig_cum"] = max(c.get("elig_cum", 0.0), float(r.get("eligible_cumulative") or 0))
    return sec, cash


def ratio_ci(off: list, on: list, n=4000, seed=20260928):
    """(비 %, lo, hi, 부호확실성) — 사람 단위로 **쌍을 함께** 재표집한다.

    부호확실성 = 재표집 중 점추정과 같은 부호가 나온 비율. 0.975 를 넘으면
    95% 구간이 0 을 지나지 않는다는 말과 같다. 구간 대신 이 값을 쓰면
    '얼마나 확실한가' 를 이분법이 아니라 정도로 읽을 수 있다.
    """
    if not off or sum(off) == 0:
        return (None, None, None, None)
    base = 100 * (sum(on) / sum(off) - 1)
    rnd = random.Random(seed)
    k, out = len(off), []
    for _ in range(n):
        a = b = 0.0
        for _ in range(k):
            i = rnd.randrange(k)
            a += off[i]
            b += on[i]
        if a > 0:
            out.append(100 * (b / a - 1))
    out.sort()
    if not out:
        return (base, None, None, None)
    same = sum(1 for v in out if (v > 0) == (base > 0)) / len(out)
    return (base, out[int(len(out) * 0.025)], out[int(len(out) * 0.975)], same)


def sign_split(off: list, on: list):
    up = sum(1 for a, b in zip(off, on) if b > a)
    dn = sum(1 for a, b in zip(off, on) if b < a)
    return up, dn, len(off) - up - dn


def needed_n(pct, lo, hi, n_now: int):
    """이 지표를 판정 가능하게 만들 표본 크기.

    구간 반폭은 √n 에 반비례한다. 지금 표본에서 잰 표준오차를 그 법칙으로
    늘려, |효과| > 1.96·표준오차 가 되는 n 을 돌려준다. **효과 크기가
    지금 값 그대로 유지된다는 가정**이고, 그 가정을 표에 함께 적는다.
    """
    if pct is None or lo is None or hi is None or pct == 0:
        return None
    se = (hi - lo) / 3.92
    if se <= 0:
        return n_now
    return max(n_now, int(math.ceil(n_now * (1.96 * se / abs(pct)) ** 2)))


def sign_test_p(up: int, dn: int) -> float | None:
    """쌍체부호검정 양측 p — 동점은 버린다(버린 수는 표에 함께 낸다).

    잡음 제거의 두 번째 자: 합계가 올라도 '몇 명이' 올랐는지가 갈린다.
    """
    m = up + dn
    if m == 0:
        return None
    k = min(up, dn)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2.0 ** m)
    return min(1.0, 2.0 * tail)


def series(sec, cash, aids, key):
    """지표 키 -> 에이전트별 값 목록 (aids 순서)."""
    if key == "total":
        return [sec[a]["total"] for a in aids]
    if key == "eligible":
        return [sec[a]["eligible"] for a in aids]
    if key == "excluded":
        return [sec[a]["total"] - sec[a]["eligible"] for a in aids]
    if key.startswith("exclkdi:"):
        # 그 업종의 적립 제외분 = (오프라인 업종총액 - 오프라인 적립분)
        #                        + (제외 채널에 BDC 구성비로 안분된 몫)
        nm = key.split(":", 1)[1]
        k, e, t = "kdi:" + nm, "kdiE:" + nm, "attr:" + nm
        return [sec[a][k] - sec[a][e] + sec[a][t] for a in aids]
    if key.startswith("kdi:"):
        return [sec[a][key] for a in aids]
    if key == "cashback":
        return [cash[a]["cashback"] for a in aids]
    return None


# 계약서의 sim_metric -> 원장 키 (팔 사이 비를 내는 것들)
METRIC_KEY = {
    "total_spend_arm_diff": "total",
    "excluded_spend_arm_diff": "excluded",
    "sector_arm_diff:가전·가구": "kdi:가전·가구",
    "sector_arm_diff:여행·레저": "kdi:여행·레저",
    "sector_arm_diff:유통": "kdi:유통",
    "sector_arm_diff:요식": "kdi:요식",
    "sector_arm_diff:학원": "kdi:학원",
    "sector_arm_diff:이·미용": "kdi:이·미용",
    "excluded_sector_arm_diff:유통": "exclkdi:유통",
}

# 팔 사이 비가 아닌 지표 — 각자 다른 셈이다
LEVEL_METRIC = {"cashback_per_capita", "cap_reach_rate", "fiscal_multiplier",
                "recipient_rate"}
RANK_METRIC = {"rank:eligible_vs_excluded": ("eligible", "excluded"),
               "rank:가전·가구_vs_이·미용": ("kdi:가전·가구", "kdi:이·미용")}
# 우리 설계·자료에 대응물이 없는 것 — '미구현' 이 아니라 이유를 적는다
NO_COUNTERPART = {
    "pre_policy_group_gap": "해당없음 — 같은 사람을 두 팔로 돌려 정책 전 격차가 0 이다(설계의 장점)",
    "heterogeneity_household_size": "자료없음 — 우리 모델은 개인 단위로 가구원수가 없다",
    "total_spend_arm_diff_nov": "미측정 — 11월을 돌리지 않았다(10월만)",
}


def level_value(metric, sec_off, sec_on, cash_on, aids):
    """(값, 표시단위, 실측과 맞댈 수 있는가)."""
    n = len(aids) or 1
    if metric == "cashback_per_capita":
        return sum(cash_on[a]["cashback"] for a in aids) / n, "원", True
    if metric == "cap_reach_rate":
        # 실측은 한도값(원)이고 우리는 도달률(%)이다 — 자가 다르니 단위를 함께 적는다.
        return 100 * sum(1 for a in aids if cash_on[a].get("cap")) / n, "% 도달", False
    if metric == "recipient_rate":
        return 100 * sum(1 for a in aids if cash_on[a]["cashback"] > 0) / n, "%", False
    if metric == "fiscal_multiplier":
        gain = sum(sec_on[a]["total"] for a in aids) - sum(sec_off[a]["total"] for a in aids)
        cb = sum(cash_on[a]["cashback"] for a in aids)
        return (100 * gain / cb if cb > 0 else None), "%", True
    return None, "", False


def multiplier_ci(sec_off, sec_on, cash_on, aids, n=4000, seed=20260928):
    """재정배수의 구간 — 사람 단위로 쌍을 함께 재표집한다."""
    rnd = random.Random(seed)
    k, out = len(aids), []
    for _ in range(n):
        g = cb = 0.0
        for _ in range(k):
            a = aids[rnd.randrange(k)]
            g += sec_on[a]["total"] - sec_off[a]["total"]
            cb += cash_on[a]["cashback"]
        if cb > 0:
            out.append(100 * g / cb)
    out.sort()
    if not out:
        return (None, None)
    return (out[int(len(out) * 0.025)], out[int(len(out) * 0.975)])


def gap_ci(sec_off, sec_on, cash_off, cash_on, aids, ka, kb, n=4000, seed=20260928):
    """(격차 %p, lo, hi, 부호확실성) — A 의 증가율 빼기 B 의 증가율.

    쌍(한 사람의 off/on/두 업종)을 **함께** 재표집한다. 그래서 '둘 중 어느 쪽이
    더 올랐나' 를 잡음과 가를 수 있다.
    """
    oa = series(sec_off, cash_off, aids, ka)
    na_ = series(sec_on, cash_on, aids, ka)
    ob = series(sec_off, cash_off, aids, kb)
    nb = series(sec_on, cash_on, aids, kb)
    if sum(oa) == 0 or sum(ob) == 0:
        return (None, None, None, None)
    base = 100 * (sum(na_) / sum(oa) - 1) - 100 * (sum(nb) / sum(ob) - 1)
    rnd, k, out = random.Random(seed), len(aids), []
    for _ in range(n):
        a1 = b1 = a2 = b2 = 0.0
        for _ in range(k):
            i = rnd.randrange(k)
            a1 += oa[i]; b1 += na_[i]; a2 += ob[i]; b2 += nb[i]
        if a1 > 0 and a2 > 0:
            out.append(100 * (b1 / a1 - 1) - 100 * (b2 / a2 - 1))
    out.sort()
    if not out:
        return (base, None, None, None)
    same = sum(1 for v in out if (v > 0) == (base > 0)) / len(out)
    return (base, out[int(len(out) * 0.025)], out[int(len(out) * 0.975)], same)


def arm_pct(sec_off, sec_on, sub, key):
    """부분집합 sub 안에서 off→on 증가율(%). off 합이 0 이면 None."""
    o = sum(series(sec_off, None, sub, key))
    n = sum(series(sec_on, None, sub, key))
    return (100 * (n / o - 1)) if o > 0 else None


def income_sector_grid(sec_off, sec_on, cash_off, aids):
    """K17 — 소비앵커 중위로 상/하를 갈라 업종별 증가율 격자를 낸다.

    앵커는 BDC 에서 온 정책 전 값이고 두 팔에서 같다. 정책 결과로 계층을
    가르면 순환이 되므로 **정책 전 값**만 쓴다.
    """
    anc = {a: cash_off[a]["anchor"] for a in aids}
    med = st.median(anc.values()) if anc else 0
    hi = [a for a in aids if anc[a] > med]
    lo = [a for a in aids if anc[a] <= med]
    cells = {}
    for nm in ("가전·가구", "학원", "요식", "유통"):
        cells[nm] = (arm_pct(sec_off, sec_on, hi, "kdi:" + nm),
                     arm_pct(sec_off, sec_on, lo, "kdi:" + nm))
    # 원문의 방향: 가전·가구·학원은 고소득이 크고, 요식·유통은 저소득이 크다
    want = {"가전·가구": "hi", "학원": "hi", "요식": "lo", "유통": "lo"}
    ok = tot = 0
    for nm, (h, l) in cells.items():
        if h is None or l is None or h == l:
            continue
        tot += 1
        ok += 1 if ((h > l) == (want[nm] == "hi")) else 0
    return cells, ok, tot, len(hi), len(lo), med


def region_evenness(sec_off, sec_on, aids, n=2000, seed=20260928):
    """K19 — 자치구별 증가율의 퍼짐이 **무작위로 갈랐을 때보다 큰가**.

    원문은 '고르게 나타났다' 고만 적었다. 그래서 수치를 맞대는 대신
    '고르지 않다고 말할 수 있는가' 를 본다. 같은 사람들을 구 크기만 유지한 채
    무작위로 다시 갈라 퍼짐의 영분포를 만들고, 실제 퍼짐을 그 안에 놓는다.
    """
    gu = defaultdict(list)
    for a in aids:
        parts = a.split("_")
        code = parts[1] if len(parts) > 1 else ""
        gu[code[:5] or "?"].append(a)
    sizes = [len(v) for v in gu.values()]
    obs = [arm_pct(sec_off, sec_on, v, "total") for v in gu.values()]
    obs = [x for x in obs if x is not None]
    if len(obs) < 2:
        return (len(gu), None, None, None)
    spread = st.pstdev(obs)
    rnd, null = random.Random(seed), []
    pool = list(aids)
    for _ in range(n):
        rnd.shuffle(pool)
        i, vals = 0, []
        for s in sizes:
            v = arm_pct(sec_off, sec_on, pool[i:i + s], "total")
            i += s
            if v is not None:
                vals.append(v)
        if len(vals) >= 2:
            null.append(st.pstdev(vals))
    null.sort()
    if not null:
        return (len(gu), spread, None, None)
    p = sum(1 for v in null if v >= spread) / len(null)
    return (len(gu), spread, null[int(len(null) * 0.95)], p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/experiments/p012_v53_pilot")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()
    d = ROOT / a.dir
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))

    sec_off, cash_off = load_arm(d, "off")
    sec_on, cash_on = load_arm(d, "on")
    aids = sorted(set(sec_off) & set(sec_on))

    print("# P012 — 두 팔 대조 (정책 없음 vs 있음, 같은 사람)")
    print()
    print("  원장 %s · 짝지은 시민 **%d명**" % (a.dir, len(aids)))
    w0, w1, wn = arm_window(d, "on")
    v0, v1, vn = arm_window(d, "off")
    print("  창   %s ~ %s · **%d일** (off 팔 %s ~ %s · %d일)" % (w0, w1, wn, v0, v1, vn))
    if (w0, w1) != (v0, v1):
        print("  ** 두 팔의 창이 다르다 — 이 대조는 무효다 **")
    if wn != 31:
        print("  ** 창이 31일이 아니다. 실측은 10월 한 달 누적이므로 비율 지표의 **크기**는")
        print("     맞댈 수 없다(파일럿 측정: 7일 +18.9% / 14일 +15.5% / 31일 +11.3%,")
        print("     실측 월 +11.25%). 방향·순위·재정배수(무차원)는 맞댈 수 있다. **")
    print()
    print("## 1. 비율 지표 — 두 팔의 차 (실측과 같은 자로 맞댄다)")
    print()
    print("  %-4s %-20s %8s %10s %17s %11s %7s %6s %6s %s"
          % ("지표", "이름", "실측", "시뮬", "95% 구간", "증:감:동", "부호확실",
             "쌍체p", "필요n", "판정"))
    print("-" * 134)

    rows = []
    hit = miss = undecid = na = lvl_in = lvl_out = 0
    has_elig = any(sec_off[a].get("_has_elig_sub") for a in aids)
    has_attr = any(sec_off[a].get("_has_attr") for a in aids)
    note_attributed: list[str] = []

    def verdict_of(iid, nm, tv, pct, lo, hi, conf, up, dn, eq, unit="%"):
        """한 줄 찍고 (판정, 행) 을 돌려준다 — 잡음 두 자를 함께 본다."""
        nonlocal hit, miss, undecid, na, lvl_in, lvl_out
        p = sign_test_p(up, dn)
        strong = (conf is not None and conf >= 0.975)
        paired = (p is not None and p < 0.05)
        same = (tv is not None) and ((pct > 0) == (tv > 0))
        # 자 3 — 실측이 시뮬의 95% 구간 안에 있나. 방향보다 강한 판정이다.
        inband = None
        if tv is not None and lo is not None and hi is not None:
            inband = (lo <= tv <= hi)
            if inband:
                lvl_in += 1
            else:
                lvl_out += 1
        if not strong:
            v, st_ = "판정불가 (부호확실 %.1f%% < 97.5%%)" % (100 * (conf or 0)), "판정불가"
            undecid += 1
        elif tv is None:
            v, st_ = "방향만 — 실측 수치 없음", "방향만"
            na += 1
        elif same:
            tag = "**방향+수준 일치**" if inband else "방향 일치·수준 벗어남"
            if not paired:
                tag += "(쌍체 약함)"
            v, st_ = "%s (오차 %.1f%%p)" % (tag, abs(pct - tv)), "일치"
            hit += 1
        else:
            v, st_ = "방향 불일치", "불일치"
            miss += 1
        nn = needed_n(pct, lo, hi, len(aids))
        print("  %-4s %-20s %8s %+9.2f%s %7.1f ~ %+7.1f %11s %6s %6s %6s %s"
              % (iid, nm, ("%+.2f" % tv) if isinstance(tv, (int, float)) else "없음",
                 pct, unit,
                 lo if lo is not None else float("nan"),
                 hi if hi is not None else float("nan"),
                 "%d:%d:%d" % (up, dn, eq),
                 ("%.1f%%" % (100 * conf)) if conf is not None else "-",
                 ("%.3f" % p) if p is not None else "-",
                 ("%d명" % nn) if nn is not None else "-", v))
        return {"id": iid, "name": nm, "truth": tv, "sim": pct, "ci": [lo, hi],
                "sign": [up, dn, eq], "sign_conf": conf, "sign_test_p": p,
                "truth_in_band": inband, "needed_n": nn, "status": st_}

    def skip(iid, nm, tv, why, status):
        nonlocal na
        na += 1
        print("  %-4s %-20s %8s %10s %17s %11s %7s %6s %6s %s"
              % (iid, nm, ("%+.0f" % tv) if isinstance(tv, (int, float)) else "-",
                 "-", "-", "-", "-", "-", "-", why))
        rows.append({"id": iid, "name": nm, "truth": tv, "status": status, "why": why})

    ratio_ids = [i for i in c["indicators"] if i.get("sim_metric") in METRIC_KEY]
    other = [i for i in c["indicators"] if i.get("sim_metric") not in METRIC_KEY]

    for ind in ratio_ids:
        iid, nm, tv = ind["id"], ind["name"][:20], ind.get("truth")
        key = METRIC_KEY[ind["sim_metric"]]
        if key.startswith("exclkdi:") and not has_elig:
            skip(iid, nm, tv, "출력부족 — 원장에 eligible_by_sub 가 없다(본런 원장에는 있다)",
                 "출력부족")
            continue
        if key.startswith("exclkdi:") and has_attr:
            note_attributed.append(iid)
        off = series(sec_off, cash_off, aids, key)
        on = series(sec_on, cash_on, aids, key)
        pct, lo, hi, conf = ratio_ci(off, on)
        if pct is None:
            skip(iid, nm, tv, "관측 0 — off 팔에 이 업종 지출이 없다", "관측없음")
            continue
        rows.append(verdict_of(iid, nm, tv, pct, lo, hi, conf, *sign_split(off, on)))

    print()
    print("## 2. 순위 지표 — 어느 쪽이 더 올랐나 (격차 %p, 실측 수치 대신 부호를 맞댄다)")
    print()
    for ind in other:
        m = ind.get("sim_metric")
        if m not in RANK_METRIC:
            continue
        iid, nm = ind["id"], ind["name"][:20]
        ka, kb = RANK_METRIC[m]
        g, lo, hi, conf = gap_ci(sec_off, sec_on, cash_off, cash_on, aids, ka, kb)
        tg = ind.get("truth_gap")
        if g is None:
            skip(iid, nm, tg, "관측 0 — 한쪽 업종에 off 팔 지출이 없다", "관측없음")
            continue
        oa, na_ = series(sec_off, cash_off, aids, ka), series(sec_on, cash_on, aids, ka)
        ob, nb = series(sec_off, cash_off, aids, kb), series(sec_on, cash_on, aids, kb)
        up = sum(1 for i in range(len(aids))
                 if (na_[i] - oa[i]) > (nb[i] - ob[i]))
        dn = sum(1 for i in range(len(aids))
                 if (na_[i] - oa[i]) < (nb[i] - ob[i]))
        r = verdict_of(iid, nm, tg, g, lo, hi, conf, up, dn,
                       len(aids) - up - dn, unit="%p")
        r["truth_gap"] = tg
        r["unit"] = "%p"
        rows.append(r)

    print()
    print("## 3. 수준 지표 — 비가 아니라 값 자체 (자가 달라 방향 셈에서 뺀다)")
    print()
    for ind in other:
        m = ind.get("sim_metric")
        if m not in LEVEL_METRIC:
            continue
        iid, nm, tv = ind["id"], ind["name"][:24], ind.get("truth")
        val, unit, cmpable = level_value(m, sec_off, sec_on, cash_on, aids)
        if val is None:
            skip(iid, nm, tv, "산출불가 — on 팔 캐시백이 0 이다", "관측없음")
            continue
        extra = ""
        if m == "fiscal_multiplier":
            mlo, mhi = multiplier_ci(sec_off, sec_on, cash_on, aids)
            extra = ("구간 %.0f ~ %.0f%%" % (mlo, mhi)) if mlo is not None else ""
            same = (tv is not None) and ((val > 100) == (tv > 100))
            decided = mlo is not None and not (mlo <= 100 <= mhi)
            inband = mlo is not None and mlo <= tv <= mhi
            if decided and same and inband:
                extra += "  → **방향+수준 일치** (실측 %.0f%%)" % tv
                hit += 1
                lvl_in += 1
            elif decided and same:
                extra += ("  → 방향만 일치 · **실측 %.0f%% 가 구간 밖** — 시뮬이 %.1f배 과대"
                          % (tv, val / tv))
                hit += 1
                lvl_out += 1
            elif decided:
                extra += "  → 배수 방향 불일치"
                miss += 1
            else:
                extra += "  → 판정불가 (구간이 100% 를 지난다)"
                undecid += 1
        elif m == "cashback_per_capita":
            half = (tv or 0) / 2.0
            extra = ("실측은 10·11월 **두 달 합**이다. 10월만 돌렸으니 균등가정 월평균 "
                     "{:,.0f}원과 본다 (배 {:.2f})".format(half, val / half if half else 0))
            na += 1
        elif m == "cap_reach_rate":
            extra = "원문에 도달률 수치가 없다(한도값 10만원만). 장치 확인용."
            na += 1
        else:
            extra = "전국 인원(%s명)과 직접 비교 못 한다. 분모를 우리가 붙인 참고값." % f"{int(tv):,}"
            na += 1
        shown = "{:,.0f}{}".format(val, unit) if unit == "원" else "{:.1f}{}".format(val, unit)
        tunit = ind.get("unit") or unit      # 실측의 자는 계약서가 들고 있다
        print("  %-4s %-24s 실측 %13s   시뮬 %13s  %s"
              % (iid, nm, "{:,}{}".format(int(tv), tunit) if isinstance(tv, (int, float)) else "없음",
                 shown, extra))
        rows.append({"id": iid, "name": nm, "truth": tv, "sim": val, "unit": unit,
                     "status": "수준", "note": extra})

    print()
    print("## 4. 이질성 지표 — 미시 반응이 계층·지역별로 갈라지나 (원문은 방향만 적었다)")
    print()
    for ind in other:
        m = ind.get("sim_metric")
        if m == "heterogeneity_income_sector":
            cells, ok, tot, nh, nl, med = income_sector_grid(sec_off, sec_on, cash_off, aids)
            print("  %-4s %s" % (ind["id"], ind["name"]))
            print("       소비앵커 중위 {:,.0f}원으로 갈랐다 — 상위 {}명 / 하위 {}명"
                  .format(med, nh, nl))
            print("       %-10s %12s %12s  %s" % ("업종", "고앵커", "저앵커", "원문이 기대한 쪽"))
            want = {"가전·가구": "고", "학원": "고", "요식": "저", "유통": "저"}
            for k2, (h, l) in cells.items():
                print("       %-10s %11s %12s  %s"
                      % (k2, ("%+.1f%%" % h) if h is not None else "관측0",
                         ("%+.1f%%" % l) if l is not None else "관측0", want[k2]))
            v = ("방향 %d/%d 칸 일치" % (ok, tot)) if tot else "판정불가 — 비교 가능한 칸이 없다"
            print("       → %s" % v)
            rows.append({"id": ind["id"], "name": ind["name"], "status": "이질성",
                         "cells": cells, "dir_ok": ok, "dir_total": tot})
            na += 1
        elif m == "heterogeneity_region":
            ngu, spread, p95, p = region_evenness(sec_off, sec_on, aids)
            print("  %-4s %s — 자치구 %d개" % (ind["id"], ind["name"], ngu))
            if spread is None:
                print("       판정불가 — 자치구가 2개 미만이다")
            else:
                even = (p is not None and p >= 0.05)
                print("       구 간 퍼짐 %.1f%%p · 무작위로 갈랐을 때 95분위 %s · p=%s"
                      % (spread, ("%.1f%%p" % p95) if p95 is not None else "-",
                         ("%.3f" % p) if p is not None else "-"))
                print("       → %s" % ("**고르다** (무작위 분할과 구분되지 않는다 = 원문과 같은 방향)"
                                       if even else "고르지 않다 (구별로 갈린다)"))
            rows.append({"id": ind["id"], "name": ind["name"], "status": "이질성",
                         "n_gu": ngu, "spread": spread, "null_p95": p95, "p": p})
            na += 1
        elif m in NO_COUNTERPART:
            print("  %-4s %-28s %s" % (ind["id"], ind["name"][:28], NO_COUNTERPART[m]))
            rows.append({"id": ind["id"], "name": ind["name"], "status": "해당없음",
                         "why": NO_COUNTERPART[m]})
            na += 1
        elif m not in RANK_METRIC and m not in LEVEL_METRIC:
            print("  %-4s %-28s 미구현 — %s" % (ind["id"], ind["name"][:28], m))
            rows.append({"id": ind["id"], "status": "미구현"})
            na += 1

    print()
    print("## 5. 잡음을 걷어낸 방향 — **두 자를 모두 통과한 것만 센다**")
    print()
    print("     자 1  부호확실성 ≥ 97.5%   (부트스트랩 재표집에서 부호가 뒤집히지 않는다)")
    print("     자 2  쌍체부호검정 p < 0.05 (사람 단위로도 한쪽으로 쏠린다 — 동점은 버렸다)")
    print("     자 3  실측이 시뮬 95% 구간 안 (방향보다 강하다 — 수준까지 맞았다는 말이다)")
    print()
    both = [r for r in rows if r.get("sign_conf") is not None
            and r["sign_conf"] >= 0.975 and (r.get("sign_test_p") or 1) < 0.05
            and r.get("status") in ("일치", "불일치")]
    one = [r for r in rows if r.get("status") in ("일치", "불일치") and r not in both]
    for r in both:
        print("     %-4s %-22s 시뮬 %+8.2f  실측 %+8.2f  확실 %.1f%%  p=%.3f  %s · 수준 %s"
              % (r["id"], r["name"], r["sim"], r["truth"], 100 * r["sign_conf"],
                 r["sign_test_p"], "방향 일치" if r["status"] == "일치" else "방향 불일치",
                 "안" if r.get("truth_in_band") else "밖"))
    if one:
        print()
        print("     아래는 자 하나만 통과 — 방향을 **주장하지 않는다**:")
        for r in one:
            print("     %-4s %-22s 확실 %.1f%%  p=%s"
                  % (r["id"], r["name"], 100 * (r.get("sign_conf") or 0),
                     ("%.3f" % r["sign_test_p"]) if r.get("sign_test_p") else "-"))

    print()
    if note_attributed:
        print()
        print("   * %s 의 제외분에는 BDC 업종 구성비를 **안분**한 몫이 들어 있다."
              % ", ".join(note_attributed))
        print("     안분값은 시뮬의 선택이 아니므로 증가율이 제외분 전체와 같다 —")
        print("     값은 맞댈 수 있지만 **독립된 정보는 아니다.**")

    print()
    print("## 6. 합계")
    dec = hit + miss
    print("   방향 일치      %2d" % hit)
    print("   방향 불일치    %2d" % miss)
    print("   판정 불가      %2d   (부호가 맞아도 세지 않는다)" % undecid)
    print("   대조 불가      %2d   (해당없음·자료없음·출력부족·참고)" % na)
    if dec:
        print()
        print("   판정 가능한 %d개 중 방향 적중 **%d/%d = %.0f%%**" % (dec, hit, dec, 100 * hit / dec))
    print("   그 중 자 1·2 를 모두 통과한 것 **%d개**" % len(both))
    print()
    print("   수준(자 3): 실측이 구간 **안 %d개** · **밖 %d개**  — 방향이 맞아도 크기가 틀릴 수 있다"
          % (lvl_in, lvl_out))
    print()
    print("   짝지은 시민이 %d명이다. 구간이 넓은 것은 규모 탓이고, 본런에서 좁아지면 넘어온다." % len(aids))

    if a.json_out:
        io.open(ROOT / a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps(
            {"n_agents": len(aids),
             # 계약서의 창이 아니라 **원장이 실제로 담은 창**을 적는다.
             "window": "%s ~ %s" % (w0, w1), "window_days": wn,
             "window_off": "%s ~ %s" % (v0, v1), "window_days_off": vn,
             "rows": rows,
             "tally": {"일치": hit, "불일치": miss, "판정불가": undecid, "대조불가": na,
                       "두자통과": len(both)}},
            ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
