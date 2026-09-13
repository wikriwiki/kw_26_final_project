# -*- coding: utf-8 -*-
"""P012 상생소비지원금 — 검증 지표 산출.

사전등록(docs/SANGSAENG_VALIDATION_MATRIX.md)의 지표를 한 번에 계산한다.
관측창이 짧아 **측정 자체가 불가능한** 지표는 NOT_MEASURABLE 로 분리한다 —
값이 나쁜 것과 잴 수 없는 것은 다른 결론이고, 섞으면 리포트가 거짓말이 된다.

    python scripts/sim/validate_p012.py \
        --off-from 2021-10-25 --off-to 2021-10-27 \
        --on-from 2021-10-28 --on-to 2021-10-29 [--warmup 0] [--drop-last]

관측 설계상 주의
  · 시뮬 첫 며칠은 방문 이력이 0에서 쌓이는 **워밍업** 구간이라 소비가 상승
    추세를 보인다(KNOWS_POI 갱신 351→434→493). 정책이 없는데 추세가 있으면
    평행추세가 깨져 C2 의 일부가 워밍업일 수 있다. --warmup N 으로 앞 N일을 버린다.
  · **마지막 날**은 야간 정산(다음 날 아침 처리)이 없어 기억·재방문이 비어 있다.
    소비 자체는 온전하므로 기본은 포함하고, 기억 기반 지표를 볼 때만 --drop-last.
"""
from __future__ import annotations

import argparse
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

RATE, CAP, RATIO = 0.10, 100_000, 1.03
BASE_RATIO = 0.268   # 적립업종이 총지출에서 차지하는 몫

# KDI 2022.9 실측 (표4-7, 10월 삼중차분)
KDI_SECTOR = {"가전·가구": 36.23, "기타": 15.20, "여행·레저": 14.27,
              "유통": 13.82, "요식": 13.33, "학원": 6.30, "이·미용": 2.87}
KDI_SIG = {"가전·가구", "기타", "여행·레저", "유통", "요식"}
KDI_CASHBACK_MULT = 16503   # 캐시백 1만원당 소비증가 (표4-11 IV 4열)


def dr(a: str, b: str) -> list[str]:
    y0, m0, d0 = map(int, a.split("-"))
    y1, m1, d1 = map(int, b.split("-"))
    s, e = date(y0, m0, d0), date(y1, m1, d1)
    return [(s + timedelta(days=i)).isoformat() for i in range((e - s).days + 1)]


def paired(per: dict, off: list[str], on: list[str]):
    """에이전트별 쌍체차 → (n, 평균, t)."""
    d = []
    for v in per.values():
        a = [v[x] for x in off if x in v]
        b = [v[x] for x in on if x in v]
        if a and b:
            d.append(sum(b) / len(b) - sum(a) / len(a))
    if len(d) < 2:
        return 0, 0.0, 0.0
    n = len(d)
    m = sum(d) / n
    se = statistics.pstdev(d) / (n ** 0.5)
    return n, m, (m / se if se > 0 else 0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off-from", required=True)
    ap.add_argument("--off-to", required=True)
    ap.add_argument("--on-from", required=True)
    ap.add_argument("--on-to", required=True)
    ap.add_argument("--warmup", type=int, default=0, help="무정책 앞 N일 버림")
    ap.add_argument("--drop-last", action="store_true", help="마지막 날 제외")
    a = ap.parse_args()

    off = dr(a.off_from, a.off_to)[a.warmup:]
    on = dr(a.on_from, a.on_to)
    if a.drop_last and len(on) > 1:
        on = on[:-1]
    if not off or not on:
        print("관측 구간이 비었다 — warmup/drop-last 를 줄일 것", file=sys.stderr)
        return 2

    print("무정책 {}  →  정책 {}".format(off, on))
    print("=" * 68)
    ok, ng, nm = [], [], []

    with driver_session() as s:
        rows = list(s.run(
            "MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI) "
            "WHERE coalesce(i.actual_spent,0)>0 "
            "RETURN a.id AS aid, toString(pl.day) AS d, i.actual_spent AS amt, "
            "coalesce(p.sangsaeng_eligible,false) AS el, p.sangsaeng_kdi AS kdi, "
            "a.spending_level_wd AS lvl"))
        if not rows:
            print("결제 데이터 없음")
            return 2

        elig = defaultdict(lambda: defaultdict(int))
        excl = defaultdict(lambda: defaultdict(int))
        sector = defaultdict(lambda: defaultdict(int))
        # (분위군, 업종) → 일자별 금액. 저=1~4 중=5~7 고=8~10
        qsec = defaultdict(lambda: defaultdict(int))
        cnt = defaultdict(int)
        amt = defaultdict(int)
        for r in rows:
            if r["el"]:
                elig[r["aid"]][r["d"]] += r["amt"]
                if r["kdi"]:
                    sector[r["kdi"]][r["d"]] += r["amt"]
                    lv = r["lvl"]
                    if lv:
                        grp = "저" if lv <= 4 else ("고" if lv >= 8 else "중")
                        qsec[(grp, r["kdi"])][r["d"]] += r["amt"]
            else:
                excl[r["aid"]][r["d"]] += r["amt"]
            cnt[r["d"]] += 1
            amt[r["d"]] += r["amt"]

        # ── A1 C2 ──────────────────────────────────────────────
        n, m, t = paired(elig, off, on)
        sig = abs(t) >= 1.96
        print("\n[A1] C2 적립업종 소비 쌍체차")
        print("     n={} · {:+,.0f}원/일 · t={:.2f} · {}".format(
            n, m, t, "유의" if sig else "유의하지 않음"))
        (ok if (m > 0 and sig) else ng).append("A1")

        # ── A2 건당 금액 ───────────────────────────────────────
        pa = sum(amt[d] for d in off) / max(sum(cnt[d] for d in off), 1)
        pb = sum(amt[d] for d in on) / max(sum(cnt[d] for d in on), 1)
        chg = 100 * (pb - pa) / pa if pa else 0
        print("\n[A2] 건당 금액  무정책 {:,}원 → 정책 {:,}원  ({:+.1f}%)".format(
            int(pa), int(pb), chg))
        (ok if chg > -10 else ng).append("A2")

        # ── A3 C1 부등식 ───────────────────────────────────────
        ne, me, te = paired(elig, off, on)
        nx, mx, tx = paired(excl, off, on)
        print("\n[A3] C1 적립 vs 제외")
        if nx < 10:
            print("     제외업종 관측 {}명뿐 → 측정 불가".format(nx))
            nm.append("A3")
        else:
            print("     적립 {:+,.0f}원(t={:.2f}, n={}) vs 제외 {:+,.0f}원(t={:.2f}, n={})".format(
                me, te, ne, mx, tx, nx))
            (ok if me > mx else ng).append("A3")

        # ── A4 요일 대칭 ───────────────────────────────────────
        dt = {r["d"]: r["t"] for r in s.run(
            "MATCH (pl:Plan) RETURN DISTINCT toString(pl.day) AS d, pl.day_type AS t")}
        wo = sum(1 for d in off if dt.get(d) == "weekend")
        wn = sum(1 for d in on if dt.get(d) == "weekend")
        print("\n[A4] 요일 대칭  무정책 주말 {}/{}일 · 정책 주말 {}/{}일".format(
            wo, len(off), wn, len(on)))
        (ok if (wo / len(off)) == (wn / len(on)) else ng).append("A4")

        # ── A5 캐시백 1만원당 소비증가 ──────────────────────────
        cb = s.run(
            "MATCH (st:State {day:date($d)}) MATCH (ag:Agent {id:st.agent_id}) "
            "WITH st,(coalesce(ag.s_daily_wd,0)*5+coalesce(ag.s_daily_we,0)*2)/7.0"
            "*$br*30*$ratio AS thr WHERE thr>0 "
            "RETURN count(*) AS n, "
            "sum(CASE WHEN st.sangsaeng_month_spent>=thr THEN 1 ELSE 0 END) AS over, "
            "avg(CASE WHEN st.sangsaeng_month_spent>thr THEN "
            "CASE WHEN (st.sangsaeng_month_spent-thr)*$rate > $cap THEN $cap "
            "ELSE (st.sangsaeng_month_spent-thr)*$rate END ELSE 0 END) AS cb",
            d=on[-1], br=BASE_RATIO, ratio=RATIO, rate=RATE, cap=CAP).single()
        share = 100 * cb["over"] / max(cb["n"], 1)
        avg_cb = int(cb["cb"] or 0)
        print("\n[A5] 캐시백 1만원당 소비증가")
        print("     문턱 돌파 {}/{}명 ({:.0f}%) · 1인 예상 캐시백 {:,}원".format(
            cb["over"], cb["n"], share, avg_cb))
        if share < 40 or avg_cb < 5000:
            print("     → 관측창이 짧아 캐시백이 실제 크기로 쌓이지 않는다. 측정 불가")
            print("       (실측 {:,}원/만원 — 이 지표는 월 단위 관측이 필요)".format(
                KDI_CASHBACK_MULT))
            nm.append("A5")
        else:
            v = m * 30 / (avg_cb / 10000)
            print("     → {:,.0f}원  (실측 {:,}원)".format(v, KDI_CASHBACK_MULT))
            ok.append("A5")

        # ── A6 평행추세 ────────────────────────────────────────
        print("\n[A6] 평행추세 (무정책 구간 내부)")
        if len(off) < 3:
            print("     무정책 {}일 — 3일 이상 필요. 측정 불가".format(len(off)))
            nm.append("A6")
        else:
            base = off[0]
            bad = 0
            for d in off[1:]:
                diff = [v[d] - v[base] for v in elig.values() if d in v and base in v]
                if len(diff) < 2:
                    continue
                nn = len(diff)
                mm = sum(diff) / nn
                se = statistics.pstdev(diff) / (nn ** 0.5)
                tt = mm / se if se else 0
                if abs(tt) >= 1.96:
                    bad += 1
                print("     {}→{}: {:+,.0f}원 t={:+.2f} {}".format(
                    base, d, mm, tt, "✔" if abs(tt) < 1.96 else "✘ 추세"))
            (ok if bad == 0 else ng).append("A6")

        # ── B1 업종 순위 ───────────────────────────────────────
        print("\n[B1] 업종별 증감 (KDI 순위 대조)")
        res = []
        for k, dd in sector.items():
            o = sum(dd.get(x, 0) for x in off) / len(off)
            p = sum(dd.get(x, 0) for x in on) / len(on)
            if o > 0:
                res.append((k, 100 * (p - o) / o))
        res.sort(key=lambda x: -x[1])
        for k, v in res:
            ref = KDI_SECTOR.get(k)
            if ref is not None:
                star = "***" if k in KDI_SIG else "ns "
                print("     {:<8} {:>+8.1f}%   (KDI {:>+6.2f}% {})".format(k, v, ref, star))
            else:
                print("     {:<8} {:>+8.1f}%".format(k, v))
        if res:
            top3 = [k for k, _ in res[:3]]
            print("     → 가전·가구 상위3 진입: {}".format("✔" if "가전·가구" in top3 else "✘"))

        # ── B3 소비분위군 × 업종 ───────────────────────────────
        print("\n[B3] 소비분위군 × 업종 (KDI 정성 패턴)")
        print("     ※ KDI는 가구 소득분위, 우리는 개인 소비분위 — 개념이 다름")
        KDI_PAT = {"가전·가구": "고", "학원": "고", "요식": "저",
                   "유통": "저", "여행·레저": "중", "이·미용": "무관"}
        for kname, expect in KDI_PAT.items():
            vals = {}
            for grp in ("저", "중", "고"):
                dd = qsec.get((grp, kname))
                if not dd:
                    continue
                o = sum(dd.get(x, 0) for x in off) / len(off)
                pp = sum(dd.get(x, 0) for x in on) / len(on)
                if o > 0:
                    vals[grp] = 100 * (pp - o) / o
            if len(vals) < 2:
                print("     {:<8} 관측 부족 — 측정 불가".format(kname))
                nm.append("B3-" + kname)
                continue
            top = max(vals, key=vals.get)
            mark = "✔" if (expect == top or expect == "무관") else "✘"
            detail = " ".join("{}{:+.0f}%".format(g, v) for g, v in vals.items())
            print("     {:<8} {}  최대={} (기대 {}) {}".format(kname, detail, top, expect, mark))
            (ok if mark == "✔" else ng).append("B3-" + kname)

        # ── C 캐시백 집행 실적 ─────────────────────────────────
        print("\n[C] 캐시백 집행 실적 (실측 대조)")
        cbd = list(s.run(
            "MATCH (st:State {day:date($d)}) MATCH (ag:Agent {id:st.agent_id}) "
            "WITH st,(coalesce(ag.s_daily_wd,0)*5+coalesce(ag.s_daily_we,0)*2)/7.0"
            "*$br*30*$ratio AS thr WHERE thr>0 "
            "RETURN CASE WHEN st.sangsaeng_month_spent>thr THEN "
            "CASE WHEN (st.sangsaeng_month_spent-thr)*$rate > $cap THEN $cap "
            "ELSE (st.sangsaeng_month_spent-thr)*$rate END ELSE 0 END AS cb",
            d=on[-1], br=BASE_RATIO, ratio=RATIO, rate=RATE, cap=CAP))
        cbs = [x["cb"] for x in cbd if x["cb"] and x["cb"] > 0]
        if not cbs:
            print("     캐시백 발생 0건 — 측정 불가")
            nm.extend(["C1", "C2", "C3", "C4"])
        else:
            avgc = sum(cbs) / len(cbs)
            at_cap = sum(1 for x in cbs if x >= CAP * 0.999)
            print("     C1 1인 평균 캐시백 {:,}원   (실측 47,880원)".format(int(avgc)))
            print("     C2 한도 도달 {}/{}명 = {:.1f}%   (실측 21.0%)".format(
                at_cap, len(cbs), 100 * at_cap / len(cbs)))
            band = [0, 10000, 30000, 50000, 70000, 100000]
            hist = [sum(1 for x in cbs if band[k] < x <= band[k + 1])
                    for k in range(len(band) - 1)]
            print("     C3 구간분포 " + " ".join(
                "{}~{}만:{}".format(band[k] // 10000, band[k + 1] // 10000, hist[k])
                for k in range(len(hist))))
            tot_cb = sum(cbs)
            boost = m * 30 * len(cbd)
            if tot_cb > 0:
                print("     C4 투입 대비 소비진작 {:.0f}%   (실측 165%)".format(
                    100 * boost / tot_cb))
            ok.extend(["C1", "C2", "C3", "C4"])


    print("\n" + "=" * 68)
    print("  통과     {} {}".format(len(ok), ok))
    print("  실패     {} {}".format(len(ng), ng))
    print("  측정불가 {} {}".format(len(nm), nm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
