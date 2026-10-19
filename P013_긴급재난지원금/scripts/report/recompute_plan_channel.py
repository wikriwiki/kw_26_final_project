"""**원장 위에서** 회계 고침을 다시 셈한다 — GPU 를 한 시간도 더 쓰지 않는다.

    python scripts/report/recompute_plan_channel.py --metrics /data/p012_28d \
        --baseline 2021-10-04,2021-10-05,2021-10-06,2021-10-07,2021-10-08 \
        --off 2021-10-13,2021-10-14 --on 2021-10-27,2021-10-28

## 무엇을 다시 세는가

엔진은 총액을 `max(앵커, 계획)` 으로 정한다. 정책 반응은 **계획액에** 실려 있는데
(P012 +9.76% p=0.0015 · P013 +6.36% p=0.0000) 계획이 앵커를 넘는 날이 4분의 1뿐이라
반응의 대부분이 앵커에 먹힌다. 고침은 그 자리를 이렇게 바꾼다.

    현행   총액 = max(앵커, 계획)
    고침   총액 = 앵커 x clamp(계획 / 그 사람의 평소 계획, LO, HI) x SCALE

계층·수준은 앵커가 잡고, **그 사람 자신의 평소 대비 오늘의 변동**만 총액에 실린다.

## 왜 다시 돌리지 않고 다시 세도 되는가

고침은 LLM 출력을 바꾸지 않는다. `cm_anchor_total` 과 `cm_planned_total` 은 이미
원장에 있고, 둘 다 고침 이전 단계에서 정해진다. 그러므로 날마다의 곱수

    m = 앵커 x clamp(계획/기준선) x SCALE / max(앵커, 계획)

를 기록된 총액에 곱하면 된다. **온라인 분리·바스켓 지수는 손대지 않는다** — 그
둘은 `m` 의 분자와 분모에 똑같이 들어가 약분된다.

**되돌릴 수 없는 것 하나**: 총액이 달라졌다면 잔고가 달라지고, 달라진 잔고를 보고
에이전트가 다음 날 다르게 계획했을 수 있다. 그 되먹임은 다시 셈으로 복원되지
않는다. v44 에서 모델이 문턱까지의 거리에 반응하지 않는 것이 확인됐으므로 그
통로는 약하다고 보지만, **약하다는 것이지 없다는 것이 아니다.** 보고에 적는다.

## 기준선은 비교 창과 같은 요일종류에서 와야 한다

앵커와 계획이 요일에 따라 **반대로** 움직인다(금 계획/앵커 0.500 · 토 1.568,
같은 439/500 명이 양쪽으로 갈린다). 섞인 요일로 기준선을 세우면 `계획/기준선` 이
클램프에 걸려 신호가 눌린다. `--baseline-mode daytype` 이 평일·주말을 갈라 세운다.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import random
import statistics as st
from datetime import date


def _is_weekend(d: str) -> bool:
    return date.fromisoformat(d).weekday() >= 5


def load_ledger(root: str, arm: str = "") -> dict:
    """{aid: {날짜: 행}} — status=ok 만. arm 이 주어지면 그 경로만 본다."""
    pat = os.path.join(root, "**", "day_*.jsonl")
    per: dict = {}
    for f in sorted(glob.glob(pat, recursive=True)):
        if arm and arm not in f:
            continue
        d = os.path.basename(f)[4:14]
        for line in io.open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("status") != "ok":
                continue
            per.setdefault(r["aid"], {})[d] = r
    return per


def build_baseline(per: dict, days: list, mode: str = "abs") -> dict:
    """{aid: 기준선} 또는 daytype 이면 {aid: {"wd": x, "we": y}}.

    중앙값을 쓴다 — 평균은 하루의 큰 계획에 끌려간다.
    """
    out: dict = {}
    for aid, byday in per.items():
        if mode == "daytype":
            wd = [byday[d]["cm_planned_total"] for d in days
                  if d in byday and not _is_weekend(d)]
            we = [byday[d]["cm_planned_total"] for d in days
                  if d in byday and _is_weekend(d)]
            cell = {}
            if wd:
                cell["wd"] = st.median(wd)
            if we:
                cell["we"] = st.median(we)
            if cell:
                out[aid] = cell
        else:
            v = [byday[d]["cm_planned_total"] for d in days if d in byday]
            if v:
                out[aid] = st.median(v)
    return out


def _base_for(base, aid: str, day: str, mode: str):
    b = base.get(aid)
    if b is None:
        return None
    if mode != "daytype":
        return b or None
    # 그 요일종류의 기준선이 없으면 다른 쪽으로 대신하지 않는다 — 그러면 요일
    # 효과가 그대로 곱수로 들어간다. 차라리 그 칸을 빼고 몇 칸인지 밝힌다.
    return b.get("we" if _is_weekend(day) else "wd") or None


def multiplier(row: dict, base_val, lo: float, hi: float, scale: float):
    """(곱수, 클램프에 걸렸나). 되살릴 수 없으면 (1.0, None)."""
    anchor = row.get("cm_anchor_total") or 0
    plan = row.get("cm_planned_total") or 0
    cur = max(anchor, plan)
    if not base_val or anchor <= 0 or cur <= 0:
        return 1.0, None
    raw = plan / base_val
    clamped = max(lo, min(hi, raw))
    return anchor * clamped * scale / cur, (raw != clamped)


# 어느 자로 재는가 — **이것을 말하지 않으면 수가 뜻을 갖지 못한다.**
#
# 엔진은 총액을 정한 뒤 `ELIGIBLE_SHARE_SEOUL`(0.2535) 로 갈라 '온라인' 몫을
# 떼어낸다. 그래서 같은 런에서도 어느 필드를 읽느냐에 따라 수준이 약 4배,
# 때로는 **부호까지** 달라진다(P013 에서 현행이 −1.03% / +0.32% 로 갈렸다).
# 셋을 함께 찍는다. 하나만 고르면 다음 사람이 옛 수와 비교하다 속는다.
RULERS = {
    "incl_online": ("cm_today_total_incl_online",
                    "가게+배송 전부 — 실측 '카드 소비'에 맞는 자. **기본으로 이것을 읽는다**"),
    "personal": ("cm_personal_total",
                 "분리 뒤 가게 몫(계획 기준)"),
    "instore": ("cm_today_total",
                "분리 뒤 가게 몫(실제 결제) — 적립업종 실적이 세는 자"),
}


def window_totals(per: dict, aids: list, days: list, base, mode, lo, hi, scale,
                  field: str = "cm_today_total_incl_online"):
    """각 사람의 창 평균 총액을 현행·고침 두 벌로. 클램프·결손도 센다."""
    cur, fix, clamp_hits, n_cells, missing = [], [], 0, 0, 0
    for aid in aids:
        byday = per[aid]
        c, f = [], []
        for d in days:
            r = byday.get(d)
            if not r:
                continue
            t = r.get(field) or 0
            m, hit = multiplier(r, _base_for(base, aid, d, mode), lo, hi, scale)
            c.append(t)
            f.append(t * m)
            n_cells += 1
            if hit is None:
                missing += 1
            elif hit:
                clamp_hits += 1
        if c:
            cur.append(st.mean(c))
            fix.append(st.mean(f))
    return cur, fix, clamp_hits, n_cells, missing


def broke_rate(per: dict, aids: list, day: str) -> float:
    """그 날 잔고가 0 인 사람의 비율.

    소득 주입이 없으면 긴 창에서 지갑이 마르고, **파산이 정책 효과로 읽힌다.**
    실제로 28일 런 하나에서 마지막 날 67%가 0원이 되어 총소비가 −57.70% 로
    나왔다(평균). 같은 날 중앙값은 오르고 있었다 — 분포 한쪽이 0 으로 쌓인 것이다.
    `experiments/error_budget/wallet_audit.md`
    """
    bal = [per[a][day].get("balance") for a in aids if day in per[a]]
    bal = [b for b in bal if isinstance(b, (int, float))]
    return (sum(1 for b in bal if b <= 0) / len(bal)) if bal else 0.0


def solve_scale(per: dict, aids: list, off_days: list, base, mode, lo, hi,
                field: str = "cm_today_total_incl_online") -> float:
    """**무정책 창의 수준이 보존되도록** SCALE 을 푼다 — 정답지를 보지 않는다.

    고침이 수준을 통째로 밀어 올리면 "정책 효과" 와 구별되지 않는다. 그래서 정책이
    꺼진 창에서 고침 총액의 평균이 현행과 같아지도록 SCALE 하나를 정한다. 조건이
    하나고 미지수가 하나라 값이 **유일하게** 결정된다 — 고를 여지가 없다.

    이러면 OFF 는 현행과 고침이 같은 수에 서고, 움직이는 것은 ON 뿐이다. 그것이
    노리는 바다: 고침은 수준을 옮기는 장치가 아니라 **반응을 통과시키는** 장치다.
    """
    cur, at_one, _, _, _ = window_totals(per, aids, off_days, base, mode, lo, hi,
                                         1.0, field)
    a, b = st.mean(cur), st.mean(at_one)
    return a / b if b else 1.0


def daily_means(per: dict, aids: list, days: list, field: str) -> list:
    """정책 없는 날들의 하루 평균 — 표류를 재는 재료."""
    out = []
    for d in days:
        v = [per[a][d].get(field) or 0 for a in aids if d in per[a]]
        if v:
            out.append((d, st.mean(v)))
    return out


def _slope(ys: list) -> float:
    xs = list(range(len(ys)))
    mx, my = st.mean(xs), st.mean(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return (sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den) if den else 0.0


def drift_per_day(points: list) -> tuple:
    """(하루당 상대 표류, R², 앞뒤 절반의 기울기). **직선이 아니면 쓰면 안 된다.**

    OFF 와 ON 사이가 14일이면 그 사이에 정책이 아닌 이유로도 수준이 움직인다 —
    지갑이 줄고, `stage2_poi.py` 가 잔액을 프롬프트에 그대로 넣으므로 그것이 다시
    소비 판단에 들어간다. 정책 전 날들에서 기울기를 재면 그 몫을 덜어낼 수 있다.

    **그런데 시뮬의 앞 며칠은 표류가 아니라 예열이다.** P013 에서 첫 3일이 +7.1%
    오르고 그 뒤 8일은 +0.2% 로 평평했는데, 전 구간에 직선을 맞추면 하루 +1.285%
    (7일에 +9.35%) 가 나온다. 그것을 덜면 고침의 +8.99% 가 −0.36% 로 지워진다.
    **없는 표류로 있는 효과를 지우는 것이다.**

    그래서 값만 내지 않고 **직선인지**를 함께 낸다. 앞뒤 절반의 기울기가 크게
    다르면 예열이 섞인 것이고, 호출한 쪽이 적용을 거부해야 한다.
    """
    if len(points) < 4:
        return 0.0, 0.0, (0.0, 0.0)
    ys = [y for _, y in points]
    my = st.mean(ys)
    if my <= 0:
        return 0.0, 0.0, (0.0, 0.0)
    b = _slope(ys)
    xs = list(range(len(ys)))
    mx = st.mean(xs)
    pred = [my + b * (x - mx) for x in xs]
    ss_res = sum((y - p) ** 2 for y, p in zip(ys, pred))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    h = len(ys) // 2
    return b / my, r2, (_slope(ys[:h]) / my, _slope(ys[h:]) / my)


def boot_pct(off: list, on: list, n: int, seed: int = 20260924):
    """쌍이 아니다 — 같은 사람의 두 창이므로 **사람 단위로** 함께 재표집한다."""
    rnd = random.Random(seed)
    k = len(off)
    out = []
    for _ in range(n):
        idx = [rnd.randrange(k) for _ in range(k)]
        a = st.mean([off[i] for i in idx])
        b = st.mean([on[i] for i in idx])
        if a > 0:
            out.append(100 * (b / a - 1))
    out.sort()
    if not out:
        return (float("nan"), float("nan"))
    return (out[int(len(out) * 0.025)], out[int(len(out) * 0.975)])


def sign_test(off: list, on: list) -> tuple:
    up = sum(1 for a, b in zip(off, on) if b > a)
    dn = sum(1 for a, b in zip(off, on) if b < a)
    return up, dn, len(off) - up - dn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--arm", default="")
    ap.add_argument("--baseline", required=True, help="쉼표로 구분한 날짜")
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    ap.add_argument("--baseline-mode", default="abs", choices=("abs", "daytype"))
    ap.add_argument("--clamp-lo", type=float, default=0.5)
    ap.add_argument("--clamp-hi", type=float, default=2.0)
    ap.add_argument("--scale", default="auto",
                    help="수 또는 auto(무정책 창의 수준이 보존되도록 푼다)")
    ap.add_argument("--placebo-on", default="",
                    help="**정책 전** 가짜 ON 창. OFF→여기 가 곧 정책 아닌 몫이다. "
                         "ON 과 같은 요일종류로 고른다")
    ap.add_argument("--trend-days", default="",
                    help="정책이 꺼진 날들. 여기서 하루당 표류를 재어 OFF→ON 에서 덜어낸다")
    ap.add_argument("--max-broke", type=float, default=0.10,
                    help="ON 창 마지막 날 잔고 0 비율의 한계. 넘으면 거부한다")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    def split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    b_days, off_days, on_days = split(a.baseline), split(a.off), split(a.on)
    trend_days = split(a.trend_days)
    placebo_on = split(a.placebo_on)

    if set(placebo_on) & set(on_days):
        print("거부: 가짜 ON 창에 진짜 ON 날짜가 섞였다 — %s"
              % sorted(set(placebo_on) & set(on_days)))
        print("  가짜 ON 은 **정책이 꺼진 날**이어야 한다. 아니면 정책 효과를 뺀다.")
        return 2
    if placebo_on and set(placebo_on) & set(off_days):
        print("거부: 가짜 ON 창이 OFF 창과 겹친다 — %s"
              % sorted(set(placebo_on) & set(off_days)))
        return 2

    if set(trend_days) & set(on_days):
        print("거부: 표류를 재는 날에 정책 창이 섞였다 — %s"
              % sorted(set(trend_days) & set(on_days)))
        print("  표류는 **정책이 꺼진 날로만** 잰다. 아니면 정책 효과를 표류로 덜어낸다.")
        return 2

    overlap = (set(b_days) & set(off_days)) | (set(b_days) & set(on_days))
    if overlap:
        print("거부: 기준선 날짜가 비교 창과 겹친다 — %s" % sorted(overlap))
        print("  겹치면 그 창의 계획/기준선이 1 로 못 박혀 그 팔만 안 움직인다.")
        return 2

    per = load_ledger(a.metrics, a.arm)
    need = set(b_days) | set(off_days) | set(on_days) | set(placebo_on)
    aids = sorted(x for x, v in per.items() if need <= set(v))
    base = build_baseline(per, b_days, a.baseline_mode)

    print("# 회계 고침을 원장 위에서 다시 셈한다")
    print()
    print("  원장        %s%s" % (a.metrics, (" [arm %s]" % a.arm) if a.arm else ""))
    print("  기준선      %s  (%s)" % (",".join(b_days), a.baseline_mode))
    print("  OFF / ON    %s  /  %s" % (",".join(off_days), ",".join(on_days)))
    print("  모든 날짜를 가진 에이전트  %d / %d" % (len(aids), len(per)))
    if not aids:
        print("  ** 쓸 에이전트가 없다 — 날짜를 확인하라")
        return 1

    # 지갑 관문 — **수를 내기 전에** 막는다. 사람이 기억할 일로 두면 잊는다.
    worst_day = max(on_days)
    broke = broke_rate(per, aids, worst_day)
    print("  %s 잔고 0 인 사람  %.1f%%" % (worst_day, 100 * broke))
    if broke > a.max_broke:
        print()
        print("거부: 지갑이 말랐다 (%.1f%% > 한계 %.1f%%)." % (100 * broke, 100 * a.max_broke))
        print("  이 원장에서 총소비를 읽으면 **파산을 정책 효과로 읽는다.**")
        print("  실제로 28일 런 하나가 67% 파산으로 −57.70% 를 냈다(평균). 같은 날")
        print("  중앙값은 오르고 있었다 — experiments/error_budget/wallet_audit.md")
        print("  고치는 법: EXP_DAILY_INCOME=anchor · EXP_BALANCE_DAYS 를 올려 다시 건다.")
        print("  (그래도 읽어야 하면 --max-broke 로 한계를 올려라. 보고에 적을 것.)")
        return 3

    def pct(x, y):
        return 100 * (st.mean(y) / st.mean(x) - 1) if st.mean(x) else float("nan")

    out_rulers, tot, hits, miss = {}, 0, 0, 0
    for key, (field, why) in RULERS.items():
        if str(a.scale).strip().lower() == "auto":
            scale = solve_scale(per, aids, off_days, base, a.baseline_mode,
                                a.clamp_lo, a.clamp_hi, field)
            how = "auto"
        else:
            scale, how = float(a.scale), "고정"
        args = (base, a.baseline_mode, a.clamp_lo, a.clamp_hi, scale, field)
        c_off, f_off, h1, n1, m1 = window_totals(per, aids, off_days, *args)
        c_on, f_on, h2, n2, m2 = window_totals(per, aids, on_days, *args)
        tot, hits, miss = n1 + n2, h1 + h2, m1 + m2

        print()
        print("## 자: %s — %s" % (key, why))
        print("   SCALE %.5f (%s)   clamp [%.2f, %.2f]"
              % (scale, how, a.clamp_lo, a.clamp_hi))
        print("   %-6s %11s %11s %9s %19s   %s"
              % ("", "OFF", "ON", "변화", "95% 구간", "증가:감소:동점"))
        rows = {}
        for nm, o, n in (("현행", c_off, c_on), ("고침", f_off, f_on)):
            p, ci, s = pct(o, n), boot_pct(o, n, a.boot), sign_test(o, n)
            print("   %-6s %11.0f %11.0f %+8.2f%% %8.2f ~ %+8.2f   %d:%d:%d"
                  % (nm, st.mean(o), st.mean(n), p, ci[0], ci[1], *s))
            rows[nm] = {"off": st.mean(o), "on": st.mean(n), "pct": p,
                        "ci": list(ci), "sign": list(s)}

        if placebo_on:
            # **정책이 아닌 몫.** OFF→가짜ON 은 같은 런·같은 사람·정책 없는 두 창이다.
            # 직선을 가정하지 않으므로 예열이 굽어 있어도 쓸 수 있다 — P013 에서
            # 인용하던 "+6.36% 정책 반응" 의 +3.70%p 가 이렇게 드러났다.
            # **두 행 모두에 댄다.** 고침은 계획액을 총액으로 끌어오므로 계획 쪽
            # 드리프트가 고침에만 실린다 — 현행만 보면 앵커가 가려서 안 보인다.
            # P013 이 그 자리였다: 현행의 드리프트는 +0.58% 인데 고침은 훨씬 컸다.
            p_off, p_fix, _, _, _ = window_totals(per, aids, placebo_on, *args)
            print("   가짜ON  %s -> %s  (정책 아닌 몫 · 정책이 꺼진 두 창)"
                  % (",".join(off_days), ",".join(placebo_on)))
            for nm, series in (("현행", p_off), ("고침", p_fix)):
                # `base` 로 쓰지 않는다 — 바깥의 기준선 dict 을 덮어쓴다(그렇게 해서
                # 다음 자 반복에서 'list' has no attribute 'get' 로 터졌다).
                ref = c_off if nm == "현행" else f_off
                pp = pct(ref, series)
                s = sign_test(ref, series)
                real = rows[nm]["pct"]
                net = (100 * ((1 + real / 100) / (1 + pp / 100) - 1)
                       if pp > -100 else float("nan"))
                flag = ""
                if abs(real) < 1.0:
                    flag = "  <- 진짜 효과가 0 근처라 뺄 것이 없다"
                elif abs(pp) > abs(real) * 0.5:
                    flag = "  <- **절반을 넘는다. 이 비교를 믿지 마라**"
                print("            %-4s 드리프트 %+7.2f%%  %d:%d:%d   덜면 **%+.2f%%**%s"
                      % (nm, pp, *s, net, flag))
                rows[nm]["placebo_pct"] = pp
                rows[nm]["net_pct"] = net

        if trend_days:
            pts = daily_means(per, aids, trend_days, field)
            g, r2, (h1_, h2_) = drift_per_day(pts)
            gap = (date.fromisoformat(min(on_days))
                   - date.fromisoformat(min(off_days))).days
            drift = 100 * ((1 + g) ** gap - 1)
            print("   표류   정책 전 %d일: %s"
                  % (len(pts), " ".join("%.0f" % y for _, y in pts)))
            print("          하루 %+.3f%% · R² %.2f · 앞절반 %+.3f%% / 뒤절반 %+.3f%%"
                  % (100 * g, r2, 100 * h1_, 100 * h2_))
            # 세 가지를 갈라 말한다 — 안 덜기로 한 **이유가 다르면 다음 수가 다르다.**
            #  · 표류가 없다        덜 것이 없다. 날짜를 바꿔도 소용없다
            #  · 예열이 섞였다      앞 며칠을 빼고 다시 주면 된다
            #  · 흩어져 있다        기울기를 못 믿는다. 날을 늘려야 한다
            # 예열 기준은 실제 값에서 잡았다: P013 예열 구간이 앞 2.359% / 뒤 0.193%
            # 로 12.2배였고 진짜 직선은 1.0배다. 3배로 끊는다.
            if abs(g) < 5e-4:
                why = ("표류가 없다 (하루 ±0.05% 미만). **덜 것이 없다.**", None)
            elif abs(h1_) > 3 * abs(h2_):
                why = ("직선이 아니다 — **예열이 섞였다.** 덜지 않는다.",
                       "예열 뒤 날짜만 --trend-days 로 주면 적용된다."
                       "  (P013 은 첫 3일 +7.1% · 그 뒤 8일 +0.2% 였고,"
                       " 전 구간에 직선을 맞추면 없는 표류로 있는 효과를 지운다)")
            elif r2 < 0.5:
                why = ("흩어져 있다 (R² %.2f) — 기울기를 못 믿는다. 덜지 않는다." % r2,
                       "정책 전 날을 더 넣어라.")
            else:
                why = None
            bent = why is not None
            if bent:
                print("          ** %s" % why[0])
                if why[1]:
                    print("             %s" % why[1])
            else:
                print("          %d일 뒤 = %+.2f%% 를 덜면:" % (gap, drift))
                for nm in ("현행", "고침"):
                    print("            %-6s %+.2f%%" % (nm, rows[nm]["pct"] - drift))
                    rows[nm]["detrended_pct"] = rows[nm]["pct"] - drift
            rows["drift"] = {"pct": drift, "per_day": g, "r2": r2,
                             "halves": [h1_, h2_], "applied": not bent}
        out_rulers[key] = {"field": field, "scale": scale, **rows}

    signs = {k: (v["현행"]["pct"] > 0) for k, v in out_rulers.items()}
    if len(set(signs.values())) > 1:
        print()
        print("** 자에 따라 부호가 갈린다. 어느 자로 읽었는지 밝히지 않은 수는 못 쓴다.")

    print()
    print("## 다시 셈이 성한지 — 이걸 먼저 본다")
    print("  클램프에 걸린 칸   %d / %d  (%.1f%%)" % (hits, tot, 100 * hits / tot))
    print("  기준선이 없는 칸   %d / %d  (%.1f%%)  <- 이 칸은 곱수 1 로 둔다"
          % (miss, tot, 100 * miss / tot))
    if 100 * hits / tot > 40:
        print("  ** 클램프가 40%를 넘는다. 기준선이 창과 다른 요일종류일 수 있다 —")
        print("     --baseline-mode daytype 으로 다시 재라.")
    if 100 * miss / tot > 10:
        print("  ** 기준선 결손이 10%를 넘는다. 기준선 날짜를 늘려라.")

    print()
    print("## 다시 셈으로 복원되지 않는 것")
    print("  총액이 달라지면 잔고가 달라지고, 그 잔고를 본 다음 날 계획이 달라졌을 수")
    print("  있다. 그 되먹임은 여기 없다. v44 에서 모델이 문턱 거리에 반응하지 않는")
    print("  것이 확인돼 통로는 약하다고 보지만, 약한 것이지 없는 것이 아니다.")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps({
            "metrics": a.metrics, "arm": a.arm, "n_agents": len(aids),
            "baseline_days": b_days, "baseline_mode": a.baseline_mode,
            "off_days": off_days, "on_days": on_days,
            "clamp": [a.clamp_lo, a.clamp_hi], "rulers": out_rulers,
            "clamp_hit_rate": hits / tot, "missing_base_rate": miss / tot,
            "broke_rate": broke, "broke_day": worst_day,
        }, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
