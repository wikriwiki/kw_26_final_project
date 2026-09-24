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
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    def split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    b_days, off_days, on_days = split(a.baseline), split(a.off), split(a.on)

    overlap = (set(b_days) & set(off_days)) | (set(b_days) & set(on_days))
    if overlap:
        print("거부: 기준선 날짜가 비교 창과 겹친다 — %s" % sorted(overlap))
        print("  겹치면 그 창의 계획/기준선이 1 로 못 박혀 그 팔만 안 움직인다.")
        return 2

    per = load_ledger(a.metrics, a.arm)
    need = set(b_days) | set(off_days) | set(on_days)
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
        }, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
