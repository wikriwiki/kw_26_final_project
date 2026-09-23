"""**정책이 켜지면 배송 몫이 달라지는가** — 사전등록 2 의 1차 질문.

    python scripts/report/online_share_policy_response.py <metrics_dir> \
        --off 2021-10-21:2021-10-22 --on 2021-10-25:2021-10-26

## 왜 이것이 1차인가

프롬프트 문구로는 이 필드가 안 움직인다는 것을 96호출로 확인했다 — 질문 범위를
3배 넓혀도 0.1448 → 0.1427(p=0.711)이고 27%는 집합 논리에 어긋났다
(`experiments/split_anchor/calib_02.md`).

확인하지 **못한** 것은 **정책 맥락**에도 안 움직이는가다. 문구는 지시문 안의
낱말이고 정책은 맥락 안의 사실이다. P010 에서 본 것도 "서술은 무효, 계산된
사실에는 반응" 이었다. 그리고 이것은 탐침으로 못 본다 — 정책이 켜진 맥락은
그래프에서만 만들어지기 때문이다. **런 하나 안에 정책 전 창과 후 창이 다
있으므로 런이 곧 그 시험이다.**

## 읽는 법

    쌍별 부호검정(**양측**). 방향을 미리 고정하지 않는다 — 정책이 배송을 늘릴
    수도 있다(외출을 꺼리는 쪽으로). 동점 수를 숨기지 않고 함께 적는다.

    동점이 압도적이면(80% 이상) 필드가 사실상 상수라는 뜻이고, 그러면
    2차(P012-1)를 이 레버의 공으로 읽지 않는다.

## 순환이 아니다

재는 것은 에이전트가 적은 행동 값이고, 가르는 것은 날짜다. 측정 공식도 목표
수치도 프롬프트에 없다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from collections import defaultdict
from math import comb


def _days(spec: str) -> list[str]:
    a, b = spec.split(":")
    return [a, b] if a != b else [a]


def load(mdir: str) -> dict[tuple[str, str], float]:
    """(에이전트, 날짜) -> s1_online_share. status=ok 만."""
    out: dict[tuple[str, str], float] = {}
    for f in sorted(glob.glob(os.path.join(mdir, "*.jsonl"))):
        day = os.path.basename(f).replace("day_", "").replace(".jsonl", "")
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("status") != "ok":
                continue
            v = d.get("s1_online_share")
            aid = d.get("agent_id") or d.get("aid")
            if isinstance(v, (int, float)) and aid:
                out[(str(aid), str(d.get("day") or day))] = float(v)
    return out


def mean_over(src, aid, days):
    v = [src[(aid, d)] for d in days if (aid, d) in src]
    return st.mean(v) if v else None


def sign_test(pairs):
    up = sum(1 for a, b in pairs if b > a)
    dn = sum(1 for a, b in pairs if b < a)
    tie = sum(1 for a, b in pairs if b == a)
    n = up + dn
    p = 1.0
    if n:
        p = min(1.0, sum(comb(n, i) for i in range(min(up, dn) + 1)) / 2 ** n * 2)
    return up, dn, tie, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_dir")
    ap.add_argument("--off", required=True)
    ap.add_argument("--on", required=True)
    a = ap.parse_args()
    off_d, on_d = _days(a.off), _days(a.on)
    src = load(a.metrics_dir)
    if not src:
        print("s1_online_share 가 하나도 없다 — v5offsite 로 안 돌았거나 파싱이 안 됐다")
        return 2
    aids = sorted({k[0] for k in src})
    pairs = []
    for aid in aids:
        x, y = mean_over(src, aid, off_d), mean_over(src, aid, on_d)
        if x is not None and y is not None:
            pairs.append((x, y))

    print("# 정책이 켜지면 배송 몫이 달라지는가 — 사전등록 2 의 1차 질문")
    print()
    print("  정책 전 창 %s · 정책 후 창 %s" % (off_d, on_d))
    print("  값이 있는 (사람,날짜) %d · 쌍이 맞는 사람 %d" % (len(src), len(pairs)))
    if not pairs:
        print("  **쌍이 없다** — 창 날짜가 런 기간 밖일 수 있다")
        return 2
    xs = [x for x, _ in pairs]
    ys = [y for _, y in pairs]
    print()
    print("  정책 전  평균 %.4f · 중앙 %.4f · sd %.4f" % (st.mean(xs), st.median(xs), st.pstdev(xs)))
    print("  정책 후  평균 %.4f · 중앙 %.4f · sd %.4f" % (st.mean(ys), st.median(ys), st.pstdev(ys)))
    print("  차이(후-전) 평균 %+.4f" % (st.mean(ys) - st.mean(xs)))
    up, dn, tie, p = sign_test(pairs)
    print()
    print("  쌍별 부호검정 (**양측** — 방향을 미리 안 고정했다)")
    print("    늘어남 %d · 줄어듦 %d · 동점 %d" % (up, dn, tie))
    print("    p=%.4f  %s" % (p, "구별 안 됨" if p > 0.05 else "**다르다**"))
    tie_share = tie / len(pairs)
    print()
    print("  동점 비율 %.1f%%  %s" % (100 * tie_share,
          "**필드가 사실상 상수다 — 2차를 이 레버의 공으로 읽지 않는다**"
          if tie_share >= 0.80 else "상수는 아니다"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
