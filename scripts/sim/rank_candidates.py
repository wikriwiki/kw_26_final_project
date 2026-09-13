"""후보 프롬프트 순위 — 순차반감 단계별 집계.

`score_policy.py --json-out` 이 남긴 결과를 모아 후보별 부호 일치율을 낸다.
다음 단계로 올릴 후보를 고르는 근거이며, 고른 이유를 그대로 기록에 남긴다.

## 순위 기준

1. **부호 일치율** — 정책을 가로질러 합산한다. 한 정책만 잘 맞추는 후보는
   일반화 주장에 쓸 수 없다.
2. **정책별 최저 일치율** — 동률이면 못 맞춘 정책이 덜 심한 쪽을 올린다.
   평균이 같아도 한 정책에서 0% 면 그 기전을 아예 못 다루는 것이다.
3. **무효 지표 적중** — "정책이면 효과가 난다"로 수렴한 후보를 거른다.
   가중치를 따로 주지는 않되 함께 보고한다.

## 1단계는 점추정 부호로, 뒷단계는 신뢰구간으로 가른다

순차반감은 **이른 단계에 예산을 적게 주고 잡음 섞인 추정으로 거르는** 것이
설계 자체다(Jamieson & Talwalkar 2016). 100명 x 2일에서 부트스트랩 구간이
0 을 포함하는 것은 당연하고, 그걸로 판정하면 모든 후보가 "0" 으로 읽혀 구분이
사라진다 — 거르기가 목적인 단계에서 아무것도 못 거르게 된다.

    1단계   증가·감소 지표는 **평균의 부호**로 판정 (거르기)
    2단계~  표본이 커지면 **부트스트랩 구간**으로 판정 (판정)

**무효(0) 가 정답인 지표는 단계와 무관하게 구간으로 본다.** 평균이 정확히 0 인
경우는 없으므로 점추정으로는 영원히 못 맞춘다.

구간 기준 적중도 함께 보고한다 — 1단계 순위가 잡음일 수 있다는 사실을 숨기지
않기 위해서다.

## 이항검정

지표 k개 x 정책 n개 부호 비교에서 무작위면 일치율 50% 다. 그보다 유의하게
높은지 단측 이항검정으로 본다. 표본이 작으면 유의하지 않게 나오는 것이 정상이며,
1단계에서 유의를 요구하지 않는다.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def binom_p(hits: int, n: int, p: float = 0.5) -> float:
    """단측 이항검정 p-value — P(X >= hits)."""
    if n <= 0:
        return 1.0
    tot = 0.0
    for k in range(hits, n + 1):
        tot += math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    return min(1.0, tot)


def hit_of(r: dict, use_point: bool) -> bool | None:
    """적중 여부. 1단계는 평균 부호로, 뒷단계는 구간으로.

    무효(0) 정답은 언제나 구간으로 본다 — 평균이 정확히 0 인 경우는 없다.
    """
    if r.get("hit") is None:
        return None
    exp = r.get("expect")
    if exp in ("0", "rank") or not use_point or "mean" not in r:
        return r["hit"]
    m = r.get("mean") or 0.0
    got = "+" if m > 0 else ("-" if m < 0 else "0")
    return got == exp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/data/stage1")
    ap.add_argument("--stage", default="1")
    ap.add_argument("--keep", type=int, default=3, help="다음 단계로 올릴 후보 수")
    ap.add_argument("--by", choices=("point", "ci"), default=None,
                    help="판정 기준. 기본은 1단계=point, 그 외=ci")
    a = ap.parse_args()
    use_point = (a.by == "point") if a.by else (str(a.stage) == "1")
    print(f"판정 기준: {'평균 부호(거르기)' if use_point else '부트스트랩 구간(판정)'}")

    files = sorted(Path(a.dir).glob("*.json"))
    if not files:
        print(f"{a.dir} 에 채점 결과가 없다", file=sys.stderr)
        return 2

    # 후보 → 정책 → (적중, 채점수, 무효적중, 무효수)
    agg: dict[str, dict[str, list[int]]] = defaultdict(dict)
    detail: dict[str, list[str]] = defaultdict(list)
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        cand, pol = d.get("label") or f.stem.split("_")[0], d["policy"]
        hs = [(r, hit_of(r, use_point)) for r in d["results"]]
        hits = sum(1 for _, h in hs if h)
        scored = sum(1 for _, h in hs if h is not None)
        ci_hits = sum(1 for r in d["results"] if r.get("hit"))
        nul = [r for r in d["results"] if r.get("expect") == "0"
               and r.get("hit") is not None]
        agg[cand][pol] = [hits, scored, sum(1 for r in nul if r["hit"]),
                          len(nul), ci_hits]
        for r, h in hs:
            if h is False:
                detail[cand].append(f"{pol}/{r['id']} 기대 {r['expect']} 실측 {r.get('got')}")

    rows = []
    for cand, bypol in agg.items():
        h = sum(v[0] for v in bypol.values())
        n = sum(v[1] for v in bypol.values())
        nh = sum(v[2] for v in bypol.values())
        nn = sum(v[3] for v in bypol.values())
        worst = min((v[0] / v[1] if v[1] else 0.0) for v in bypol.values())
        ci = sum(v[4] for v in bypol.values())
        rows.append({"cand": cand, "hits": h, "n": n,
                     "rate": h / n if n else 0.0, "worst": worst,
                     "null_hits": nh, "null_n": nn, "ci_hits": ci,
                     "p": binom_p(h, n), "bypol": bypol})
    rows.sort(key=lambda r: (-r["rate"], -r["worst"], r["cand"]))

    pols = sorted({p for r in rows for p in r["bypol"]})
    print(f"=== 순차반감 {a.stage}단계 — 후보 {len(rows)}개 / 정책 {len(pols)}개 ===")
    head = "{:<6}{:>9}{:>8}{:>9}{:>10}".format("후보", "일치율", "적중", "최저정책", "이항 p")
    for p in pols:
        head += "{:>18}".format(p[:16])
    print(head)
    print("-" * len(head))
    for r in rows:
        line = "{:<6}{:>8.0f}%{:>8}{:>8.0f}%{:>10.3f}".format(
            r["cand"], 100 * r["rate"], f"{r['hits']}/{r['n']}",
            100 * r["worst"], r["p"])
        for p in pols:
            v = r["bypol"].get(p)
            line += "{:>18}".format(f"{v[0]}/{v[1]}" if v else "-")
        print(line)

    print()
    print("구간 기준 적중 (참고 — 1단계 순위가 잡음일 수 있음을 숨기지 않기 위해)")
    for r in rows:
        print(f"  {r['cand']}: {r['ci_hits']}/{r['n']}")

    print()
    print("무효(0) 지표 적중 — '정책이면 효과가 난다'로 수렴한 후보를 거르는 방어선")
    for r in rows:
        if r["null_n"]:
            print(f"  {r['cand']}: {r['null_hits']}/{r['null_n']}")
        else:
            print(f"  {r['cand']}: 무효 지표 관측 없음 (표본 부족)")

    keep = [r["cand"] for r in rows[:a.keep]]
    print()
    print(f"다음 단계로 올릴 후보 {a.keep}개: {', '.join(keep)}")
    print("탈락 후보가 틀린 지표:")
    for r in rows[a.keep:]:
        ds = detail.get(r["cand"]) or []
        print(f"  {r['cand']}: " + ("; ".join(ds[:4]) or "없음"))
    print()
    print("※ 실효 자유도는 글자 수가 아니라 시도한 후보 수다. 여기서 고른 것은")
    print("  사전등록된 6개 안에서만 나온 결과이며, 새 후보를 추가하면 그만큼 늘어난다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
