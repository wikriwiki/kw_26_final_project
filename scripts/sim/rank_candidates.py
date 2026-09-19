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
    # 사전등록된 교란 표시 — 해당 단계에서 못 쓰는 지표를 순위에서 뺀다.
    # 사후 지표 선택이 아니라 채점표에 미리 적어둔 사유를 적용하는 것이다.
    drop_conf = str(a.stage) == "1"
    conf: dict[str, bool] = {}
    # 현재 채점표에 살아 있는 지표만 인정한다 — 결함이 드러나 제거된 지표가
    # 과거 결과 파일에 남아 있어도 순위에 들어가지 않게 한다.
    import json as _j0
    _tb = _j0.loads((Path(__file__).resolve().parents[2] / "data" / "experiments"
                     / "scoring_table.json").read_text(encoding="utf-8"))
    live_ids = {i["id"] for k, v in _tb.items() if not k.startswith("_")
                for i in v.get("indicators", [])}
    stage_excl = set()
    if str(a.stage) == "2":
        stage_excl = set((_tb.get("_meta") or {}).get("stage2_excluded") or {})
        if stage_excl:
            print("2단계 제외(채점 방식 변경): " + ", ".join(sorted(stage_excl)))
    if drop_conf:
        import json as _j
        _t = _j.loads((Path(__file__).resolve().parents[2] / "data" /
                       "experiments" / "scoring_table.json").read_text(encoding="utf-8"))
        for _k, _v in _t.items():
            if _k.startswith("_"):
                continue
            for _i in _v.get("indicators", []):
                if _i.get("stage1_confounded"):
                    conf[_i["id"]] = True
        if conf:
            print("1단계 교란으로 제외: " + ", ".join(sorted(conf)))
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
        rs = d["results"]
        if drop_conf:
            rs = [r for r in rs if not conf.get(r.get("id"))]
        # 본런 규모 지표는 짧은 런에서 구조적으로 못 낸다(예: 한도 10만원 도달은
        # 정책 2일로 불가). 3단계 전까지는 채점에서 뺀다.
        if str(a.stage) in ("1", "2"):
            rs = [r for r in rs if r.get("scale") != "main"]
        # 채점표에서 빠진 지표(사유와 함께 not_scorable 로 이동)는 무시한다.
        rs = [r for r in rs if r.get("id") in live_ids or not live_ids]
        # 단계별 제외 — 채점 방식이 중간에 바뀌어 후보 간 기준이 달라진 지표.
        if stage_excl:
            rs = [r for r in rs if r.get("id") not in stage_excl]
        hs = [(r, hit_of(r, use_point)) for r in rs]
        hits = sum(1 for _, h in hs if h)
        scored = sum(1 for _, h in hs if h is not None)
        ci_hits = sum(1 for r in rs if r.get("hit"))
        nul = [r for r in rs if r.get("expect") == "0"
               and r.get("hit") is not None]
        agg[cand][pol] = [hits, scored, sum(1 for r in nul if r["hit"]),
                          len(nul), ci_hits]
        for r, h in hs:
            if h is False:
                detail[cand].append(f"{pol}/{r['id']} 기대 {r['expect']} 실측 {r.get('got')}")

    # [공정 비교] 후보마다 끝난 정책 수가 다르면 어려운 정책이 빠진 후보가
    # 유리해진다. 모든 후보가 가진 정책으로만 순위를 낸다. 부분 결과를 중간에
    # 들여다볼 때 순위가 뒤집히는 것을 막는 장치다.
    common = set.intersection(*(set(v) for v in agg.values())) if agg else set()
    skipped = sorted({p for v in agg.values() for p in v} - common)
    if skipped:
        print("공통 정책만 비교 — 제외: " + ", ".join(skipped))
    if not common:
        print("공통 정책이 없어 후보 순위를 계산할 수 없습니다.", file=sys.stderr)
        return 2
    if common:
        agg = {c: {p: v for p, v in bp.items() if p in common}
               for c, bp in agg.items()}

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

    print("주의: 지표 간 독립성과 50% 귀무확률은 검증되지 않았습니다. "
          "이항 p는 참고값이며 일반화 유의성의 증거가 아닙니다.")
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
