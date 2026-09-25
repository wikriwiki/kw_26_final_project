"""행정동별 **적립업종 몫**을 BDC 실측에서 낸다 — 상수 0.2535 를 대신할 근거.

    python scripts/report/dong_eligible_share.py --zip output/stats/bdc_based_results.zip

## 무엇을 고치려는 것인가

`consumption.py` 는 적립업종 몫을 **전원 같은 상수**로 쓴다.

    ELIGIBLE_SHARE_SEOUL = 0.2535     # 모든 사람, 모든 날, 모든 동에서 같다

그래서 `적립업종 지출 변화 == 총지출 변화` 가 되고, 정답지가 갈라 놓은 두 수
(적립 +20.82% vs 제외 +2.85%)를 **원리적으로 재현할 수 없다.** 우리 간격은 항상 0 이다.
`experiments/error_budget/p012_2_why_unproducible.md`

## 자료 — 이미 갖고 있었다

`output/stats/bdc_based_results.zip :: dong_consumption.json` 에 **행정동 426개**의
`industry_ratio` 가 있다. 업종 95종, 합 1.0, 평일·주말 따로. 그리고 이 어휘는
**대형마트를 슈퍼마켓과 분리한다** — 에이전트의 `spending_top_wd_json` 은 상위 7종만
담아 합이 0.672 여서 쓸 수 없었지만, 이쪽은 완전하다.

주의: 최상위 `*` 키의 `industry_ratio` 는 **지출 구성이 아니다**(한식 0.0003 ·
세금공과금 0.125). 다른 정규화이므로 쓰지 않는다. 동 코드 키만 쓴다.

## 모르는 것은 모른다고 센다

업종명이 제외·적립을 묶어 놓은 경우가 있다(가전 = 대형전자전문점 + 동네가전).
그리고 원천에 `ZZ_나머지`·`결제대행(PG)` 라는 잔여 버킷이 있다. **이것들을 임의로
한쪽에 넣으면 그 지표는 우리가 만든 수가 된다.** 그래서 세 갈래로 낸다.

    적립 몫      근거가 있는 적립업종의 합
    제외 몫      근거가 있는 제외업종의 합
    모름 몫      나머지 — **지우지 않고 그대로 보고한다**

`모름` 이 크면 그만큼이 이 자료로 낼 수 있는 정확도의 상한이다.
"""
from __future__ import annotations

import argparse
import io
import json
import statistics as st
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MAP = ROOT / "data" / "sangsaeng" / "bdc_industry_arm.json"

EXCLUDED_ARMS = ("excluded_other", "excluded_luxury", "excluded_vice",
                 "excluded_nonconsumption")


def load_arms() -> dict:
    """{업종명: arm}. 매핑 파일의 근거를 버리지 않고 이름만 뽑는다."""
    m = json.loads(MAP.read_text(encoding="utf-8"))
    out = {}
    for arm, table in m.items():
        if arm.startswith("_") or not isinstance(table, dict):
            continue
        for name in table:
            out[name] = arm
    return out


def load_dongs(zip_path: Path, key: str = "industry_ratio") -> dict:
    """{동코드: {업종: 비중}} — 동 코드 키만. '*' 는 쓰지 않는다."""
    with zipfile.ZipFile(zip_path) as z:
        with z.open("dong_consumption.json") as f:
            d = json.loads(f.read().decode("utf-8"))
    return {k: (v.get(key) or {}) for k, v in d.items()
            if k.isdigit() and isinstance(v, dict)}


def split(ratio: dict, arms: dict) -> dict:
    """한 동의 구성을 적립·제외·모름으로. 합은 원본 합을 보존한다."""
    out = {"eligible": 0.0, "excluded": 0.0, "unknown": 0.0, "total": 0.0}
    miss = {}
    for name, w in ratio.items():
        try:
            w = float(w)
        except (TypeError, ValueError):
            continue
        out["total"] += w
        arm = arms.get(name)
        if arm == "eligible":
            out["eligible"] += w
        elif arm in EXCLUDED_ARMS:
            out["excluded"] += w
        else:
            out["unknown"] += w
            if arm is None:
                miss[name] = miss.get(name, 0.0) + w
    out["_unmapped"] = miss
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="output/stats/bdc_based_results.zip")
    ap.add_argument("--key", default="industry_ratio",
                    choices=("industry_ratio", "industry_weekday_ratio",
                             "industry_weekend_ratio"))
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    arms = load_arms()
    dongs = load_dongs(ROOT / a.zip, a.key)
    print("# 행정동별 적립업종 몫 — BDC 실측")
    print()
    print("  자료  %s :: dong_consumption.json [%s]" % (a.zip, a.key))
    print("  동 %d개 · 매핑된 업종명 %d종" % (len(dongs), len(arms)))

    rows = {k: split(v, arms) for k, v in dongs.items()}
    unmapped: dict = {}
    for r in rows.values():
        for n, w in r["_unmapped"].items():
            unmapped[n] = unmapped.get(n, 0.0) + w

    def q(vals, p):
        s = sorted(vals)
        return s[int(len(s) * p)] if s else float("nan")

    print()
    print("  %-10s %8s %8s %8s %8s %8s" % ("", "최소", "10%", "중앙", "90%", "최대"))
    for lbl, k in (("적립 몫", "eligible"), ("제외 몫", "excluded"), ("모름 몫", "unknown")):
        v = [r[k] for r in rows.values()]
        print("  %-10s %8.4f %8.4f %8.4f %8.4f %8.4f"
              % (lbl, min(v), q(v, .1), st.median(v), q(v, .9), max(v)))

    el = [r["eligible"] for r in rows.values()]
    un = [r["unknown"] for r in rows.values()]
    print()
    print("## 지금 쓰는 상수와 대면")
    print("  ELIGIBLE_SHARE_SEOUL = 0.2535  (전원·전일·전동 동일)")
    print("  BDC 실측 적립 몫      중앙 %.4f · 동별로 %.4f ~ %.4f 로 **갈린다**"
          % (st.median(el), min(el), max(el)))
    print("  -> 상수를 이 값으로 바꾸면 적립 몫이 **사람이 사는 동에 따라 달라진다**")

    print()
    print("## 이 자료의 한계 — 숨기지 않는다")
    print("  모름 몫 중앙 **%.4f** (%.1f%%). 그만큼이 정확도의 상한이다."
          % (st.median(un), 100 * st.median(un)))
    tot = sum(unmapped.values()) or 1.0
    print("  매핑 파일이 unknown 으로 둔 것 + 이름이 없던 것, 비중 큰 순:")
    for n in sorted(unmapped, key=lambda x: -unmapped[x])[:8]:
        print("     %-24s 동 평균 %.4f" % (n, unmapped[n] / len(rows)))
    if unmapped:
        print("  (이름이 매핑 파일에 아예 없으면 새 업종이다 — 파일에 근거와 함께 추가할 것)")

    if a.json_out:
        io.open(a.json_out, "w", encoding="utf-8", newline="\n").write(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
             for k, v in rows.items()}, ensure_ascii=False, indent=1))
        print()
        print("→ %s" % a.json_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
