"""배송 몫을 **말하기는 하는가**, 그리고 기준 평균은 얼마인가 — 사전등록 단계 가.

    python scripts/sim/online_share_probe.py --frozen <frozen_inputs.json> --out <dir>
    python scripts/sim/online_share_probe.py --out <dir> --responses <jsonl>

사전등록: `experiments/split_anchor/prereg.md`

## 무엇을 재나

v5 는 `online_share` 를 묻지 않아 계획 2,977건 전부 None 이었다(diagnosis_03).
v5online 은 묻는다. 본런에 12시간을 붓기 전에 두 가지만 싸게 확인한다.

    (a) 응답률   JSON 최상위에 online_share 가 실제로 오는가.
                 **90% 미만이면 중단한다** — 안 오는 필드로는 회계를 못 돌린다
    (b) 기준평균  mean(1 - online_share). 이 값이 KEEP_MEAN 이 되어 회계에서
                 쏠림을 지운다. 동결 맥락은 **정책이 없으므로** 기준선이 맞다

## 왜 정책 대조를 여기서 안 하나

동결 맥락은 전부 "(활성 정책 없음)" 이다. 정책이 켜진 맥락은 그래프에서
빌드되므로 탐침으로 못 만든다. **정책에 움직이는가는 본런에서** 정책 시작 전후
날짜의 online_share 를 맞대어 본다(사전등록 단계 다·라).

## 쏠림은 예상돼 있다

예시 숫자 근처로 몰리는 것은 이미 측정된 현상이다(daily_propensity 가 0.68 에
54%). 그래서 회계는 수준이 아니라 **편차만** 받고, 여기서 잰 평균으로 나눈다.
쏠림 자체는 중단 사유가 아니다 — **분산이 0 이면** 레버가 죽은 것이므로 그것을
따로 적는다.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import statistics as st
from pathlib import Path


def parse_online(raw: str) -> float | None:
    """응답에서 최상위 online_share 하나. 못 찾으면 None."""
    if not raw:
        return None
    m = re.search(r'"online_share"\s*:\s*([0-9]*\.?[0-9]+)', raw)
    if not m:
        return None
    try:
        v = float(m.group(1))
    except ValueError:
        return None
    return v if 0.0 <= v <= 1.0 else None


def build(frozen: dict) -> list[dict]:
    """동결 맥락을 그대로 쓴다 — 한 글자도 안 끼운다. 바뀌는 것은 SYSTEM 뿐이다."""
    out = []
    for c in frozen.get("cells") or []:
        if c.get("side") == "on":      # off/on 쌍이면 off 쪽만 (정책 없는 기준선)
            continue
        out.append({k: c[k] for k in ("aid", "case", "date", "user") if k in c})
    return out


def report(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r.get("error")]
    vals = [v for v in (parse_online(r.get("raw") or "") for r in ok) if v is not None]
    rate = len(vals) / len(ok) if ok else 0.0
    keep = [1.0 - v for v in vals]
    out = {
        "n_called": len(rows), "n_ok": len(ok), "n_parsed": len(vals),
        "response_rate": round(rate, 4),
        "online_mean": round(st.mean(vals), 4) if vals else None,
        "online_median": round(st.median(vals), 4) if vals else None,
        "online_sd": round(st.pstdev(vals), 4) if len(vals) > 1 else None,
        "distinct": len(set(vals)),
        "keep_mean": round(st.mean(keep), 4) if keep else None,
        "gate_response_rate": bool(rate >= 0.90),
        "gate_has_variation": bool(len(set(vals)) > 1 and (st.pstdev(vals) if len(vals) > 1 else 0) > 0.01),
    }
    print("호출 %d · 성공 %d · 파싱 %d" % (out["n_called"], out["n_ok"], out["n_parsed"]))
    print()
    print("  (a) 응답률        %.1f%%   %s (기준 90%%)"
          % (100 * rate, "통과" if out["gate_response_rate"] else "**중단**"))
    if vals:
        print("  (b) online_share  평균 %.4f · 중앙 %.4f · 표준편차 %s · 고유값 %d"
              % (out["online_mean"], out["online_median"],
                 ("%.4f" % out["online_sd"]) if out["online_sd"] is not None else "-",
                 out["distinct"]))
        print("      → KEEP_MEAN = %.4f   (= mean(1 - online_share), 이 값을 얼린다)"
              % out["keep_mean"])
        print("      분산          %s" % ("있음" if out["gate_has_variation"]
                                          else "**없음 — 레버가 죽었다**"))
        import collections
        for v, n in collections.Counter(vals).most_common(6):
            print("        %.2f  %4d  %s" % (v, n, "#" * int(50 * n / len(vals))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--responses", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.responses:
        rows = [json.loads(l) for l in io.open(a.responses, encoding="utf-8") if l.strip()]
        res = report(rows)
        io.open(out / "calibration.json", "w", encoding="utf-8", newline="\n").write(
            json.dumps(res, ensure_ascii=False, indent=1))
        print()
        print("→", out / "calibration.json")
        return 0
    frozen = json.loads(io.open(a.frozen, encoding="utf-8").read())
    cells = build(frozen)
    io.open(out / "cells.json", "w", encoding="utf-8", newline="\n").write(
        json.dumps({"cells": cells}, ensure_ascii=False, indent=1))
    print("정책 없는 기준 맥락 %d칸 (시민 %d)"
          % (len(cells), len({c["aid"] for c in cells})))
    print("wrote", out / "cells.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
