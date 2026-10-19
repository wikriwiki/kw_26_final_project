# -*- coding: utf-8 -*-
"""서울시 상권분석서비스(추정매출-행정동, OA-22175) → 세부 업종별 카드 결제 1건당 금액표.

    python scripts/prep/build_ticket_price.py   →  output/stats/ticket_price.json

행정동 × 서비스 업종마다 (분기 합계) 당월_매출_금액 ÷ 당월_매출_건수 = 그 동네 그 업종의 결제 1건당 평균.
서울 전체로는 동별 값의 하위 10% · 중앙 · 평균 · 상위 10% 를 둔다(최소·최대는 연회 등 극단 결제라 쓰지 않음).
'1인분 가격'이 아니다 — 여럿이 함께 낸 결제가 섞여 있다. 2단계 프롬프트에는 이 이름 그대로 보인다.
(2026-10-07, 사용자 결정: 2025년 자료 사용, 범위는 하위 10%~상위 10%)
"""
from __future__ import annotations

import csv
import glob
import io
import json
import statistics as st
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "stats" / "ticket_price.json"


def _rows(path):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with io.open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise SystemExit(f"읽을 수 없는 인코딩: {path}")


def main():
    paths = sorted(glob.glob(str(ROOT / "data" / "golmok" / "*.csv")))
    if not paths:
        raise SystemExit("data/golmok/*.csv 가 없다")
    amt, cnt, quarters = {}, {}, set()
    for p in paths:
        for r in _rows(p):
            try:
                a, c = float(r["당월_매출_금액"]), float(r["당월_매출_건수"])
            except (KeyError, TypeError, ValueError):
                continue
            if a <= 0 or c <= 0:
                continue
            k = (r["행정동_코드"].strip(), r["서비스_업종_코드_명"].strip())
            amt[k] = amt.get(k, 0.0) + a
            cnt[k] = cnt.get(k, 0.0) + c
            quarters.add(r.get("기준_년분기_코드"))
    dong: dict[str, dict[str, int]] = {}
    per_svc: dict[str, list[float]] = {}
    for (d, svc), a in amt.items():
        t = a / cnt[(d, svc)]
        dong.setdefault(d, {})[svc] = int(round(t))
        per_svc.setdefault(svc, []).append(t)
    svc_stats = {}
    for svc, v in per_svc.items():
        v = sorted(v)
        q = lambda f: int(round(v[min(len(v) - 1, int(f * len(v)))]))
        svc_stats[svc] = {"n_dong": len(v), "p10": q(0.10), "p50": int(round(st.median(v))),
                          "mean": int(round(st.mean(v))), "p90": q(0.90)}
    out = {"_meta": {"built": date.today().isoformat(), "source": "서울시 상권분석서비스(추정매출-행정동, OA-22175)",
                     "files": [Path(p).name for p in paths], "quarters": sorted(q for q in quarters if q),
                     "definition": "행정동×서비스업종 분기 합계 당월_매출_금액 ÷ 당월_매출_건수 (카드 결제 1건당, 여럿이 함께 낸 결제 포함)",
                     "range": "서울 동별 값의 하위 10% ~ 상위 10%"},
           "svc": svc_stats, "dong": dong}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"세부 업종 {len(svc_stats)} · 동 {len(dong)} · 분기 {sorted(q for q in quarters if q)} → {OUT}")


if __name__ == "__main__":
    main()
