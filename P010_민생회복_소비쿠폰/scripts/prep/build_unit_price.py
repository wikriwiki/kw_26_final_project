# -*- coding: utf-8 -*-
"""실측 건단가 → 가격 prior 빌더  (output/stats/unit_price.json 생성)

소비 알고리즘의 가격 축을 합성값에서 실측으로 교체하는 전처리.

Layer A-lite (항상, 레포 내 데이터):
  dong_consumption.json 의 (카드이용금액계, 카드이용건수계) 쌍은 동 내부에서
  합=1로 정규화돼 있지만 두 값의 '비율'은 정규화가 상쇄되어
      건단가(평균 결제단가) = 금액계 / 건수계   (BDC 신용카드 원천, 실측)
  로 복원된다. 이것을 동 가격계수로 변환:
      z = clamp( (ln u − mean ln u) / std ln u , ±2 )     # 소표본 꼬리 강건화
      factor = 1 + ELASTICITY × z  → 평균 1.0 정규화 → clamp [0.80, 1.25]
  ⇒ 기존 b069_sales(발달지수 매출) prior를 대체 — Huff prior·공간 검증 정답과
    같은 변수를 쓰던 순환성 제거 (Pearson(건단가, b069_sales) ≈ 0.61: 관련되나 다른 축).

Layer A (선택, 공공 CSV 있으면 자동 승격):
  data/golmok/*.csv — 서울시 상권분석서비스(추정매출-행정동, OA-22175):
  행정동×서비스업종 분기 당월_매출_금액 / 당월_매출_건수 → 업종(L1)×동 건단가.
  → l1_unit_price(서울 중위 업종 단가) + dong_factor(업종가중 동 계수)로 대체.
  https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do

출력 스키마 (poi_price.py 가 소비):
{
  "_meta": {...provenance/params...},
  "source": "bdc_dong_consumption" | "golmok_csv",
  "dong_factor": {"11110515": 0.97, ...},      # 평균 1.0
  "l1_unit_price": {"식사": 9800, ...} | {}     # golmok 있을 때만 (원)
}
"""
from __future__ import annotations

import csv
import glob
import io
import json
import math
import re
import statistics as st
import sys
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "stats" / "unit_price.json"

ELASTICITY = 0.08          # z 1σ당 ±8% — 동 간 단가차의 가격계수 반영 강도
Z_CLAMP = 2.0              # 소표본 동(평창동류 양극단) 꼬리 절단
FACTOR_CLAMP = (0.80, 1.25)

# golmok 서비스업종명 → 시뮬 L1 (순서 중요: 구체 키워드 먼저)
GOLMOK_L1_MAP: list[tuple[str, str]] = [
    ("치과", "건강"), ("한의원", "건강"), ("일반의원", "건강"), ("약국", "건강"),
    ("한식", "식사"), ("중식", "식사"), ("일식", "식사"), ("양식", "식사"),
    ("패스트푸드", "식사"), ("치킨", "식사"), ("분식", "식사"),
    ("호프", "주점"), ("간이주점", "주점"),
    ("커피", "카페"), ("음료", "카페"),
    ("제과", "디저트"),
    ("편의점", "편의점"), ("슈퍼마켓", "마트"),
    ("미용실", "미용"), ("네일", "미용"), ("피부관리", "미용"), ("화장품", "미용"),
    ("노래방", "여가"), ("PC방", "여가"), ("피시방", "여가"), ("당구장", "여가"),
    ("골프연습장", "여가"), ("스포츠", "여가"), ("볼링장", "여가"),
    ("학원", "교육"), ("교습", "교육"),
    ("의류", "쇼핑"), ("신발", "쇼핑"), ("가방", "쇼핑"), ("안경", "쇼핑"),
    ("서적", "쇼핑"), ("문구", "쇼핑"), ("완구", "쇼핑"), ("귀금속", "쇼핑"),
]


def _industry_to_l1(name: str) -> str | None:
    for kw, l1 in GOLMOK_L1_MAP:
        if kw in (name or ""):
            return l1
    return None


def _zlog_factors(unit: dict[str, float]) -> dict[str, float]:
    """건단가(원) → 동 가격계수. 로그-z 클램프 + 평균 1.0 정규화."""
    logs = {k: math.log(v) for k, v in unit.items() if v and v > 0}
    mu, sd = st.mean(logs.values()), st.pstdev(logs.values())
    if sd <= 0:
        return {k: 1.0 for k in logs}
    raw = {k: 1.0 + ELASTICITY * max(-Z_CLAMP, min(Z_CLAMP, (x - mu) / sd))
           for k, x in logs.items()}
    m = st.mean(raw.values())
    lo, hi = FACTOR_CLAMP
    return {k: round(max(lo, min(hi, v / m)), 4) for k, v in raw.items()}


# ─────────────────────────────────────────────────────────
# Layer A-lite: dong_consumption.json (BDC 카드 — 레포 내)
# ─────────────────────────────────────────────────────────
def build_from_bdc() -> tuple[dict[str, float], dict]:
    dcs = json.load(io.open(ROOT / "output/stats/dong_consumption.json", encoding="utf-8"))
    unit: dict[str, float] = {}
    for k, v in dcs.items():
        if not re.fullmatch(r"\d{8}", k) or not isinstance(v, dict):
            continue  # 전처리 잔재 키(업종명 등) 제외
        r = v.get("hourly_consumption_ratio") or {}
        amt, cnt = r.get("카드이용금액계"), r.get("카드이용건수계")
        if amt and cnt and cnt > 0:
            unit[k] = amt / cnt
    factors = _zlog_factors(unit)
    vals = sorted(unit.values())
    meta = {
        "n_dong": len(unit),
        "unit_price_ratio_p50": round(vals[len(vals) // 2], 2),
        "note": "비율척도(원단위 소실, 천원 추정) — 동 간 상대비교만 사용",
    }
    return factors, meta


# ─────────────────────────────────────────────────────────
# Layer A: golmok 추정매출-행정동 CSV (있으면 승격)
# ─────────────────────────────────────────────────────────
def _read_csv_any(path: str):
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            with io.open(path, encoding=enc, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows and any("행정동" in (c or "") or "매출" in (c or "") for c in rows[0]):
                return rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    return []


def _col(row: dict, *kws: str) -> str | None:
    for c in row:
        if c and all(k in c for k in kws):
            return c
    return None


def build_from_golmok(paths: list[str]) -> tuple[dict[str, float], dict[str, int], dict]:
    """행정동×업종 당월_매출_금액/건수 → dong×L1 건단가 → (동계수, L1 중위단가)."""
    acc: dict[tuple[str, str], list[float]] = {}   # (dong, l1) -> [건단가...]
    n_rows = used = 0
    for p in paths:
        rows = _read_csv_any(p)
        if not rows:
            continue
        c_dong = _col(rows[0], "행정동", "코드")
        c_ind = _col(rows[0], "서비스", "업종", "명") or _col(rows[0], "업종", "명")
        c_amt = _col(rows[0], "매출", "금액")
        c_cnt = _col(rows[0], "매출", "건수")
        if not all((c_dong, c_ind, c_amt, c_cnt)):
            continue
        for r in rows:
            n_rows += 1
            l1 = _industry_to_l1(r.get(c_ind, ""))
            dong = (r.get(c_dong) or "").strip()
            try:
                amt, cnt = float(r[c_amt]), float(r[c_cnt])
            except (TypeError, ValueError, KeyError):
                continue
            if not l1 or not re.fullmatch(r"\d{8}", dong) or cnt <= 0 or amt <= 0:
                continue
            acc.setdefault((dong, l1), []).append(amt / cnt)
            used += 1
    if not acc:
        return {}, {}, {}

    # L1 중위 단가 (서울 전역) — Stage2 앵커의 기준선
    by_l1: dict[str, list[float]] = {}
    for (d, l1), v in acc.items():
        by_l1.setdefault(l1, []).extend(v)
    l1_unit = {l1: int(round(st.median(v) / 100.0) * 100) for l1, v in by_l1.items()}

    # 동 계수: 업종 구성차 통제 — 동×L1 단가를 L1 중위로 나눈 뒤 동 중위
    rel_by_dong: dict[str, list[float]] = {}
    for (d, l1), v in acc.items():
        rel_by_dong.setdefault(d, []).append(st.median(v) / max(l1_unit[l1], 1))
    dong_unit = {d: st.median(v) for d, v in rel_by_dong.items()}
    factors = _zlog_factors(dong_unit)
    meta = {"csv_rows": n_rows, "rows_used": used,
            "n_dong": len(factors), "n_l1": len(l1_unit)}
    return factors, l1_unit, meta


def main() -> None:
    golmok_paths = sorted(glob.glob(str(ROOT / "data" / "golmok" / "*.csv")))
    out: dict = {"_meta": {"built": date.today().isoformat(),
                           "elasticity": ELASTICITY, "z_clamp": Z_CLAMP,
                           "factor_clamp": FACTOR_CLAMP}}

    if golmok_paths:
        factors, l1_unit, meta = build_from_golmok(golmok_paths)
        if factors:
            out.update(source="golmok_csv", dong_factor=factors, l1_unit_price=l1_unit)
            out["_meta"]["golmok"] = meta
            out["_meta"]["provenance"] = "서울시 상권분석서비스(추정매출-행정동, OA-22175) 당월_매출_금액/건수"
    if "dong_factor" not in out:
        factors, meta = build_from_bdc()
        out.update(source="bdc_dong_consumption", dong_factor=factors, l1_unit_price={})
        out["_meta"]["bdc"] = meta
        out["_meta"]["provenance"] = "BDC 신용카드(집계구 결제) 동별 카드이용금액계/건수계 비율 복원"

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    f = out["dong_factor"]
    vals = sorted(f.values())
    print(f"source={out['source']}  동 {len(f)}개")
    print(f"dong_factor: min={vals[0]} p50={vals[len(vals)//2]} max={vals[-1]} mean={st.mean(vals):.4f}")
    if out["l1_unit_price"]:
        print("l1_unit_price:", out["l1_unit_price"])
    print("→", OUT)


if __name__ == "__main__":
    main()
