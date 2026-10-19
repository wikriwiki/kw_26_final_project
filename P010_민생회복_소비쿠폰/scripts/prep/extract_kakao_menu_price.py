# -*- coding: utf-8 -*-
"""카카오 플레이스 메뉴가격 추출 → POI 개별 가격대 (Layer B)  [크롤 머신에서 실행]

kakao_enrich.db 의 panel3_raw.raw_json(카카오 상세 패널 원본)에는 리뷰 외에
매장이 등록한 메뉴·가격이 들어있는 경우가 많다(음식점은 가격정보 필수 입력).
이 스크립트는:
  1) raw_json 전체를 재귀 탐색해 {이름, 가격} 리스트 형태의 메뉴 필드를 자동 발견
     (스키마 하드코딩 없음 — menu/가격/price 냄새 키 + 값 패턴 검증)
  2) POI별 대표가 = 메뉴가 중앙값(median)
  3) 카카오 카테고리 → 시뮬 L1 매핑 후, L1 내 백분위로 밴드(₩30%/₩₩50%/₩₩₩20%)
     + rel = 대표가 / L1 중앙값 (clamp [0.6, 1.6])
  4) output/stats/poi_menu_price.json 출력 → scripts/sim/poi_price.py 가 자동 사용
     (rel 은 절대 상대배율이라 동 가격계수와 곱하지 않음 — 이중계상 방지)

사용:  python extract_kakao_menu_price.py [--db PATH] [--out PATH] [--probe]
  --probe : 추출 없이 가격 필드 후보·커버리지만 리포트 (스키마 확인용, 먼저 실행 권장)
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DEFAULT_DB = r"C:/Users/Administrator/naver_crawl/sqlite/kakao_enrich.db"
DEFAULT_OUT = Path(__file__).resolve().parents[2] / "output" / "stats" / "poi_menu_price.json"

BAND_CUT = (30, 80)          # 백분위 → ₩/₩₩/₩₩₩ (poi_price.py 와 동일 30/50/20)
REL_CLAMP = (0.60, 1.60)
MIN_PRICE, MAX_PRICE = 500, 500_000   # 메뉴가 sanity (원)

# 카카오 카테고리명 키워드 → 시뮬 L1 (순서 중요: 구체 키워드 먼저)
KAKAO_L1_MAP: list[tuple[str, str]] = [
    ("치과", "건강"), ("한의원", "건강"), ("병원", "건강"), ("의원", "건강"), ("약국", "건강"),
    ("술집", "주점"), ("호프", "주점"), ("포장마차", "주점"), ("바(BAR)", "주점"), ("이자카야", "주점"),
    ("카페", "카페"), ("커피", "카페"), ("찻집", "카페"),
    ("제과", "디저트"), ("베이커리", "디저트"), ("떡", "디저트"), ("도넛", "디저트"), ("아이스크림", "디저트"),
    ("편의점", "편의점"), ("슈퍼", "마트"), ("마트", "마트"),
    ("미용", "미용"), ("헤어", "미용"), ("네일", "미용"), ("피부", "미용"),
    ("노래", "여가"), ("PC방", "여가"), ("볼링", "여가"), ("당구", "여가"), ("골프", "여가"),
    ("스포츠", "여가"), ("영화", "여가"),
    ("학원", "교육"), ("교습", "교육"),
    ("음식점", "식사"), ("한식", "식사"), ("중식", "식사"), ("일식", "식사"), ("양식", "식사"),
    ("분식", "식사"), ("치킨", "식사"), ("패스트푸드", "식사"), ("고기", "식사"), ("국수", "식사"),
]

_PRICE_KEY_RE = re.compile(r"menu|price|가격|메뉴", re.IGNORECASE)
_NAMEISH = ("name", "menu", "title", "메뉴", "상품")
_PRICEISH = ("price", "가격", "cost", "금액")


def _parse_price(v) -> int | None:
    """'12,000원' / '12000' / 12000 → int. '변동'·범위 등은 None."""
    if isinstance(v, (int, float)):
        p = int(v)
    else:
        s = re.sub(r"[,\s원~]", "", str(v or ""))
        if not s.isdigit():
            return None
        p = int(s)
    return p if MIN_PRICE <= p <= MAX_PRICE else None


def _extract_prices(obj, path="", hits=None, probe: Counter | None = None) -> list[int]:
    """raw_json 재귀 탐색 — {이름, 가격} dict 리스트에서 가격 수집."""
    if hits is None:
        hits = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            # dict 안에 name류 + price류 키가 같이 있으면 메뉴 항목으로 간주
            if any(t in kl for t in _PRICEISH) and not isinstance(v, (dict, list)):
                sib_name = any(any(t in str(k2).lower() for t in _NAMEISH) for k2 in obj)
                p = _parse_price(v)
                if p and sib_name:
                    hits.append(p)
                    if probe is not None:
                        probe[path + "." + k] += 1
            _extract_prices(v, f"{path}.{k}", hits, probe)
    elif isinstance(obj, list):
        for x in obj[:80]:
            _extract_prices(x, path + "[]", hits, probe)
    return hits


def _kakao_cat_to_l1(cat: str | None) -> str | None:
    for kw, l1 in KAKAO_L1_MAP:
        if kw in (cat or ""):
            return l1
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--probe", action="store_true", help="가격 필드 탐지 리포트만")
    ap.add_argument("--limit", type=int, default=0, help="처리 행 제한 (0=전체)")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    q = ("SELECT s.poi_id, p.raw_json FROM poi_status s "
         "JOIN panel3_raw p ON s.kakao_pid = p.kakao_pid WHERE s.status='fetched'")
    if args.limit:
        q += f" LIMIT {args.limit}"

    probe_counter: Counter = Counter()
    per_poi: dict[str, dict] = {}
    n_rows = n_json_err = 0

    for row in conn.execute(q):
        n_rows += 1
        try:
            panel = json.loads(row["raw_json"])
        except (json.JSONDecodeError, TypeError):
            n_json_err += 1
            continue
        prices = _extract_prices(panel, probe=probe_counter if args.probe else None)
        if not prices:
            continue
        cat = ((panel.get("summary") or {}).get("category") or {}).get("name")
        com_id = row["poi_id"] or ""
        neo_id = "C_" + com_id[4:] if com_id.startswith("COM_") else com_id
        per_poi[neo_id] = {
            "median_won": int(st.median(prices)),
            "n_menu": len(prices),
            "kakao_cat": cat,
        }

    print(f"패널 {n_rows:,}건 스캔 / JSON 오류 {n_json_err} / 메뉴가 확보 POI {len(per_poi):,} "
          f"({100 * len(per_poi) / max(n_rows, 1):.1f}%)")

    if args.probe:
        print("\n=== 가격 필드 발견 경로 top 20 ===")
        for k, n in probe_counter.most_common(20):
            print(f"  {n:8,}  {k}")
        cats = Counter(v["kakao_cat"] for v in per_poi.values())
        print("\n=== 메뉴가 보유 카카오 카테고리 top 15 ===")
        for c, n in cats.most_common(15):
            print(f"  {n:6,}  {c}")
        return

    # L1별 백분위 → band/rel
    by_l1: dict[str, list[int]] = defaultdict(list)
    for v in per_poi.values():
        l1 = _kakao_cat_to_l1(v["kakao_cat"])
        v["l1"] = l1
        if l1:
            by_l1[l1].append(v["median_won"])
    l1_sorted = {l1: sorted(v) for l1, v in by_l1.items() if len(v) >= 30}
    l1_median = {l1: v[len(v) // 2] for l1, v in l1_sorted.items()}

    out_poi: dict[str, dict] = {}
    lo, hi = REL_CLAMP
    for pid, v in per_poi.items():
        l1 = v.get("l1")
        if not l1 or l1 not in l1_sorted:
            continue  # L1 미매핑/표본부족 업종은 제외 → 해시 밴드 fallback
        arr = l1_sorted[l1]
        # 백분위(0~100)
        import bisect
        pct = 100.0 * bisect.bisect_left(arr, v["median_won"]) / len(arr)
        band = 1 if pct < BAND_CUT[0] else (2 if pct < BAND_CUT[1] else 3)
        rel = round(max(lo, min(hi, v["median_won"] / max(l1_median[l1], 1))), 3)
        out_poi[pid] = {"band": band, "rel": rel,
                        "median_won": v["median_won"], "n_menu": v["n_menu"], "l1": l1}

    out = {
        "_meta": {
            "built": date.today().isoformat(), "source_db": args.db,
            "n_poi": len(out_poi), "band_cut": BAND_CUT, "rel_clamp": REL_CLAMP,
            "l1_median_won": l1_median,
            "provenance": "카카오 플레이스 매장 등록 메뉴가격 (panel3_raw)",
        },
        "poi": out_poi,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"L1 중앙단가: {l1_median}")
    print(f"→ {args.out}  (POI {len(out_poi):,}개)")
    print("이 파일을 시뮬 레포 output/stats/ 에 두면 poi_price.py 가 자동 사용합니다.")


if __name__ == "__main__":
    main()
