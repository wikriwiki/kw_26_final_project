# -*- coding: utf-8 -*-
"""commerce POI 전체에 상생소비지원금 적립/제외 여부 백필.

기록 속성:
  p.sangsaeng_eligible : bool  — 적립업종이면 True (실적 인정). 제외면 False.
  p.sangsaeng_arm      : str   — eligible / excluded_luxury / excluded_vice /
                                 excluded_nonconsumption / excluded_other
  p.sangsaeng_kdi      : str   — 적립 POI의 KDI 8분류(가전·가구/학원/이·미용/여행·레저/
                                 요식/유통/기타). 제외 POI는 NULL. (D1/D2 채점용)
  p.sangsaeng_src      : str   — upjong_code / rule_name / rule_fallback (판정 근거 출처)
  p.upjong_l3          : str   — 원천 L3 업종코드 (조인된 POI만). 사후 감사·분석용.

판정: scripts/sim/sangsaeng_eligibility.sangsaeng_arm
      ① 원천 CSV의 L3 업종코드 조인 → arm 확정
      ② 코드가 못 닿는 항목만 상호명 룰 (카지노류·대형마트 브랜드)
      ③ 코드 있고 ①②에 안 걸림 → 적립 확정
      ④ 코드 미확보 POI → 기존 L2 룰 fallback
      KDI 매핑: data/sangsaeng/industry_map.json (l1_default + sub_override)

원천 CSV 조인 근거 (docs/SANGSAENG_UPJONG_CODE_FIX.md §5.2):
  03_pois.py 가 commerce POI id 를 f"C_{상가아이디}" 로 결정론적 생성하므로,
  같은 원천 CSV를 다시 읽어 (id → upjong_l3_cd) 인덱스를 만들면 100% 정확히 붙는다.
  그래프 구조(노드·관계·카테고리)는 건드리지 않으며, 재적재도 불필요하다.

  왜 코드가 필요한가: 우리 L2 '일반주점' 하나에 요리주점(I21104, 10,193·적립)과
  유흥주점(I21101/I21102, 2,365·제외)이 병합돼 있어 카테고리로도 상호명으로도
  갈라지지 않는다. L3 코드만이 정확한 분리 축이다.

런타임은 이 속성을 우선 사용하고, 없으면 같은 룰로 fallback 하므로 백필 전에도
시뮬은 동작한다 (백필은 조회 일관성·집계 편의·야간 State 집계용).

사용:
  python scripts/neo4j_load/11_sangsaeng_eligibility.py --check-csv   # CSV만 검증 (DB 불필요)
  python scripts/neo4j_load/11_sangsaeng_eligibility.py --dry-run     # DB 읽기만
  python scripts/neo4j_load/11_sangsaeng_eligibility.py               # 실제 백필
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
from sangsaeng_eligibility import (  # noqa: E402
    ARM_ELIGIBLE,
    ARM_EXCLUDED_LUXURY,
    ARM_EXCLUDED_NONCONSUMPTION,
    ARM_EXCLUDED_OTHER,
    ARM_EXCLUDED_VICE,
    sangsaeng_arm,
    src_of,
)

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
_MAP_PATH = ROOT / "data" / "sangsaeng" / "industry_map.json"
_POIS_DIR = ROOT / "data" / "neo4j_load" / "pois"

# commerce 원천 CSV 컬럼 인덱스 — 03_pois.py 의 C_COL 과 반드시 같아야 한다 (같은 파일).
C_ID_COL = 0        # 상가아이디
C_UPJONG_COL = 7    # 상권업종소분류코드

# --check-csv 기대값 (docs §8.3). 원천 데이터 기준.
_EXPECT_CSV = {
    "G21701": 2404,     # 시계/귀금속 — excluded_luxury
    "I21101": 2269,     # 일반 유흥 주점 — excluded_vice
    "I21102": 96,       # 무도 유흥 주점 — excluded_vice
    "R10410": 1038,     # 복권 — excluded_vice
    "I21104": 10193,    # 요리 주점 — ★ 적립 유지 확인용
}

# --dry-run 기대값 (docs §8.4). DB 적재 시 좌표·동코드 누락분이 빠져 ±5% 허용.
_EXPECT_ARM = {
    ARM_EXCLUDED_VICE: 3403,
    ARM_EXCLUDED_LUXURY: 2404,
    ARM_EXCLUDED_NONCONSUMPTION: 40614,
}
_TOL = 0.05


def find_commerce_csv(explicit: str | None) -> Path | None:
    """원천 상가 CSV 경로. 명시값 우선, 없으면 data/neo4j_load/pois/ 에서 자동 탐색."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    if not _POIS_DIR.is_dir():
        return None
    cands = [p for p in sorted(_POIS_DIR.glob("*.csv"))
             if p.stem not in ("residence", "workplace")]
    # '소상공인'이 들어간 파일을 우선, 없으면 가장 큰 CSV
    for p in cands:
        if "소상공인" in p.name:
            return p
    return max(cands, key=lambda p: p.stat().st_size) if cands else None


def load_upjong_index(csv_path: Path) -> tuple[dict[str, str], int]:
    """원천 상가 CSV → ({POI id: L3코드}, 읽은 행수). POI id 규약은 03_pois.py 와 동일."""
    idx: dict[str, str] = {}
    n_rows = 0
    for enc in ("utf-8-sig", "cp949", "euc-kr"):
        idx.clear()
        n_rows = 0
        try:
            with io.open(csv_path, encoding=enc, newline="") as f:
                rd = csv.reader(f)
                next(rd, None)                      # 헤더
                for row in rd:
                    n_rows += 1
                    if len(row) <= C_UPJONG_COL:
                        continue
                    sid = (row[C_ID_COL] or "").strip()
                    code = (row[C_UPJONG_COL] or "").strip().upper()
                    if sid and code:
                        idx[f"C_{sid}"] = code
            return idx, n_rows
        except (UnicodeDecodeError, UnicodeError):
            continue
    return {}, 0


def load_industry_map() -> tuple[dict, dict]:
    """(l1_default, sub_override) 반환."""
    raw = json.loads(_MAP_PATH.read_text(encoding="utf-8"))
    return raw.get("l1_default", {}), raw.get("sub_override", {})


def kdi_category(l1: str | None, sub: str | None, l1_default: dict, sub_override: dict) -> str | None:
    """적립 POI의 KDI 8분류. sub_override 가 l1_default 보다 우선. 미매핑이면 None."""
    s = (sub or "").strip()
    if s in sub_override:
        return sub_override[s]
    return l1_default.get((l1 or "").strip())


FETCH = """
MATCH (p:POI {type:'commerce'})-[:IN_CATEGORY]->(c:Category)
RETURN p.id AS id, p.name AS name, c.name AS sub, c.parent AS l1
"""
UPDATE = """
UNWIND $rows AS r
MATCH (p:POI {id: r.id})
SET p.sangsaeng_eligible = r.el,
    p.sangsaeng_arm = r.arm,
    p.sangsaeng_kdi = r.kdi,
    p.sangsaeng_src = r.src,
    p.upjong_l3 = r.code
"""
BATCH = 20_000

_ARM_ORDER = (ARM_ELIGIBLE, ARM_EXCLUDED_VICE, ARM_EXCLUDED_LUXURY,
              ARM_EXCLUDED_NONCONSUMPTION, ARM_EXCLUDED_OTHER)


def check_csv(csv_path: Path) -> int:
    """CSV만 읽어 컬럼·인코딩·코드 분포 검증 (DB 불필요). 반환값 = 실패 건수."""
    idx, n_rows = load_upjong_index(csv_path)
    print(f"원천 CSV: {csv_path}")
    print(f"  행수 {n_rows:,} / 인덱스 {len(idx):,}건 "
          f"(id 규약 C_{{상가아이디}}, 컬럼 {C_ID_COL}·{C_UPJONG_COL})")
    if not idx:
        print("  ✘ 인덱스가 비었다 — 컬럼 인덱스 또는 인코딩 확인 (03_pois.py C_COL 과 대조)")
        return 1

    dist = Counter(idx.values())
    print(f"  고유 L3 코드 {len(dist)}종\n")
    print("  코드별 건수 (기대 ±3%):")
    bad = 0
    for code, exp in _EXPECT_CSV.items():
        got = dist.get(code, 0)
        ok = abs(got - exp) <= exp * 0.03
        if not ok:
            bad += 1
        print(f"    {'✔' if ok else '✘'} {code} = {got:>7,}  (기대 {exp:,})")
    if bad:
        print("\n  ✘ 기대 범위를 벗어났다 — 컬럼 인덱스나 인코딩 문제일 가능성 (§7.1로 복귀)")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="판정만 하고 DB에 쓰지 않음")
    ap.add_argument("--check-csv", action="store_true",
                    help="원천 CSV만 검증하고 종료 (DB 불필요)")
    ap.add_argument("--commerce-csv", default=None,
                    help="원천 상가 CSV 경로 (기본: data/neo4j_load/pois/ 자동탐색)")
    args = ap.parse_args()

    csv_path = find_commerce_csv(args.commerce_csv)

    if args.check_csv:
        if not csv_path:
            print(f"✘ 원천 CSV를 찾지 못했다 ({_POIS_DIR}). --commerce-csv 로 경로 지정.")
            sys.exit(1)
        sys.exit(1 if check_csv(csv_path) else 0)

    # ── 업종코드 인덱스 ──
    if csv_path:
        upjong_idx, n_rows = load_upjong_index(csv_path)
        print(f"업종코드 인덱스: {len(upjong_idx):,}건 (원천 {csv_path.name}, {n_rows:,}행)")
    else:
        upjong_idx = {}
        print(f"⚠ 원천 CSV 없음({_POIS_DIR}) — 코드 조인 없이 상호명·L2 룰로만 판정한다.")
        print("  이 경우 유흥주점·복권이 적립으로 남아 C1 대조군이 성립하지 않는다.")
        print("  --commerce-csv 로 경로를 지정할 것.")

    l1_default, sub_override = load_industry_map()
    print(f"industry_map 로드: L1 {len(l1_default)}종 / sub_override {len(sub_override)}종\n")

    from _common import driver_session  # noqa: E402  (DB 필요한 경로에서만 import)

    arms: Counter = Counter()
    srcs: Counter = Counter()
    kdi_dist: Counter = Counter()
    excluded_by_sub: Counter = Counter()
    rows, total, n_coded = [], 0, 0

    with driver_session() as s:
        for rec in s.run(FETCH):
            total += 1
            code = upjong_idx.get(rec["id"])
            if code:
                n_coded += 1
            arm, why = sangsaeng_arm(rec["name"], rec["sub"], rec["l1"], upjong_l3=code)
            el = (arm == ARM_ELIGIBLE)
            kdi = kdi_category(rec["l1"], rec["sub"], l1_default, sub_override) if el else None
            src = src_of(why)
            arms[arm] += 1
            srcs[src] += 1
            if el:
                kdi_dist[kdi or "(미매핑)"] += 1
            else:
                excluded_by_sub[f"{rec['l1']}/{rec['sub']}"] += 1
            rows.append({"id": rec["id"], "el": el, "arm": arm,
                         "kdi": kdi, "src": src, "code": code})

        cov = n_coded / max(total, 1)
        print(f"commerce POI {total:,}개 판정 완료")
        print(f"  코드 커버리지: {n_coded:,} / {total:,} ({cov*100:.1f}%)"
              f"  {'✔' if cov >= 0.95 else '✘ 목표 95% 미만 — CSV와 DB 적재본 세대 불일치 의심'}")
        print(f"  src 분포: {dict(srcs)}")
        n_el = arms.get(ARM_ELIGIBLE, 0)
        print(f"  적립 비율: {100 * n_el / max(total, 1):.2f}%  (상생=네거티브, 대부분 적립이 정상)")
        print("  arm 분포:")
        for a in _ARM_ORDER:
            print(f"    {a:24s} {arms.get(a, 0):>8,}")
        print("  KDI 8분류 분포:", dict(kdi_dist))
        print("  제외 상위 (L1/sub):", excluded_by_sub.most_common(10))

        print("\n  기대값 대조 (±5%):")
        bad = 0
        for a, exp in _EXPECT_ARM.items():
            got = arms.get(a, 0)
            ok = abs(got - exp) <= exp * _TOL
            if not ok:
                bad += 1
            print(f"    {'✔' if ok else '✘'} {a:24s} {got:>8,}  (기대 {exp:,})")
        if bad:
            print("  ✘ 기대 범위 이탈 — 로직이 아니라 CSV↔DB 조인 문제일 가능성이 높다")
            print("    (로직은 sangsaeng_eligibility.py --audit 로 이미 검증됨)")

        if args.dry_run:
            print("\n[dry-run] DB 미기록")
            return
        for i in range(0, len(rows), BATCH):
            s.run(UPDATE, rows=rows[i:i + BATCH])
        print(f"\n→ p.sangsaeng_eligible / arm / kdi / src / upjong_l3 기록 완료 ({len(rows):,}개)")


if __name__ == "__main__":
    main()
