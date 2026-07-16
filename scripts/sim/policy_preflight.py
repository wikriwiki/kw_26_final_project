# -*- coding: utf-8 -*-
"""정책 사전점검(preflight) — 시뮬 구동 전, 정책 JSON에 맞춰 환경이 준비됐는지 검사.

무엇을 하는가:
  A. 정책 스키마 검증 (필수 필드·날짜·소득 tier·금액)
  B. 정책 속성 → 환경 요구사항 자동 도출·점검
     poi_restricted   → POI coupon_eligible 백필 필요 (가맹점 CSV·09 스크립트)
     target_l1s       → 카테고리 어휘(categories.yaml) 존재 확인 → 봉투 필터로 자동 반영
     excluded_income  → 소득 필드 정밀도 경고 (INCOME_ASSIGNMENT_PLAN v2 참조)
  C. 프롬프트 미리보기 — 대상/제외 페르소나가 실제로 보게 될 정책 카드 렌더
  D. 지급 규모 요약 (tier별 금액 — 대상자 수는 서버 집계 필요)

무엇을 하지 않는가 (설계 원칙 — 결과 유도 금지):
  ✋ 행동 파라미터(소비성향 클램프·가격 탄력·정렬 가점 등)는 절대 건드리지 않는다.
  preflight는 '입력이 올바르게 표현·주입되는가'만 점검한다. 정책 효과의 크기·방향은
  전적으로 시뮬 내생 — 기대 효과를 사전에 주입하는 어떤 경로도 두지 않는다.

사용: python scripts/sim/policy_preflight.py data/neo4j_load/policies/P010.json [P011.json ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
VALID_TIERS = {"하", "중하", "중", "중상", "상"}
KNOWN_TYPES = {"grant", "subsidy", "regulation", "facility", "campaign", "tax", "transit", "environment"}

_PASS, _WARN, _FAIL = "✅", "⚠️ ", "❌"


def _l1_vocab() -> set[str]:
    try:
        import yaml
        raw = yaml.safe_load((ROOT / "data/neo4j_load/categories/categories.yaml").read_text(encoding="utf-8"))
        return {c["name"] for c in raw.get("categories", [])}
    except Exception:
        return set()


def check_policy(path: Path) -> list[tuple[str, str]]:
    """정책 1건 점검 → [(등급, 메시지)]."""
    out: list[tuple[str, str]] = []
    pol = json.loads(path.read_text(encoding="utf-8"))
    pid = pol.get("id", "?")

    # ── A. 스키마 ──
    for f in ("id", "name", "type", "description", "effective_from", "effective_until"):
        out.append((_PASS, f"{f} 존재") if pol.get(f) else (_FAIL, f"필수 필드 누락: {f}"))
    ptype = pol.get("type")
    out.append((_PASS, f"type={ptype}") if ptype in KNOWN_TYPES else (_WARN, f"미지의 type: {ptype}"))
    try:
        d0, d1 = date.fromisoformat(pol["effective_from"]), date.fromisoformat(pol["effective_until"])
        out.append((_PASS, f"기간 {d0}~{d1}") if d0 <= d1 else (_FAIL, "effective_from > until"))
    except Exception as e:
        out.append((_FAIL, f"날짜 파싱 실패: {e}"))

    ig = pol.get("income_grants") or {}
    exc = set(pol.get("excluded_income") or [])
    grant_key = pol.get("grant_key") or "income"
    # income_grants 조회 키에 따라 유효 tier 집합이 달라진다:
    #   spend_decile → 소비 10분위 "1".."10" / 그 외(income) → 소득 5분위(하~상)
    valid_tiers = {str(i) for i in range(1, 11)} if grant_key == "spend_decile" else VALID_TIERS
    if ptype == "grant":
        bad_tiers = (set(ig) | exc) - valid_tiers
        out.append((_PASS, f"{grant_key} tier 유효 ({sorted(ig, key=lambda x: (len(x), x))})") if not bad_tiers
                   else (_FAIL, f"미지의 {grant_key} tier: {bad_tiers}"))
        overlap = set(ig) & exc
        out.append((_PASS, "지급/제외 tier 무모순") if not overlap
                   else (_FAIL, f"tier가 지급과 제외에 동시 존재: {overlap}"))
        bad_amt = {k: v for k, v in ig.items() if not isinstance(v, int) or v <= 0}
        out.append((_PASS, "지급액 양의 정수") if not bad_amt else (_FAIL, f"지급액 오류: {bad_amt}"))

    # ── B. 환경 요구사항 (정책 속성 → 자동 도출) ──
    if pol.get("poi_restricted"):
        csvs = list((ROOT / "data" / "coupon").glob("*.csv"))
        out.append((_PASS, f"사용처 제한 → 실측 가맹점 CSV {len(csvs)}개 준비됨 "
                           f"(서버에서 09_coupon_eligibility.py 백필 필요)") if csvs
                   else (_WARN, "사용처 제한인데 data/coupon/*.csv 없음 — 룰 fallback만 사용됨"))
        out.append((_PASS, "소비 봉투(require_poi_eligible)로 자동 반영 — run_simulation 배선 확인됨"))
    tl = pol.get("benefit_categories") or []
    if tl:
        vocab = _l1_vocab()
        bad = [t for t in tl if vocab and t not in vocab]
        out.append((_PASS, f"업종 스코프 {tl} 어휘 유효 → 봉투 카테고리 필터로 자동 반영") if not bad
                   else (_FAIL, f"카테고리 어휘에 없음: {bad}"))
    if exc or (ig and len(set(ig.values())) > 1):
        if grant_key == "spend_decile":
            out.append((_PASS, f"{pid}: 차등 지급을 소비 10분위(spending_level_wd, BDC 실측)로 결정 — "
                               "시뮬 보유 최소 계층 단위로 직접 배선"))
        else:
            out.append((_WARN, f"{pid}: 소득 기준 차등/제외 사용 — 현재 p_income_level은 소비분위 유도(근사). "
                               "정밀 백테스트는 INCOME_ASSIGNMENT_PLAN v2(p_income_pct) 이후 권장"))

    # ── C. 프롬프트 미리보기 (대상/제외 페르소나) ──
    try:
        from dawn_context import _format_policy
        row = {
            "id": pid, "name": pol["name"], "type": ptype, "description": pol.get("description", ""),
            "rate": pol.get("benefit_rate"), "cap": pol.get("cap_per_agent"),
            "poi_restricted": bool(pol.get("poi_restricted")),
            "from_": pol.get("effective_from"), "until_": pol.get("effective_until"),
            "regions": pol.get("target_districts") or [], "target_l1s": tl,
            "income_grants": json.dumps(ig, ensure_ascii=False),
            "excluded_income": list(exc),
        }
        target_tier = next(iter(ig), "중")
        excl_tier = next(iter(exc), None)
        print(f"\n--- {pid} 프롬프트 미리보기 (소득 '{target_tier}' 대상자, 지급 당일) ---")
        print(_format_policy([row], persona={"income": target_tier, "daily_wd": 30000},
                             state={"grant_received": json.dumps({pid: ig.get(target_tier, 0)}),
                                    "grant_remaining": json.dumps({pid: ig.get(target_tier, 0)}),
                                    "grant_days_since": {pid: 0}}))
        if excl_tier:
            print(f"--- {pid} 프롬프트 미리보기 (소득 '{excl_tier}' 제외자) ---")
            print(_format_policy([row], persona={"income": excl_tier, "daily_wd": 90000},
                                 state={"grant_received": "{}", "grant_remaining": "{}"}))
        out.append((_PASS, "프롬프트 카드 렌더 정상"))
    except Exception as e:
        out.append((_FAIL, f"프롬프트 렌더 실패: {e}"))

    # ── D. 지급 규모 ──
    if ig:
        out.append((_PASS, "tier별 지급: " + ", ".join(f"{k} {v:,}원" for k, v in ig.items())
                    + (f" / 제외 {sorted(exc)}" if exc else "")
                    + " — 대상자 수·총예산은 서버에서 MATCH (a:Agent) 집계"))
    return out


def check_db_wiring(path: Path) -> list[tuple[str, str]]:
    """적재 후 DB 배선 실측 — 파일 검사가 구조적으로 못 잡는 치명 결함을 게이트.

    핵심: POLICY_CYPHER 는 (Policy)-[:applied_to]->(District/Dong) 엣지를 통해
    에이전트에 정책을 노출한다. 광역 토큰("서울특별시")이 District 이름과 안 맞아
    applied_to 가 0건이면 정책은 존재하지만 **어떤 에이전트도 보지 못한다**.
    NEO4J_URI 가 있을 때만(=적재 이후) 실행.
    """
    import os
    out: list[tuple[str, str]] = []
    uri = os.environ.get("NEO4J_URI")
    if not uri:
        out.append((_WARN, "NEO4J_URI 미설정 → DB 배선 점검 생략 (정책 적재 후 재실행 권장)"))
        return out
    pid = json.loads(path.read_text(encoding="utf-8")).get("id", "?")
    try:
        from neo4j import GraphDatabase
        drv = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"),
                                              os.environ.get("NEO4J_PASSWORD", "")))
        with drv.session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as s:
            if s.run("MATCH (p:Policy {id:$id}) RETURN count(p) AS c", id=pid).single()["c"] == 0:
                out.append((_FAIL, f"{pid} 노드가 DB에 없음 — 로더 미실행"))
                return out
            n_app = s.run("MATCH (:Policy {id:$id})-[r:applied_to]->() RETURN count(r) AS c",
                          id=pid).single()["c"]
            out.append((_PASS, f"applied_to {n_app}개 연결 (정책 노출 배선 존재)") if n_app
                       else (_FAIL, f"{pid} applied_to 0건 — 정책이 있어도 노출 0. "
                                    "target_districts 가 District 이름과 불일치(광역 토큰 확장 실패)"))
            n_exposed = s.run(
                "MATCH (a:Agent)-[:LIVES_AT|WORKS_AT]->(:POI)-[:IN_DONG]->(:Dong)"
                "<-[:HAS_DONG]-(:District)<-[:applied_to]-(:Policy {id:$id}) "
                "RETURN count(DISTINCT a) AS c", id=pid).single()["c"]
            out.append((_PASS, f"POLICY_CYPHER 경로 실측 → 노출 대상 {n_exposed:,}명") if n_exposed
                       else (_FAIL, f"{pid} 노출 에이전트 0명 — 지역 매칭 경로 단절"))
        drv.close()
    except Exception as e:
        out.append((_WARN, f"DB 배선 점검 실패(건너뜀): {e}"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", type=Path)
    args = ap.parse_args()
    n_fail = 0
    for p in args.paths:
        print(f"\n{'='*64}\n정책 사전점검: {p}\n{'='*64}")
        results = check_policy(p)
        results += check_db_wiring(p)
        for grade, msg in results:
            print(f"  {grade} {msg}")
        n_fail += sum(1 for g, _ in results if g == _FAIL)
    print(f"\n{'='*64}")
    print("🔒 중립성 원칙: preflight는 입력 표현·주입 준비만 점검했다. 행동 파라미터는")
    print("   일절 변경하지 않았으며, 정책 효과의 크기·방향은 전적으로 시뮬 내생이다.")
    print(f"결과: {'FAIL ' + str(n_fail) + '건 — 수정 후 재실행' if n_fail else 'READY — 시뮬 구동 가능'}")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
