# -*- coding: utf-8 -*-
"""상생 백테스트 기간 집계 (G4) — 관측창 전체의 에이전트별 적립/제외 소비 산출.

한 런(OFF 또는 ON)의 관측창(전1주 또는 후1주) INCLUDES를 POI 라벨(sangsaeng_eligible/
sangsaeng_arm/sangsaeng_kdi, 11_sangsaeng_eligibility.py 백필)과 조인해 에이전트별로:

  total_spent            : 전체 actual_spent
  eligible_spent         : 적립업종(sangsaeng_eligible=true) — C1의 '적립' arm ≈ 사실상 총소비
  excluded_luxury_spent  : 시계·귀금속(명품 proxy) — C1의 깨끗한 '제외' arm
  excluded_other_spent   : 유흥·사행·대형·비소비 (오염 대조군, 참고)
  by_kdi                 : 적립업종의 KDI 8분류별 소비 (D1/D2용)
  n_events, n_policy_trigger : trigger='policy' 자기보고 건수 (§4.5 ③ 반응 로그)
  spend_decile           : 소비 10분위 (D2 소득 재배분용)

OFF/ON 각각 1회 실행해 off.json / on.json 으로 저장한 뒤 validate_sangsaeng.py 가
에이전트별 B−A 쌍체차를 계산한다(§5.4). A/B런은 같은 관측창이라 같은 --start/--days.

사용:
  python scripts/sim/aggregate_period.py --start 2026-07-13 --days 7 --tag off --out off.json
  python scripts/sim/aggregate_period.py --start 2026-07-20 --days 7 --tag on  --out on.json
  python scripts/sim/aggregate_period.py --selftest        # DB 없이 집계 로직 검증
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ARM_LUXURY = "excluded_luxury"


def _blank_agent() -> dict:
    return {
        "total_spent": 0,
        "eligible_spent": 0,
        "excluded_luxury_spent": 0,
        "excluded_other_spent": 0,
        "by_kdi": defaultdict(int),
        "n_events": 0,
        "n_policy_trigger": 0,
        "spend_decile": None,
    }


def aggregate_rows(rows) -> dict[str, dict]:
    """순수 집계 — INCLUDES 행 리스트 → {aid: 집계}. DB 없이 단위테스트 가능.

    각 row 는 dict/Record 형태로 다음 키를 가진다:
      aid, decile, elig(bool), arm(str), kdi(str|None), spent(int|float), trigger(str|None)
    """
    agg: dict[str, dict] = {}
    for r in rows:
        aid = r["aid"]
        a = agg.get(aid)
        if a is None:
            a = _blank_agent()
            agg[aid] = a
        if a["spend_decile"] is None and r.get("decile") is not None:
            a["spend_decile"] = int(r["decile"])
        spent = int(r.get("spent") or 0)
        a["total_spent"] += spent
        a["n_events"] += 1
        if (r.get("trigger") or "") == "policy":
            a["n_policy_trigger"] += 1
        if r.get("elig") is True:
            a["eligible_spent"] += spent
            kdi = r.get("kdi")
            if kdi:
                a["by_kdi"][kdi] += spent
        else:
            if (r.get("arm") or "") == ARM_LUXURY:
                a["excluded_luxury_spent"] += spent
            else:
                a["excluded_other_spent"] += spent
    # defaultdict → dict (JSON 직렬화용)
    for a in agg.values():
        a["by_kdi"] = dict(a["by_kdi"])
    return agg


FETCH = """
MATCH (a:Agent)-[hp:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
WHERE date($start) <= hp.day AND hp.day <= date($end)
RETURN a.id AS aid,
       a.spending_level_wd AS decile,
       p.sangsaeng_eligible AS elig,
       p.sangsaeng_arm AS arm,
       p.sangsaeng_kdi AS kdi,
       coalesce(i.actual_spent, 0) AS spent,
       i.trigger AS trigger
"""


def fetch_rows(start: date, end: date) -> list[dict]:
    """Neo4j에서 관측창 INCLUDES 행 수집 (driver 필요)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
    from _common import driver_session  # noqa: E402
    with driver_session() as s:
        return [dict(r) for r in s.run(FETCH, start=start.isoformat(), end=end.isoformat())]


def build_payload(rows, start: date, end: date, tag: str) -> dict:
    agents = aggregate_rows(rows)
    return {
        "meta": {
            "tag": tag,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_agents": len(agents),
            "n_rows": len(rows) if hasattr(rows, "__len__") else None,
        },
        "agents": agents,
    }


def _selftest() -> None:
    rows = [
        # agent A1 (decile 3): 적립 식사 10000 + 적립 가전 50000 + 제외명품 30000 + 유흥 8000
        {"aid": "A1", "decile": 3, "elig": True, "arm": "eligible", "kdi": "요식", "spent": 10000, "trigger": "policy"},
        {"aid": "A1", "decile": 3, "elig": True, "arm": "eligible", "kdi": "가전·가구", "spent": 50000, "trigger": "lifestyle"},
        {"aid": "A1", "decile": 3, "elig": False, "arm": "excluded_luxury", "kdi": None, "spent": 30000, "trigger": "none"},
        {"aid": "A1", "decile": 3, "elig": False, "arm": "excluded_other", "kdi": None, "spent": 8000, "trigger": "none"},
        # agent A2 (decile 9): 적립 유통 20000 (trigger policy)
        {"aid": "A2", "decile": 9, "elig": True, "arm": "eligible", "kdi": "유통", "spent": 20000, "trigger": "policy"},
    ]
    agg = aggregate_rows(rows)
    a1 = agg["A1"]
    assert a1["total_spent"] == 98000, a1
    assert a1["eligible_spent"] == 60000, a1
    assert a1["excluded_luxury_spent"] == 30000, a1
    assert a1["excluded_other_spent"] == 8000, a1
    assert a1["by_kdi"] == {"요식": 10000, "가전·가구": 50000}, a1
    assert a1["n_events"] == 4 and a1["n_policy_trigger"] == 1, a1
    assert a1["spend_decile"] == 3
    a2 = agg["A2"]
    assert a2["eligible_spent"] == 20000 and a2["by_kdi"] == {"유통": 20000}
    assert a2["n_policy_trigger"] == 1
    # 빈 입력
    assert aggregate_rows([]) == {}
    print("  ✔ aggregate_rows 적립/제외/KDI/trigger 집계 정확")
    print("aggregate_period SELFTEST ALL OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", help="관측창 시작일 YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=7, help="관측창 일수 (기본 7 = 1주)")
    ap.add_argument("--tag", default="run", help="off / on 등 런 태그")
    ap.add_argument("--out", type=Path, help="출력 JSON 경로")
    ap.add_argument("--selftest", action="store_true", help="DB 없이 집계 로직 검증")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.start or not args.out:
        ap.error("--start 와 --out 은 필수 (또는 --selftest)")

    start = date.fromisoformat(args.start)
    end = start + timedelta(days=args.days - 1)
    rows = fetch_rows(start, end)
    payload = build_payload(rows, start, end, args.tag)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    m = payload["meta"]
    print(f"[{args.tag}] {m['start']}~{m['end']} · 에이전트 {m['n_agents']:,}명 · INCLUDES {m['n_rows']:,}건 → {args.out}")


if __name__ == "__main__":
    main()
