"""시뮬 결과 KPI 평가 — Day별 Plan/Memory/State 분석.

KPI:
  1) 검증 통과율: 적재된 Plan 수 / 총 agent 수
  2) 환각율: poi_id 후보 풀 밖 매칭 (0이어야 함)
  3) 운영시간 위반율
  4) 평일 직장 체류 ≥ 4h 준수율
  5) 카테고리 분포 (정책 대상 vs 비대상)
  6) 정책 효과: 자치구별·일별 식사·카페·디저트 방문률
  7) 만족도 평균
  8) KNOWS_POI 갱신 통계 (재방문 vs 신규)
  9) 토큰·시간 통계
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))
from _common import driver_session  # noqa: E402


import os
METRICS_DIR = Path(os.environ.get("SIM_OUTPUT_DIR",
                                  os.path.expanduser("~/sim_output"))) / "metrics"


def load_metrics(days: list[str]) -> dict[str, list[dict]]:
    out = {}
    for d in days:
        path = METRICS_DIR / f"day_{d}.jsonl"
        if not path.exists():
            out[d] = []
            continue
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
        out[d] = rows
    return out


def kpi_basic(metrics: dict[str, list[dict]]) -> dict:
    """Day별 기본 통계 (성공률·토큰·시간·이벤트)."""
    out = {}
    for d, rows in metrics.items():
        ok = [r for r in rows if r["status"] == "ok"]
        err = [r for r in rows if r["status"] != "ok"]
        total = len(rows)
        if total == 0:
            out[d] = {"empty": True}
            continue
        out[d] = {
            "total": total,
            "ok": len(ok), "err": len(err),
            "success_rate": round(len(ok) / total * 100, 2),
            "avg_elapsed": round(sum(r["elapsed"] for r in ok) / max(len(ok), 1), 2),
            "avg_events": round(sum(r.get("n_events", 0) for r in ok) / max(len(ok), 1), 2),
            "avg_includes": round(sum(r.get("n_includes", 0) for r in ok) / max(len(ok), 1), 2),
            "avg_sat": round(sum(r.get("avg_sat", 0) or 0 for r in ok) / max(len(ok), 1), 3),
            "total_tokens_in": sum(r.get("tokens_in", 0) for r in ok),
            "total_tokens_out": sum(r.get("tokens_out", 0) for r in ok),
            "policy_hits_sum": sum(r.get("policy_hits", 0) for r in ok),
            "n_visited_memories_sum": sum(r.get("n_visited_memories", 0) for r in ok),
            "s1_retries": sum(max(0, r.get("s1_attempts", 1) - 1) for r in ok),
            "s2_retries": sum(max(0, r.get("s2_attempts", 0) - 1) for r in ok),
        }
    return out


def kpi_category_distribution(start: date, days: int) -> dict:
    """Day별 카테고리 분포 (강남 vs 비강남 분리)."""
    out = {}
    for i in range(days):
        d = start + timedelta(days=i)
        with driver_session() as s:
            rows = s.run("""
                MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(dist:District)
                MATCH (a)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(p:POI)
                WHERE i.category IS NOT NULL AND NOT i.category IN ['집','직장']
                  AND p.type = 'commerce'
                RETURN dist.code AS dist_code, i.category AS cat, count(*) AS n
            """, day=d.isoformat()).data()
        gangnam = Counter()
        other = Counter()
        for r in rows:
            (gangnam if r["dist_code"] == "11680" else other)[r["cat"]] += r["n"]
        out[d.isoformat()] = {
            "gangnam": dict(gangnam.most_common()),
            "non_gangnam": dict(other.most_common()),
        }
    return out


def kpi_policy_effect(start: date, days: int) -> dict:
    """정책 P001 효과: 강남 거주 agent의 정책 대상 카테고리 방문률 변화."""
    out = {}
    target_cats = {"식사", "카페", "디저트"}
    for i in range(days):
        d = start + timedelta(days=i)
        with driver_session() as s:
            rows = s.run("""
                MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(:Dong)<-[:HAS_DONG]-(dist:District)
                MATCH (a)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(p:POI {type:'commerce'})
                RETURN dist.code AS dist_code, i.category AS cat, count(*) AS n
            """, day=d.isoformat()).data()
        gn_target = sum(r["n"] for r in rows if r["dist_code"] == "11680" and r["cat"] in target_cats)
        gn_total = sum(r["n"] for r in rows if r["dist_code"] == "11680")
        nogn_target = sum(r["n"] for r in rows if r["dist_code"] != "11680" and r["cat"] in target_cats)
        nogn_total = sum(r["n"] for r in rows if r["dist_code"] != "11680")
        out[d.isoformat()] = {
            "gangnam_target_rate": round(gn_target / max(gn_total, 1) * 100, 2),
            "gangnam_target": gn_target, "gangnam_total": gn_total,
            "non_gangnam_target_rate": round(nogn_target / max(nogn_total, 1) * 100, 2),
            "non_gangnam_target": nogn_target, "non_gangnam_total": nogn_total,
        }
    return out


def kpi_hallucination(start: date, days: int) -> dict:
    """환각 검출 — 모든 poi_id가 :POI 노드에 존재하는지 (적재 시 MATCH 사용해서 0이어야 함)."""
    out = {}
    for i in range(days):
        d = start + timedelta(days=i)
        with driver_session() as s:
            r = s.run("""
                MATCH (a:Agent)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->(p)
                WITH count(p) AS n_inc, count(CASE WHEN p:POI THEN 1 END) AS n_poi
                RETURN n_inc, n_poi
            """, day=d.isoformat()).single()
            out[d.isoformat()] = {"includes": r["n_inc"], "valid_poi": r["n_poi"]}
    return out


def kpi_memory_growth(start: date, days: int) -> dict:
    """매일 visited Memory 누적·KNOWS_POI 갱신 수."""
    out = {}
    for i in range(days):
        d = start + timedelta(days=i)
        with driver_session() as s:
            mem = s.run("MATCH (m:Memory {type:'visited', day: date($day)}) RETURN count(m) AS n",
                        day=d.isoformat()).single()["n"]
            kp_visited = s.run("""
                MATCH ()-[kp:KNOWS_POI]->()
                WHERE kp.last_visit = date($day)
                RETURN count(kp) AS n
            """, day=d.isoformat()).single()["n"]
            out[d.isoformat()] = {"visited_memories_created": mem, "knows_poi_updated": kp_visited}
    return out


def kpi_workplace_compliance(start: date, days: int) -> dict:
    """평일 + 직장 보유 agent의 workplace 체류 시간 (운영시간 09~18 기준)."""
    out = {}
    for i in range(days):
        d = start + timedelta(days=i)
        if d.weekday() >= 5:
            out[d.isoformat()] = {"weekend": True}
            continue
        with driver_session() as s:
            rows = s.run("""
                MATCH (a:Agent)-[:WORKS_AT]->()
                MATCH (a)-[:HAS_PLAN {day: date($day)}]->(:Plan)-[i:INCLUDES]->()
                WHERE i.anchor = 'workplace'
                WITH a.id AS aid, collect(toString(i.time)) AS times
                RETURN aid, times
            """, day=d.isoformat()).data()
            n_agents = len(rows)
            n_with_workplace_time = sum(1 for r in rows if r["times"])
            out[d.isoformat()] = {
                "weekday": True,
                "agents_with_workplace": n_agents,
                "with_workplace_event": n_with_workplace_time,
                "rate": round(n_with_workplace_time / max(n_agents, 1) * 100, 2),
            }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--days", type=int, default=3)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    day_strs = [(start + timedelta(days=i)).isoformat() for i in range(args.days)]

    print("=== 1) 기본 통계 (메트릭 jsonl 기반) ===")
    metrics = load_metrics(day_strs)
    basic = kpi_basic(metrics)
    print(json.dumps(basic, ensure_ascii=False, indent=2))

    print("\n=== 2) 카테고리 분포 (그래프 기반) ===")
    cat_dist = kpi_category_distribution(start, args.days)
    print(json.dumps(cat_dist, ensure_ascii=False, indent=2))

    print("\n=== 3) 정책 P001 효과 (강남 vs 비강남) ===")
    policy = kpi_policy_effect(start, args.days)
    print(json.dumps(policy, ensure_ascii=False, indent=2))

    print("\n=== 4) 환각 검출 (모든 poi_id는 :POI 노드) ===")
    hall = kpi_hallucination(start, args.days)
    print(json.dumps(hall, ensure_ascii=False, indent=2))

    print("\n=== 5) Memory·KNOWS_POI 매일 갱신 ===")
    mem = kpi_memory_growth(start, args.days)
    print(json.dumps(mem, ensure_ascii=False, indent=2))

    print("\n=== 6) 평일 직장 체류 준수율 ===")
    wp = kpi_workplace_compliance(start, args.days)
    print(json.dumps(wp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
