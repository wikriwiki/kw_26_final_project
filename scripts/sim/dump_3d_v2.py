"""3일 v2 baseline 시뮬 데이터 dump.

Neo4j 시뮬 관련 노드·엣지를 JSONL로 export.
출력: output/sim/dumps/3d_baseline_v2/
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "neo4j_load"))
from _common import driver_session  # noqa: E402

DAYS = ["2026-05-01", "2026-05-02", "2026-05-03"]
OUT_DIR = Path("G:/내 드라이브/Kw/final_project/output/sim/dumps/3d_baseline_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _serialize(v):
    """Neo4j 타입을 JSON 직렬화 가능하게."""
    if hasattr(v, "to_native"):
        v = v.to_native()
    if hasattr(v, "isoformat"):
        return v.isoformat()
    return v


def _node_to_dict(node) -> dict:
    return {k: _serialize(v) for k, v in dict(node).items()}


def dump_nodes(s, label: str, where: str = "", out_name: str = None) -> int:
    out_name = out_name or f"{label.lower()}.jsonl"
    out_path = OUT_DIR / out_name
    q = f"MATCH (n:{label}) {where} RETURN n"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in s.run(q):
            f.write(json.dumps(_node_to_dict(r["n"]), ensure_ascii=False, default=str) + "\n")
            n += 1
    print(f"  {label}: {n:,} → {out_name}")
    return n


def dump_includes(s) -> int:
    """Plan-INCLUDES-POI relation (시뮬 day만)."""
    out_path = OUT_DIR / "includes.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for day in DAYS:
            q = f'''
                MATCH (p:Plan {{day: date("{day}")}})-[r:INCLUDES]->(poi:POI)
                RETURN p.agent_id AS agent_id, p.day AS day, poi.id AS poi_id, properties(r) AS props
            '''
            for row in s.run(q):
                rec = {
                    "agent_id": row["agent_id"],
                    "day": str(row["day"]),
                    "poi_id": row["poi_id"],
                    **{k: _serialize(v) for k, v in (row["props"] or {}).items()},
                }
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                n += 1
    print(f"  INCLUDES (Plan→POI): {n:,} → includes.jsonl")
    return n


def dump_conversation(s) -> int:
    """Conversation (시뮬 day만, 모든 properties)."""
    out_path = OUT_DIR / "conversation.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for day in DAYS:
            for r in s.run(f'MATCH (c:Conversation {{day: date("{day}")}}) RETURN c'):
                f.write(json.dumps(_node_to_dict(r["c"]), ensure_ascii=False, default=str) + "\n")
                n += 1
    print(f"  Conversation: {n:,} → conversation.jsonl")
    return n


def dump_agent_summary(s) -> int:
    """Agent 페르소나 메타 (분석용 — 모든 필드)."""
    out_path = OUT_DIR / "agents.jsonl"
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in s.run("MATCH (a:Agent) RETURN a"):
            f.write(json.dumps(_node_to_dict(r["a"]), ensure_ascii=False, default=str) + "\n")
            n += 1
    print(f"  Agent: {n:,} → agents.jsonl")
    return n


def dump_poi_referenced(s) -> int:
    """시뮬 plan에서 참조된 POI만."""
    out_path = OUT_DIR / "pois_referenced.jsonl"
    n = 0
    referenced = set()
    for day in DAYS:
        for r in s.run(f'MATCH (:Plan {{day: date("{day}")}})-[:INCLUDES]->(poi:POI) RETURN DISTINCT poi.id AS id'):
            referenced.add(r["id"])
    with open(out_path, "w", encoding="utf-8") as f:
        for r in s.run('MATCH (poi:POI) WHERE poi.id IN $ids RETURN poi', ids=list(referenced)):
            f.write(json.dumps(_node_to_dict(r["poi"]), ensure_ascii=False, default=str) + "\n")
            n += 1
    print(f"  POI (referenced): {n:,} → pois_referenced.jsonl")
    return n


def main():
    t0 = time.time()
    print(f"\n=== 3D v2 baseline dump 시작 → {OUT_DIR} ===\n")
    counts = {}
    with driver_session() as s:
        print("[1] 시뮬 결과 노드")
        counts["plan"] = dump_nodes(s, "Plan", where='WHERE p.day IN [date("2026-05-01"), date("2026-05-02"), date("2026-05-03")]'.replace("p.day", "n.day"))
        counts["state"] = dump_nodes(s, "State", where='WHERE n.day IN [date("2026-05-01"), date("2026-05-02"), date("2026-05-03")]')
        counts["conversation"] = dump_conversation(s)
        counts["memory"] = dump_nodes(s, "Memory")
        print()
        print("[2] 시뮬 결과 엣지")
        counts["includes"] = dump_includes(s)
        print()
        print("[3] 메타 (Agent + 참조 POI)")
        counts["agents"] = dump_agent_summary(s)
        counts["pois"] = dump_poi_referenced(s)

    # 메타 정보
    meta = {
        "sim_name": "3d_baseline_v2",
        "description": "7,500 agent × 3일 무정책 baseline (memory zero + SYSTEM_S2 v2)",
        "sim_dates": DAYS,
        "sim_weekdays": ["Friday", "Saturday", "Sunday"],
        "model": "LGAI-EXAONE/EXAONE-4.0-32B-AWQ",
        "vllm_version": "0.11",
        "workers": 32,
        "max_model_len": 8192,
        "dump_generated_at": datetime.now().isoformat(),
        "node_counts": counts,
        "files": {
            "plan.jsonl": "Plan 노드 (agent별 day plan)",
            "state.jsonl": "State 노드 (day별 agent 상태: balance, fatigue, yesterday_satisfaction 등)",
            "conversation.jsonl": "Conversation (Night phase 매칭 + LLM 분류 결과)",
            "memory.jsonl": "Memory 노드 (issue/recommendation 기반)",
            "includes.jsonl": "Plan→POI 외출 picks (pick_factor, pick_reason, actual_spent 등)",
            "agents.jsonl": "Agent 페르소나 메타 (전체 필드)",
            "pois_referenced.jsonl": "시뮬에서 실제 visit된 POI 메타",
        },
        "known_data_issues": [
            "workplace_dong_code_raw 0% 충원 — 15,000 Agent 모두 누락",
            "KNOWS 엣지 0개 — 06_social.py 미실행으로 사회 관계 시드 부재",
            "review_lookup_count metric 누락 — run_simulation.py:298 dict에 키 빠짐",
            "resume retry로 jsonl 일부 중복 (Day 2 status=ok 7,701건, limit 7,500 초과)",
        ],
    }
    with open(OUT_DIR / "dump_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print()
    print(f"  dump_meta.json (시뮬 메타 + 알려진 이슈)")
    print(f"\n=== 완료 ({time.time() - t0:.1f}s) ===")

    # 파일 크기 요약
    print("\n=== 파일 크기 ===")
    total = 0
    for f in sorted(OUT_DIR.glob("*")):
        sz = f.stat().st_size
        total += sz
        print(f"  {f.name:30s} {sz/1024/1024:>8.2f} MB")
    print(f"  {'합계':30s} {total/1024/1024:>8.2f} MB")


if __name__ == "__main__":
    main()
