"""
load_fusion_to_neo4j.py — A+LLM 봉합본 페르소나의 NVIDIA 풍부 필드를
Neo4j 의 :Agent 노드에 적재한다.

기존 Agent 노드는 BDC 통계로 만들어진 정량 필드만 보유.
이 스크립트는 봉합본 jsonl 에서 다음 필드만 추출해 Agent 노드에 머지:

- personality_lifestyle_raw  : 봉합된 200자 fused lifestyle (기존 BDC lifestyle 덮어쓰기)
- nvidia_summary             : 1줄 요약 (한 줄)
- nvidia_hobbies             : 취미 JSON 배열 (string으로 직렬화)
- nvidia_cultural_background : 문화·배경 설명
- nvidia_education_level     : 학력 라벨
- nvidia_marital_status      : 혼인 라벨
- nvidia_family_type         : 가족 구성 라벨
- nvidia_career_goals        : 커리어 목표·야망
- nvidia_skills              : 전문성·기술 요약
- persona_uuid               : NVIDIA Nemotron 원본 uuid (감사 추적용)
- match_level                : gu_sex_age / sex_age / sex / any
- llm_audited / llm_reconciled / llm_consistent : 봉합 메타

봉합본 jsonl 와 Neo4j Agent 의 agent_id 매칭. 매칭 안 되는 jsonl 행은 skip.

사용:
  python scripts/persona/load_fusion_to_neo4j.py \\
      --jsonl output/personas/full/A_rank_coupling_full_llm.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "neo4j_load"))
from _common import driver_session, bulk_run  # noqa: E402


UPDATE_CYPHER = """
UNWIND $batch AS r
MATCH (a:Agent {id: r.agent_id})
SET a.personality_lifestyle_raw    = r.lifestyle,
    a.nvidia_summary               = r.nvidia_summary,
    a.nvidia_hobbies               = r.nvidia_hobbies,
    a.nvidia_cultural_background   = r.nvidia_cultural_background,
    a.nvidia_education_level       = r.nvidia_education_level,
    a.nvidia_marital_status        = r.nvidia_marital_status,
    a.nvidia_family_type           = r.nvidia_family_type,
    a.nvidia_career_goals          = r.nvidia_career_goals,
    a.nvidia_skills                = r.nvidia_skills,
    a.persona_uuid                 = r.persona_uuid,
    a.match_level                  = r.match_level,
    a.llm_audited                  = r.llm_audited,
    a.llm_consistent               = r.llm_consistent,
    a.llm_reconciled               = r.llm_reconciled
"""


def _project_row(p: dict) -> dict:
    nv = p.get("nvidia_persona") or {}
    nr = p.get("nvidia_reserved") or {}
    m = p.get("_match") or {}
    return {
        "agent_id":                   p["agent_id"],
        "lifestyle":                  (p.get("personality", {}).get("lifestyle") or "")[:300],
        "nvidia_summary":             (nv.get("summary") or "")[:300],
        "nvidia_hobbies":             json.dumps(nv.get("hobbies") or [], ensure_ascii=False),
        "nvidia_cultural_background": (nv.get("cultural_background") or "")[:300],
        "nvidia_education_level":     nv.get("education_level") or "",
        "nvidia_marital_status":      nv.get("marital_status") or "",
        "nvidia_family_type":         nv.get("family_type") or "",
        "nvidia_career_goals":        (nr.get("career_goals_and_ambitions") or "")[:300],
        "nvidia_skills":              (nr.get("skills_and_expertise") or "")[:300],
        "persona_uuid":               m.get("nvidia_uuid") or "",
        "match_level":                m.get("match_level") or "",
        "llm_audited":                bool(m.get("llm_audited")),
        "llm_consistent":             bool(m.get("llm_consistent", True)),
        "llm_reconciled":             bool(m.get("llm_reconciled")),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path,
                    default=PROJECT_ROOT / "output" / "personas" / "full" / "A_rank_coupling_full_llm.jsonl")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    if not args.jsonl.exists():
        print(f"[!] 파일 없음: {args.jsonl}", file=sys.stderr)
        return 2

    rows: list[dict] = []
    with args.jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(_project_row(json.loads(line)))
    print(f"[load] jsonl {len(rows):,} 행")

    with driver_session() as s:
        # 매칭 가능 agent 수 사전 점검
        ids = [r["agent_id"] for r in rows]
        sample_size = min(2000, len(ids))
        found = s.run(
            "UNWIND $ids AS aid MATCH (a:Agent {id: aid}) RETURN count(a) AS n",
            ids=ids[:sample_size]
        ).single()["n"]
        match_pct = found / sample_size * 100
        print(f"[check] 샘플 {sample_size:,}건 중 매칭 {found:,} ({match_pct:.1f}%)")

        # 실제 적재
        print(f"[load] batch_size={args.batch_size} 적용 중 ...")
        bulk_run(s, UPDATE_CYPHER, rows, batch_size=args.batch_size)

        # 사후 검증
        after = s.run("""
            MATCH (a:Agent)
            RETURN count(a) AS total,
                   sum(CASE WHEN a.nvidia_summary IS NOT NULL THEN 1 ELSE 0 END) AS with_nvidia,
                   sum(CASE WHEN a.llm_reconciled THEN 1 ELSE 0 END) AS reconciled
        """).single()
        print(f"[done] Agent 총 {after['total']:,} / NVIDIA 적재 {after['with_nvidia']:,} / 봉합 {after['reconciled']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
