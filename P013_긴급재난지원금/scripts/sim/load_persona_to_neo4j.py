"""페르소나 jsonl → Neo4j Agent 노드 업데이트.

build_rank_coupling --summarize 결과 (5줄 lifestyle 봉합본)을
Agent.personality_lifestyle_raw에 저장.

기존 1줄 lifestyle은 personality_lifestyle_raw_v1로 백업.
NVIDIA uuid도 nvidia_uuid_v2에 저장 (재현·중복방지 추적용).

CLI:
  python scripts/sim/load_persona_to_neo4j.py \\
      --input output/personas/full/A_rank_coupling_bdc_nvidia_v2.jsonl \\
      --backup-old
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "neo4j_load"))

from _common import driver_session


UPDATE_CYPHER = """
UNWIND $rows AS r
MATCH (a:Agent {id: r.aid})
SET a.personality_lifestyle_raw = r.lifestyle,
    a.nvidia_uuid_v2 = r.nvidia_uuid,
    a.persona_match_level = r.match_level
"""


def _bump_seq(aid: str) -> str:
    """agent_id 마지막 _NNN seq를 +1 보정.
    jsonl은 0-indexed(000부터), Neo4j Agent.id는 1-indexed(001부터)라
    매칭을 위해 +1 보정.
    AGT_11110515_F_20대_000 → AGT_11110515_F_20대_001
    """
    parts = aid.rsplit("_", 1)
    if len(parts) != 2:
        return aid
    try:
        n = int(parts[1])
        return f"{parts[0]}_{n+1:03d}"
    except ValueError:
        return aid

BACKUP_CYPHER = """
MATCH (a:Agent)
WHERE a.personality_lifestyle_raw IS NOT NULL
  AND a.personality_lifestyle_raw_v1 IS NULL
SET a.personality_lifestyle_raw_v1 = a.personality_lifestyle_raw
RETURN count(a) AS n
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--backup-old", action="store_true",
                    help="기존 lifestyle을 _v1 백업 컬럼으로 복사 후 새 5줄로 덮어쓰기")
    ap.add_argument("--bump-seq", action="store_true",
                    help="jsonl agent_id seq(_NNN)를 +1 보정 후 Neo4j 매칭")
    ap.add_argument("--batch", type=int, default=500)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"ERR: input not found: {p}", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    n_in = 0
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            n_in += 1
            aid = r.get("agent_id")
            lifestyle = (r.get("personality") or {}).get("lifestyle") or ""
            match_meta = r.get("_match") or r.get("match_meta") or {}
            if not aid or not lifestyle:
                continue
            if args.bump_seq:
                aid = _bump_seq(aid)
            rows.append({
                "aid": aid,
                "lifestyle": lifestyle,
                "nvidia_uuid": match_meta.get("nvidia_uuid") or "",
                "match_level": match_meta.get("match_level") or "",
            })

    print(f"[load_persona] input rows={n_in} / valid={len(rows)}")

    with driver_session() as s:
        if args.backup_old:
            r = s.run(BACKUP_CYPHER).single()
            print(f"  backup: {r['n']:,} agents → personality_lifestyle_raw_v1")

        updated = 0
        for i in range(0, len(rows), args.batch):
            batch = rows[i:i + args.batch]
            s.run(UPDATE_CYPHER, rows=batch)
            updated += len(batch)
            if (i // args.batch) % 5 == 0:
                print(f"  updated {updated:,}/{len(rows):,}")
        print(f"[load_persona] done: {updated:,} agents updated")

        # 검증
        r = s.run("MATCH (a:Agent) WHERE a.nvidia_uuid_v2 IS NOT NULL "
                  "RETURN count(a) AS n_v2, "
                  "count(DISTINCT a.nvidia_uuid_v2) AS unique_uuids").single()
        print(f"  Neo4j 확인: v2 적재된 agent {r['n_v2']:,} / unique uuid {r['unique_uuids']:,}")


if __name__ == "__main__":
    main()
