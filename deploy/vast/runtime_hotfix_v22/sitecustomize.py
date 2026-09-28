"""Scoped repair for v22 skipped-agent State creation.

Activated only by NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX=1. The frozen v22
simulator files remain unchanged on the server so its existing Day2 receipts
can be resumed. Every new skipped metric records this file's SHA256 separately.
"""

import hashlib
import os
from pathlib import Path


if os.environ.get("NO_SMOKING_V22_SKIP_PERSISTENCE_HOTFIX") == "1":
    import agent_day_store

    _patch_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def _save_skipped_day(tx, result):
        from datetime import date, timedelta

        aid, day = result["aid"], result["experience_day"]
        previous = (date.fromisoformat(day) - timedelta(days=1)).isoformat()
        row = tx.run(
            """MATCH (a:Agent {id:$aid})-[:HAS_STATE {day:date($previous)}]->(prev:State)
            CREATE (s)
            SET s = properties(prev), s.id=$sid, s.agent_id=$aid, s.day=date($day),
                s.execution_receipts_json='[]', s.appraisal_changes_json='[]',
                s.online_spent=null, s.skipped_agent_day=true,
                s.skip_attempts=$attempts, s.skip_error=$error
            SET s:State
            MERGE (a)-[:HAS_STATE {day:date($day)}]->(s)
            RETURN count(s) AS n""",
            aid=aid, previous=previous, day=day, sid=f"{aid}_{day}",
            attempts=result["attempts"], error=result["last_error"],
        ).single()
        if not row or row["n"] != 1:
            raise agent_day_store.EvidenceError(
                "Previous State missing while recording skipped agent day"
            )
        tagged = dict(result, persistence_hotfix_sha256=_patch_sha256)
        return agent_day_store.save_result(tx, tagged)

    agent_day_store.save_skipped_day = _save_skipped_day
