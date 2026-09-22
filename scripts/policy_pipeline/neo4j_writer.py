"""
neo4j_writer.py
===============
ValidatedPolicy + PolicyScope → Neo4j.

기존 load_p007.py 와 동일한 Cypher 구조를 유지하되, **하드코딩 JSON** 대신
타입 안전한 ValidatedPolicy 를 입력으로 받는다.

생성/갱신되는 노드·엣지:
  (:Policy {id, name, type, description, benefit_rate, cap_per_agent,
            announce_date, effective_from, effective_until,
            raw_json_ref, extracted_at, validation_notes})
  (:Policy)-[:applied_to]->(:District)           # ValidatedPolicy.target_districts
  (:Policy)-[:applied_to]->(:Dong)               # scope.affected_dongs (확장됐을 때)
  (:Policy)-[:targets]->(:Category)              # benefit_categories → parent IN [...] L2 노드

state.PolicyStatus.APPLIED 로 전이는 호출자(pipeline) 책임.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from scripts.policy_pipeline.models import ValidatedPolicy
from scripts.policy_pipeline.scope import PolicyScope


log = logging.getLogger(__name__)

# neo4j_load 모듈 path 등록
_NEO4J_LOAD_DIR = Path(__file__).resolve().parents[1] / "neo4j_load"
if str(_NEO4J_LOAD_DIR) not in sys.path:
    sys.path.insert(0, str(_NEO4J_LOAD_DIR))


@dataclass
class WriteResult:
    policy_id: str
    districts_linked: int = 0
    dongs_linked: int = 0
    categories_linked: int = 0


# ---------------------------------------------------------------------------
# Cypher 상수
# ---------------------------------------------------------------------------
_MERGE_POLICY_CYPHER = """
MERGE (p:Policy {id: $id})
SET p.name = $name,
    p.type = $type,
    p.description = $description,
    p.benefit_rate = $benefit_rate,
    p.cap_per_agent = $cap_per_agent,
    p.announce_date = CASE WHEN $announce_date IS NULL THEN p.announce_date
                           ELSE date($announce_date) END,
    p.effective_from = CASE WHEN $effective_from IS NULL THEN p.effective_from
                            ELSE date($effective_from) END,
    p.effective_until = CASE WHEN $effective_until IS NULL THEN p.effective_until
                             ELSE date($effective_until) END,
    p.raw_json_ref = $raw_json_ref,
    p.extracted_at = datetime(),
    p.validation_notes = $validation_notes
"""

_LINK_DISTRICTS_CYPHER = """
MATCH (p:Policy {id: $id})
MATCH (d:District) WHERE d.name IN $names
MERGE (p)-[:applied_to]->(d)
RETURN count(DISTINCT d) AS n
"""

_LINK_DONGS_CYPHER = """
MATCH (p:Policy {id: $id})
MATCH (d:Dong) WHERE d.code IN $codes
MERGE (p)-[:applied_to]->(d)
RETURN count(DISTINCT d) AS n
"""

# L1 카테고리 이름들로 :targets 엣지를 만들 때, Neo4j 의 Category 노드는
# L2 만 있으므로 parent 가 일치하는 모든 L2 에 엣지를 건다.
_LINK_CATEGORIES_CYPHER = """
MATCH (p:Policy {id: $id})
MATCH (c:Category) WHERE c.parent IN $l1s
MERGE (p)-[:targets]->(c)
RETURN count(DISTINCT c) AS n
"""

_DEACTIVATE_OTHERS_CYPHER = """
MATCH (p:Policy)
WHERE p.id <> $keep_id
  AND p.effective_until >= date($cutoff)
SET p.effective_until = date($cutoff) - duration({days: 1}),
    p.notes = coalesce(p.notes,'') + ' [DEACTIVATED for ' + $keep_id + ' sim]'
RETURN count(p) AS n
"""


# ---------------------------------------------------------------------------
# 메인 API
# ---------------------------------------------------------------------------
def write_policy_to_neo4j(
    validated: ValidatedPolicy,
    scope: PolicyScope,
    *,
    session=None,
    deactivate_others_from: str | None = None,
) -> WriteResult:
    """ValidatedPolicy + PolicyScope 를 Neo4j 에 MERGE.

    Args:
      session: neo4j 드라이버 세션. None 이면 `_common.driver_session()` 사용.
      deactivate_others_from: 'YYYY-MM-DD'. 지정 시 다른 모든 Policy 의
        effective_until 을 이 날짜 직전으로 당겨 단일 정책 시뮬 환경 구성.
    """
    if session is None:
        from _common import driver_session  # type: ignore
        with driver_session() as s:
            return _write(validated, scope, s, deactivate_others_from)
    return _write(validated, scope, session, deactivate_others_from)


def _write(
    validated: ValidatedPolicy,
    scope: PolicyScope,
    session,
    deactivate_others_from: str | None,
) -> WriteResult:
    result = WriteResult(policy_id=validated.policy_id)

    # 1) :Policy MERGE
    session.run(
        _MERGE_POLICY_CYPHER,
        id=validated.policy_id,
        name=validated.title,
        type=validated.policy_type.value,
        description=validated.summary,
        benefit_rate=validated.benefit_rate,
        cap_per_agent=validated.cap_per_agent,
        announce_date=_to_iso(validated.announce_date),
        effective_from=_to_iso(validated.effective_from),
        effective_until=_to_iso(validated.effective_until),
        raw_json_ref=validated.source_file_hash,
        validation_notes=list(validated.validation_notes),
    )
    log.info("MERGE :Policy %s", validated.policy_id)

    # 2) :applied_to → :District
    if validated.target_districts:
        r = session.run(
            _LINK_DISTRICTS_CYPHER,
            id=validated.policy_id,
            names=validated.target_districts,
        ).single()
        result.districts_linked = (r and r["n"]) or 0
        log.info("applied_to → %d districts", result.districts_linked)

    # 3) :applied_to → :Dong (scope 분석에서 확장된 동들)
    if scope.affected_dongs:
        r = session.run(
            _LINK_DONGS_CYPHER,
            id=validated.policy_id,
            codes=scope.affected_dongs,
        ).single()
        result.dongs_linked = (r and r["n"]) or 0
        log.info("applied_to → %d dongs", result.dongs_linked)

    # 4) :targets → :Category (L2 노드, parent IN [L1 이름들])
    if validated.benefit_categories:
        r = session.run(
            _LINK_CATEGORIES_CYPHER,
            id=validated.policy_id,
            l1s=validated.benefit_categories,
        ).single()
        result.categories_linked = (r and r["n"]) or 0
        log.info("targets → %d category nodes", result.categories_linked)

    # 5) (옵션) 다른 정책 비활성화
    if deactivate_others_from:
        r = session.run(
            _DEACTIVATE_OTHERS_CYPHER,
            keep_id=validated.policy_id,
            cutoff=deactivate_others_from,
        ).single()
        n = (r and r["n"]) or 0
        if n:
            log.info("deactivated %d other policies (cutoff=%s)", n, deactivate_others_from)

    return result


def _to_iso(d) -> str | None:
    if d is None:
        return None
    return d.isoformat()
