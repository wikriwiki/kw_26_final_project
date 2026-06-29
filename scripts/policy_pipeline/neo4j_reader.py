"""
neo4j_reader.py
===============
`scope.GraphReader` 의 Neo4j 구현.

agent_persona 브랜치 스키마:
  (:District {code, name})-[:HAS_DONG]->(:Dong {code, name})
  (:POI {id})-[:IN_DONG]->(:Dong)
  (:POI)-[:IN_CATEGORY]->(:Category {name, parent})    # parent = L1 이름
  (:Agent {id, p_age_group, p_income_level, p_life_stage, ...})-[:LIVES_AT|WORKS_AT]->(:POI)

`scripts.neo4j_load._common.driver_session()` 를 재사용해 .env 기반 인증.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)


# scripts/neo4j_load/_common.py 를 import 하려면 path 등록
_NEO4J_LOAD_DIR = Path(__file__).resolve().parents[1] / "neo4j_load"
if str(_NEO4J_LOAD_DIR) not in sys.path:
    sys.path.insert(0, str(_NEO4J_LOAD_DIR))


class Neo4jGraphReader:
    """`with ... as reader:` 사용. 매 호출마다 새 세션을 열지 않고 컨텍스트 매니저로
    한번에 묶는다.

    GraphReader Protocol 을 구조적으로 만족 (`dongs_in_districts` 등 메서드 제공).
    """

    def __init__(self, session=None) -> None:
        self._owned_session_cm = None
        self._session = session

    def __enter__(self):
        if self._session is None:
            from _common import driver_session  # type: ignore
            self._owned_session_cm = driver_session()
            self._session = self._owned_session_cm.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._owned_session_cm is not None:
            self._owned_session_cm.__exit__(exc_type, exc, tb)
            self._owned_session_cm = None
            self._session = None
        return False

    # ------------------------------------------------------------------
    # GraphReader Protocol
    # ------------------------------------------------------------------
    def dongs_in_districts(self, district_names: list[str]) -> list[str]:
        if not district_names:
            return []
        q = """
        MATCH (dist:District)-[:HAS_DONG]->(d:Dong)
        WHERE dist.name IN $names
        RETURN d.code AS code
        """
        return [r["code"] for r in self._session.run(q, names=district_names)]

    def pois_in_dongs_for_categories(
        self, dong_codes: list[str], l1_categories: list[str],
    ) -> list[str]:
        if not dong_codes or not l1_categories:
            return []
        q = """
        MATCH (p:POI)-[:IN_DONG]->(d:Dong),
              (p)-[:IN_CATEGORY]->(c:Category)
        WHERE d.code IN $dongs AND c.parent IN $l1s
        RETURN DISTINCT p.id AS id
        """
        return [r["id"] for r in self._session.run(q, dongs=dong_codes, l1s=l1_categories)]

    def agents_in_dongs_for_groups(
        self, dong_codes: list[str], target_groups: list[str],
    ) -> list[str]:
        """대상 그룹 어휘를 Agent 노드 속성에 매핑해 조회.

        매핑은 보수적으로: '청년' → 20·30대, '노인'/'어르신' → 60대 이상, 등.
        매치 안 되는 대상 그룹은 무시 (downstream 이 추후 어휘 보강).
        """
        if not dong_codes:
            return []

        age_groups = _map_target_groups_to_age(target_groups)
        income_levels = _map_target_groups_to_income(target_groups)

        # 어떤 매핑도 안 잡히면 "전체" 로 해석 — 동 거주자 전부.
        match_all = not age_groups and not income_levels

        if match_all:
            q = """
            MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(d:Dong)
            WHERE d.code IN $dongs
            RETURN DISTINCT a.id AS id
            """
            return [r["id"] for r in self._session.run(q, dongs=dong_codes)]

        q = """
        MATCH (a:Agent)-[:LIVES_AT]->(:POI)-[:IN_DONG]->(d:Dong)
        WHERE d.code IN $dongs
          AND (
            ($ages = [] OR a.p_age_group IN $ages)
            OR ($incomes = [] OR a.p_income_level IN $incomes)
          )
        RETURN DISTINCT a.id AS id
        """
        return [
            r["id"]
            for r in self._session.run(
                q, dongs=dong_codes, ages=age_groups, incomes=income_levels,
            )
        ]


# ---------------------------------------------------------------------------
# 어휘 → Agent 속성 매핑 (확장 여지 있음)
# ---------------------------------------------------------------------------
_AGE_MAP = {
    "청년": ["20", "30"],
    "학생": ["20", "10"],
    "노인": ["60", "70", "80"],
    "어르신": ["60", "70", "80"],
}

_INCOME_MAP = {
    "저소득층": ["하"],
    "취약계층": ["하"],
}


def _map_target_groups_to_age(groups: list[str]) -> list[str]:
    out: set[str] = set()
    for g in groups:
        for code in _AGE_MAP.get(g, []):
            out.add(code)
    return sorted(out)


def _map_target_groups_to_income(groups: list[str]) -> list[str]:
    out: set[str] = set()
    for g in groups:
        for code in _INCOME_MAP.get(g, []):
            out.add(code)
    return sorted(out)
