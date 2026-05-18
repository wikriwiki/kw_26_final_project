"""neo4j_writer: Cypher 호출 시그니처 검증 (mock session).

실제 Neo4j 연결은 통합 테스트에서. 여기서는 ValidatedPolicy + Scope 가 들어가면
올바른 Cypher 와 파라미터가 호출되는지만 본다.
"""

from datetime import date

from scripts.policy_pipeline.models import PolicyType, ValidatedPolicy
from scripts.policy_pipeline.neo4j_writer import write_policy_to_neo4j
from scripts.policy_pipeline.scope import PolicyScope, ScopeUnit


class _StubSingle:
    def __init__(self, value):
        self._value = value
    def single(self):
        return {"n": self._value}


class _StubSession:
    def __init__(self, link_counts=None):
        self.calls = []
        self._counts = link_counts or {}
    def run(self, cypher, **params):
        self.calls.append((cypher, params))
        # MATCH ... RETURN count 류 쿼리만 single() 가능
        if "RETURN count" in cypher:
            # 어떤 단계인지 추정
            if "applied_to" in cypher and "District" in cypher:
                return _StubSingle(self._counts.get("districts", 0))
            if "applied_to" in cypher and "Dong" in cypher:
                return _StubSingle(self._counts.get("dongs", 0))
            if "targets" in cypher:
                return _StubSingle(self._counts.get("categories", 0))
            if "DEACTIVATED" in cypher:
                return _StubSingle(self._counts.get("deactivated", 0))
        return _StubResult()


class _StubResult:
    def single(self):
        return None


def _validated_p007() -> ValidatedPolicy:
    return ValidatedPolicy(
        policy_id="P007",
        title="서울시민 소상공인 응원 쿠폰",
        summary="서울 전역",
        source_file="P007.json",
        source_file_hash="deadbeefcafebabe1234567890abcdef" * 2,
        policy_type=PolicyType.SUBSIDY,
        benefit_rate=1.0,
        cap_per_agent=100000,
        announce_date=date(2026, 4, 25),
        effective_from=date(2026, 5, 1),
        effective_until=date(2026, 6, 30),
        target_districts=["강남구", "마포구"],
        benefit_categories=[],
    )


def test_writer_merges_policy_and_links_districts():
    p = _validated_p007()
    scope = PolicyScope(policy_id="P007", scope_units=[ScopeUnit.DISTRICT],
                       affected_districts=["강남구", "마포구"])
    session = _StubSession(link_counts={"districts": 2})

    result = write_policy_to_neo4j(p, scope, session=session)

    # 첫 호출 = Policy MERGE
    cypher_first, params_first = session.calls[0]
    assert "MERGE (p:Policy {id: $id})" in cypher_first
    assert params_first["id"] == "P007"
    assert params_first["type"] == "subsidy"
    assert params_first["benefit_rate"] == 1.0
    assert params_first["cap_per_agent"] == 100000
    # 날짜는 ISO 문자열로
    assert params_first["effective_from"] == "2026-05-01"

    # 두번째 호출 = applied_to District
    cypher_second, params_second = session.calls[1]
    assert "(:District)" in cypher_second or "d:District" in cypher_second
    assert params_second["names"] == ["강남구", "마포구"]

    assert result.districts_linked == 2


def test_writer_links_dongs_from_scope():
    p = _validated_p007()
    scope = PolicyScope(
        policy_id="P007",
        scope_units=[ScopeUnit.DONG],
        affected_districts=["강남구"],
        affected_dongs=["1168010100", "1168010200"],
    )
    session = _StubSession(link_counts={"districts": 1, "dongs": 2})

    result = write_policy_to_neo4j(p, scope, session=session)

    # Dong link Cypher 가 호출됐는지
    dong_calls = [c for c, params in session.calls if "Dong" in c and "applied_to" in c]
    assert dong_calls
    assert result.dongs_linked == 2


def test_writer_links_categories_for_l1_names():
    p = ValidatedPolicy(
        policy_id="P_FOOD",
        title="식사·카페 정책",
        summary="식사·카페 업종 한정",
        source_file="P_FOOD.json",
        source_file_hash="ab" * 32,
        policy_type=PolicyType.DISCOUNT,
        target_districts=["강남구"],
        benefit_categories=["식사", "카페"],
    )
    scope = PolicyScope(
        policy_id="P_FOOD",
        scope_units=[ScopeUnit.DISTRICT, ScopeUnit.CATEGORY],
        affected_districts=["강남구"],
        affected_categories=["식사", "카페"],
    )
    session = _StubSession(link_counts={"districts": 1, "categories": 23})

    result = write_policy_to_neo4j(p, scope, session=session)

    cat_call_params = [params for c, params in session.calls if ":targets" in c]
    assert cat_call_params
    assert cat_call_params[0]["l1s"] == ["식사", "카페"]
    assert result.categories_linked == 23


def test_deactivate_others_when_flag_set():
    p = _validated_p007()
    scope = PolicyScope(policy_id="P007", scope_units=[ScopeUnit.DISTRICT],
                       affected_districts=["강남구"])
    session = _StubSession(link_counts={"districts": 1, "deactivated": 3})

    write_policy_to_neo4j(p, scope, session=session, deactivate_others_from="2026-05-01")

    deact_calls = [c for c, params in session.calls if "DEACTIVATED" in c]
    assert deact_calls
    deact_params = next(params for c, params in session.calls if "DEACTIVATED" in c)
    assert deact_params["keep_id"] == "P007"
    assert deact_params["cutoff"] == "2026-05-01"
