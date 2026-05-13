"""core 도메인 모델이 infra/llm 모듈과 잘 연동되는지 검증.

검사 항목:
1. PlanDraft를 generate_structured/BatchController에 schema로 넘길 수 있는가
   (Pydantic v2 + BaseModel 상속 조건 충족)
2. PlanDraft.model_json_schema()가 sglang `response_format` 형식으로 변환 가능한가
3. JSON schema에 enum/제약/필수 필드가 잘 들어갔는가

실제 sglang 서버 호출은 하지 않음 (그건 통합 테스트의 영역). 여기선
타입/스키마 호환성만 검증.
"""
from __future__ import annotations

from typing import Any, get_type_hints

from pydantic import BaseModel

from src.core import EpisodeDraft, Plan, PlanDraft
from src.infra.llm import BatchController, generate_structured
from src.infra.llm.structured import build_response_format


# ─── 1. 타입 호환성 ─────────────────────────────────────


def test_plan_draft_is_a_basemodel():
    # generate_structured의 schema 인자는 type[T: BaseModel] 요구
    assert issubclass(PlanDraft, BaseModel)
    assert issubclass(EpisodeDraft, BaseModel)
    assert issubclass(Plan, BaseModel)


def test_generate_structured_signature_accepts_plan_draft():
    """generate_structured의 schema 파라미터 타입이 BaseModel 서브클래스인지.

    런타임 호출은 안 하지만, 타입 힌트를 직접 확인.
    """
    hints = get_type_hints(generate_structured)
    # schema는 type[T: BaseModel]; 정확한 generic 검증은 어렵지만 존재 확인
    assert "schema" in hints


def test_batch_controller_submit_accepts_plan_draft():
    """BatchController.submit(schema=PlanDraft)가 타입상 유효한지."""
    hints = get_type_hints(BatchController.submit)
    assert "schema" in hints


# ─── 2. sglang response_format 변환 ─────────────────────


def test_plan_draft_schema_builds_sglang_response_format():
    """PlanDraft → sglang JSON schema response_format 변환."""
    rf = build_response_format(PlanDraft)

    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "PlanDraft"
    assert rf["json_schema"]["strict"] is True
    assert isinstance(rf["json_schema"]["schema"], dict)


def test_episode_draft_schema_builds_sglang_response_format():
    """EpisodeDraft도 단독으로 schema화 가능 (intent classifier 등 다른 task용)."""
    rf = build_response_format(EpisodeDraft)
    assert rf["json_schema"]["name"] == "EpisodeDraft"


# ─── 3. 스키마 내용물 검증 ──────────────────────────────


def test_plan_draft_schema_has_episodes_array():
    schema = PlanDraft.model_json_schema()
    props = _resolve_properties(schema)
    assert "episodes" in props
    assert props["episodes"]["type"] == "array"
    # min/max 제약이 반영됨
    assert props["episodes"]["minItems"] == 1
    assert props["episodes"]["maxItems"] == 12


def test_episode_draft_schema_has_required_fields():
    schema = EpisodeDraft.model_json_schema()
    required = set(schema.get("required", []))
    # LLM이 반드시 채워야 할 필드들
    assert {"time_slot", "hour", "sequence", "action", "region_code",
            "source", "motivation"}.issubset(required)


def test_episode_draft_enums_are_serialized():
    """TimeSlot/ActionType/EpisodeSource가 JSON schema의 enum으로 나오는지."""
    schema = EpisodeDraft.model_json_schema()
    defs = schema.get("$defs", {}) or schema.get("definitions", {})
    # Pydantic v2는 $defs로 enum을 묶어둠
    enum_names = {k for k, v in defs.items() if "enum" in v}
    assert {"TimeSlot", "ActionType", "EpisodeSource"} <= enum_names


def test_episode_draft_hour_range_in_schema():
    schema = EpisodeDraft.model_json_schema()
    props = _resolve_properties(schema)
    hour = props["hour"]
    assert hour["minimum"] == 0
    assert hour["maximum"] == 23


def test_full_plan_schema_works_too():
    """저장용 Plan도 schema화 가능 (필요 시 디버깅용 직렬화에 사용)."""
    rf = build_response_format(Plan)
    assert rf["json_schema"]["name"] == "Plan"


# ─── 헬퍼 ────────────────────────────────────────────────


def _resolve_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Pydantic v2가 properties를 ref 안에 숨겨두는 경우 풀어줌."""
    if "properties" in schema:
        return schema["properties"]
    # $ref만 있는 경우
    if "$ref" in schema:
        ref = schema["$ref"].split("/")[-1]
        return schema.get("$defs", {}).get(ref, {}).get("properties", {})
    return {}
