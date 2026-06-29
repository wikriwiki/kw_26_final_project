"""pipeline: end-to-end (LLM + Neo4j writer 모킹)."""

from pathlib import Path

import pytest

from scripts.policy_pipeline.models import ExtractedPolicy, PolicyType
from scripts.policy_pipeline.neo4j_writer import WriteResult
from scripts.policy_pipeline.pipeline import (
    inject_validated_payload,
    process_policy_file,
)
from scripts.policy_pipeline.scope import NullGraphReader
from scripts.policy_pipeline.state import PolicyStatus, calculate_file_hash


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _StubLLM:
    def __init__(self, factory):
        self.factory = factory

    def call_chat_structured(self, response_model, system_prompt, user_prompt):
        return self.factory()


class _StubWriter:
    def __init__(self) -> None:
        self.calls = []

    def write(self, validated, scope, *, deactivate_others_from=None):
        self.calls.append({
            "policy_id": validated.policy_id,
            "districts": list(validated.target_districts),
            "categories": list(validated.benefit_categories),
            "deactivate_others_from": deactivate_others_from,
        })
        return WriteResult(
            policy_id=validated.policy_id,
            districts_linked=len(validated.target_districts),
            dongs_linked=len(scope.affected_dongs),
            categories_linked=len(validated.benefit_categories),
        )


def _happy_factory():
    return ExtractedPolicy(
        policy_id="placeholder",
        title="서울시 소비쿠폰",
        summary="강남구 식사 업종 10% 할인",
        source_file="placeholder",
        target_districts=["강남구"],
        benefit_categories=["식사"],
        policy_type=PolicyType.COUPON,
        benefit_rate=0.1,
        confidence=0.95,
    )


def _ambiguous_factory():
    return ExtractedPolicy(
        policy_id="placeholder",
        title="모호한 정책",
        summary="일부 지역 일부 업종",
        source_file="placeholder",
        target_districts=["강남구"],
        benefit_categories=["일부 업종"],  # 모호어
        policy_type=PolicyType.COUPON,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Isolation fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated(tmp_path: Path, monkeypatch) -> Path:
    from scripts.policy_pipeline import (
        archive, dedup, invalidator, state, summary_jobs, validator,
    )

    state_log = tmp_path / "state.jsonl"
    monkeypatch.setattr(archive, "DEFAULT_PROCESSED_DIR", tmp_path / "processed")
    monkeypatch.setattr(archive, "DEFAULT_FAILED_DIR", tmp_path / "failed")
    monkeypatch.setattr(state, "DEFAULT_STATE_LOG_PATH", state_log)
    monkeypatch.setattr(dedup, "DEFAULT_STATE_LOG_PATH", state_log)
    monkeypatch.setattr(validator, "DEFAULT_VALIDATION_LOG_PATH", tmp_path / "val.jsonl")
    monkeypatch.setattr(validator, "DEFAULT_EXTRACTED_POLICY_LOG_PATH", tmp_path / "ext.jsonl")
    monkeypatch.setattr(invalidator, "DEFAULT_INVALIDATION_LOG_PATH", tmp_path / "inv.jsonl")
    monkeypatch.setattr(invalidator, "DEFAULT_REGISTRY_PATH", tmp_path / "reg.json")
    monkeypatch.setattr(summary_jobs, "DEFAULT_JOB_LOG_PATH", tmp_path / "jobs.jsonl")
    return tmp_path


@pytest.fixture
def inbox_file(isolated: Path) -> Path:
    inbox = isolated / "inbox"
    inbox.mkdir()
    f = inbox / "policy_sample.txt"
    f.write_text("정책 원문 텍스트", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_happy_path_ends_applied(inbox_file, isolated):
    writer = _StubWriter()
    result = process_policy_file(
        inbox_file,
        llm_client=_StubLLM(_happy_factory),
        graph_reader=NullGraphReader(),
        writer=writer,
    )
    assert result.final_status == PolicyStatus.APPLIED
    assert writer.calls
    assert writer.calls[0]["districts"] == ["강남구"]
    assert writer.calls[0]["categories"] == ["식사"]
    assert result.neo4j_districts_linked == 1
    assert result.cache_keys_invalidated >= 1
    assert (isolated / "processed") in result.archived_to.parents


def test_ambiguous_marks_needs_review_no_neo4j_write(inbox_file, isolated):
    writer = _StubWriter()
    result = process_policy_file(
        inbox_file,
        llm_client=_StubLLM(_ambiguous_factory),
        graph_reader=NullGraphReader(),
        writer=writer,
    )
    assert result.final_status == PolicyStatus.NEEDS_REVIEW
    assert not writer.calls   # 검토 대상은 Neo4j 안 씀
    assert (isolated / "failed") in result.archived_to.parents


def test_duplicate_hash_skipped(inbox_file, isolated):
    writer = _StubWriter()
    process_policy_file(
        inbox_file,
        llm_client=_StubLLM(_happy_factory),
        graph_reader=NullGraphReader(),
        writer=writer,
    )

    # 동일 내용 다시 떨굼
    inbox = inbox_file.parent
    second = inbox / "policy_sample.txt"
    second.write_text("정책 원문 텍스트", encoding="utf-8")

    result = process_policy_file(
        second,
        llm_client=_StubLLM(_happy_factory),
        graph_reader=NullGraphReader(),
        writer=writer,
    )
    assert result.skipped_as_duplicate is True
    assert len(writer.calls) == 1  # 두 번째는 LLM/Writer 모두 호출 안됨


def test_inject_validated_payload_p007(tmp_path, isolated):
    """수기 검수 JSON 주입 — P007.json 포맷."""
    payload = {
        "id": "P007",
        "name": "서울시민 소상공인 응원 쿠폰",
        "type": "subsidy",
        "description": "서울 전역",
        "benefit_rate": 1.0,
        "cap_per_agent": 100000,
        "announce_date": "2026-04-25",
        "effective_from": "2026-05-01",
        "effective_until": "2026-06-30",
        "target_districts": [
            "강남구", "강동구", "강북구", "강서구", "관악구", "광진구", "구로구",
            "금천구", "노원구", "도봉구", "동대문구", "동작구", "마포구", "서대문구",
            "서초구", "성동구", "성북구", "송파구", "양천구", "영등포구", "용산구",
            "은평구", "종로구", "중구", "중랑구",
        ],
        "benefit_categories": [],
    }
    writer = _StubWriter()
    result = inject_validated_payload(
        payload,
        source_file="P007.json",
        file_hash="deadbeefcafebabe" * 8,
        graph_reader=NullGraphReader(),
        writer=writer,
        deactivate_others_from="2026-05-01",
    )
    assert result.final_status == PolicyStatus.APPLIED
    assert writer.calls
    assert writer.calls[0]["policy_id"] == "P007"
    assert writer.calls[0]["deactivate_others_from"] == "2026-05-01"
    assert len(writer.calls[0]["districts"]) == 25
