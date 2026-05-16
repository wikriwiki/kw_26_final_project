"""pipeline: end-to-end (mock LLM + mock graph reader)."""

from pathlib import Path

import pytest

from src.policy_pipeline.models import BenefitType, ExtractedPolicy
from src.policy_pipeline.pipeline import process_policy_file
from src.policy_pipeline.state import PolicyStatus


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------
class _StubStructuredLLM:
    def __init__(self, factory):
        self.factory = factory

    def complete_structured(self, prompt: str, response_model, *, system_prompt=None):
        return self.factory()


def _happy_factory():
    return ExtractedPolicy(
        policy_id="policy_x",
        title="서울시 소비쿠폰",
        summary="소상공인 결제 시 10% 할인",
        source_file="x",
        target_regions=["서울시"],
        target_industries=["전체 업종"],
        target_groups=["전체 시민"],
        benefit_type=BenefitType.COUPON,
        benefit_rate=0.1,
        confidence=0.95,
    )


def _review_factory():
    return ExtractedPolicy(
        policy_id="policy_x",
        title="모호한 정책",
        summary="일부 지역 일부 업종 지원",
        source_file="x",
        target_regions=["서울시"],
        target_industries=["일부 업종"],  # 모호어
        benefit_type=BenefitType.COUPON,
        confidence=0.95,
    )


# ---------------------------------------------------------------------------
# Isolation fixture — 모든 IO 경로를 tmp_path 로 격리
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_io(tmp_path: Path, monkeypatch) -> Path:
    from src.policy_pipeline import (
        archive,
        dedup,
        invalidator,
        state,
        summary_jobs,
        validator,
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
def inbox_file(isolated_io: Path) -> Path:
    inbox = isolated_io / "inbox"
    inbox.mkdir()
    f = inbox / "policy_sample.txt"
    f.write_text("정책 원문 텍스트", encoding="utf-8")
    return f


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_happy_path_returns_validated(inbox_file, isolated_io):
    llm = _StubStructuredLLM(_happy_factory)
    result = process_policy_file(inbox_file, llm)

    assert result.final_status == PolicyStatus.VALIDATED
    assert result.cache_keys_invalidated >= 1
    assert result.summary_jobs_enqueued >= 1
    # 파일이 processed 로 이동됨
    assert not inbox_file.exists()
    assert result.archived_to is not None
    assert result.archived_to.exists()
    assert (isolated_io / "processed") in result.archived_to.parents


def test_needs_review_archives_to_failed_bucket(inbox_file, isolated_io):
    llm = _StubStructuredLLM(_review_factory)
    result = process_policy_file(inbox_file, llm)

    assert result.final_status == PolicyStatus.NEEDS_REVIEW
    # 검토 대상은 보존 폴더(failed/)로 이동
    assert result.archived_to is not None
    assert (isolated_io / "failed") in result.archived_to.parents


def test_duplicate_hash_is_skipped(inbox_file, isolated_io):
    llm = _StubStructuredLLM(_happy_factory)
    first = process_policy_file(inbox_file, llm)
    assert first.final_status == PolicyStatus.VALIDATED

    # 같은 내용의 파일을 다시 떨군다
    inbox = inbox_file.parent
    second_file = inbox / "policy_sample.txt"
    second_file.write_text("정책 원문 텍스트", encoding="utf-8")

    second = process_policy_file(second_file, llm)
    assert second.skipped_as_duplicate is True
