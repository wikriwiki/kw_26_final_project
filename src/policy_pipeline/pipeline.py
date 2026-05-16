"""
pipeline.py
===========
Section 9 — Watchdog 전체 파이프라인.

한 정책 파일에 대해 다음 흐름을 직선으로 실행:

  파일 stable 확인  (watcher 가 이미 보장)
  → file_hash 계산 (loader 안에서)
  → dedup 확인     (같은 hash 가 이미 처리됨? skip)
  → PolicyDocument 적재
  → LLM structured extraction
  → 도메인 룰 검증 (validator)
  → [VALIDATED] scope analysis → invalidation → summary jobs enqueue → archive(processed)
  → [NEEDS_REVIEW] state.NEEDS_REVIEW 로그 + archive(failed) (검토 보존용)
  → [FAILED] state.FAILED 로그 + archive(failed)

Neo4j 적재(요구사항 7번)는 이번 PR 범위 밖.

이 함수는 worker thread 가 큐에서 path 를 꺼낼 때마다 호출.
실패 시 raise 하지 않고 PipelineResult 로 반환 — 워커가 다음 파일 처리를 계속하도록.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.policy_pipeline.archive import archive_failure, archive_success
from src.policy_pipeline.dedup import is_duplicate_hash
from src.policy_pipeline.extractor import (
    LLMClient,
    PolicyExtractionError,
    StructuredLLMClient,
    extract_policy,
)
from src.policy_pipeline.invalidator import invalidate_for_scope
from src.policy_pipeline.loader import PolicyDocumentLoadError, load_policy_document
from src.policy_pipeline.scope import GraphReader, NullGraphReader, analyze_scope
from src.policy_pipeline.state import (
    PolicyStatus,
    append_policy_status,
    append_status_for_file,
    build_policy_id,
    calculate_file_hash,
)
from src.policy_pipeline.summary_jobs import enqueue_summary_jobs_for_scope
from src.policy_pipeline.validator import validate_and_record_policy


log = logging.getLogger("policy_pipeline")


# ---------------------------------------------------------------------------
# Result 객체
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    file: Path
    final_status: PolicyStatus
    policy_id: str | None = None
    error: str | None = None
    archived_to: Path | None = None
    cache_keys_invalidated: int = 0
    summary_jobs_enqueued: int = 0
    skipped_as_duplicate: bool = False


# ---------------------------------------------------------------------------
# 의존성 주입용 Protocol
# ---------------------------------------------------------------------------
class LLMClientLike(Protocol):
    """structured / legacy 둘 다 받게."""
    def complete(self, prompt: str) -> str: ...  # noqa


# ---------------------------------------------------------------------------
# 메인 함수
# ---------------------------------------------------------------------------
def process_policy_file(
    path: Path,
    llm_client: StructuredLLMClient | LLMClient,
    *,
    graph_reader: GraphReader | None = None,
) -> PipelineResult:
    """한 정책 파일을 처음부터 끝까지 처리.

    워커가 큐에서 path 하나를 꺼낼 때마다 호출한다.
    """
    graph_reader = graph_reader or NullGraphReader()

    # 1) 해시 계산 + dedup
    try:
        file_hash = calculate_file_hash(path)
    except FileNotFoundError:
        log.warning("file vanished before processing: %s", path)
        return PipelineResult(file=path, final_status=PolicyStatus.FAILED,
                              error="file not found")

    if is_duplicate_hash(file_hash):
        log.info("duplicate hash skip: %s (%s)", path.name, file_hash[:12])
        archived = archive_success(path, file_hash)
        return PipelineResult(
            file=path,
            final_status=PolicyStatus.VALIDATED,  # 이전 처리에서 도달했던 단계 그대로 보존
            policy_id=build_policy_id(file_hash),
            archived_to=archived,
            skipped_as_duplicate=True,
        )

    append_status_for_file(path, PolicyStatus.EXTRACTING)

    # 2) 문서 로드
    try:
        document = load_policy_document(path)
    except PolicyDocumentLoadError as exc:
        return _terminate_failure(path, file_hash, f"load error: {exc}")

    # 3) LLM 추출
    try:
        extracted = extract_policy(document, llm_client)
    except PolicyExtractionError as exc:
        return _terminate_failure(path, file_hash, f"extract error: {exc}")
    except Exception as exc:  # noqa  catch broad: LLM SDK 가 다양한 예외 던짐
        return _terminate_failure(path, file_hash, f"unexpected extract error: {exc}")

    # 4) 도메인 검증
    outcome = validate_and_record_policy(extracted)

    if outcome.status == PolicyStatus.FAILED:
        archived = archive_failure(path, file_hash)
        return PipelineResult(
            file=path,
            final_status=PolicyStatus.FAILED,
            policy_id=extracted.policy_id,
            error="validator returned FAILED",
            archived_to=archived,
        )

    if outcome.status == PolicyStatus.NEEDS_REVIEW:
        # 검토 보존용으로 failed/ 폴더에 둔다 (이름은 'failed' 지만 실패 의미는 아니고
        # "자동 처리 불가" 버킷). 별도 review/ 폴더로 분리하고 싶다면 archive.py 확장.
        archived = archive_failure(path, file_hash)
        return PipelineResult(
            file=path,
            final_status=PolicyStatus.NEEDS_REVIEW,
            policy_id=extracted.policy_id,
            archived_to=archived,
            error="; ".join(extracted.review_reasons),
        )

    # 5) VALIDATED — 후속 처리
    validated = outcome.validated_policy
    assert validated is not None  # type narrowing

    # 5a) scope analysis (textual + optional graph)
    scope = analyze_scope(validated, graph_reader)

    # 5b) cache invalidation (version bump + key 열거)
    invalidation = invalidate_for_scope(scope)

    # 5c) summary 재생성 job enqueue
    jobs = enqueue_summary_jobs_for_scope(scope)

    # 5d) Neo4j 적재는 이번 PR 범위 밖. 자리만 표시.
    # TODO(neo4j): src/graph/queries/policy_writer.write(validated, scope) 후
    # state APPLIED 로 전이. 현재는 VALIDATED 에서 종료.

    # 5e) archive 이동
    archived = archive_success(path, file_hash)

    append_policy_status(
        policy_id=validated.policy_id,
        file_hash=file_hash,
        source_path=str(path),
        status=PolicyStatus.VALIDATED,
    )

    return PipelineResult(
        file=path,
        final_status=PolicyStatus.VALIDATED,
        policy_id=validated.policy_id,
        archived_to=archived,
        cache_keys_invalidated=len(invalidation.invalidated_keys),
        summary_jobs_enqueued=len(jobs),
    )


# ---------------------------------------------------------------------------
# 실패 경로 공통
# ---------------------------------------------------------------------------
def _terminate_failure(
    path: Path,
    file_hash: str,
    reason: str,
) -> PipelineResult:
    log.warning("pipeline failure for %s: %s", path.name, reason)
    archived = archive_failure(path, file_hash)
    append_policy_status(
        policy_id=build_policy_id(file_hash),
        file_hash=file_hash,
        source_path=str(path),
        status=PolicyStatus.FAILED,
        error_message=reason,
    )
    return PipelineResult(
        file=path,
        final_status=PolicyStatus.FAILED,
        policy_id=build_policy_id(file_hash),
        error=reason,
        archived_to=archived,
    )
