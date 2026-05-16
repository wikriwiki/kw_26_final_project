"""
pipeline.py
===========
정책 1건의 end-to-end 처리.

흐름:
  파일 stable 확인 (watcher 책임)
  → file_hash 계산
  → dedup 확인 (같은 hash 가 이미 처리됨? skip)
  → PolicyDocument 적재
  → LLM Structured Output 추출 → ExtractedPolicy
  → 도메인 룰 검증 → ValidatedPolicy / NEEDS_REVIEW / FAILED
  → [VALIDATED] scope 분석 (textual + Neo4j GraphReader)
       → Neo4j MERGE (Policy + applied_to + targets)
       → cache invalidation + summary job enqueue
       → state APPLIED
       → archive(processed/)
  → [NEEDS_REVIEW / FAILED] archive(failed/)

manual JSON 주입 경로(`inject_validated_payload`)는 LLM 단계를 건너뛰고
extracted_from_curated_json() 으로 직접 변환.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from scripts.policy_pipeline.archive import archive_failure, archive_success
from scripts.policy_pipeline.dedup import is_duplicate_hash
from scripts.policy_pipeline.extractor import (
    PolicyExtractionError,
    StructuredLLMClient,
    extract_policy,
)
from scripts.policy_pipeline.invalidator import invalidate_for_scope
from scripts.policy_pipeline.loader import (
    PolicyDocumentLoadError,
    load_policy_document,
)
from scripts.policy_pipeline.models import (
    ExtractedPolicy,
    ValidatedPolicy,
    extracted_from_curated_json,
)
from scripts.policy_pipeline.scope import GraphReader, NullGraphReader, analyze_scope
from scripts.policy_pipeline.state import (
    PolicyStatus,
    append_policy_status,
    append_status_for_file,
    build_policy_id,
    calculate_file_hash,
)
from scripts.policy_pipeline.summary_jobs import enqueue_summary_jobs_for_scope
from scripts.policy_pipeline.validator import validate_and_record_policy


log = logging.getLogger("policy_pipeline")


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    file: Path
    final_status: PolicyStatus
    policy_id: str | None = None
    error: str | None = None
    archived_to: Path | None = None
    skipped_as_duplicate: bool = False
    cache_keys_invalidated: int = 0
    summary_jobs_enqueued: int = 0
    neo4j_districts_linked: int = 0
    neo4j_dongs_linked: int = 0
    neo4j_categories_linked: int = 0


# ---------------------------------------------------------------------------
# Neo4j 쓰기 Protocol — 테스트가 mock 으로 갈아끼울 수 있도록
# ---------------------------------------------------------------------------
class PolicyWriter(Protocol):
    def write(
        self,
        validated: ValidatedPolicy,
        scope,  # PolicyScope
        *,
        deactivate_others_from: str | None = None,
    ): ...


# ---------------------------------------------------------------------------
# Default Neo4j 어댑터 — 호출 시점에 lazy import
# ---------------------------------------------------------------------------
class _Neo4jDefaultWriter:
    def __init__(self) -> None:
        self._session_cm = None
        self._session = None

    def __enter__(self):
        from scripts.policy_pipeline.neo4j_writer import write_policy_to_neo4j
        from sys import path as _sys_path
        neo4j_dir = Path(__file__).resolve().parents[1] / "neo4j_load"
        if str(neo4j_dir) not in _sys_path:
            _sys_path.insert(0, str(neo4j_dir))
        from _common import driver_session  # type: ignore
        self._session_cm = driver_session()
        self._session = self._session_cm.__enter__()
        self._write_fn = write_policy_to_neo4j
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._session_cm is not None:
            self._session_cm.__exit__(exc_type, exc, tb)
            self._session_cm = None
            self._session = None
        return False

    def write(self, validated, scope, *, deactivate_others_from=None):
        return self._write_fn(
            validated, scope, session=self._session,
            deactivate_others_from=deactivate_others_from,
        )


class _Neo4jDefaultReader:
    def __init__(self) -> None:
        self._reader = None

    def __enter__(self):
        from scripts.policy_pipeline.neo4j_reader import Neo4jGraphReader
        self._reader = Neo4jGraphReader()
        self._reader.__enter__()
        return self._reader

    def __exit__(self, exc_type, exc, tb):
        if self._reader is not None:
            self._reader.__exit__(exc_type, exc, tb)
            self._reader = None
        return False


# ---------------------------------------------------------------------------
# 메인 함수 (자연어 파일 경로)
# ---------------------------------------------------------------------------
def process_policy_file(
    path: Path,
    *,
    llm_client: StructuredLLMClient | None = None,
    graph_reader: GraphReader | None = None,
    writer: PolicyWriter | None = None,
    deactivate_others_from: str | None = None,
) -> PipelineResult:
    """한 정책 파일을 처음부터 끝까지 처리.

    `writer=None` 이면 기본 Neo4j 어댑터 사용 (NEO4J_PASSWORD 환경변수 필요).
    `graph_reader=None` 이면 기본 Neo4j reader.
    오프라인 테스트는 graph_reader=NullGraphReader(), writer=<mock> 으로 호출.
    """
    try:
        file_hash = calculate_file_hash(path)
    except FileNotFoundError:
        return PipelineResult(file=path, final_status=PolicyStatus.FAILED, error="file not found")

    if is_duplicate_hash(file_hash):
        log.info("duplicate hash skip: %s", path.name)
        return PipelineResult(
            file=path,
            final_status=PolicyStatus.VALIDATED,
            policy_id=build_policy_id(file_hash),
            archived_to=archive_success(path, file_hash),
            skipped_as_duplicate=True,
        )

    append_status_for_file(path, PolicyStatus.EXTRACTING)

    try:
        document = load_policy_document(path)
    except PolicyDocumentLoadError as exc:
        return _terminate_failure(path, file_hash, f"load error: {exc}")

    try:
        extracted = extract_policy(document, client=llm_client)
    except PolicyExtractionError as exc:
        return _terminate_failure(path, file_hash, f"extract error: {exc}")
    except Exception as exc:  # noqa  LLM SDK 가 다양한 예외
        return _terminate_failure(path, file_hash, f"unexpected extract error: {exc}")

    return _post_extraction(
        path, file_hash, extracted,
        graph_reader=graph_reader,
        writer=writer,
        deactivate_others_from=deactivate_others_from,
    )


# ---------------------------------------------------------------------------
# 수기 검수 JSON 주입 경로 (load_p007.py 대체)
# ---------------------------------------------------------------------------
def inject_validated_payload(
    payload: dict,
    *,
    source_file: str,
    file_hash: str,
    graph_reader: GraphReader | None = None,
    writer: PolicyWriter | None = None,
    deactivate_others_from: str | None = None,
) -> PipelineResult:
    """수기 검수된 JSON dict 를 직접 inject. LLM 단계 건너뛴다."""
    extracted = extracted_from_curated_json(
        payload, source_file=source_file, file_hash=file_hash,
    )
    return _post_extraction(
        Path(source_file), file_hash, extracted,
        graph_reader=graph_reader,
        writer=writer,
        deactivate_others_from=deactivate_others_from,
        skip_archive_if_missing=True,
    )


# ---------------------------------------------------------------------------
# 공통 후속 처리
# ---------------------------------------------------------------------------
def _post_extraction(
    path: Path,
    file_hash: str,
    extracted: ExtractedPolicy,
    *,
    graph_reader: GraphReader | None,
    writer: PolicyWriter | None,
    deactivate_others_from: str | None,
    skip_archive_if_missing: bool = False,
) -> PipelineResult:
    outcome = validate_and_record_policy(extracted)

    if outcome.status == PolicyStatus.FAILED:
        archived = _maybe_archive(path, file_hash, "failed", skip_archive_if_missing)
        return PipelineResult(
            file=path, final_status=PolicyStatus.FAILED,
            policy_id=extracted.policy_id,
            error="validator returned FAILED",
            archived_to=archived,
        )

    if outcome.status == PolicyStatus.NEEDS_REVIEW:
        archived = _maybe_archive(path, file_hash, "failed", skip_archive_if_missing)
        return PipelineResult(
            file=path, final_status=PolicyStatus.NEEDS_REVIEW,
            policy_id=extracted.policy_id,
            archived_to=archived,
            error="; ".join(extracted.review_reasons),
        )

    validated = outcome.validated_policy
    assert validated is not None

    # 1) Scope 분석 — graph_reader 가 컨텍스트 매니저면 with 로 묶음
    if graph_reader is None:
        reader_ctx = _Neo4jDefaultReader()
        try:
            with reader_ctx as reader:
                scope = analyze_scope(validated, reader)
        except Exception as exc:  # noqa: Neo4j 인증 실패 등
            log.warning("graph reader unavailable, falling back to textual scope: %s", exc)
            scope = analyze_scope(validated, NullGraphReader())
    else:
        scope = analyze_scope(validated, graph_reader)

    # 2) Neo4j 적재
    write_result_districts = 0
    write_result_dongs = 0
    write_result_categories = 0

    if writer is None:
        writer_ctx = _Neo4jDefaultWriter()
        try:
            with writer_ctx as w:
                wr = w.write(validated, scope, deactivate_others_from=deactivate_others_from)
                write_result_districts = wr.districts_linked
                write_result_dongs = wr.dongs_linked
                write_result_categories = wr.categories_linked
        except Exception as exc:  # noqa
            log.error("Neo4j write failed for %s: %s", validated.policy_id, exc)
            archived = _maybe_archive(path, file_hash, "failed", skip_archive_if_missing)
            append_policy_status(
                policy_id=validated.policy_id, file_hash=file_hash,
                source_path=str(path), status=PolicyStatus.FAILED,
                error_message=f"neo4j write error: {exc}",
            )
            return PipelineResult(
                file=path, final_status=PolicyStatus.FAILED,
                policy_id=validated.policy_id,
                error=f"neo4j write error: {exc}",
                archived_to=archived,
            )
    else:
        wr = writer.write(validated, scope, deactivate_others_from=deactivate_others_from)
        write_result_districts = getattr(wr, "districts_linked", 0)
        write_result_dongs = getattr(wr, "dongs_linked", 0)
        write_result_categories = getattr(wr, "categories_linked", 0)

    # 3) cache invalidation + summary jobs
    invalidation = invalidate_for_scope(scope)
    jobs = enqueue_summary_jobs_for_scope(scope)

    # 4) state APPLIED
    append_policy_status(
        policy_id=validated.policy_id,
        file_hash=file_hash,
        source_path=str(path),
        status=PolicyStatus.APPLIED,
    )

    # 5) archive
    archived = _maybe_archive(path, file_hash, "processed", skip_archive_if_missing)

    return PipelineResult(
        file=path,
        final_status=PolicyStatus.APPLIED,
        policy_id=validated.policy_id,
        archived_to=archived,
        cache_keys_invalidated=len(invalidation.invalidated_keys),
        summary_jobs_enqueued=len(jobs),
        neo4j_districts_linked=write_result_districts,
        neo4j_dongs_linked=write_result_dongs,
        neo4j_categories_linked=write_result_categories,
    )


# ---------------------------------------------------------------------------
# 헬퍼
# ---------------------------------------------------------------------------
def _terminate_failure(path: Path, file_hash: str, reason: str) -> PipelineResult:
    log.warning("pipeline failure for %s: %s", path.name, reason)
    archived = archive_failure(path, file_hash) if path.exists() else None
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


def _maybe_archive(path: Path, file_hash: str, kind: str, skip_if_missing: bool) -> Path | None:
    if not path.exists():
        return None
    if skip_if_missing and not path.is_file():
        return None
    if kind == "processed":
        return archive_success(path, file_hash)
    return archive_failure(path, file_hash)
