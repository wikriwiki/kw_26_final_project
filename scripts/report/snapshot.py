"""Immutable run-snapshot checks for the DASOL report boundary.

The existing DASOL renderer reads Neo4j, while the console selects a run from
the file-backed simulation outputs.  This module makes that boundary explicit:
the selected run must be complete, its source files are hashed before the job,
and the CLI verifies the same hashes again before importing the renderer.

This is intentionally an adapter contract.  It does not alter the protected
simulation or report-engine files and it never fabricates a missing source.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


class SnapshotError(ValueError):
    """A selected run cannot be treated as an immutable report snapshot."""


STATIC_FILES = ("summary.json", "events.jsonl", "poi_summary.json")
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_inside(path: Path, root: Path, *, label: str) -> Path:
    candidate = path.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise SnapshotError(f"{label}이 허용된 run root 밖에 있습니다") from exc
    return candidate


def _hash_file(path: Path) -> tuple[int, str]:
    try:
        before = path.stat()
    except OSError as exc:
        raise SnapshotError(f"snapshot 원본을 읽을 수 없습니다: {path.name}") from exc
    if not path.is_file() or before.st_size <= 0:
        raise SnapshotError(f"snapshot 원본이 비어 있거나 파일이 아닙니다: {path.name}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(block)
        after = path.stat()
    except OSError as exc:
        raise SnapshotError(f"snapshot 원본을 읽는 중 실패했습니다: {path.name}") from exc
    if before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns:
        raise SnapshotError(f"snapshot 원본이 검사 중 변경되었습니다: {path.name}")
    return before.st_size, digest.hexdigest()


def _date_list(start: date, days: int) -> list[str]:
    return [(start + timedelta(days=index)).isoformat() for index in range(days)]


def _plan_days(run: dict[str, Any]) -> tuple[int, date, list[str]]:
    plan = run.get("plan") if isinstance(run.get("plan"), dict) else {}
    planned = plan.get("planned_days")
    start_raw = plan.get("start_day")
    days_present = run.get("days_present")
    if not isinstance(planned, int) or planned < 1:
        raise SnapshotError("run snapshot의 planned_days가 확인되지 않습니다")
    if not isinstance(start_raw, str):
        raise SnapshotError("run snapshot의 start_day가 확인되지 않습니다")
    try:
        start = date.fromisoformat(start_raw)
    except ValueError as exc:
        raise SnapshotError("run snapshot의 start_day가 올바른 날짜가 아닙니다") from exc
    if not isinstance(days_present, list) or not all(isinstance(item, str) for item in days_present):
        raise SnapshotError("run snapshot의 days_present가 올바르지 않습니다")
    expected = _date_list(start, planned)
    if days_present != expected:
        raise SnapshotError("완료 run의 days_present가 계획 범위와 일치하지 않습니다")
    for key in ("days_with_timing", "days_with_done_checkpoint"):
        values = run.get(key)
        if not isinstance(values, list) or set(values) != set(expected):
            raise SnapshotError(f"완료 run의 {key}가 전체 계획 일자를 덮지 않습니다")
    return planned, start, expected


def _source_paths(days: Iterable[str]) -> list[str]:
    paths = list(STATIC_FILES)
    for day in days:
        paths.extend(
            (
                f"metrics/day_{day}.jsonl",
                f"timing/day_{day}.json",
                f"checkpoints/done_{day}.json",
            )
        )
    return paths


def snapshot_readiness(*, run_id: str, run: dict[str, Any], data_root: Path) -> dict[str, Any]:
    """Cheap catalog-time readiness check; it deliberately does not hash files."""

    try:
        root = Path(str(run.get("root", ""))).resolve()
        data_root = data_root.resolve()
        _resolve_inside(root, data_root, label="run root")
        if run_id != run.get("run_id", run_id):
            raise SnapshotError("run_id와 run detail이 다릅니다")
        if run.get("status") != "completed":
            raise SnapshotError("완료된 run만 report snapshot으로 사용할 수 있습니다")
        _, _, days = _plan_days(run)
        paths = _source_paths(days)
        missing = [rel for rel in paths if not (root / rel).is_file() or (root / rel).stat().st_size <= 0]
        if missing:
            raise SnapshotError(f"snapshot 원본이 없습니다: {', '.join(missing[:4])}")
        return {
            "ready": True,
            "run_id": run_id,
            "root_relative": root.relative_to(data_root).as_posix(),
            "days": days,
            "source_count": len(paths),
            "unknown": [],
        }
    except (OSError, SnapshotError) as exc:
        return {
            "ready": False,
            "run_id": run_id,
            "root_relative": None,
            "days": run.get("days_present", []),
            "source_count": 0,
            "reason": str(exc),
            "unknown": [],
        }


def build_manifest(
    *,
    run_id: str,
    run: dict[str, Any],
    data_root: Path,
    requested_start: date,
    requested_days: int,
) -> dict[str, Any]:
    """Hash a completed run and return a manifest suitable for job input."""

    if requested_days < 1 or requested_days > 365:
        raise SnapshotError("report 기간은 1~365일이어야 합니다")
    if run_id != run.get("run_id", run_id):
        raise SnapshotError("run_id와 run detail이 다릅니다")
    if run.get("status") != "completed":
        raise SnapshotError("완료된 run snapshot에서만 report를 생성할 수 있습니다")

    root = Path(str(run.get("root", ""))).resolve()
    data_root = data_root.resolve()
    _resolve_inside(root, data_root, label="run root")
    _, plan_start, available_days = _plan_days(run)
    requested = _date_list(requested_start, requested_days)
    if not set(requested).issubset(set(available_days)):
        raise SnapshotError("요청한 report 기간이 완료 run snapshot 범위를 벗어났습니다")
    if requested_start < plan_start:
        raise SnapshotError("요청한 report 시작일이 run snapshot보다 빠릅니다")

    files: list[dict[str, Any]] = []
    for relative in _source_paths(available_days):
        path = _resolve_inside(root / relative, root, label="snapshot source")
        size, digest = _hash_file(path)
        files.append({"path": relative, "bytes": size, "sha256": digest})

    manifest_core = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "root": str(root),
        "root_relative": root.relative_to(data_root).as_posix(),
        "run_completed_at": run.get("completed_at"),
        "plan": run.get("plan", {}),
        "available_days": available_days,
        "requested": {
            "start": requested_start.isoformat(),
            "days": requested_days,
            "end": requested[-1],
        },
        "files": files,
    }
    snapshot_id = hashlib.sha256(
        json.dumps(
            {"run_id": run_id, "files": files},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    return {**manifest_core, "snapshot_id": snapshot_id, "created_at": _utc_now(), "unknown": []}


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write a manifest atomically inside the already controlled output root."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as fp:
        temporary = Path(fp.name)
        json.dump(manifest, fp, ensure_ascii=False, indent=2)
        fp.write("\n")
    temporary.replace(path)
    return path


def verify_manifest(
    path: Path,
    *,
    expected_run_id: str,
    requested_start: date,
    requested_days: int,
    data_root: Path | None = None,
) -> dict[str, Any]:
    """Re-read and hash every source in a manifest immediately before running."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SnapshotError("snapshot manifest를 읽지 못했습니다") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotError("snapshot manifest schema가 지원되지 않습니다")
    if payload.get("run_id") != expected_run_id:
        raise SnapshotError("snapshot manifest의 run_id가 요청과 다릅니다")
    requested = payload.get("requested")
    if not isinstance(requested, dict) or requested.get("start") != requested_start.isoformat() or requested.get("days") != requested_days:
        raise SnapshotError("snapshot manifest의 report 범위가 요청과 다릅니다")
    root = Path(str(payload.get("root", ""))).resolve()
    if data_root is not None:
        _resolve_inside(root, data_root.resolve(), label="manifest run root")
    available_days = payload.get("available_days")
    plan = payload.get("plan")
    if not isinstance(available_days, list) or not all(isinstance(day, str) for day in available_days):
        raise SnapshotError("snapshot manifest의 available_days가 올바르지 않습니다")
    if not isinstance(plan, dict) or not isinstance(plan.get("planned_days"), int) or not isinstance(plan.get("start_day"), str):
        raise SnapshotError("snapshot manifest의 plan이 올바르지 않습니다")
    expected_days = _date_list(date.fromisoformat(plan["start_day"]), plan["planned_days"])
    if available_days != expected_days:
        raise SnapshotError("snapshot manifest의 available_days가 plan과 일치하지 않습니다")
    entries = payload.get("files")
    if not isinstance(entries, list) or not entries:
        raise SnapshotError("snapshot manifest에 source file이 없습니다")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise SnapshotError("snapshot manifest의 source entry가 올바르지 않습니다")
        relative = entry["path"].replace("\\", "/")
        if relative in seen or relative.startswith("/") or "/../" in f"/{relative}/" or relative.startswith("../"):
            raise SnapshotError("snapshot manifest의 source 경로가 안전하지 않습니다")
        seen.add(relative)
        path_on_disk = _resolve_inside(root / relative, root, label="manifest source")
        size, digest = _hash_file(path_on_disk)
        if size != entry.get("bytes") or digest != entry.get("sha256"):
            raise SnapshotError(f"snapshot source가 변경되었습니다: {relative}")
        normalized.append({"path": relative, "bytes": size, "sha256": digest})

    expected_paths = set(_source_paths(available_days))
    if seen != expected_paths:
        missing = sorted(expected_paths - seen)
        extra = sorted(seen - expected_paths)
        detail = f"누락={', '.join(missing[:3])}" if missing else f"추가={', '.join(extra[:3])}"
        raise SnapshotError(f"snapshot manifest의 source 집합이 run 계약과 다릅니다: {detail}")

    expected_id = hashlib.sha256(
        json.dumps(
            {"run_id": expected_run_id, "files": normalized},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    if payload.get("snapshot_id") != expected_id:
        raise SnapshotError("snapshot manifest의 snapshot_id가 source hash와 일치하지 않습니다")
    return payload


def report_source_paths(manifest: dict[str, Any]) -> list[str]:
    """Return compact, report-relevant source paths for provenance links."""

    requested = manifest.get("requested") if isinstance(manifest.get("requested"), dict) else {}
    start_raw = requested.get("start")
    days = requested.get("days")
    try:
        requested_days = _date_list(date.fromisoformat(str(start_raw)), int(days))
    except (TypeError, ValueError):
        requested_days = []
    wanted = set(STATIC_FILES)
    wanted.update(
        path
        for day in requested_days
        for path in (
            f"metrics/day_{day}.jsonl",
            f"timing/day_{day}.json",
            f"checkpoints/done_{day}.json",
        )
    )
    return [
        str(entry.get("path"))
        for entry in manifest.get("files", [])
        if isinstance(entry, dict) and entry.get("path") in wanted
    ]
