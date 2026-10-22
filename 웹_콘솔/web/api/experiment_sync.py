"""실험 서버 → 이 PC 의 미러. **읽기 전용** 동기화.

서버에서 실행하는 명령은 ``find``(목록), ``cat``·``tail -c``(내용), ``head -c | sha256sum``(앞부분 대조)
그리고 근거 묶음을 받을 때의 ``tar -c`` 뿐이다. 서버의 파일·프로세스·설정을 바꾸지 않는다.

접속 정보는 저장소에 두지 않는다. 실험 정의의 ``remote.profile`` 이름으로 이 PC 의 환경변수를 찾는다::

    SIM_REMOTE_NO_SMOKING_VAST="ssh -i C:/Users/<나>/.ssh/no_smoking_vast -p 45025 root@<서버>"

받는 것
-------
* 팔마다: ``experiment_run.json``, ``backup_completed_*.json``, ``night2_completed_*.json``, ``metrics/day_*.jsonl``
* 실험 번들: ``runtime.json``, ``personas.json`` (명단·흡연 라벨·프로필)
* 대상자 인터뷰 근거(``evidence/v1/...``)는 그 사람을 인터뷰할 때만 받는다 (``fetch_agent_evidence``).

진행 중인 날짜 파일은 계속 커진다. 앞부분이 이미 받은 것과 같으면(sha256 대조) 늘어난 뒷부분만 받는다.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from .experiments import Experiment, ExperimentStore, experiments_root

SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4")
ARM_PATTERNS = ("experiment_run.json", "backup_completed_*.json", "night2_completed_*.json")
COMMAND_TIMEOUT = 900
MAX_EVIDENCE_FILES = 64


class SyncError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def ssh_base(profile: str) -> list[str]:
    name = f"SIM_REMOTE_{profile}"
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise SyncError(f"원격 접속이 설정되지 않았습니다 ({name})")
    parts = shlex.split(raw, posix=True)
    if not parts or Path(parts[0]).name.lower() not in {"ssh", "ssh.exe"}:
        raise SyncError(f"{name} 는 ssh 명령이어야 합니다")
    return [parts[0], *SSH_OPTIONS, *parts[1:]]


def _remote_path(root: str, *parts: str) -> str:
    joined = posixpath.normpath(posixpath.join(root, *parts))
    if not joined.startswith(posixpath.normpath(root) + "/") and joined != posixpath.normpath(root):
        raise SyncError(f"원격 경로가 허용 범위를 벗어납니다: {joined}")
    return joined


class Remote:
    """읽기 전용 원격 명령 몇 가지. 실행 함수는 테스트에서 바꿔 끼울 수 있다."""

    def __init__(self, base: list[str], runner: Callable[..., subprocess.CompletedProcess] | None = None) -> None:
        self.base = base
        self.runner = runner or subprocess.run
        self.bytes_received = 0

    def run(self, command: str, *, compress: bool = False) -> bytes:
        args = [self.base[0], *(["-C"] if compress else []), *self.base[1:], command]
        result = self.runner(args, capture_output=True, timeout=COMMAND_TIMEOUT)
        if result.returncode != 0:
            message = (result.stderr or b"").decode("utf-8", "replace").strip().splitlines()
            raise SyncError(f"원격 명령 실패 (코드 {result.returncode}): {message[-1] if message else ''}")
        self.bytes_received += len(result.stdout)
        return result.stdout

    def list_files(self, directory: str, patterns: tuple[str, ...], subdir: str | None = None) -> dict[str, tuple[int, float]]:
        target = _remote_path(directory, subdir) if subdir else directory
        output = self.run(
            f"test -d {shlex.quote(target)} && find {shlex.quote(target)} -maxdepth 1 -type f -printf '%s %T@ %f\\n' || true"
        )
        files: dict[str, tuple[int, float]] = {}
        for line in output.decode("utf-8", "replace").splitlines():
            size, mtime, name = (line.split(" ", 2) + ["", ""])[:3]
            if not name or not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                continue
            files[f"{subdir}/{name}" if subdir else name] = (int(size), float(mtime))
        return files

    def cat(self, path: str) -> bytes:
        return self.run(f"cat {shlex.quote(path)}", compress=True)

    def tail_from(self, path: str, offset: int) -> bytes:
        return self.run(f"tail -c +{offset + 1} {shlex.quote(path)}", compress=True)

    def head_sha256(self, path: str, length: int) -> str:
        output = self.run(f"head -c {int(length)} {shlex.quote(path)} | sha256sum")
        return output.decode("ascii", "replace").split()[0]

    def tar(self, directory: str, members: list[str]) -> bytes:
        quoted = " ".join(shlex.quote(member) for member in members)
        return self.run(f"tar -C {shlex.quote(directory)} -cf - -- {quoted}", compress=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_atomic(path: Path, data: bytes, *, append_to: Path | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".part{os.getpid()}")
    with temp.open("wb") as handle:
        if append_to is not None:
            with append_to.open("rb") as source:
                for block in iter(lambda: source.read(1 << 20), b""):
                    handle.write(block)
        handle.write(data)
    # 윈도에서는 다른 요청이 그 파일을 읽는 순간 바꿔 끼우기가 실패한다. 잠깐 뒤 다시 한다.
    for attempt in range(10):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.2)


def _mirror_dir(remote: Remote, remote_dir: str, local_dir: Path, patterns: dict[str | None, tuple[str, ...]], state: dict[str, Any]) -> dict[str, int]:
    """한 폴더의 파일을 맞춘다. 상태 파일에 원격 크기·시각과 로컬 sha256 을 남긴다."""
    stats = {"checked": 0, "downloaded": 0, "appended": 0, "unchanged": 0}
    known = state.setdefault("files", {})
    for subdir, names in patterns.items():
        for relative, (size, mtime) in remote.list_files(remote_dir, names, subdir).items():
            stats["checked"] += 1
            local = local_dir / relative
            record = known.get(relative)
            if local.is_file() and record and record.get("remote_size") == size and record.get("remote_mtime") == mtime and local.stat().st_size == size:
                stats["unchanged"] += 1
                continue
            source = _remote_path(remote_dir, relative)
            local_size = local.stat().st_size if local.is_file() else 0
            appended = False
            if relative.endswith(".jsonl") and 0 < local_size < size:
                # 시뮬레이터는 날짜 파일에 줄을 덧붙이기만 한다. 앞부분이 같으면 뒤만 받는다.
                if remote.head_sha256(source, local_size) == _sha256(local):
                    _write_atomic(local, remote.tail_from(source, local_size), append_to=local)
                    appended = True
            if not appended:
                _write_atomic(local, remote.cat(source))
            stats["appended" if appended else "downloaded"] += 1
            known[relative] = {
                "remote_size": size,
                "remote_mtime": mtime,
                "local_size": local.stat().st_size,
                "sha256": _sha256(local),
                "synced_at": now(),
            }
    return stats


class SyncLock:
    def __init__(self, path: Path, stale_after: float = 3 * 3600) -> None:
        self.path = path
        self.stale_after = stale_after
        self.fd: int | None = None

    def __enter__(self) -> "SyncLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and time.time() - self.path.stat().st_mtime > self.stale_after:
            self.path.unlink(missing_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise SyncError("이미 동기화 중입니다") from exc
        os.write(self.fd, str(os.getpid()).encode())
        return self

    def __exit__(self, *_exc: Any) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)


def sync_experiment(experiment: Experiment, *, remote: Remote | None = None, store: ExperimentStore | None = None,
                    progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """한 실험의 팔·번들을 맞춘다. 끝나면 하루 집계 캐시를 미리 채운다."""
    profile = experiment.remote.get("profile")
    results_root = experiment.remote.get("results_root")
    if not profile or not results_root:
        raise SyncError("이 실험 정의에는 원격 위치가 없습니다")
    remote = remote or Remote(ssh_base(str(profile)))
    say = progress or (lambda _message: None)
    report: dict[str, Any] = {"experiment_id": experiment.experiment_id, "started_at": now(), "arms": {}, "bundle": None}
    with SyncLock(experiment.local_dir / "_sync.lock"):
        for arm in experiment.arms:
            say(f"{arm.arm_id} 확인 중")
            local_dir = experiment.arm_dir(arm)
            state_path = local_dir / "_sync.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            remote_dir = _remote_path(str(results_root), arm.directory)
            stats = _mirror_dir(remote, remote_dir, local_dir, {None: ARM_PATTERNS, "metrics": ("day_*.jsonl",)}, state)
            if stats["checked"]:
                state["synced_at"] = now()
                state["remote_profile"] = profile
                _write_atomic(state_path, json.dumps(state, ensure_ascii=False, indent=1).encode("utf-8"))
            report["arms"][arm.arm_id] = stats
        bundle = experiment.bundle or {}
        if bundle.get("directory") and experiment.remote.get("bundles_root"):
            say("실험 번들 확인 중")
            state_path = experiment.bundle_dir / "_sync.json"
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
            remote_dir = _remote_path(str(experiment.remote["bundles_root"]), str(bundle["directory"]))
            stats = _mirror_dir(remote, remote_dir, experiment.bundle_dir, {None: tuple(bundle.get("files") or ())}, state)
            state["synced_at"] = now()
            _write_atomic(state_path, json.dumps(state, ensure_ascii=False, indent=1).encode("utf-8"))
            report["bundle"] = stats
    if store is not None:
        say("하루 집계 계산 중")
        for arm in experiment.arms:
            for day in arm.day_list():
                store.cache.get(experiment, arm, day)
    report["finished_at"] = now()
    report["bytes_received"] = remote.bytes_received
    return report


def fetch_agent_evidence(store: ExperimentStore, experiment: Experiment, agent_id: str, *, remote: Remote | None = None,
                         through_day: str | None = None) -> dict[str, Any]:
    """한 사람의 인터뷰 근거(일별 저널)를 받는다. 이미 있는 파일은 건너뛴다."""
    profile = experiment.remote.get("profile")
    results_root = experiment.remote.get("results_root")
    if not profile or not results_root:
        raise SyncError("이 실험 정의에는 원격 위치가 없습니다")
    fetched = 0
    missing: dict[str, list[str]] = {}
    for arm in experiment.arms:
        wanted = []
        for day in arm.day_list():
            if through_day and day > through_day:
                break
            row = store.agent_row(experiment, arm, day, agent_id)
            reference = (row or {}).get("interview_evidence") or {}
            relative = reference.get("path")
            if not isinstance(relative, str) or not relative.startswith("evidence/"):
                continue
            pure = PurePosixPath(relative)
            if pure.is_absolute() or ".." in pure.parts:
                continue
            if not (experiment.arm_dir(arm) / relative).is_file():
                wanted.append(relative)
        if wanted:
            missing[arm.arm_id] = wanted[:MAX_EVIDENCE_FILES]
    if not missing:
        return {"fetched": 0, "bytes_received": 0}
    remote = remote or Remote(ssh_base(str(profile)))
    for arm_id, members in missing.items():
        arm = experiment.arm(arm_id)
        target = experiment.arm_dir(arm)
        payload = remote.tar(_remote_path(str(results_root), arm.directory), members)
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive.getmembers():
                name = PurePosixPath(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts or str(name) not in members:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                _write_atomic(target / str(name), handle.read())
                fetched += 1
    return {"fetched": fetched, "bytes_received": remote.bytes_received}


class SyncJobs:
    """콘솔의 "지금 받기" 버튼 — 한 실험에 하나씩, 배경에서 돈다."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def status(self, experiment_id: str) -> dict[str, Any]:
        with self.lock:
            job = dict(self.jobs.get(experiment_id) or {"state": "idle"})
        experiment = self.store.experiment(experiment_id)
        profile = experiment.remote.get("profile")
        job["remote_configured"] = bool(profile and os.environ.get(f"SIM_REMOTE_{profile}", "").strip())
        job["remote_profile"] = profile
        job["mirror_root_exists"] = experiments_root().is_dir()
        return job

    def start(self, experiment_id: str) -> dict[str, Any]:
        experiment = self.store.experiment(experiment_id)
        with self.lock:
            if (self.jobs.get(experiment_id) or {}).get("state") == "running":
                return dict(self.jobs[experiment_id])
            job = {"state": "running", "started_at": now(), "message": "시작", "report": None, "error": None}
            self.jobs[experiment_id] = job

        def update(message: str) -> None:
            with self.lock:
                job["message"] = message

        def work() -> None:
            try:
                report = sync_experiment(experiment, store=self.store, progress=update)
            except Exception as exc:  # 사용자에게 그대로 보여 줄 메시지만 남긴다
                with self.lock:
                    job.update({"state": "failed", "error": str(exc)[:300], "finished_at": now()})
                return
            with self.lock:
                job.update({"state": "done", "report": report, "message": "완료", "finished_at": now()})

        threading.Thread(target=work, name=f"sync-{experiment_id}", daemon=True).start()
        return dict(job)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="실험 서버에서 결과를 읽기 전용으로 받아 온다")
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args(argv)
    from scripts.report.llm import load_dotenv

    load_dotenv()
    store = ExperimentStore()
    experiment = store.experiment(args.experiment)
    report = sync_experiment(experiment, store=store, progress=lambda message: print("  ", message, flush=True))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    raise SystemExit(main())
