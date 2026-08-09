"""Process ownership and the physical B8 execution lock."""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .store import StoreError


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def pid_exists(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


class RunLock:
    """Cross-process lock backed by an atomic create.

    A second process cannot win the same lock even if its UI has stale state.
    We never overwrite an existing lock from the API; an operator must release
    a lock that is no longer owned by a live child process explicitly.
    """

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> dict | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(500, "실행 lock을 읽지 못했습니다", str(exc)) from exc
        return payload if isinstance(payload, dict) else None

    def status(self) -> dict:
        payload = self.read()
        if payload is None:
            return {"locked": False, "owner": None, "unknown": []}
        pid = payload.get("pid")
        return {
            "locked": True,
            "owner": payload,
            "process_alive": pid_exists(pid) if pid else None,
            "stale": bool(pid) and not pid_exists(pid),
            "unknown": [],
        }

    def acquire(
        self,
        *,
        run_id: str,
        policy_id: str,
        owner: str = "web-console",
        plan: dict | None = None,
    ) -> dict:
        existing = self.read()
        if existing is not None:
            raise StoreError(
                409,
                "이미 실행 중인 run이 있습니다. 기존 lock을 해제할 때까지 재실행할 수 없습니다.",
                json.dumps(self.status(), ensure_ascii=False),
            )
        payload = {
            "run_id": run_id,
            "policy_id": policy_id,
            "owner": owner,
            "started_at": utc_now(),
            "pid": None,
            "state": "starting",
            "lock_file": str(self.path),
            "plan": {key: value for key, value in (plan or {}).items() if value is not None},
        }
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            # The race is intentional: another worker won the physical lock.
            raise StoreError(409, "이미 다른 실행 요청이 lock을 보유했습니다") from exc
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(payload, fp, ensure_ascii=False, indent=2)
                fp.write("\n")
        except Exception:
            self.path.unlink(missing_ok=True)
            raise
        return payload

    def attach_process(self, pid: int, *, state: str = "running") -> dict:
        payload = self.read()
        if payload is None:
            raise StoreError(409, "실행 lock이 사라졌습니다")
        payload = {**payload, "pid": pid, "state": state}
        self._atomic_write(payload)
        return payload

    def _atomic_write(self, payload: dict) -> None:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=self.path.parent, prefix=f".{self.path.name}.", suffix=".tmp", delete=False
        ) as fp:
            temp = Path(fp.name)
            json.dump(payload, fp, ensure_ascii=False, indent=2)
            fp.write("\n")
        temp.replace(self.path)

    def release(self, *, expected_pid: int | None = None, force: bool = False) -> dict:
        payload = self.read()
        if payload is None:
            return {"locked": False, "released": False, "unknown": []}
        current_pid = payload.get("pid")
        if expected_pid is not None and current_pid not in (None, expected_pid):
            raise StoreError(409, "현재 lock 소유 프로세스와 요청이 일치하지 않습니다")
        if current_pid and pid_exists(current_pid) and not force:
            raise StoreError(409, "실행 프로세스가 살아 있습니다. 먼저 명시적 중단을 요청해야 합니다.")
        self.path.unlink(missing_ok=True)
        return {"locked": False, "released": True, "previous": payload, "unknown": []}


class Runner:
    """Starts only the explicitly configured command and owns its lock."""

    def __init__(self, *, repo_root: Path, lock: RunLock):
        self.repo_root = repo_root.resolve()
        self.lock = lock

    def configured_command(self) -> list[str] | None:
        raw_json = os.environ.get("SIM_RUN_COMMAND_JSON")
        if raw_json:
            try:
                command = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                raise StoreError(500, "SIM_RUN_COMMAND_JSON이 유효한 JSON이 아닙니다", str(exc)) from exc
            if isinstance(command, list) and all(isinstance(item, str) and item for item in command):
                return command
            raise StoreError(500, "SIM_RUN_COMMAND_JSON은 문자열 배열이어야 합니다")
        raw = os.environ.get("SIM_RUN_COMMAND")
        if raw:
            return shlex.split(raw, posix=False)
        return None

    def start(self, *, run_id: str, policy_id: str, plan: dict | None = None) -> dict:
        command = self.configured_command()
        if not command:
            raise StoreError(
                503,
                "실행 명령이 구성되지 않았습니다. SIM_RUN_COMMAND_JSON을 운영자가 설정해야 합니다.",
            )
        lock = self.lock.acquire(run_id=run_id, policy_id=policy_id, plan=plan)
        # 실행 파라미터는 명령줄을 조립해 넘기지 않고 **환경변수로만** 전달한다.
        # 사용자가 임의 인자를 프로세스에 밀어 넣을 경로를 만들지 않기 위해서다.
        environment = dict(os.environ)
        environment["SIM_RUN_ID"] = str(run_id)
        environment["SIM_POLICY_ID"] = str(policy_id)
        for key, env_name in (("start_day", "SIM_START_DAY"), ("days", "SIM_DAYS"), ("agents", "SIM_AGENTS")):
            value = (plan or {}).get(key)
            if value is not None:
                environment[env_name] = str(value)
        try:
            proc = subprocess.Popen(
                command,
                cwd=self.repo_root,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            lock = self.lock.attach_process(proc.pid)
        except Exception as exc:
            self.lock.release(force=True)
            raise StoreError(500, "시뮬레이션 프로세스를 시작하지 못했습니다", str(exc)) from exc
        return {"accepted": True, "lock": lock, "command": command, "unknown": []}

    def request_stop(self) -> dict:
        payload = self.lock.read()
        if payload is None:
            raise StoreError(409, "중단할 실행이 없습니다")
        pid = payload.get("pid")
        if not pid or not pid_exists(pid):
            return self.lock.release(force=True)
        try:
            # SIGINT is a graceful request. The console never sends SIGKILL,
            # kills by name, or touches a process it does not own.
            os.kill(pid, signal.SIGINT)
        except OSError as exc:
            raise StoreError(409, "소유 프로세스에 graceful 중단 신호를 보내지 못했습니다", str(exc)) from exc
        updated = {**payload, "state": "stop_requested", "stop_requested_at": utc_now()}
        self.lock._atomic_write(updated)
        return {"accepted": True, "lock": updated, "unknown": []}

