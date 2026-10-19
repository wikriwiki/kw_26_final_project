"""The console must never terminate a simulation while reading its status."""
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

from web.api.runner import RunLock, Runner, pid_exists
from web.api.store import StoreError


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process API")
def test_windows_status_and_stop_keep_owned_process_alive(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    lock = RunLock(tmp_path / "run.lock")
    runner = Runner(repo_root=tmp_path, lock=lock)
    try:
        lock.acquire(run_id="test", policy_id="P010")
        lock.attach_process(child.pid)
        with patch("web.api.runner.os.kill", side_effect=AssertionError("unsafe signal")):
            assert pid_exists(child.pid)
            assert lock.status()["process_alive"] is True
            with pytest.raises(StoreError, match="Windows"):
                runner.request_stop()
            with pytest.raises(StoreError):
                lock.release()
        assert child.poll() is None
        assert lock.read()["pid"] == child.pid
    finally:
        child.terminate()
        child.wait(timeout=10)
        lock.release(force=True)
    assert not pid_exists(child.pid)


def test_permission_denied_does_not_release_live_lock():
    with patch("web.api.runner.sys.platform", "linux"), patch(
        "web.api.runner.os.kill", side_effect=PermissionError
    ):
        assert pid_exists(12345)


def test_start_uses_fixed_command_and_passes_plan_only_in_child_environment(tmp_path):
    runner = Runner(repo_root=tmp_path, lock=RunLock(tmp_path / "run.lock"))
    env = {"SIM_RUN_COMMAND_JSON": '["python", "fixed_runner.py"]'}
    with patch.dict(os.environ, env), patch("web.api.runner.subprocess.Popen") as spawn:
        spawn.return_value.pid = 12345
        runner.start(run_id="trial", policy_id="P010", plan={"days": 3, "agents": 10})
        assert spawn.call_args.args == (["python", "fixed_runner.py"],)
        options = spawn.call_args.kwargs
        assert options["cwd"] == tmp_path.resolve()
        assert options["env"]["SIM_RUN_ID"] == "trial"
        assert options["env"]["SIM_POLICY_ID"] == "P010"
        assert options["env"]["SIM_DAYS"] == "3"
        assert options["env"]["SIM_AGENTS"] == "10"
        assert options.get("shell", False) is False
        assert runner.lock.read()["pid"] == 12345
        with pytest.raises(StoreError):
            runner.start(run_id="second", policy_id="P010")
        assert spawn.call_count == 1


def test_spawn_failure_releases_starting_lock(tmp_path):
    runner = Runner(repo_root=tmp_path, lock=RunLock(tmp_path / "run.lock"))
    with patch.dict(os.environ, {"SIM_RUN_COMMAND_JSON": '["missing"]'}), patch(
        "web.api.runner.subprocess.Popen", side_effect=FileNotFoundError
    ):
        with pytest.raises(StoreError):
            runner.start(run_id="trial", policy_id="P010")
    assert runner.lock.read() is None


def test_empty_configured_command_is_rejected_before_lock(tmp_path):
    runner = Runner(repo_root=tmp_path, lock=RunLock(tmp_path / "run.lock"))
    with patch.dict(os.environ, {"SIM_RUN_COMMAND_JSON": "[]"}):
        with pytest.raises(StoreError):
            runner.start(run_id="trial", policy_id="P010")
    assert runner.lock.read() is None


def test_posix_stop_signals_only_owned_pid_and_keeps_lock(tmp_path):
    import signal

    lock = RunLock(tmp_path / "run.lock")
    runner = Runner(repo_root=tmp_path, lock=lock)
    lock.acquire(run_id="trial", policy_id="P010")
    lock.attach_process(12345)
    with patch("web.api.runner.sys.platform", "linux"), patch(
        "web.api.runner.pid_exists", return_value=True
    ), patch("web.api.runner.os.kill") as kill:
        assert runner.request_stop()["accepted"]
        kill.assert_called_once_with(12345, signal.SIGINT)
    assert lock.read()["state"] == "stop_requested"


def test_start_without_command_does_not_save_policy(tmp_path):
    from pathlib import Path
    from fastapi.testclient import TestClient
    from web.api.app import create_app
    from web.api.store import ArtifactStore

    root = Path(__file__).resolve().parents[2]
    store = ArtifactStore(repo_root=root, data_root=tmp_path, fixture_dir=root / "web/fixtures")
    runner = Runner(repo_root=tmp_path, lock=RunLock(tmp_path / "run.lock"))
    client = TestClient(create_app(store=store, runner=runner, read_only=False))
    with patch.object(runner, "configured_command", return_value=None), patch.object(
        store, "save_policy"
    ) as save:
        response = client.post("/api/runner/start", json={
            "run_id": "trial", "policy_id": "P010", "policy": {"id": "P010"}
        })
    assert response.status_code == 503
    save.assert_not_called()
    assert runner.lock.read() is None
