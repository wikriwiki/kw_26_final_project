"""검증 런의 정책 노출 배선 검사를 조용히 건너뛰지 않는다."""

from pathlib import Path
import sys
from types import SimpleNamespace

from scripts.sim import policy_preflight


def test_missing_db_is_a_failure_only_when_db_gate_is_required(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    path = Path("unused_policy.json")

    static = policy_preflight.check_db_wiring(path)
    validation = policy_preflight.check_db_wiring(path, require_db=True)

    assert static[0][0] == policy_preflight._WARN
    assert validation[0][0] == policy_preflight._FAIL
    assert "DB 배선 점검 불가" in validation[0][1]


def test_off_arm_rejects_leftover_policy(monkeypatch):
    monkeypatch.setenv("NEO4J_URI", "bolt://test:7687")

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def run(self, query):
            count = 1 if "count(p)" in query else 25
            return SimpleNamespace(single=lambda: {"c": count})

    class Driver:
        def session(self, **_kwargs):
            return Session()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(
        GraphDatabase=SimpleNamespace(driver=lambda *_args, **_kwargs: Driver())
    ))

    result = policy_preflight.check_no_policy_db()

    assert result[0][0] == policy_preflight._FAIL
    assert "Policy 1개, applied_to 25개" in result[0][1]
