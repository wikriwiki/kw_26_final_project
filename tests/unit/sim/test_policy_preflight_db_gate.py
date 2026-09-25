"""검증 런의 정책 노출 배선 검사를 조용히 건너뛰지 않는다."""

from pathlib import Path

from scripts.sim import policy_preflight


def test_missing_db_is_a_failure_only_when_db_gate_is_required(monkeypatch):
    monkeypatch.delenv("NEO4J_URI", raising=False)
    path = Path("unused_policy.json")

    static = policy_preflight.check_db_wiring(path)
    validation = policy_preflight.check_db_wiring(path, require_db=True)

    assert static[0][0] == policy_preflight._WARN
    assert validation[0][0] == policy_preflight._FAIL
    assert "DB 배선 점검 불가" in validation[0][1]
