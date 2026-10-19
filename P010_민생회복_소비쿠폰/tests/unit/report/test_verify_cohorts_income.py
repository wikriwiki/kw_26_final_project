"""소득표가 없는 런도 결제원장을 뽑을 수 있어야 한다 — 단, 런 내내 같은 방식일 때만.

앵커 비례 소득(EXP_DAILY_INCOME=anchor:<k>)은 표가 없고 설정이 실행 지문에 들어간다.
표가 없다는 이유로 원장 추출을 막으면 30일 넘는 런을 보존할 수 없다. 대신 날마다
방식이 바뀌면(어떤 날은 표, 어떤 날은 없음) 여전히 거부한다.
"""
from __future__ import annotations

import json

import pytest

from scripts.report.export_cashback_month import verify_cohorts

H = "a" * 64


def write(tmp_path, days, roster, income):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for d, inc in zip(days, income):
        c = {"agent_ids": roster, "execution_fingerprint": "fp", "run_id": "r",
             "prompt_variant": "v53", "system_prompt_sha256": H,
             "stage2_system_prompt_sha256": H}
        if inc is not None:
            c["baseline_income_map_sha256"] = inc
        (tmp_path / ("cohort_%s.json" % d)).write_text(json.dumps(c), encoding="utf-8")
    return metrics


def test_a_run_with_no_income_map_on_every_day_passes(tmp_path):
    days = ["2021-10-01", "2021-10-02"]
    m = write(tmp_path, days, ["a", "b"], [None, None])
    r = verify_cohorts(m, days, ["a", "b"])
    assert r["baseline_income_map_sha256"] is None
    assert "fingerprint" in r["income_mode"]


def test_a_run_with_the_same_map_every_day_still_passes(tmp_path):
    days = ["2021-10-01", "2021-10-02"]
    m = write(tmp_path, days, ["a"], ["m1", "m1"])
    assert verify_cohorts(m, days, ["a"])["baseline_income_map_sha256"] == "m1"


def test_switching_between_map_and_no_map_mid_run_is_refused(tmp_path):
    days = ["2021-10-01", "2021-10-02"]
    m = write(tmp_path, days, ["a"], ["m1", None])
    with pytest.raises(ValueError, match="changed within run"):
        verify_cohorts(m, days, ["a"])
