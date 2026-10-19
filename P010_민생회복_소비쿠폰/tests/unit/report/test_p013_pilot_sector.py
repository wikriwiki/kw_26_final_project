from __future__ import annotations

import pytest

from scripts.report.p013_pilot_sector import aggregate, pair


def test_sector_aggregation_is_by_frozen_citizen_and_ignores_other_sectors():
    rows = [{"aid": "a", "day": "2020-05-11", "amount": 10, "sector": "가구"},
            {"aid": "a", "day": "2020-05-11", "amount": 7, "sector": "미용실"},
            {"aid": "a", "day": "2020-05-11", "amount": 9, "sector": "마트"}]
    got = aggregate(rows, ["a", "b"], ["2020-05-11"], (["가구"], ["미용실"]))
    assert got == {"a": {"semidurable_won": 10, "face_service_won": 7},
                   "b": {"semidurable_won": 0, "face_service_won": 0}}


def test_paired_sector_rank_is_exploratory_proxy_and_requires_matching_arms():
    base = {"schema": "p013_sector_arm_v1", "roster": ["a", "b"],
            "days": ["2020-05-11"], "groups": [["가구"], ["미용실"]],
            "scoring_table_sha256": "abc"}
    off = {**base, "arm": "off", "by_aid": {
        "a": {"semidurable_won": 100, "face_service_won": 100},
        "b": {"semidurable_won": 100, "face_service_won": 100}}}
    on = {**base, "arm": "on", "by_aid": {
        "a": {"semidurable_won": 120, "face_service_won": 110},
        "b": {"semidurable_won": 120, "face_service_won": 110}}}
    got = pair(on, off, draws=20)
    assert got["rank_gap_percentage_points"] == pytest.approx(10)
    assert got["rank_same_as_registered_direction"] is True
    assert "not_external" in got["comparison"]
    with pytest.raises(ValueError, match="definitions differ"):
        pair({**on, "days": ["2020-05-12"]}, off)
