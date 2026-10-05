from scripts.experiments.select_report_district_cohort import select


def _agent(aid, code, gu):
    return {"agent_id": aid, "residence": {"dong_code": code + "101", "gu": gu}}


def test_preserves_existing_roster_and_report_districts():
    agents = [
        _agent("n", "11350", "노원구"), _agent("s", "11650", "서초구"),
        _agent("p", "11710", "송파구"), _agent("other", "11110", "종로구"),
        _agent("extra", "11350", "노원구"),
    ]
    selected, audit = select(agents, [a["agent_id"] for a in agents],
                             ["other", "p", "n", "s"])
    assert selected == ["n", "p", "s"]
    assert audit["selected_by_district"] == {"11350": 1, "11650": 1, "11710": 1}
    assert audit["eligible_by_district"] == {"11350": 2, "11650": 1, "11710": 1}


def test_rejects_conflicting_residence_code_and_name():
    import pytest

    with pytest.raises(ValueError, match="District code/name mismatch"):
        select([_agent("a", "11350", "서초구")], ["a"], ["a"])
