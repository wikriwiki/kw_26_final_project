"""Date-boundary and leakage checks for the preparation-only context builder."""
from datetime import date
import pytest
from scripts.experiments.preview_covid_context import build_context


def test_support_launch_also_changes_dining_hours():
    before = build_context(date(2021, 9, 5), "마포구")
    after = build_context(date(2021, 9, 6), "마포구")
    assert before["distancing"]["dine_in_cutoff"] == "21:00"
    assert after["distancing"]["dine_in_cutoff"] == "22:00"
    assert after["latest_city_reference_date"] < after["day"]
    assert after["latest_case_source_date"] == "2021-09-04"
    assert "710만" not in after["prompt_preview"]
    assert "국민지원금" not in after["prompt_preview"]


def test_chuseok_family_exception_is_date_and_venue_scoped():
    assert not build_context(date(2021, 9, 16), "종로구")["overrides"]
    during = build_context(date(2021, 9, 17), "종로구")
    assert during["overrides"][0]["scope"] == "가정 내 가족모임만"
    assert "가정 밖 식당 모임에는 적용되지 않는다" in during["prompt_preview"]
    assert not build_context(date(2021, 9, 24), "종로구")["overrides"]


def test_december_rules_supersede_initial_reopening():
    early = build_context(date(2021, 12, 5), "강남구")
    middle = build_context(date(2021, 12, 6), "강남구")
    late = build_context(date(2021, 12, 18), "강남구")
    assert early["distancing"]["private_meetings"]["total_max"] == 10
    assert middle["distancing"]["private_meetings"]["total_max"] == 6
    assert late["distancing"]["private_meetings"]["total_max"] == 4
    assert late["distancing"]["dine_in_cutoff"] == "21:00"


@pytest.mark.parametrize("day,district", [(date(2022, 1, 1), "강남구"), (date(2021, 9, 6), "타시도"), (date(2021, 10, 1), "마포구")])
def test_uncovered_or_inconsistent_context_is_not_silently_accepted(day, district):
    with pytest.raises(ValueError):
        build_context(day, district)


def test_missing_vaccination_is_not_imputed_as_observed_or_ready():
    context = build_context(date(2021, 9, 6), "강남구")
    assert context["vaccination_status"] == "unknown"
    assert context["constraints_enforced"] is False
    assert context["runtime_integration"] == "not_implemented"
