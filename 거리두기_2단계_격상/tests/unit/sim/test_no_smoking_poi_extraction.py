import copy

import pytest

from scripts.experiments import extract_no_smoking_pois as extraction
from scripts.experiments.no_smoking_zone import normalize_pois


def candidate(name, category="스포츠", code=None):
    return {"poi_id": "C_source_id", "name": name, "upjong_l3": code,
            "district_code": "11350", "category": category, "parent": "여가",
            "props": {"id": "C_source_id", "name": name, "type": "commerce"}}


@pytest.mark.parametrize("name,category,facility", [
    ("동원당구장", "당구", "billiard"),
    ("매니아BilliardClub", "당구", "billiard"),
    ("시카고포켓볼", "당구", "billiard"),
    ("골프존다원실내", "스포츠", "indoor_golf"),
    ("골프존M클럽스크린", "스포츠", "screen_golf"),
    ("와이비 골프 스크린", "스포츠", "screen_golf"),
    ("초안산스크린골프카페", "스포츠", "screen_golf"),
])
def test_explicit_names_with_matching_category_can_enter_registry(name, category, facility):
    decision = extraction.classify(candidate(name, category))
    assert decision["status"] == "included"
    assert decision["facility_type"] == facility
    assert not decision["flags"]["registration_verified"]


@pytest.mark.parametrize("name,category", [
    ("용인대탑복싱", "당구"), ("프렌즈골프", "당구"),
    ("브릭스클라이밍짐", "당구"), ("챔프3000", "당구"),
    ("레저로스크린파크골프수락산점", "스포츠"),
    ("한스파크스크린골프", "스포츠"),
    ("골프존파크서초", "스포츠"), ("프렌즈스크린송파", "스포츠"),
    ("일반골프연습장", "스포츠"), ("GDR아카데미", "스포츠"),
    ("스크린골프시스템판매", "스포츠"), ("실외스크린골프", "스포츠"),
    ("실내테니스", "스포츠"), ("테스트당구용품", "당구"),
])
def test_ambiguous_brands_other_sports_and_conflicting_markers_are_not_included(name, category):
    decision = extraction.classify(candidate(name, category))
    assert decision["status"] != "included"
    assert decision["facility_type"] is None


def test_code_neither_overrides_name_uncertainty_nor_conflicting_categories():
    assert extraction.classify(candidate("일반골프연습장", code="R10311"))["status"] == "pending"
    assert extraction.classify(candidate("스크린골프", code="R10312"))["rule_id"] == "sports_code_not_golf_practice"
    assert extraction.classify(candidate("명시당구장", "당구", "R10311"))["rule_id"] == "category_code_conflict"


def test_duplicate_and_contradictory_source_identifiers_are_quarantined():
    row = candidate("스크린골프")
    registry, audit = extraction.extract([row, copy.deepcopy(row)], "a" * 64, "audit.json")
    assert not registry["pois"]
    assert audit["summary"]["rule_counts"] == {"duplicate_poi_identity": 2}
    row["props"]["name"] = "다른상호"
    assert extraction.classify(row)["rule_id"] == "conflicting_source_identity_or_name"


def test_output_normalizes_and_keeps_raw_evidence_out_of_runtime_registry():
    registry, audit = extraction.extract([candidate("스크린골프")], "b" * 64, "audit.json")
    assert normalize_pois(registry) == registry["pois"]
    assert set(registry["pois"][0]) == {"poi_id", "district_code", "facility_type", "classification_source"}
    assert audit["included"][0]["raw_name"] == "스크린골프"
    assert audit["source_sha256"] in registry["pois"][0]["classification_source"]


def test_district_and_commerce_scope_are_required():
    row = candidate("스크린골프")
    row["district_code"] = "11680"
    assert extraction.classify(row)["status"] == "excluded"
    row["district_code"] = "11350"
    row["props"]["type"] = "residence"
    assert extraction.classify(row)["status"] == "excluded"
