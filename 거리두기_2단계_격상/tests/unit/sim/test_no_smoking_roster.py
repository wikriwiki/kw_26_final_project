import hashlib
import json

import pytest

from scripts.experiments import select_no_smoking_roster as roster


def test_preserves_full_eligible_intersection_and_returns_exact_sorted_size():
    source = ["removed-a", "retained-b", "removed-c", "retained-d"]
    eligible = ["retained-b", "retained-d", "new-a", "new-b", "new-c"]
    selected, audit = roster.select_roster(source, eligible, 4, 20171203)
    assert selected == sorted(selected) and len(selected) == 4
    assert {"retained-b", "retained-d"} <= set(selected) <= set(eligible)
    assert audit["removed_source_ids"] == ["removed-a", "removed-c"]
    assert audit["added_count"] == audit["removed_count"] == 2
    assert set(audit["added_replacement_ids"]) == set(selected) - set(source)
    assert audit["demographics_smoking_and_outcomes_used"] is False
    assert audit["selected_ids_sha256"] == roster.canonical_hash(selected)


def test_order_changes_do_not_change_selection_or_audit():
    source = ["a", "b", "c"]
    eligible = ["a", "d", "e", "f", "g"]
    actual = roster.select_roster(source, eligible, 3, 17)
    assert actual == roster.select_roster(list(reversed(source)), list(reversed(eligible)), 3, 17)
    ranking = sorted(eligible[1:], key=lambda aid: (hashlib.sha256(f"no-smoking-roster:17:{aid}".encode()).hexdigest(), aid))
    assert actual[1]["added_replacement_ids"] == sorted(ranking[:2])


def test_no_replacements_when_entire_original_roster_is_eligible():
    selected, audit = roster.select_roster(["b", "a"], ["c", "b", "a"], 2, 42)
    assert selected == ["a", "b"]
    assert audit["added_replacement_ids"] == audit["removed_source_ids"] == []


@pytest.mark.parametrize("invalid", [["a", "a"], [], [None], [1], [" "], [" a"], {"id": "a"}])
@pytest.mark.parametrize("side", ["source", "eligible"])
def test_invalid_or_duplicate_ids_are_rejected(invalid, side):
    source, eligible = ([invalid, ["a", "b"]] if side == "source" else [["a", "b"], invalid])
    with pytest.raises(ValueError, match="unique nonblank"):
        roster.select_roster(source, eligible, 2, 42)


def test_inadequate_eligible_pool_is_rejected_without_dropping_retained_members():
    with pytest.raises(ValueError, match="insufficient"):
        roster.select_roster(["a", "b", "c"], ["a", "d"], 3, 42)


@pytest.mark.parametrize("size", [0, -1, True, 2.5])
def test_invalid_requested_size_is_rejected(size):
    with pytest.raises(ValueError, match="positive integer"):
        roster.select_roster(["a"], ["a"], size, 42)


def test_fixed_source_roster_size_must_match_request():
    with pytest.raises(ValueError, match="source roster size"):
        roster.select_roster(["a", "b"], ["a", "b", "c"], 3, 42)


def test_cli_freezes_exact_file_hashes_and_refuses_overwrite(tmp_path):
    source, eligible, out = tmp_path / "source.json", tmp_path / "eligible.json", tmp_path / "frozen"
    source.write_text('["a", "b"]', encoding="utf-8")
    eligible.write_text('["a", "c", "d"]', encoding="utf-8")
    args = ["--source-ids", str(source), "--eligible-ids", str(eligible), "--size", "2", "--seed", "42", "--out", str(out)]
    assert roster.main(args) == 0
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["source_file_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["eligible_file_sha256"] == hashlib.sha256(eligible.read_bytes()).hexdigest()
    selected = (out / "cohort_ids.json").read_bytes()
    with pytest.raises(SystemExit):
        roster.main(args)
    assert (out / "cohort_ids.json").read_bytes() == selected
