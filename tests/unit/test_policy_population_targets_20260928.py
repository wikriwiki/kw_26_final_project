"""Independent source-code and aggregate oracles for pre-policy census targets."""
from copy import deepcopy

import pytest

from tools.build_policy_population_targets_20260928 import demographic_counts


# KOSIS source age codes are upper-exclusive five-year edges; 101 means 100+.
# This literal oracle comes from the source labels, not the builder arithmetic.
AGE_ORACLE = {
    25: ("20 - 24세", "20대"), 30: ("25 - 29세", "20대"),
    35: ("30 - 34세", "30대"), 40: ("35 - 39세", "30대"),
    45: ("40 - 44세", "40대"), 50: ("45 - 49세", "40대"),
    55: ("50 - 54세", "50대"), 60: ("55 - 59세", "50대"),
    65: ("60 - 64세", "60세 이상"), 70: ("65 - 69세", "60세 이상"),
    75: ("70 - 74세", "60세 이상"), 80: ("75 - 79세", "60세 이상"),
    85: ("80 - 84세", "60세 이상"), 90: ("85 - 89세", "60세 이상"),
    95: ("90 - 94세", "60세 이상"), 100: ("95 - 99세", "60세 이상"),
    101: ("100+", "60세 이상"),
}


def census_rows():
    rows = []
    for dong_index, code in enumerate(("1111051500", "1111053000")):
        for sex_index, sex in enumerate(("남자", "여자")):
            for age_index, (age_code, (label, _)) in enumerate(AGE_ORACLE.items()):
                rows.append({"stats_ym": "202004", "admdong_cd": code,
                             "sigungu_nm": "종로구", "epmndn_nm": "동" + str(dong_index),
                             "sex_nm": sex, "agegrd_cd": str(age_code),
                             "agegrd_nm": label,
                             "popl_cnt": str(1 + age_index + 100 * sex_index + 1000 * dong_index)})
    return rows


def test_native_age_code_oracle_includes_20_24_and_100_plus_once():
    rows = census_rows()
    counts, joint, total = demographic_counts(rows, "202004")
    assert total == sum(int(row["popl_cnt"]) for row in rows)
    assert len(joint) == 2 * 2 * 17
    for band in ("20대", "30대", "40대", "50대", "60세 이상"):
        expected = sum(int(row["popl_cnt"]) for row in rows
                       if AGE_ORACLE[int(row["agegrd_cd"])][1] == band)
        assert counts["age_band"][band] == expected
    assert counts["sex"]["M"] == sum(int(r["popl_cnt"]) for r in rows if r["sex_nm"] == "남자")
    assert counts["sex"]["F"] == sum(int(r["popl_cnt"]) for r in rows if r["sex_nm"] == "여자")
    assert set(counts["admin_dong"]) == {"1111051500", "1111053000"}


def test_city_gu_aggregate_and_under20_do_not_double_count_adults():
    rows = census_rows()
    adults = sum(int(row["popl_cnt"]) for row in rows)
    for code, name, sex, age in (("1100000000", "계", "계", -1),
                                 ("1111000000", "계", "남자", 25),
                                 ("1111051500", "동0", "계", 25),
                                 ("1111051500", "동0", "남자", 20)):
        rows.append({"stats_ym": "202004", "admdong_cd": code,
                     "sigungu_nm": "계", "epmndn_nm": name, "sex_nm": sex,
                     "agegrd_cd": str(age), "agegrd_nm": "계" if age == -1 else "15 - 19세",
                     "popl_cnt": "9000000"})
    assert demographic_counts(rows, "202004")[2] == adults


def test_duplicate_adult_cell_is_rejected():
    rows = census_rows()
    rows.append(deepcopy(rows[0]))
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")


@pytest.mark.parametrize("missing_index", [0, 16, 17, 33])
def test_sparse_adult_sex_age_grid_is_rejected(missing_index):
    rows = census_rows()
    rows.pop(missing_index)
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")


@pytest.mark.parametrize("age_code", [24, 26, 102, 999])
def test_age_code_outside_official_source_vocabulary_is_rejected(age_code):
    rows = census_rows()
    rows[0]["agegrd_cd"] = str(age_code)
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")


def test_foreign_period_even_on_an_aggregate_row_is_rejected():
    rows = census_rows()
    rows.append({**rows[0], "stats_ym": "202005", "epmndn_nm": "계"})
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")


@pytest.mark.parametrize("bad_code", ["11110515", "11110515000", "11110515AA"])
def test_native_ten_digit_dong_namespace_is_not_relabelled(bad_code):
    rows = census_rows()
    rows[0]["admdong_cd"] = bad_code
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")


def test_negative_population_is_rejected():
    rows = census_rows()
    rows[0]["popl_cnt"] = "-1"
    with pytest.raises(ValueError):
        demographic_counts(rows, "202004")
