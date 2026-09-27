from scripts.report.pair_preperiod_balance import window


def test_window_reports_daily_paired_gap_and_relative_gap():
    ids = ["a", "b"]
    days = ["2020-05-09", "2020-05-10"]
    index = {"on": {}, "off": {}}
    for aid in ids:
        for day in days:
            index["on"][aid, day] = {"offline_spent": 30, "online_spent": 10,
                                      "eligible_offline_spent": 20}
            index["off"][aid, day] = {"offline_spent": 20, "online_spent": 10,
                                       "eligible_offline_spent": 10}
    total = window(index, ids, days, "total")
    eligible = window(index, ids, days, "eligible_offline")
    assert total["on_won"] == 160
    assert total["off_won"] == 120
    assert total["relative_gap"] == 1 / 3
    assert total["gap_per_citizen_day_won"] == 10
    assert eligible["relative_gap"] == 1
