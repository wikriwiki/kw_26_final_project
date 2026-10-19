import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts/sim"))
import pytest
from run_integrity import require_complete_day


def row(aid,offline=0,online=0):
    return dict(aid=aid,status="ok",cm_today_total=offline,cm_online_total=online,cm_today_total_incl_online=offline+online)


def test_zero_consumer_retained_and_online_counted():
    assert require_complete_day(["a","b"],[row("a"),row("b",20,10)])==dict(complete_agents=2,total_including_online=30)


@pytest.mark.parametrize("rows",[[row("a")],[row("a"),row("a")],[row("a"),dict(row("b"),status="error")]])
def test_missing_duplicate_or_failed_is_not_zero_consumption(rows):
    with pytest.raises(ValueError): require_complete_day(["a","b"],rows)


def test_incomplete_or_inconsistent_online_accounting_fails():
    for bad in [dict(row("a"),cm_online_total=None),dict(row("a"),cm_online_total=1),dict(row("a"),cm_today_total=float("nan"))]:
        with pytest.raises(ValueError): require_complete_day(["a"],[bad])


def test_no_commerce_engine_branch_emits_real_zero_ledger():
    from consumption import apply_consumption_model
    meta=apply_consumption_model([],daily=10000,income_tier="중",tendency="보통",balance=100000)
    assert meta["reason"]=="no_commerce"
    ledger=dict(aid="a",status="ok",**{"cm_"+k:meta[k] for k in ["today_total","online_total","today_total_incl_online"]})
    assert require_complete_day(["a"],[ledger])["total_including_online"]==0
