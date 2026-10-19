import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts/sim"))
import json
import pytest
from relative_planning import decode

CELL={"date":"2021-10-25","zones":["11680521"],"has_work":False,"arm":"off","case":"cashback"}


def native():
    home=dict(anchor="residence",category="집",intent="집안 활동",reasoning="집에서 생활",trigger="none")
    activity=dict(anchor="zone:11680521",category="식사",intent="점심",reasoning="일상 식사",trigger="lifestyle")
    return dict(start_minute=360,start=home,activities=[dict(home,minutes_after_previous=120),dict(activity,minutes_after_previous=240),
                dict(home,minutes_after_previous=60),dict(home,minutes_after_previous=240)],finish=dict(home,minutes_after_previous=120),daily_propensity=.4)


def test_relative_serialization_preserves_choices_and_exact_requested_time():
    source=native();obj,errors,flags=decode(json.dumps(source),CELL)
    assert errors==[] and flags==[]
    assert [e["time"] for e in obj["events"]]==["06:00","08:00","12:00","13:00","17:00","19:00"]
    assert obj["events"][2]["intent"]=="점심" and obj["daily_propensity"]==.4


@pytest.mark.parametrize("mutation",["overflow","small_gap","boolean_gap"])
def test_invalid_relative_plan_is_rejected_not_repaired(mutation):
    source=native()
    if mutation=="overflow": source["finish"]["minutes_after_previous"]=1000
    elif mutation=="small_gap": source["activities"][0]["minutes_after_previous"]=15
    else: source["activities"][0]["minutes_after_previous"]=True
    with pytest.raises(ValueError): decode(json.dumps(source),CELL)
