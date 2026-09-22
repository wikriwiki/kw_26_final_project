import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts/sim"))
import copy
from planning_repair import preserve_choices,home_boundaries


def plan():
    return {"daily_propensity":.4,"events":[
        {"time":"08:00","anchor":"zone:11680521","category":"식사","intent":"식사","reasoning":"아침 식사","trigger":"lifestyle"}]}


def test_time_and_home_boundary_edits_preserve_choice_content():
    before=plan(); after=copy.deepcopy(before)
    after["events"][0]["time"]="08:30"
    after["events"]=home_boundaries()[:1]+after["events"]+home_boundaries()[1:]
    assert preserve_choices(before,after)


def test_dropped_duplicated_or_changed_purchase_is_not_a_format_fix():
    before=plan()
    for mutation in ["drop","duplicate","intent","propensity"]:
        after=copy.deepcopy(before)
        if mutation=="drop": after["events"]=[]
        elif mutation=="duplicate": after["events"]*=2
        elif mutation=="intent": after["events"][0]["intent"]="대량 구매"
        else: after["daily_propensity"]=.8
        assert not preserve_choices(before,after)
