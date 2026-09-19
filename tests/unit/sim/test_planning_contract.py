import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/"scripts/sim"))
import json
from planning_contract import inspect_schedule, schedule_schema


def home_day():
    return {"events":[{"time":t,"anchor":"residence","category":"집","intent":"집안 활동",
                       "reasoning":"오늘 집에서 생활한다.","trigger":"none"}
                      for t in ["07:00","09:00","12:00","15:00","18:00","23:00"]],
            "daily_propensity":.3}


CELL={"date":"2021-10-25","zones":["11680521"],"has_work":False,"arm":"off","case":"cashback"}


def test_staying_home_is_not_a_failure_or_forced_commerce():
    _, errors, flags=inspect_schedule(json.dumps(home_day()),CELL)
    assert errors==[] and flags==[]


def test_numeric_anchor_cannot_silently_become_valid():
    value=home_day(); value["events"][2]["anchor"]="11680521"
    assert "zone" in inspect_schedule(json.dumps(value),CELL)[1]


def test_unknown_workplace_and_whitespace_reason_are_rejected():
    value=home_day(); value["events"][2].update(anchor="workplace",category="직장",reasoning=" ")
    errors=inspect_schedule(json.dumps(value),CELL)[1]
    assert "nonexistent_workplace" in errors and "explanation" in errors


def test_schema_does_not_create_a_workplace_or_require_outings():
    schema=schedule_schema(CELL["zones"],has_work=False)
    kinds=schema["properties"]["events"]["items"]["anyOf"]
    assert kinds[0]["properties"]["anchor"]["enum"]==["residence"]
    assert len(kinds)==2 and kinds[1]["properties"]["anchor"]["enum"]==["zone:11680521"]


def test_factual_flags_are_separate_from_json_contract():
    value=home_day(); value["events"][2]["reasoning"]="어제 카페에 방문했던 기억 때문에 집에서 쉰다."
    _,errors,flags=inspect_schedule(json.dumps(value),CELL)
    assert errors==[] and flags[0]["kind"]=="unsupported_memory_screen"


def test_universal_candidate_has_no_policy_names_or_effect_targets():
    from prompts.v11 import SYSTEM_PROMPT
    for forbidden in ["P010","P012","P013","P014","P015","캐시백","상품권","쿠폰","지원금","%"]:
        assert forbidden not in SYSTEM_PROMPT
