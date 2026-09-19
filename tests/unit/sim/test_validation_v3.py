import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("validate_prompt_v3", ROOT / "scripts/sim/validate_prompt_v3.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def good():
    events = []
    for i, t in enumerate(["07:00", "08:00", "12:00", "15:00", "18:00", "23:00"]):
        home = i in (0, 5)
        events.append(dict(time=t, anchor="residence" if home else "zone:11680670",
                           category="집" if home else "식사", intent="일상", reasoning="오늘 일정", trigger="lifestyle"))
    return dict(events=events, daily_propensity=.5)


def test_raw_contract_rejects_repairable_time_and_zone_errors():
    value=good()
    assert pilot.contract(json.dumps(value), ["11680670"])[1] == []
    value["events"][2]["time"]="08:00"
    value["events"][3]["anchor"]="zone:00000000"
    errors=pilot.contract(json.dumps(value), ["11680670"])[1]
    assert "time" in errors and "zone" in errors


def test_nonfinite_propensity_is_not_a_valid_response():
    value=good(); value["daily_propensity"]=float("nan")
    assert "propensity" in pilot.contract(json.dumps(value), ["11680670"])[1]


def test_incomplete_experiment_cannot_pass_gate():
    config=json.loads((ROOT/"data/experiments/validation_v3.json").read_text(encoding="utf-8"))
    result=pilot.summarize([], config)
    assert all(not v["rollout_gate_pass"] for v in result["variants"].values())


def test_v10_has_no_fixed_policy_facts_or_historical_effect_targets():
    from scripts.sim.prompts.v10 import SYSTEM_PROMPT
    for text in ["10%", "3%", "20.82", "14.1", "P012", "P015", "그 때문에 줄이지는 않는다"]:
        assert text not in SYSTEM_PROMPT


def test_policy_facts_preserve_late_restrictions_and_do_not_claim_unrestricted():
    from scripts.sim.dawn_context import _format_policy_facts
    text=_format_policy_facts([dict(id="X", type="grant", description="가"*300+"온라인 사용 불가", poi_restricted=True)])
    assert "온라인 사용 불가" in text
    assert "업종 제한 없음" not in text


def test_cashback_threshold_label_uses_actual_parameter():
    from scripts.sim.dawn_context import _format_cashback_status
    text=_format_cashback_status("X", {"threshold_ratio":1.05}, {"daily_wd":50000, "daily_we":50000}, {})
    assert "5% 문턱" in text
    assert "3% 문턱" not in text
