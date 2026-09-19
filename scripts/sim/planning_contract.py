"""Policy-independent Stage1 wire contract. Constrains representation, not effects."""
import json
import re
from validate_prompt_v3 import contract

CATEGORIES = ["식사", "카페", "디저트", "주점", "편의점", "마트", "미용", "쇼핑", "여가", "건강", "교육", "기타"]
TRIGGERS = ["appointment", "rumor", "policy", "lifestyle", "mood", "none"]


def schedule_schema(zones, weekend=False, has_work=True):
    def event(anchors, categories):
        props = {
            "time": {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9]$"},
            "anchor": {"type": "string", "enum": anchors},
            "category": {"type": "string", "enum": categories},
            "intent": {"type": "string", "minLength": 1},
            "reasoning": {"type": "string", "minLength": 1},
            "trigger": {"type": "string", "enum": TRIGGERS},
            "sub_category": {"type": "string"},
        }
        return {"type": "object", "properties": props, "required": list(props)[:6], "additionalProperties": False}
    kinds = [event(["residence"], ["집"])]
    if has_work:
        kinds.append(event(["workplace"], ["직장"]))
    if zones:
        kinds.append(event(["zone:" + str(z) for z in zones], CATEGORIES))
    return {"type": "object", "properties": {
        "events": {"type": "array", "minItems": 4 if weekend else 6,
                   "maxItems": 8 if weekend else 10, "items": {"anyOf": kinds}},
        "daily_propensity": {"type": "number", "minimum": 0, "maximum": 1}},
        "required": ["events", "daily_propensity"], "additionalProperties": False}


def inspect_schedule(raw, cell):
    from datetime import date
    weekend = date.fromisoformat(cell["date"]).weekday() >= 5
    obj, errors = contract(raw, cell["zones"], weekend)
    if obj is None:
        return None, errors, []
    events = obj.get("events", [])
    if len(events) > (8 if weekend else 10):
        errors.append("event_count_max")
    if set(obj) != {"events", "daily_propensity"}:
        errors.append("top_level_fields")
    flags = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for field in ["reasoning", "intent"]:
            if not isinstance(event.get(field), str) or not event.get(field, "").strip():
                errors.append("explanation")
        if not cell.get("has_work", True) and event.get("anchor") == "workplace":
            errors.append("nonexistent_workplace")
        reason = str(event.get("reasoning", ""))
        if re.search(r"어제.{0,55}(들렀|방문했|먹었|마셨|구입했|구매했|다녀왔|메뉴가 새로|메뉴.*기억)", reason):
            flags.append({"kind": "unsupported_memory_screen", "event": event})
        if cell["arm"] == "off" and cell["case"] != "distancing" and event.get("trigger") == "policy" and re.search(r"P01[234]|캐시백|쿠폰|바우처|지원금|상품권", reason):
            flags.append({"kind": "unsupported_fiscal_benefit_screen", "event": event})
    return obj, sorted(set(errors)), flags
