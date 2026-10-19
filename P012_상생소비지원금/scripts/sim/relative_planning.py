"""Lossless relative-time serialization: no movement, amount or timing repair."""
import copy
import json
from planning_contract import schedule_schema,inspect_schedule


def relative_schema(cell):
    from datetime import date
    weekend=date.fromisoformat(cell["date"]).weekday()>=5
    base=schedule_schema(cell["zones"],weekend,cell["has_work"])
    kinds=copy.deepcopy(base["properties"]["events"]["items"]["anyOf"])
    for kind in kinds:
        del kind["properties"]["time"]
        kind["required"].remove("time")
        kind["properties"]["minutes_after_previous"]={"type":"integer","minimum":20,"maximum":1439}
        kind["required"].append("minutes_after_previous")
    start=copy.deepcopy(kinds[0])
    del start["properties"]["minutes_after_previous"]
    start["required"].remove("minutes_after_previous")
    return {"type":"object","properties":{
        "start_minute":{"type":"integer","minimum":0,"maximum":1439},
        "start":start,"activities":{"type":"array","minItems":2 if weekend else 4,"maxItems":6 if weekend else 8,"items":{"anyOf":kinds}},
        "finish":kinds[0],"daily_propensity":{"type":"number","minimum":0,"maximum":1}},
        "required":["start_minute","start","activities","finish","daily_propensity"],"additionalProperties":False}


def decode(raw,cell):
    obj=json.loads(raw)
    minute=obj["start_minute"]
    if isinstance(minute,bool) or not isinstance(minute,int) or not 0<=minute<1440: raise ValueError("invalid start")
    events=[]
    for idx,original in enumerate([obj["start"]]+obj["activities"]+[obj["finish"]]):
        e=dict(original)
        if idx:
            gap=e.pop("minutes_after_previous")
            if isinstance(gap,bool) or not isinstance(gap,int) or gap<20: raise ValueError("invalid gap")
            minute+=gap
        if minute>=1440: raise ValueError("day_overflow")
        e["time"]=f"{minute//60:02d}:{minute%60:02d}"
        events.append(e)
    absolute={"events":events,"daily_propensity":obj["daily_propensity"]}
    _,errors,flags=inspect_schedule(json.dumps(absolute,ensure_ascii=False),cell)
    return absolute,errors,flags


def relative_user(user):
    # Remove legacy output-shape footer, keeping all input facts unchanged.
    return user.split("====================================================================")[0].rstrip()+"\n위 사실로 start_minute, start, activities, finish, daily_propensity JSON을 출력한다."
