"""One explicit representation repair, with economic choices held fixed."""
from collections import Counter
import copy
import json
from datetime import date
from planning_contract import schedule_schema,inspect_schedule
from validate_prompt_v4 import post


REPAIR_INSTRUCTION = """앞 응답의 일정 형식 검사에서 아래 오류가 확인됐다. 원래 개인의 선택을 유지하여 일정 표현만 고친다.
daily_propensity와 각 외출의 장소·업종·의도·이유·trigger는 그대로 유지한다. 새로운 외출을 만들거나 기존 외출을 삭제하지 않는다.
시간은 반드시 20분 이상 간격으로 증가시키고 같은 시각을 반복하지 않는다. 마지막 외출 후에는 집에서 마무리한다.
개수가 넘으면 중복된 집안 준비·휴식 활동만 합칠 수 있다. 입력에 확정된 약속이나 이용 시간 제약도 지킨다.
스키마에 있는 원래 이벤트 또는 집 시작/마무리 이벤트만 사용할 수 있다. JSON 객체 하나만 다시 출력한다.
"""


def signature(event):
    return json.dumps({k:v for k,v in event.items() if k!="time"},ensure_ascii=False,sort_keys=True)


def preserve_choices(before, after):
    # Require each non-home activity exactly once; times are the only editable field.
    a=Counter(signature(e) for e in before["events"] if e["anchor"]!="residence")
    b=Counter(signature(e) for e in after["events"] if e["anchor"]!="residence")
    originals={signature(e) for e in before["events"]}
    boundaries={signature(e) for e in home_boundaries()}
    return (before["daily_propensity"]==after["daily_propensity"] and a==b
            and all(signature(e) in originals|boundaries for e in after["events"]))


def home_boundaries():
    return [{"time":"00:00","anchor":"residence","category":"집","intent":intent,
             "reasoning":reason,"trigger":"none"}
            for intent,reason in [("하루 시작","집에서 오늘 하루를 시작한다."),("하루 마무리","오늘 외출을 마치고 집에서 하루를 마무리한다.")]]


def repair_schema(obj,cell):
    schema=schedule_schema(cell["zones"],date.fromisoformat(cell["date"]).weekday()>=5,cell["has_work"])
    variants=[]
    for event in obj["events"]+home_boundaries():
        props={"time":{"type":"string","pattern":"^([01][0-9]|2[0-3]):[0-5][0-9]$"}}
        for k,v in event.items():
            if k!="time": props[k]={"const":v}
        variants.append({"type":"object","properties":props,"required":list(props),"additionalProperties":False})
    schema["properties"]["events"]["items"]={"anyOf":variants}
    schema["properties"]["daily_propensity"]={"type":"number","enum":[obj["daily_propensity"]]}
    return schema


def repair(first,cell,system,config,base):
    result=copy.deepcopy(first)
    result["first_response"]=first
    result["first_valid"]=first["valid"]
    result["repair_attempted"]=False
    if first["valid"] or "raw" not in first: return result
    try:
        obj=json.loads(first["raw"])
        # Repair only representation. Malformed or invalid values need a new generation.
        if not isinstance(obj.get("events"),list) or not isinstance(obj.get("daily_propensity"),(int,float)):
            return result
        result["repair_attempted"]=True
        errors={"violations":first["errors"],"event_times":[[i,e.get("time"),e.get("anchor")] for i,e in enumerate(obj["events"])]}
        payload={"model":config["model"],"messages":[{"role":"system","content":system},
                 {"role":"user","content":cell["user"]},{"role":"assistant","content":first["raw"]},
                 {"role":"user","content":REPAIR_INSTRUCTION+json.dumps(errors,ensure_ascii=False)}],
                 "temperature":0,"max_tokens":config["max_tokens"],"seed":first["seed"],
                 "chat_template_kwargs":{"enable_thinking":False},
                 "response_format":{"type":"json_schema","json_schema":{"name":"same_choices_correct_schedule","schema":repair_schema(obj,cell)}}}
        response=post(base,payload)
        raw=response["choices"][0]["message"]["content"]
        updated,errors,flags=inspect_schedule(raw,cell)
        preserved=updated is not None and preserve_choices(obj,updated)
        if not preserved: errors.append("economic_choices_changed")
        result.update(raw=raw,valid=not errors,errors=errors,semantic_flags=flags,
                      choice_preservation=preserved,repair_usage=response.get("usage"),repair_finish_reason=response["choices"][0].get("finish_reason"))
    except Exception as e:
        result.update(valid=False,errors=["repair_failure"],repair_error=str(e))
    return result
