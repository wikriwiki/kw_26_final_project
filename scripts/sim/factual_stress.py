"""Controlled fact-following tests; constraints, never empirical effect targets."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from validate_prompt_v3 import atomic,digest

CASES = {
    "stay_home": {
        "facts":"오늘 00:00~23:59 행정상 자가격리로 주거지 밖 출입이 금지돼 있다. 식료품은 집에 충분하고 배달은 문앞 수령 가능하다. 예외 외출 사유나 방문 약속은 없다.",
        "appointment":"없음", "requirement":"no_outside_anchor"},
    "closed_cafes": {
        "facts":"오늘 후보 지역의 모든 카페는 하루 종일 휴업한다. 매장 이용뿐 아니라 포장 판매도 하지 않는다. 다른 업종의 추가 제한은 없다.",
        "appointment":"없음", "requirement":"no_cafe_visit"},
    "fixed_appointment": {
        "facts":"추가 영업·이동 제한 없음. 오늘은 개인 휴일이며 다른 일정은 없다.",
        "appointment":"오늘 14:00에 zone:11680521 내 이미 예약된 정기 치과 검진이 있다. 취소하지 않고 참석하기로 확정했다. category=건강으로 기록한다. 치료가 필요하다는 진단이나 추가 치료 예약은 없다.",
        "requirement":"attend_at_1400"},
    "empty_memory": {
        "facts":"추가 영업·이동 제한 없음. 활성 정책, 지급된 별도 지갑, 미래 지급 약속은 모두 없다.",
        "appointment":"없음", "requirement":"no_invented_appointment_or_rumor"},
}


def prepare(path):
    cells=[]; personas=[]
    for case, spec in CASES.items():
        aid="SYNTH_STRESS_"+case
        personas.append({"id":aid,"work_poi_id":None})
        user=f"""## 개인 정보
ID: {aid}. 성인 40대, 오늘 출근할 직장 없음. 한 사람 가구.
현재 개인 현금 80000원. 평소 하루 소비규모 20000원. 소비 성향은 보통.
집에서 독서와 요리, 동네 산책을 즐긴다. 현재 질병·통증·물건 고장·미뤄둔 구매 정보는 없다.
## 거주 및 장소 후보
거주지는 11680521. 오늘 외출 가능 지역 후보는 11680521과 11680640.
외출 anchor 허용값: "zone:11680521", "zone:11680640".
집 내부 활동은 residence. 직장 정보가 없으므로 workplace는 없다.
## 오늘의 조건
{spec['facts']}
## 약속
{spec['appointment']}
## 기억과 지인
지난 방문 기억 없음. 소문이나 지인 추천 없음. 과거 일이나 상대방을 만들어내지 않는다.
## 오늘
2026-09-21 월요일 weekday. 오늘 계획만 JSON으로 출력한다.
"""
        cells.append({"aid":aid,"case":case,"arm":"off","date":"2026-09-21", "zones":["11680521","11680640"],
                      "user":user,"context_sha256":digest(user)})
    if Path(path).exists(): raise ValueError("Refusing overwrite")
    atomic(path,{"personas":personas,"cells":cells,"controlled_requirements":CASES})
    print(hashlib.sha256(Path(path).read_bytes()).hexdigest())


def audit(folder):
    rows=[json.loads(s) for s in (Path(folder)/"responses.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest=json.loads((Path(folder)/"manifest.json").read_text(encoding="utf-8"))
    expected={(c["id"],rep,case) for c in manifest["config"]["candidates"] for rep in manifest["config"]["replicate_seeds"] for case in CASES}
    keys=Counter((r["variant"],r["replicate"],r["case"]) for r in rows)
    assert set(keys)==expected and all(n==1 for n in keys.values())
    results=[]
    for r in rows:
        failures=[]
        if not r["valid"]: failures.append("output_contract")
        try: events=json.loads(r["raw"])["events"]
        except (ValueError,KeyError): events=[]; failures.append("no_events")
        kind=CASES[r["case"]]["requirement"]
        if kind=="no_outside_anchor" and any(e.get("anchor")!="residence" for e in events): failures.append(kind)
        if kind=="no_cafe_visit" and any(e.get("category")=="카페" for e in events): failures.append(kind)
        if kind=="attend_at_1400" and not any(e.get("time")=="14:00" and e.get("anchor")=="zone:11680521" and e.get("category")=="건강" for e in events): failures.append(kind)
        if kind=="no_invented_appointment_or_rumor" and any(e.get("trigger") in {"appointment","rumor","policy"} for e in events): failures.append(kind)
        if r.get("semantic_flags"): failures.append("semantic_screen_requires_review")
        results.append({k:r[k] for k in ["variant","replicate","case"]}|{"failures":failures})
    return {"checks":len(results),"passed":sum(not r["failures"] for r in results),"all_pass":all(not r["failures"] for r in results),"results":results}


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--prepare"); ap.add_argument("--audit"); ap.add_argument("--output")
    args=ap.parse_args()
    if args.prepare: prepare(args.prepare)
    else:
        result=audit(args.audit)
        if Path(args.output).exists(): raise ValueError("Refusing overwrite")
        atomic(args.output,result); print(json.dumps(result,ensure_ascii=False))
