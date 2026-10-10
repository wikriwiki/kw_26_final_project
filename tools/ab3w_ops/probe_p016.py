"""P016 프롬프트 재현(읽기 전용, LLM 안 부름). 실제 process_one 경로로 Stage1·Stage2 프롬프트 전문을 만든다.
사용: probe_p016.py <run dir> <on|off> <day> <events_from_aid> <events_from_day> <outdir> <aid ...>
설정은 그 런의 engine.json(settings) 그대로. Stage1 일정은 events_from_aid 의 정책 있는 쪽 events_from_day 일정을 빌려 장보기가 들어가게 한다.
"""
import json, os, sys, types
from datetime import date
run, side, day_s, src_aid, src_day, outdir = sys.argv[1:7]
aids = sys.argv[7:]
REPO = "/data/repo_ab3w_20261011"
os.makedirs(outdir, exist_ok=True)
eng = json.load(open(f"{run}/engine.json"))
env = {k: str(v) for k, v in eng["settings"].items() if v is not None}
bolt = {"on": 7693, "off": 7694}[side]
pw = os.environ["NEO4J_PASSWORD_ON" if side == "on" else "NEO4J_PASSWORD_OFF"]
env.update({"PYTHONIOENCODING": "utf-8", "NEO4J_USER": "neo4j", "NEO4J_URI": f"bolt://localhost:{bolt}", "NEO4J_PASSWORD": pw,
            "LLM_BASE_URL": "http://127.0.0.1:9/v1", "LLM_MODE": "exaone_4_5", "SIM_OUTPUT_DIR": outdir + "/simout",
            "SIM_RUN_ID": "ab3w-p016-" + os.path.basename(run.rstrip("/")).split("_")[-1], "SIM_ENVIRONMENT": "covid_2021",
            "EXP_PLAN_BASELINE_FILE": f"{run}/plan_baseline_live.json"})
os.environ.update(env)
os.environ.pop("SIM_ALLOW_STAGE2_FALLBACK", None)
sys.path.insert(0, REPO + "/scripts/sim"); sys.path.insert(0, REPO); os.chdir(REPO)
from neo4j import GraphDatabase
src = GraphDatabase.driver("bolt://localhost:7693", auth=("neo4j", os.environ["NEO4J_PASSWORD_ON"]))
with src.session() as s:
    rows = s.run("""MATCH (a:Agent {id:$a})-[h:HAS_PLAN]->(p:Plan)-[r:INCLUDES]->(x) WHERE toString(h.day)=$d
      RETURN r.order AS o, r.time AS t, r.category AS cat, r.sub_category AS sub, r.anchor AS anchor,
             r.intent AS intent, r.reasoning AS why, r.trigger AS trig ORDER BY o""", a=src_aid, d=src_day).data()
src.close()
evs = [{"time": str(r["t"])[:5], "anchor": r["anchor"] or "residence", "category": r["cat"], "sub_category": r["sub"],
        "intent": r["intent"] or "-", "reasoning": r["why"] or "-", "trigger": "lifestyle"} for r in rows]
import run_simulation as rs
import stage1_intent as s1m
import stage2_poi as s2m
rs._SIM_ENV = "covid_2021"
today = date.fromisoformat(day_s)

class Captured(Exception):
    pass
def resp(content):
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content), finish_reason="stop")],
                                 usage=types.SimpleNamespace(prompt_tokens=0, completion_tokens=0))
rep = []
for aid in aids:
    cap = {}
    def fake_s1(_c, system, user, **kw):
        cap["s1_system"], cap["s1_user"] = system, user
        return resp(json.dumps({"events": evs, "daily_propensity": 0.5}, ensure_ascii=False))
    def fake_s2(_c, system, user, **kw):
        cap["s2_system"], cap["s2_user"] = system, user
        raise Captured("stage2 captured")
    s1m._llm_call = fake_s1; s2m._llm_call = fake_s2
    res = rs.process_one(aid, today, 0)
    cap["status"], cap["error"] = res.get("status"), (res.get("error") or "")[:300]
    for k in ("s1_system", "s1_user", "s2_system", "s2_user"):
        if k in cap:
            open(f"{outdir}/{aid}_{side}_{k}.txt", "w", encoding="utf-8").write(cap[k])
    rep.append({"aid": aid, "side": side, "status": cap["status"], "err": cap["error"][:160],
                **{k + "_len": len(cap.get(k, "")) for k in ("s1_user", "s2_user")}})
print(json.dumps(rep, ensure_ascii=False, indent=1))
