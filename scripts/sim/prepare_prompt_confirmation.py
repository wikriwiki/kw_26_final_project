"""Prepare disjoint-person confirmation contexts; read-only Neo4j, no LLM."""
import argparse
import copy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import random
from validate_prompt_v3 import ROOT, atomic, policy_row, digest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exclude-inputs",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--sample-n",type=int,default=24)
    args=ap.parse_args()
    if os.environ.get("PYTHONHASHSEED")!="0": raise SystemExit("Set PYTHONHASHSEED=0 before launch")
    for k,v in {"EXP_POLICY_ANONYMOUS":"1","EXP_DURABLES":"1","EXP_CATLINE":"fold","EXP_SANGSAENG_BASE_RATIO":"0.268"}.items(): os.environ[k]=v
    from neo4j_load._common import driver_session
    from dawn_context import PERSONA_CYPHER,DawnContext,_build_zone_candidates,_sangsaeng_monthly_anchor
    from environments import build_environment
    from prompts import get
    out=Path(args.out)
    if out.exists(): raise SystemExit("Refusing to replace confirmation inputs")
    excluded=json.loads(Path(args.exclude_inputs).read_text(encoding="utf-8"))
    old_ids={p["id"] for p in excluded["personas"]}
    with driver_session() as session:
        rows=list(session.run("MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() RETURN coalesce(a.spending_level_wd,0) AS d,collect(a.id) AS ids"))
        pop={int(r["d"]):sorted(set(r["ids"])-old_ids) for r in rows}
        total=sum(map(len,pop.values())); n=args.sample_n
        quota={d:int(len(ids)*n/total) for d,ids in pop.items()}
        order=sorted(pop,key=lambda d:(-(len(pop[d])*n/total-quota[d]),d))
        for d in order[:n-sum(quota.values())]: quota[d]+=1
        ids=sorted(a for d in sorted(pop) for a in random.Random(f"confirmation-20260920-{d}").sample(pop[d],quota[d]))
        assert not set(ids)&old_ids and len(ids)==n
        personas=[dict(session.run(PERSONA_CYPHER,aid=a).single()) for a in ids]
    config=json.loads((ROOT/"data/experiments/validation_v3.json").read_text(encoding="utf-8"))
    cells=[]
    for p in personas:
        for case in config["cases"]:
            day=date.fromisoformat(case["date"])
            zones=_build_zone_candidates(p,day)
            for arm in ["off","on"]:
                pol=policy_row(case["policy"],day) if arm=="on" else None
                state={"balance":int((p.get("daily_wd") or 0)*39),"month_spent":0,"energy":.7,"mood":.5,"fatigue":.3,"yest_sat":.6,
                       "sangsaeng_month_spent":int(_sangsaeng_monthly_anchor(p)/30*(day.day-1))}
                if pol and pol["type"]=="grant": state.update(grant_received={pol["id"]:280000},grant_remaining={pol["id"]:280000})
                env=build_environment("covid_2021",day)
                if case["id"]=="distancing":
                    common=[s for s in env.get("facts",[]) if s.startswith("서울 신규 확진")]
                    rules=(["식당 매장 취식은 21시까지, 이후 포장·배달만 가능","카페는 시간과 무관하게 매장 이용 불가, 포장·배달만 가능"]
                           if arm=="on" else ["식당·카페의 추가 방역 영업시간 제한 없음. 각 매장 고유 운영시간은 유지"])
                    env={"headline":"감염병 유행 중인 서울","facts":common+rules}
                ctx=DawnContext(persona=copy.deepcopy(p),state=state,zone_candidates=zones,policy=[pol] if pol else [],environment=env)
                blocks=ctx.to_prompt_blocks(day)
                user=get("v5").format_dawn_blocks(blocks,day,"weekday" if day.weekday()<5 else "weekend","월화수목금토일"[day.weekday()])
                cells.append({"aid":p["id"],"case":case["id"],"arm":arm,"date":case["date"],"zones":[z["code"] for z in zones],"user":user,"context_sha256":digest(user)})
    atomic(out,{"personas":personas,"cells":cells,"selection":{
        "seed":"confirmation-20260920-{decile}","excluded_ids":sorted(old_ids),"quota":quota,
        "source_excluded_sha256":hashlib.sha256(Path(args.exclude_inputs).read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}})
    print(json.dumps({"n":len(personas),"contexts":len(cells),"sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"overlap":0}))


if __name__=="__main__": main()
