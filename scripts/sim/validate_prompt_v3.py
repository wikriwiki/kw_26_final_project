"""Read-only, preregistered matched-context prompt pilot; never a macro-effect test."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/sim"))


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def contract(raw, zones, weekend=False):
    """Inspect raw output BEFORE production validators can repair it."""
    errors = []
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            raise ValueError("not object")
        p = obj.get("daily_propensity")
        if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 <= p <= 1:
            errors.append("propensity")
        events = obj.get("events")
        if not isinstance(events, list) or len(events) < (4 if weekend else 6):
            errors.append("event_count")
        if not events:
            return obj, errors
        if events[0].get("anchor") != "residence" or events[-1].get("anchor") != "residence":
            errors.append("endpoints")
        previous = -20
        cats = {"식사", "카페", "디저트", "주점", "편의점", "마트", "미용", "쇼핑", "여가", "건강", "교육", "기타", "집", "직장"}
        for e in events:
            t = str(e.get("time", ""))
            try:
                h, m = map(int, t.split(":"))
                minute = h * 60 + m
                if len(t) != 5 or not (0 <= h < 24 and 0 <= m < 60) or minute - previous < 20:
                    errors.append("time")
                previous = minute
            except (ValueError, TypeError):
                errors.append("time")
            anchor, cat = e.get("anchor", ""), e.get("category")
            if cat not in cats:
                errors.append("category")
            if anchor == "residence":
                if cat != "집": errors.append("anchor_category")
            elif anchor == "workplace":
                if cat != "직장": errors.append("anchor_category")
            elif anchor not in {"zone:" + str(z) for z in zones} or cat in {"집", "직장"}:
                errors.append("zone")
            if not e.get("reasoning") or not e.get("intent"):
                errors.append("explanation")
            if e.get("trigger") not in {"appointment", "rumor", "policy", "lifestyle", "mood", "none"}:
                errors.append("trigger")
        return obj, sorted(set(errors))
    except (ValueError, TypeError, AttributeError):
        return None, ["json_schema"]


def policy_row(pid, day):
    if not pid:
        return None
    p = json.loads((ROOT / f"data/neo4j_load/policies/{pid}.json").read_text(encoding="utf-8"))
    # Explicit counterfactual date, never silently edit the policy source file.
    return {**p, "from_": day.isoformat(), "effective_from": day.isoformat(),
            "until_": p["effective_until"], "regions": p.get("target_districts", []),
            "target_l1s": p.get("benefit_categories", []),
            "rate": p.get("benefit_rate"), "cap": p.get("cap_per_agent"),
            "mech_params": json.dumps(p, ensure_ascii=False)}


def prepare(config):
    from neo4j_load._common import driver_session
    from dawn_context import PERSONA_CYPHER, DawnContext, _build_zone_candidates, _sangsaeng_monthly_anchor
    from environments import build_environment
    from prompts import get
    # Proportional strata with stable tie breaking, independent of Cypher row order.
    with driver_session() as session:
        rows = list(session.run("MATCH (a:Agent) WHERE (a)-[:LIVES_AT]->() RETURN coalesce(a.spending_level_wd,0) AS d, collect(a.id) AS ids"))
        pop = {int(r["d"]): sorted(r["ids"]) for r in rows}
        n, total = config["sample_n"], sum(len(v) for v in pop.values())
        quota = {d: int(len(ids) * n / total) for d, ids in pop.items()}
        order = sorted(pop, key=lambda d: (-(len(pop[d]) * n / total - quota[d]), d))
        for d in order[:n-sum(quota.values())]: quota[d] += 1
        ids = sorted(a for d in sorted(pop) for a in random.Random(f"v3-{d}").sample(pop[d], quota[d]))
        personas = [dict(session.run(PERSONA_CYPHER, aid=a).single()) for a in ids]
    cells = []
    for p in personas:
        for case in config["cases"]:
            day = date.fromisoformat(case["date"])
            zones = _build_zone_candidates(p, day)
            for arm in config["arms"]:
                pol = policy_row(case["policy"], day) if arm == "on" else None
                state = {"balance": int((p.get("daily_wd") or 0) * 39), "month_spent": 0,
                         "energy": .7, "mood": .5, "fatigue": .3, "yest_sat": .6,
                         "sangsaeng_month_spent": int(_sangsaeng_monthly_anchor(p) / 30 * (day.day-1))}
                if pol and pol["type"] == "grant":
                    state.update(grant_received={pol["id"]: 280000}, grant_remaining={pol["id"]: 280000})
                env = build_environment("covid_2021", day)
                if case["id"] == "distancing":
                    # Micro intervention isolates specified regulations from the disease context.
                    common = [x for x in env.get("facts", []) if x.startswith("서울 신규 확진")]
                    restrictions = (["식당 매장 취식은 21시까지, 이후 포장·배달만 가능",
                                     "카페는 시간과 무관하게 매장 이용 불가, 포장·배달만 가능"]
                                    if arm == "on" else ["식당·카페의 추가 방역 영업시간 제한 없음. 각 매장 고유 운영시간은 유지"])
                    env = {"headline": "감염병 유행 중인 서울", "facts": common + restrictions}
                ctx = DawnContext(persona=copy.deepcopy(p), state=state, zone_candidates=zones,
                                  policy=[pol] if pol else [], environment=env)
                blocks = ctx.to_prompt_blocks(day)
                user = get("v5").format_dawn_blocks(blocks, day, "weekday" if day.weekday()<5 else "weekend", "월화수목금토일"[day.weekday()])
                cells.append({"aid": p["id"], "case": case["id"], "arm": arm,
                              "date": case["date"], "zones": [z["code"] for z in zones],
                              "user": user, "context_sha256": digest(user)})
    return {"personas": personas, "cells": cells,
            "systems": {v: get(v).SYSTEM_PROMPT for v in config["candidates"]}}


def invoke(job, config, systems, base):
    started = time.time()
    variant, rep, cell = job
    seed = int(digest([rep, cell["aid"], cell["case"]])[:8], 16) % 2147483647
    result = {k: cell[k] for k in ("aid", "case", "arm", "date", "context_sha256")}
    result.update(variant=variant, replicate=rep, seed=seed)
    payload = {"model": config["model"], "messages": [{"role": "system", "content": systems[variant]}, {"role": "user", "content": cell["user"]}],
               "temperature": config["temperature"], "max_tokens": config["max_tokens"], "seed": seed,
               "chat_template_kwargs": {"enable_thinking": False}}
    try:
        req = Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=300) as r: response = json.load(r)
        raw = response["choices"][0]["message"]["content"]
        obj, errors = contract(raw, cell["zones"], date.fromisoformat(cell["date"]).weekday()>=5)
        events = (obj or {}).get("events") or []
        result.update(raw=raw, errors=errors, valid=not errors,
                      finish_reason=response["choices"][0].get("finish_reason"), usage=response.get("usage"),
                      propensity=(obj or {}).get("daily_propensity"),
                      commerce_events=sum(e.get("category") not in {"집", "직장"} for e in events),
                      fiscal_policy_attributions=sum(e.get("trigger")=="policy" and bool(re.search(r"P01[234]|캐시백|쿠폰|바우처|지원금|상품권", str(e.get("reasoning", "")))) for e in events),
                      policy_triggers=sum(e.get("trigger")=="policy" for e in events))
    except Exception as e:
        result.update(valid=False, errors=["request_or_response"], error=str(e))
    result["elapsed_seconds"] = round(time.time()-started, 3)
    return result


def summarize(rows, config):
    out = {"scope": config["phase"], "causal_spending_claim": False, "variants": {}}
    for v in config["candidates"]:
        rr = [r for r in rows if r["variant"]==v]
        valid = sum(bool(r.get("valid")) for r in rr)
        failed = sum("error" in r for r in rr)
        off = [r for r in rr if r["arm"]=="off" and r["case"]!="distancing"]
        false = sum(bool(r.get("fiscal_policy_attributions")) for r in off)
        errors = {}
        for r in rr:
            for e in r.get("errors", []): errors[e] = errors.get(e, 0)+1
        diagnostics = {}
        for case in config["cases"]:
            cc = [r for r in rr if r["case"]==case["id"]]
            paired = {(r["aid"],r["replicate"],r["arm"]):r for r in cc}
            means = []
            for rep in config["replicate_seeds"]:
                diffs = []
                for aid in sorted({r["aid"] for r in cc}):
                    a,b = paired.get((aid,rep,"off"),{}), paired.get((aid,rep,"on"),{})
                    if a.get("valid") and b.get("valid"):
                        diffs.append(b["propensity"]-a["propensity"])
                means.append({"replicate":rep,"paired_n":len(diffs),"propensity_delta":statistics.mean(diffs) if diffs else None})
            diagnostics[case["id"]] = means
        expected = config["sample_n"]*len(config["replicate_seeds"])*len(config["cases"])*2
        out["variants"][v] = {"responses":len(rr), "expected":expected, "strict_valid":valid,
                               "strict_valid_rate":valid/len(rr) if rr else 0,
                               "failed":failed,"false_policy_off_responses":false,"off_responses":len(off),
                               "errors":errors,"diagnostics":diagnostics,
                               "rollout_gate_pass":len(rr)==expected and valid/len(rr)>=.95 and not failed and not false}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--prepare-only", action="store_true")
    # 등록부는 동결된 기록이다. 새 후보를 넣으려고 validation_v3.json 을 고치면 그때의
    # frozen manifest 와 어긋나 과거 런의 재현성이 사라진다. 그래서 파일을 갈아엎는 대신
    # 새 등록부를 하나 더 쓰고 여기서 가리킨다. 기본값은 예전 그대로다.
    ap.add_argument("--config", default="data/experiments/validation_v3.json",
                    help="사전등록 파일 (저장소 루트 기준 상대경로)")
    args=ap.parse_args()
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("PYTHONHASHSEED=0 required before process starts")
    os.environ["EXP_POLICY_ANONYMOUS"]="1"
    os.environ["EXP_DURABLES"]="1"
    os.environ["EXP_CATLINE"]="fold"
    os.environ["EXP_SANGSAENG_BASE_RATIO"]="0.268"
    config=json.loads((ROOT/args.config).read_text(encoding="utf-8"))
    out=Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if (out/"responses.jsonl").exists(): raise SystemExit("Existing run: refusing overwrite or implicit retry")
    frozen=out/"frozen_inputs.json"
    if frozen.exists():
        inputs=json.loads(frozen.read_text(encoding="utf-8"))
        manifest=json.loads((out/"manifest.json").read_text())
        if manifest["config_sha256"] != digest(config) or manifest["inputs_sha256"] != digest(inputs):
            raise SystemExit("preregistration changed after freezing")
        if manifest["script_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise SystemExit("runner changed after freezing")
    else:
        inputs=prepare(config); atomic(frozen, inputs)
        atomic(out/"manifest.json", {"config":config,"config_sha256":digest(config),"inputs_sha256":digest(inputs),
                                   "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                   "system_hashes":{v:digest(s) for v,s in inputs["systems"].items()},
                                   "source_hashes":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                    for base in [ROOT/"scripts/sim",ROOT/"data/experiments/covid_support_2021",ROOT/"data/neo4j_load/policies"]
                                                    for p in sorted(base.rglob("*")) if p.is_file() and p.suffix in {".py",".json"}}})
    if args.prepare_only:
        print(f"Frozen {len(inputs['cells'])} contexts. No LLM calls."); return
    base=os.environ.get("LLM_BASE_URL","http://localhost:8000/v1").rstrip("/")
    with urlopen(base+"/models",timeout=10) as r: models=json.load(r)
    if config["model"] not in [m["id"] for m in models["data"]]: raise SystemExit("wrong served model")
    jobs=[(v,rep,cell) for v in config["candidates"] for rep in config["replicate_seeds"] for cell in inputs["cells"]]
    random.Random(20260919).shuffle(jobs)
    rows=[]
    with (out/"responses.jsonl").open("x",encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures=[pool.submit(invoke,j,config,inputs["systems"],base) for j in jobs]
        for f in as_completed(futures):
            row=f.result(); rows.append(row)
            fp.write(json.dumps(row,ensure_ascii=False)+"\n"); fp.flush(); os.fsync(fp.fileno())
            if len(rows)%24==0: print(f"completed {len(rows)}/{len(jobs)}",flush=True)
    summary=summarize(rows,config); atomic(out/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
