"""Versioned development screen; frozen inputs, no DB access or implicit retry."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time
from datetime import date, datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from validate_prompt_v3 import ROOT, atomic, digest
from planning_contract import schedule_schema, inspect_schedule
from prompts import get


def post(base, payload):
    req = Request(base + "/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    try:
        with urlopen(req, timeout=300) as r:
            return json.load(r)
    except HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:1000]}") from e


def prepare(source, config):
    raw = Path(source).read_bytes()
    if hashlib.sha256(raw).hexdigest() != config["source_inputs_sha256"]:
        raise ValueError("Wrong source input bytes")
    inputs = json.loads(raw)
    personas = {p["id"]:p for p in inputs["personas"]}
    cells = []
    for original in inputs["cells"]:
        c = dict(original)
        text = c["user"].replace("평일 재택 ", "평일 집 체류 ").replace("주말 재택 ", "주말 집 체류 ")
        text = "\n".join(line for line in text.splitlines() if not line.startswith("- 판단 원칙:"))
        c["user"] = text + "\n외출 anchor 허용값: " + ", ".join('"zone:' + str(z) + '"' for z in c["zones"])
        c["context_sha256"] = digest(c["user"])
        c["has_work"] = bool(personas[c["aid"]].get("work_poi_id"))
        cells.append(c)
    return {"personas":inputs["personas"], "cells":cells,
            "systems": {c["id"]:get(c["prompt"]).SYSTEM_PROMPT for c in config["candidates"]}}


def invoke(job, config, systems, base):
    candidate, rep, cell = job
    t = time.monotonic()
    seed = int(digest([rep,cell["aid"],cell["case"]])[:8],16) % 2147483647
    out = {k:cell[k] for k in ["aid","case","arm","date","context_sha256"]}
    out.update(variant=candidate["id"], replicate=rep, seed=seed, structured=candidate["structured"])
    payload = {"model":config["model"],"messages":[{"role":"system","content":systems[candidate["id"]]},
               {"role":"user","content":cell["user"]}],"temperature":config["temperature"],
               "top_p":config["top_p"],"max_tokens":config["max_tokens"],"seed":seed,
               "chat_template_kwargs":{"enable_thinking":False}}
    if candidate["structured"]:
        schema = schedule_schema(cell["zones"], date.fromisoformat(cell["date"]).weekday()>=5, cell["has_work"])
        payload["response_format"] = {"type":"json_schema","json_schema":{"name":"citizen_day","strict":True,"schema":schema}}
        out["schema_sha256"] = digest(schema)
    try:
        response = post(base,payload)
        choice = response["choices"][0]
        raw = choice["message"]["content"]
        obj, errors, flags = inspect_schedule(raw, cell)
        out.update(raw=raw, errors=errors, valid=not errors, semantic_flags=flags,
                   finish_reason=choice.get("finish_reason"), usage=response.get("usage"),
                   propensity=(obj or {}).get("daily_propensity"))
    except Exception as e:
        out.update(valid=False, errors=["request_or_response"], error=str(e), semantic_flags=[])
    out["elapsed_seconds"] = round(time.monotonic()-t,3)
    return out


def summarize(rows, config, cells):
    expected = {(c["id"],rep,x["aid"],x["case"],x["arm"]) for c in config["candidates"] for rep in config["replicate_seeds"] for x in cells}
    counts = Counter((r["variant"],r["replicate"],r["aid"],r["case"],r["arm"]) for r in rows)
    complete = set(counts)==expected and all(n==1 for n in counts.values())
    out = {"id":config["id"],"complete_unique_matrix":complete,"macro_claim":False,"variants":{}}
    for candidate in config["candidates"]:
        rr = [r for r in rows if r["variant"]==candidate["id"]]
        valid = sum(bool(r["valid"]) for r in rr)
        failures = sum("error" in r for r in rr)
        flags = Counter(f["kind"] for r in rr for f in r["semantic_flags"])
        out["variants"][candidate["id"]] = {
            "responses":len(rr),"strict_valid":valid,"strict_valid_rate":valid/len(rr) if rr else 0,
            "request_failures":failures,"error_counts":dict(Counter(e for r in rr for e in r["errors"])),
            "semantic_screen_event_counts":dict(flags),
            "format_gate":complete and bool(rr) and valid/len(rr)>=.95 and failures==0,
            "factual_review_required":bool(flags),
            "mean_completion_tokens":statistics.mean(r["usage"]["completion_tokens"] for r in rr if r.get("usage")) if any(r.get("usage") for r in rr) else None,
            "finish_reasons":dict(Counter(r.get("finish_reason","missing") for r in rr))}
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source")
    ap.add_argument("--out",required=True)
    ap.add_argument("--config",default=str(ROOT/"data/experiments/validation_v4.json"))
    ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--probe",action="store_true")
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    base=os.environ.get("LLM_BASE_URL","http://localhost:8000/v1").rstrip("/")
    folder=Path(args.out); folder.mkdir(parents=True,exist_ok=True)
    if args.probe:
        result=post(base,{"model":config["model"],"messages":[{"role":"user","content":"Return a JSON object with ok=true."}],
                         "temperature":0,"max_tokens":40,"chat_template_kwargs":{"enable_thinking":False},
                         "response_format":{"type":"json_schema","json_schema":{"name":"probe","schema":{
                             "type":"object","properties":{"ok":{"type":"boolean","enum":[True]}},"required":["ok"],"additionalProperties":False}}}})
        atomic(folder/"capability_probe.json",result)
        assert json.loads(result["choices"][0]["message"]["content"])=={"ok":True}
        print("Structured output capability probe passed; not a policy experiment."); return
    if (folder/"responses.jsonl").exists():
        raise SystemExit("Existing responses: refusing implicit retry/overwrite")
    runner_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    contract_hash=hashlib.sha256((ROOT/"scripts/sim/planning_contract.py").read_bytes()).hexdigest()
    if (folder/"frozen_inputs.json").exists():
        inputs=json.loads((folder/"frozen_inputs.json").read_text(encoding="utf-8"))
        manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_sha256"]==digest(config) and manifest["inputs_sha256"]==digest(inputs)
        assert manifest["runner_sha256"]==runner_hash and manifest["contract_sha256"]==contract_hash
    else:
        inputs=prepare(args.source,config)
        atomic(folder/"frozen_inputs.json",inputs)
        atomic(folder/"manifest.json",{"config":config,"config_sha256":digest(config),"inputs_sha256":digest(inputs),
              "runner_sha256":runner_hash,"contract_sha256":contract_hash,
              "registered_at":datetime.now(timezone.utc).isoformat(),
              "system_hashes":{k:digest(v) for k,v in inputs["systems"].items()}})
    if args.prepare_only:
        print(f"Frozen {len(inputs['cells'])} contexts; no policy LLM calls."); return
    with urlopen(base+"/models",timeout=10) as r: models=json.load(r)
    assert config["model"] in [m["id"] for m in models["data"]]
    jobs=[(c,rep,cell) for c in config["candidates"] for rep in config["replicate_seeds"] for cell in inputs["cells"]]
    random.Random(20260920).shuffle(jobs)
    rows=[]
    with (folder/"responses.jsonl").open("x",encoding="utf-8") as fp, ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        pending=[pool.submit(invoke,j,config,inputs["systems"],base) for j in jobs]
        for future in as_completed(pending):
            row=future.result(); rows.append(row)
            fp.write(json.dumps(row,ensure_ascii=False)+"\n"); fp.flush(); os.fsync(fp.fileno())
            if len(rows)%24==0: print(f"completed {len(rows)}/{len(jobs)}",flush=True)
    summary=summarize(rows,config,inputs["cells"])
    atomic(folder/"summary.json",summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
