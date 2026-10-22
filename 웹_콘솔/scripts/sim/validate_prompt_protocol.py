"""Preregistered one-feedback generation protocol, including unrepaired first responses."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import random
from urllib.request import urlopen
from validate_prompt_v3 import ROOT,atomic,digest
from validate_prompt_v4 import prepare,invoke,summarize
from planning_repair import repair


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source"); ap.add_argument("--out",required=True)
    ap.add_argument("--config",required=True); ap.add_argument("--prepare-only",action="store_true")
    ap.add_argument("--reuse-first-responses")
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    hashes={n:hashlib.sha256((ROOT/"scripts/sim"/n).read_bytes()).hexdigest() for n in ["validate_prompt_protocol.py","planning_repair.py","validate_prompt_v4.py","planning_contract.py","validate_prompt_v3.py"]}
    if (out/"responses.jsonl").exists(): raise SystemExit("No implicit restart or overwrite")
    if (out/"manifest.json").exists():
        inputs=json.loads((out/"frozen_inputs.json").read_text(encoding="utf-8"))
        manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["config_sha256"]==digest(config) and manifest["inputs_sha256"]==digest(inputs) and manifest["source_hashes"]==hashes
    else:
        inputs=prepare(args.source,config)
        atomic(out/"frozen_inputs.json",inputs)
        atomic(out/"manifest.json",{"config":config,"config_sha256":digest(config),"inputs_sha256":digest(inputs),"source_hashes":hashes,"registered_at":datetime.now(timezone.utc).isoformat()})
    if args.prepare_only: print(f"Frozen {len(inputs['cells'])} contexts, no calls."); return
    replay={}
    if args.reuse_first_responses:
        source=Path(args.reuse_first_responses).read_bytes()
        assert hashlib.sha256(source).hexdigest()==config["source_first_responses_sha256"]
        for line in source.decode().splitlines():
            r=json.loads(line)
            if r["variant"]==config["reuse_variant"]:
                k=(r["replicate"],r["aid"],r["case"],r["arm"])
                assert k not in replay
                replay[k]=r
    elif config.get("source_first_responses_sha256"):
        raise SystemExit("Registered replay requires original first responses")
    base=os.environ.get("LLM_BASE_URL","http://localhost:8000/v1")
    with urlopen(base+"/models",timeout=10) as r: models=json.load(r)
    assert config["model"] in [x["id"] for x in models["data"]]
    def process(job):
        candidate,rep,cell=job
        if replay:
            first=dict(replay[(rep,cell["aid"],cell["case"],cell["arm"])])
            assert first["context_sha256"]==cell["context_sha256"]
            first["variant"]=candidate["id"]
        else:
            first=invoke(job,config,inputs["systems"],base)
        return repair(first,cell,inputs["systems"][candidate["id"]],config,base)
    jobs=[(c,rep,cell) for c in config["candidates"] for rep in config["replicate_seeds"] for cell in inputs["cells"]]
    random.Random(20260921).shuffle(jobs); rows=[]
    with (out/"responses.jsonl").open("x",encoding="utf-8") as fp,ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        futures=[pool.submit(process,j) for j in jobs]
        for future in as_completed(futures):
            row=future.result(); rows.append(row)
            fp.write(json.dumps(row,ensure_ascii=False)+"\n"); fp.flush(); os.fsync(fp.fileno())
            if len(rows)%24==0: print(f"completed {len(rows)}/{len(jobs)}",flush=True)
    summary=summarize(rows,config,inputs["cells"])
    summary["protocol"]="first response plus at most one choice-preserving representation repair"
    for variant,v in summary["variants"].items():
        rr=[r for r in rows if r["variant"]==variant]
        v["first_valid"]=sum(r["first_valid"] for r in rr)
        v["first_valid_rate"]=v["first_valid"]/len(rr)
        v["repair_attempts"]=sum(r["repair_attempted"] for r in rr)
        v["repair_errors"]=dict(Counter(r.get("repair_error") for r in rr if r.get("repair_error")))
        v["choice_preservation_failures"]=sum(r.get("choice_preservation") is False for r in rr)
        v["representation_pipeline_gate"]=v.pop("format_gate")
        if v["repair_errors"]: v["representation_pipeline_gate"]=False
        v["not_first_response_gate"]=True
    atomic(out/"summary.json",summary); print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
