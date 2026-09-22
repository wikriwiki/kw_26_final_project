"""Versioned relative-time protocol. Native output and exact decoded plan retained."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import time
from urllib.request import urlopen
from validate_prompt_v3 import ROOT,atomic,digest
from validate_prompt_v4 import prepare,post,summarize
from relative_planning import relative_schema,decode,relative_user


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--source"); ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--prepare-only",action="store_true")
    args=ap.parse_args(); config=json.loads(Path(args.config).read_text(encoding="utf-8"))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if (out/"responses.jsonl").exists(): raise SystemExit("Existing run, refusing overwrite")
    hashes={n:hashlib.sha256((ROOT/"scripts/sim"/n).read_bytes()).hexdigest() for n in ["relative_planning.py","validate_relative_prompt.py","validate_prompt_v4.py","planning_contract.py","validate_prompt_v3.py"]}
    if (out/"manifest.json").exists():
        inputs=json.loads((out/"frozen_inputs.json").read_text(encoding="utf-8"));manifest=json.loads((out/"manifest.json").read_text(encoding="utf-8"))
        assert manifest["source_hashes"]==hashes and manifest["config_sha256"]==digest(config) and manifest["inputs_sha256"]==digest(inputs)
    else:
        inputs=prepare(args.source,config)
        for cell in inputs["cells"]:
            cell["user"]=relative_user(cell["user"])+"\n외출 anchor 허용값: "+", ".join('"zone:'+str(z)+'"' for z in cell["zones"])
            cell["context_sha256"]=digest(cell["user"])
        atomic(out/"frozen_inputs.json",inputs)
        atomic(out/"manifest.json",{"config":config,"config_sha256":digest(config),"inputs_sha256":digest(inputs),"source_hashes":hashes})
    if args.prepare_only: print(f"Frozen {len(inputs['cells'])} contexts; no LLM calls."); return
    base=os.environ.get("LLM_BASE_URL","http://localhost:8000/v1")
    with urlopen(base+"/models",timeout=10) as r: models=json.load(r)
    assert config["model"] in [m["id"] for m in models["data"]]
    def invoke(job):
        c,rep,cell=job;t=time.monotonic()
        result={k:cell[k] for k in ["aid","case","arm","date","context_sha256"]}
        seed=int(digest([rep,cell["aid"],cell["case"]])[:8],16)%2147483647
        result.update(variant=c["id"],replicate=rep,seed=seed)
        try:
            schema=relative_schema(cell)
            payload={"model":config["model"],"messages":[{"role":"system","content":inputs["systems"][c["id"]]},
                     {"role":"user","content":cell["user"]}],"temperature":config["temperature"],"top_p":config["top_p"],
                     "max_tokens":config["max_tokens"],"seed":seed,"chat_template_kwargs":{"enable_thinking":False},
                     "response_format":{"type":"json_schema","json_schema":{"name":"relative_citizen_schedule","schema":schema}}}
            response=post(base,payload);choice=response["choices"][0];raw=choice["message"]["content"]
            result.update(native_raw=raw,usage=response.get("usage"),finish_reason=choice.get("finish_reason"),schema_sha256=digest(schema))
            try:
                obj,errors,flags=decode(raw,cell)
                result.update(raw=json.dumps(obj,ensure_ascii=False),valid=not errors,errors=errors,semantic_flags=flags,propensity=obj["daily_propensity"])
            except Exception as e:
                result.update(valid=False,errors=[str(e)],semantic_flags=[])
        except Exception as e: result.update(valid=False,errors=["request_or_response"],error=str(e),semantic_flags=[])
        result["elapsed_seconds"]=round(time.monotonic()-t,3)
        return result
    jobs=[(c,rep,cell) for c in config["candidates"] for rep in config["replicate_seeds"] for cell in inputs["cells"]]
    random.Random(20260922).shuffle(jobs);rows=[]
    with (out/"responses.jsonl").open("x",encoding="utf-8") as fp,ThreadPoolExecutor(max_workers=config["workers"]) as pool:
        pending=[pool.submit(invoke,j) for j in jobs]
        for f in as_completed(pending):
            row=f.result();rows.append(row);fp.write(json.dumps(row,ensure_ascii=False)+"\n");fp.flush();os.fsync(fp.fileno())
            if len(rows)%24==0: print(f"completed {len(rows)}/{len(jobs)}",flush=True)
    summary=summarize(rows,config,inputs["cells"])
    summary["representation"]="relative gaps exactly decoded to absolute times; no hidden timing edits, retries, or choice changes"
    atomic(out/"summary.json",summary);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
