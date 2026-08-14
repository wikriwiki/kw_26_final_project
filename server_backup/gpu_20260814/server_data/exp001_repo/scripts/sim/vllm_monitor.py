"""vLLM throughput · KV cache · queue 1초 간격 polling.

시뮬 돌아가는 동안 별도 process로 띄워서 병목 추적.
출력: C:/Users/Administrator/sim_output/vllm_metrics.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import urllib.request


def fetch_metrics(base_url: str) -> dict:
    """vLLM /metrics endpoint Prometheus 형식 파싱."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=2) as resp:
            text = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return {"_err": str(e)}

    out = {}
    # 우리가 관심 있는 vLLM metrics
    keys = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:e2e_request_latency_seconds_sum",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
    ]
    for line in text.split("\n"):
        if line.startswith("#"): continue
        for k in keys:
            if line.startswith(k):
                m = re.match(rf"{re.escape(k)}.*?\s+([\d.e+-]+)\s*$", line)
                if m:
                    try:
                        out[k.replace("vllm:", "")] = float(m.group(1))
                    except ValueError:
                        pass
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--out", default="C:/Users/Administrator/sim_output/vllm_metrics.jsonl")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--print-every", type=int, default=10, help="N samples마다 콘솔 출력")
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"=== vLLM monitor → {args.out} ===")
    print(f"  polling {args.base_url}/metrics @ {args.interval}s 간격")

    last_prompt = last_gen = last_t = None
    sample_n = 0
    try:
        with open(args.out, "a", encoding="utf-8") as fp:
            while True:
                t = time.time()
                m = fetch_metrics(args.base_url)
                m["_ts"] = datetime.now().isoformat()
                # 1초 간격 throughput (token/s) 계산
                p = m.get("prompt_tokens_total")
                g = m.get("generation_tokens_total")
                if last_t and p is not None and last_prompt is not None:
                    dt = t - last_t
                    m["_prompt_tps"] = round((p - last_prompt) / dt, 1) if dt > 0 else 0
                    m["_gen_tps"] = round((g - last_gen) / dt, 1) if dt > 0 else 0
                last_prompt, last_gen, last_t = p, g, t
                fp.write(json.dumps(m, ensure_ascii=False) + "\n")
                fp.flush()
                sample_n += 1
                if sample_n % args.print_every == 0:
                    run = int(m.get("num_requests_running", 0) or 0)
                    wait = int(m.get("num_requests_waiting", 0) or 0)
                    kv = m.get("gpu_cache_usage_perc")
                    kv_s = f"{kv*100:.1f}%" if kv is not None else "?"
                    print(f"  [{sample_n}] run={run:3d} wait={wait:3d} KV={kv_s} "
                          f"prompt={m.get('_prompt_tps',0):>6.1f}t/s gen={m.get('_gen_tps',0):>6.1f}t/s")
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n=== monitor 종료 ===")


if __name__ == "__main__":
    main()
