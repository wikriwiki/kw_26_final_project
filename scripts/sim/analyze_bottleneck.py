"""시뮬 jsonl + vLLM monitor 데이터에서 병목 구간 분석.

사용:
  python scripts/sim/analyze_bottleneck.py [--day 2026-05-01]

출력:
  단계별 평균/p50/p90/max elapsed
  Stage 어느 단계가 전체 시간의 몇 % 차지
  vLLM 통계 (KV cache 사용량 추이, throughput, queue depth)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median


def percentile(values: list[float], p: float) -> float:
    if not values: return 0.0
    s = sorted(values)
    idx = int(len(s) * p)
    return s[min(idx, len(s) - 1)]


def analyze_jsonl(day: str | None = None):
    metrics_dir = Path("C:/Users/Administrator/sim_output/metrics")
    files = sorted(metrics_dir.glob(f"day_{day or '*'}.jsonl"))
    if not files:
        print("no metrics jsonl found")
        return

    all_rows = []
    for f in files:
        seen = {}
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                try:
                    j = json.loads(line)
                    if j.get("status") == "ok":
                        seen[j["aid"]] = j
                except json.JSONDecodeError:
                    continue
        all_rows.extend([(f.stem, j) for j in seen.values()])

    if not all_rows:
        print("no ok rows")
        return

    print(f"=== {len(files)}개 jsonl, {len(all_rows):,} agents 분석 ===\n")

    # 단계별 timing 통계
    stages = ["t_dawn", "t_s1", "t_s2", "t_write_plan", "t_night_finalize"]
    print(f"{'stage':18s} {'n':>6s} {'avg':>7s} {'p50':>7s} {'p90':>7s} {'max':>7s} {'%total':>7s}")
    total_elapsed = [j[1].get("elapsed", 0) for j in all_rows]
    total_avg = mean(total_elapsed)
    for stg in stages:
        vals = [j[1].get(f"timing_{stg}") for j in all_rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            print(f"  {stg:16s} {'-':>6s} (metric 없음 — 코드 수정 후 재시뮬 필요)")
            continue
        avg = mean(vals)
        p50 = median(vals)
        p90 = percentile(vals, 0.9)
        mx = max(vals)
        pct = avg * 100 / total_avg if total_avg > 0 else 0
        print(f"  {stg:16s} {len(vals):>6,} {avg:>6.2f}s {p50:>6.2f}s {p90:>6.2f}s {mx:>6.2f}s {pct:>6.1f}%")
    print(f"  {'TOTAL elapsed':16s} {len(total_elapsed):>6,} {total_avg:>6.2f}s "
          f"{median(total_elapsed):>6.2f}s {percentile(total_elapsed, 0.9):>6.2f}s {max(total_elapsed):>6.2f}s")
    print()

    # review_lookup 발동률
    looks = [j[1].get("review_lookup_count", 0) for j in all_rows]
    nonzero = [v for v in looks if v > 0]
    if any(l > 0 for l in looks):
        print(f"review_lookup 발동: {len(nonzero):,}/{len(looks):,} ({len(nonzero)*100/max(len(looks),1):.1f}%)")
        if nonzero:
            print(f"  발동 시 평균 lookup POI 수: {mean(nonzero):.2f}")
            print(f"  최대: {max(nonzero)}")
    else:
        print(f"review_lookup 발동 0건 (metric 없음 또는 LLM이 lookup 안 함)")
    print()

    # Stage1·2 재시도
    s1_retries = sum(1 for _, j in all_rows if j.get("s1_attempts", 1) > 1)
    s2_retries = sum(1 for _, j in all_rows if j.get("s2_attempts", 1) > 1)
    print(f"Stage1 재시도: {s1_retries:,} ({s1_retries*100/len(all_rows):.1f}%)")
    print(f"Stage2 재시도: {s2_retries:,} ({s2_retries*100/len(all_rows):.1f}%)")


def analyze_vllm():
    f = Path("C:/Users/Administrator/sim_output/vllm_metrics.jsonl")
    if not f.exists():
        print("\nvLLM monitor 데이터 없음 (vllm_monitor.py를 시뮬 전에 띄워야)")
        return
    rows = []
    with open(f, encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not rows:
        print("\nvLLM rows empty")
        return
    print(f"\n=== vLLM 통계 ({len(rows):,} samples) ===")
    for key, label in [
        ("num_requests_running", "Running reqs"),
        ("num_requests_waiting", "Waiting reqs"),
        ("gpu_cache_usage_perc", "KV cache %"),
        ("_prompt_tps", "Prompt tok/s"),
        ("_gen_tps", "Generation tok/s"),
    ]:
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if not vals: continue
        avg = mean(vals)
        mx = max(vals)
        mn = min(vals)
        if "perc" in key:
            print(f"  {label:18s} avg={avg*100:>5.1f}% max={mx*100:>5.1f}% min={mn*100:>5.1f}%")
        else:
            print(f"  {label:18s} avg={avg:>7.1f} max={mx:>7.1f} min={mn:>7.1f}")

    # KV cache 포화 (>90%) 비율
    kv_vals = [r.get("gpu_cache_usage_perc") for r in rows if r.get("gpu_cache_usage_perc") is not None]
    if kv_vals:
        sat = sum(1 for v in kv_vals if v > 0.90)
        print(f"\n  KV cache > 90% 시간 비율: {sat*100/len(kv_vals):.1f}% ({sat}/{len(kv_vals)} samples)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default=None, help="특정 day jsonl만 (예: 2026-05-01)")
    args = ap.parse_args()
    analyze_jsonl(args.day)
    analyze_vllm()


if __name__ == "__main__":
    main()
