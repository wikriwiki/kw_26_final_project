"""시뮬레이션 병목 계측 집계.

에이전트 처리 경로에는 perf_counter 측정만 두고, 분위수·순위 계산은 하루가 끝난 뒤
metrics JSONL을 한 번 읽어 수행한다. 따라서 LLM/Neo4j 호출 수와 에이전트별 시간
복잡도를 늘리지 않는다.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Iterable


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def summarize(values: Iterable[float]) -> dict:
    vals = sorted(float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return {"n": 0, "total": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    total = sum(vals)
    return {
        "n": len(vals),
        "total": round(total, 6),
        "avg": round(total / len(vals), 6),
        "p50": round(_percentile(vals, 0.50), 6),
        "p95": round(_percentile(vals, 0.95), 6),
        "p99": round(_percentile(vals, 0.99), 6),
        "max": round(vals[-1], 6),
    }


def _collect_timing_leaves(obj, prefix: str, out: dict[str, list[float]]) -> None:
    if not isinstance(obj, dict):
        return
    for key, value in obj.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _collect_timing_leaves(value, path, out)
        elif (
            str(key).startswith("t_")
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            out.setdefault(path, []).append(float(value))


def build_timing_report(rows: list[dict]) -> dict:
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    leaves: dict[str, list[float]] = {}
    counters: dict[str, list[float]] = {}

    for row in ok_rows:
        phase = {
            key.removeprefix("timing_"): value
            for key, value in row.items()
            if key.startswith("timing_") and isinstance(value, (int, float))
        }
        _collect_timing_leaves(phase, "phase", leaves)
        _collect_timing_leaves(row.get("dawn_timing") or {}, "dawn", leaves)
        _collect_timing_leaves(row.get("prompt_timing") or {}, "prompt", leaves)
        _collect_timing_leaves(row.get("s1_timing") or {}, "stage1", leaves)
        _collect_timing_leaves(row.get("s2_timing") or {}, "stage2", leaves)

        dawn = row.get("dawn_timing") or {}
        s1 = row.get("s1_timing") or {}
        s2 = row.get("s2_timing") or {}
        for path, value in (
            ("dawn.n_memory_returned", dawn.get("n_memory_returned")),
            ("stage1.n_llm_calls", s1.get("n_llm_calls")),
            ("stage2.n_llm_calls", s2.get("n_llm_calls")),
        ):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                counters.setdefault(path, []).append(float(value))

    metrics = {path: summarize(values) for path, values in sorted(leaves.items())}
    count_metrics = {path: summarize(values) for path, values in sorted(counters.items())}

    # 하위 타이머의 총합 기준 순위. t_total은 다른 하위 구간을 포함하므로 제외한다.
    ranked = [
        {
            "path": path,
            "total_sec": stats["total"],
            "avg_sec": stats["avg"],
            "p95_sec": stats["p95"],
        }
        for path, stats in metrics.items()
        if not path.endswith(".t_total")
    ]
    ranked.sort(key=lambda x: x["total_sec"], reverse=True)

    persona_hits = sum(
        1 for r in ok_rows if (r.get("dawn_timing") or {}).get("persona_cache_hit") is True
    )
    policy_hits = sum(
        1 for r in ok_rows if (r.get("dawn_timing") or {}).get("policy_cache_hit") is True
    )
    n = len(ok_rows)

    return {
        "agents_ok": n,
        "agents_error": len(rows) - n,
        "timings": metrics,
        "counters": count_metrics,
        "cache": {
            "persona_hit_rate": round(persona_hits / n, 6) if n else 0.0,
            "policy_hit_rate": round(policy_hits / n, 6) if n else 0.0,
        },
        "bottleneck_rank": ranked[:30],
    }


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_day_timing_report(metrics_path: Path, report_path: Path) -> dict:
    report = build_timing_report(load_jsonl(metrics_path))
    write_json_atomic(report_path, report)
    return report


def slow_cases(rows: list[dict], *, dawn_sec: float, stage1_sec: float, stage2_sec: float) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        phases = {
            "dawn": float(row.get("timing_t_dawn") or 0.0),
            "stage1": float(row.get("timing_t_s1") or 0.0),
            "stage2": float(row.get("timing_t_s2") or 0.0),
        }
        limits = {"dawn": dawn_sec, "stage1": stage1_sec, "stage2": stage2_sec}
        slow = {name: sec for name, sec in phases.items() if sec >= limits[name]}
        if not slow:
            continue
        out.append({
            "aid": row.get("aid"),
            "slow": slow,
            "dawn_timing": row.get("dawn_timing"),
            "prompt_timing": row.get("prompt_timing"),
            "s1_timing": row.get("s1_timing"),
            "s2_timing": row.get("s2_timing"),
            "tokens_in": row.get("tokens_in"),
            "tokens_out": row.get("tokens_out"),
            "s1_attempts": row.get("s1_attempts"),
            "s2_attempts": row.get("s2_attempts"),
        })
    return out
