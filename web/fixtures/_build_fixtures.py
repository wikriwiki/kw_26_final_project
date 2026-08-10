# -*- coding: utf-8 -*-
"""web/fixtures/*.json 재생성기 (S1 소유).

실제 산출물에서만 값을 뽑는다. 하드코딩된 수치는 없다 (기준 B1).
소스: C:\\Users\\srdyh\\gpu_exp_data\\20260802\\  (SIM_DATA_ROOT 로 덮어쓸 수 있음)

사용:
    python web/fixtures/_build_fixtures.py

주의:
  - 이 스크립트는 읽기 전용이다. scripts/ data/ output/ 를 수정하지 않는다.
  - metrics/day_*.jsonl 은 최대 19MB다. 반드시 **스트리밍**으로 집계한다.
    (기준 B5 — 브라우저로 원본을 보내지 않는다는 계약의 근거 구현)
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_ROOT = Path(os.environ.get(
    "SIM_DATA_ROOT", r"C:\Users\srdyh\gpu_exp_data\20260802"))
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
POLICY_DIR = REPO_ROOT / "data" / "neo4j_load" / "policies"
PREFLIGHT = REPO_ROOT / "scripts" / "sim" / "policy_preflight.py"

# run_id -> (run root, 참고 로그 파일). 실행 시점에 실제 metrics 루트를
# 발견한다. 현재 실측은 data_root 직하(out_*)와 rescue/out_* 두 깊이지만,
# 이름 목록을 코드에 고정하지 않아 다음 산출물도 같은 계약으로 들어온다.
RUNS: dict[str, tuple[Path, Path]] = {}


def _run_id_from_root(root: Path) -> str:
    name = root.name
    return name[4:] if name.startswith("out_") else name


def discover_runs(data_root: Path) -> dict[str, tuple[Path, Path]]:
    root = Path(data_root).resolve()
    logs_dir = root / "logs_scripts"
    logs = {
        path.stem.removeprefix("run_"): path
        for path in logs_dir.glob("run_*.log")
        if path.is_file()
    }
    candidates: list[Path] = []
    if root.is_dir():
        for first in sorted(root.iterdir()):
            if not first.is_dir() or first.name == "logs_scripts":
                continue
            if (first / "metrics").is_dir():
                candidates.append(first)
            for second in sorted(first.iterdir()):
                if second.is_dir() and (second / "metrics").is_dir():
                    candidates.append(second)
    discovered: dict[str, tuple[Path, Path]] = {}
    for run_root in candidates:
        run_id = _run_id_from_root(run_root)
        if run_id in discovered:
            continue
        discovered[run_id] = (run_root, logs.get(run_id, logs_dir / f"run_{run_id}.log"))
    return discovered


def configure_runs(data_root: Path) -> None:
    global DATA_ROOT, RUNS
    DATA_ROOT = Path(data_root).resolve()
    RUNS = discover_runs(DATA_ROOT)


configure_runs(DATA_ROOT)

SLOW_PAGE = 15
FAILURE_PAGE = 12


# ─────────────────────────────────────────────────────────────
# 공통 통계 — scripts/sim/timing_metrics.summarize 와 동일 정의
# ─────────────────────────────────────────────────────────────
def _percentile(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1 - w) + vals[hi] * w


def summarize(values: list[float]) -> dict:
    vals = sorted(float(v) for v in values
                  if isinstance(v, (int, float)) and not isinstance(v, bool))
    if not vals:
        return {"n": 0, "total": 0.0, "avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    total = sum(vals)
    return {
        "n": len(vals),
        "total": round(total, 3),
        "avg": round(total / len(vals), 3),
        "p50": round(_percentile(vals, 0.50), 3),
        "p95": round(_percentile(vals, 0.95), 3),
        "max": round(vals[-1], 3),
    }


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def write(name: str, payload) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    kb = path.stat().st_size / 1024
    flag = "  <-- 200KB 초과!" if kb > 200 else ""
    print(f"  {name:<46} {kb:8.1f} KB{flag}")


# ─────────────────────────────────────────────────────────────
# run 인벤토리
# ─────────────────────────────────────────────────────────────
def scan_run(run_id: str) -> dict:
    root, log_path = RUNS[run_id]
    metrics_days = sorted(p.name[4:-6] for p in (root / "metrics").glob("day_*.jsonl"))
    timing_days = sorted(p.name[4:-5] for p in (root / "timing").glob("day_*.json"))
    done_days = sorted(p.name[5:-5] for p in (root / "checkpoints").glob("done_*.json"))
    failed_days = sorted(p.name[7:-5] for p in (root / "checkpoints").glob("failed_*.json"))
    summary = read_json(root / "summary.json")

    artifacts = {
        "summary_json": (root / "summary.json").exists(),
        "events_jsonl": (root / "events.jsonl").exists(),
        "poi_summary_json": (root / "poi_summary.json").exists(),
        "stage1_failures_jsonl": (root / "stage1_failures.jsonl").exists(),
        "timing_dir": (root / "timing").is_dir() and bool(timing_days),
        "checkpoints_dir": (root / "checkpoints").is_dir(),
        "metrics_dir": (root / "metrics").is_dir(),
    }

    # summary.json 이 있으면 계획값을 안다. 없으면 "미확인".
    if summary:
        args = summary.get("args") or {}
        planned_days = args.get("days")
        agents_target = args.get("limit")
        start_day = args.get("start")
        workers = args.get("workers")
        completed_at = summary.get("completed_at")
        updated_at = summary.get("updated_at")
        plan_source = "summary.json:args"
    else:
        planned_days = agents_target = start_day = workers = None
        completed_at = updated_at = None
        plan_source = None

    # run 디렉터리 밖의 로그로 계획값을 보완할 수 있다 (선택적 결합).
    log_hint = parse_run_log(log_path)

    status = (
        "completed" if summary and summary.get("completed_at")
        and planned_days == len(metrics_days)
        else "incomplete"
    )

    return {
        "run_id": run_id,
        "root": str(root),
        "status": status,
        "artifacts": artifacts,
        "days_present": metrics_days,
        "days_with_timing": timing_days,
        "days_with_done_checkpoint": done_days,
        "days_with_failed_checkpoint": failed_days,
        "plan": {
            "source": plan_source,
            "start_day": start_day,
            "planned_days": planned_days,
            "agents_target": agents_target,
            "workers": workers,
        },
        "completed_at": completed_at,
        "updated_at": updated_at,
        "log_hint": log_hint,
        "summary": summary,
    }


def parse_run_log(path: Path) -> dict | None:
    """run_*.log 헤더에서 계획값을 읽는다. run 디렉터리 밖 소스라 별도 표기한다."""
    if not path.exists():
        return None
    agents = days = start = workers = out = None
    last_progress = None
    with path.open(encoding="utf-8", errors="replace") as fp:
        for line in fp:
            s = line.strip()
            if s.startswith("agents:") and "days:" in s:
                for part in s.split(","):
                    k, _, v = part.partition(":")
                    k, v = k.strip(), v.strip()
                    if k == "agents":
                        agents = int(v)
                    elif k == "days":
                        days = int(v)
                    elif k == "start":
                        start = v
                    elif k == "workers":
                        workers = int(v)
            elif s.startswith("output:"):
                out = s.split(":", 1)[1].strip()
            elif "/" in s and s.startswith(tuple("0123456789")) and "ETA" in s:
                last_progress = s
    return {
        "source_file": path.name,
        "agents_target": agents,
        "planned_days": days,
        "start_day": start,
        "workers": workers,
        "output_dir": out,
        "last_progress_line": last_progress,
    }


# ─────────────────────────────────────────────────────────────
# 일자 집계 — metrics/day_*.jsonl 스트리밍 (기준 B5)
# ─────────────────────────────────────────────────────────────
SUM_FIELDS = [
    "tokens_in", "tokens_out", "n_events", "n_includes", "n_visited_memories",
    "policy_hits", "grant_applied_today", "grant_expired_today",
    "policy_spend_today", "grant_remaining_total", "policy_spend_corrected",
    "cm_planned_total", "cm_today_total", "cm_today_total_incl_online",
    "cm_online_total", "cm_personal_total", "cm_anchor_total",
    "cm_eligible_base", "cm_additional_from_grant",
    "cm_grant_carry_in", "cm_grant_carry_out", "cm_intended_grant_today",
    "cm_selected_policy_liquidity", "cm_policy_requested_total",
    "cm_policy_allocated_total", "cm_policy_eligible_spend_total",
    "cm_policy_eligible_event_count", "cm_policy_liquidity_relief",
    "cm_mechanical_policy_uplift", "cm_substituted",
]
DIST_FIELDS = ["elapsed", "timing_t_dawn", "timing_t_s1", "timing_t_s2",
               "timing_t_write_plan", "avg_sat", "balance", "mood", "fatigue"]
FALLBACK_FIELDS = [
    "review_lookup_count", "fb_resolve_dong", "fb_cand_sub_match",
    "fb_cand_l1_dong", "fb_cand_l1_district", "fb_cand_all_empty",
    "fb_hallucinations_corrected", "fb_hallucinations_dropped",
    "fb_order_mismatch", "fb_missing_picks_filled",
    "fb_pool_split_groups", "fb_pool_split_events",
]
# by_spend_decile 의 각 행이 **항상** 갖는 지표. 값이 하나도 없어도 0 으로 채운다.
# (분위마다 키 집합이 달라지면 표 컴포넌트가 열을 못 고정한다)
DECILE_FIELDS = [
    "grant_applied_today", "grant_remaining_total", "policy_spend_today",
    "cm_policy_allocated_total", "cm_today_total_incl_online",
]
# 최상위 필드지만 sums/distributions/fallback_counts 가 아닌 **다른 키로** 응답에 반영되는 것들.
# _fields_not_aggregated 는 "응답 어디에도 없는 필드" 목록이므로 여기 있는 것은 빼야 한다.
CONSUMED_FIELDS = {
    "status": "status_counts / agents_ok / agents_error",
    "spend_decile": "by_spend_decile",
    "s1_attempts": "attempt_counts",
    "s2_attempts": "attempt_counts",
    "dawn_timing": "cache (persona_cache_hit / policy_cache_hit)",
    "s1_timing": "llm_call_totals.stage1",
    "s2_timing": "llm_call_totals.stage2",
}


def aggregate_day(metrics_path: Path) -> dict:
    """19MB jsonl 을 한 줄씩 읽어 고정 크기 집계로 접는다."""
    n_rows = 0
    status = Counter()
    sums = Counter()
    dists = defaultdict(list)
    fb = Counter()
    attempts = Counter()          # s1_attempts / s2_attempts 분포
    decile = defaultdict(lambda: Counter())
    decile_n = Counter()
    cache = Counter()
    llm_calls = Counter()
    errors: list[dict] = []
    unknown_fields: set[str] = set()
    # 이 응답의 **어느 키에도** 반영되지 않는 필드만 _fields_not_aggregated 에 남긴다.
    known = (set(SUM_FIELDS) | set(DIST_FIELDS) | set(FALLBACK_FIELDS)
             | set(CONSUMED_FIELDS))

    with metrics_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                status["malformed"] += 1
                continue
            n_rows += 1
            st = r.get("status")
            status[st] += 1
            if st != "ok":
                if len(errors) < 10:
                    errors.append({k: r.get(k) for k in
                                   ("aid", "status", "elapsed", "error", "trace")})
                continue

            for f in SUM_FIELDS:
                v = r.get(f)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    sums[f] += v
            for f in DIST_FIELDS:
                v = r.get(f)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    dists[f].append(v)
            for f in FALLBACK_FIELDS:
                v = r.get(f)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    fb[f] += v

            attempts[f"s1={r.get('s1_attempts')}"] += 1
            attempts[f"s2={r.get('s2_attempts')}"] += 1

            # spend_decile 은 결측일 수 있다 (rescue Day 0 실측 4,533행 중 5행이 null).
            # 조용히 버리면 sum(by_spend_decile.agents) != agents_ok 가 되어
            # 계약이 스스로 정한 "부분 계산은 부분이라고 말한다"(§4.1.4)를 어긴다.
            # → null 도 하나의 버킷(spend_decile: null)으로 세어 항등식을 유지한다.
            d = r.get("spend_decile")
            bucket = d if isinstance(d, int) and not isinstance(d, bool) else None
            decile_n[bucket] += 1
            for f in DECILE_FIELDS:
                v = r.get(f)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    decile[bucket][f] += v

            dt = r.get("dawn_timing") or {}
            if dt.get("persona_cache_hit") is True:
                cache["persona_hit"] += 1
            if dt.get("policy_cache_hit") is True:
                cache["policy_hit"] += 1
            for key, src in (("stage1", r.get("s1_timing")), ("stage2", r.get("s2_timing"))):
                v = (src or {}).get("n_llm_calls")
                if isinstance(v, (int, float)):
                    llm_calls[key] += v

            # 전수 스캔. 표본 50행으로는 "드물게만 나타나는 필드"를 놓치고,
            # 그러면 이 목록이 '명세'가 아니라 '표본 관측'이 된다.
            unknown_fields |= (set(r) - known)

    ok = status.get("ok", 0)
    decile_rows = [
        {"spend_decile": d,
         "agents": decile_n[d],
         **{f: (round(decile[d][f], 3) if isinstance(decile[d][f], float)
                else decile[d][f]) for f in DECILE_FIELDS}}
        # None 버킷은 항상 마지막. 숫자 분위와 섞어 정렬하면 TypeError 가 난다.
        for d in sorted((k for k in decile_n if k is not None))
    ]
    if None in decile_n:
        decile_rows.append(
            {"spend_decile": None,
             "agents": decile_n[None],
             **{f: (round(decile[None][f], 3) if isinstance(decile[None][f], float)
                    else decile[None][f]) for f in DECILE_FIELDS}}
        )
    unknown: list[str] = []
    if decile_n.get(None):
        # 전량이 아니라 일부 행만 결측이어도 넣는다.
        # 화면이 "10분위 전량 분해"라고 오해하면 안 된다.
        unknown.append("spend_decile")
    if not ok:
        unknown += ["cache.persona_hit_rate", "cache.policy_hit_rate"]
    return {
        "rows": n_rows,
        "status_counts": dict(status),
        "agents_ok": ok,
        "agents_error": n_rows - ok,
        "sums": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sorted(sums.items())},
        "distributions": {k: summarize(v) for k, v in sorted(dists.items())},
        "fallback_counts": dict(sorted(fb.items())),
        "attempt_counts": dict(sorted(attempts.items())),
        "llm_call_totals": dict(llm_calls),
        "cache": {
            "persona_hit_rate": round(cache["persona_hit"] / ok, 6) if ok else None,
            "policy_hit_rate": round(cache["policy_hit"] / ok, 6) if ok else None,
        },
        "by_spend_decile": decile_rows,
        # sum(by_spend_decile[].agents) == agents_ok 항등식을 응답에서 바로 검산할 수 있게
        # 결측 버킷의 크기를 따로 실어 준다 (0 이면 전량 분해된 것).
        "spend_decile_unknown_agents": decile_n.get(None, 0),
        "error_samples": errors,
        # ok 행에 실존하지만 **이 응답 어디에도** 반영되지 않은 최상위 필드.
        # 전 행 스캔 결과이며, 다른 키로 접힌 필드(CONSUMED_FIELDS)는 제외한다.
        # S2 가 "브라우저로 보내지 않기로 한 것"의 명세다.
        "_fields_not_aggregated": sorted(unknown_fields),
        "unknown": unknown,
    }


def status_scan(metrics_path: Path) -> dict:
    """카운트 3종만 얻는 경량 경로 (기준 B4).

    JSON 파싱 없이 줄 단위 바이트 검사만 한다. `status` 는 항상 두 번째 키이고
    error 행의 `error`/`trace` 안에 같은 패턴이 나와도 따옴표가 이스케이프되므로
    (`\\"status\\": \\"ok\\"`) 오탐이 없다.
    실측 검증: 3종 run 36개 일자 전부에서 aggregate_day 결과와 rows/ok/error 완전 일치.
    실측 비용: 19.6MB 0.045초 (aggregate_day 0.87초의 1/19), 36일자 합계 0.15초 vs 2.55초.
    """
    rows = ok = err = 0
    with metrics_path.open("rb") as fp:
        for line in fp:
            if not line.strip():
                continue
            # append 중인 JSONL의 마지막 줄은 개행이 없고 JSON이 잘리지 않을
            # 수 있다. 완전하지 않은 마지막 줄은 aggregate_day가 버리는 것과
            # 같은 의미로 제외해 rows == ok + error를 유지한다.
            if not line.endswith(b"\n"):
                try:
                    record = json.loads(line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if not isinstance(record, dict) or record.get("status") not in {"ok", "error"}:
                    continue
                rows += 1
                if record["status"] == "ok":
                    ok += 1
                else:
                    err += 1
                continue
            rows += 1
            if b'"status": "ok"' in line:
                ok += 1
            elif b'"status": "error"' in line:
                err += 1
    return {"rows": rows, "agents_ok": ok, "agents_error": err}


def day_progress(run: dict, day: str) -> dict:
    """일자별 진행 — 화면이 '얼마나 됐나'를 그리는 데 필요한 최소 필드."""
    root = Path(run["root"])
    done = read_json(root / "checkpoints" / f"done_{day}.json")
    failed = read_json(root / "checkpoints" / f"failed_{day}.json")
    tm = read_json(root / "timing" / f"day_{day}.json")
    metrics_path = root / "metrics" / f"day_{day}.jsonl"

    agg = aggregate_day(metrics_path)
    target = run["plan"]["agents_target"]
    # summary.json 이 없으면 목표치를 모른다 → 진행률을 계산하지 않는다.
    pct = round(agg["agents_ok"] / target, 6) if target else None

    sm = None
    for s in ((run.get("summary") or {}).get("summary") or []):
        if s.get("day") == day:
            sm = s
            break

    return {
        "day": day,
        "agents_ok": agg["agents_ok"],
        "agents_error": agg["agents_error"],
        "metrics_rows": agg["rows"],
        # 카운트 3종의 출처. 픽스처는 전체 집계 경로로 만들었다.
        # S2 가 B4 를 지키려고 경량 경로를 쓰면 "status_scan" 이 된다 (§3.3).
        "counts_source": "metrics_aggregate",
        "checkpoint_done_count": len(done) if isinstance(done, list) else None,
        "checkpoint_failed_count": len(failed) if isinstance(failed, list) else None,
        "agents_target": target,
        "progress_ratio": pct,
        "day_complete": bool(sm),
        "elapsed_sec": (sm or {}).get("elapsed_sec"),
        "agent_elapsed_sec": (sm or {}).get("agent_elapsed_sec"),
        "night2_elapsed_sec": (sm or {}).get("night2_elapsed_sec"),
        "timing_report_present": tm is not None,
        "policy_payment": (tm or {}).get("policy_payment"),
        "metrics_bytes": metrics_path.stat().st_size,
        "unknown": _unknown_flags(run, sm, tm, target),
    }


def _unknown_flags(run, day_summary, timing_report, target) -> list[str]:
    out = []
    if target is None:
        out.append("agents_target")           # 목표 agent 수 미확인 → 진행률 계산 불가
    if day_summary is None:
        out.append("elapsed_sec")             # summary.json 에 해당 일자 없음 → 소요시간 미확인
        out.append("day_complete")            # 그 날이 끝났는지 자체를 알 수 없음
    if timing_report is None:
        out.append("timing_report")           # timing/day_*.json 없음 → 병목 순위 미확인
    return out


# ─────────────────────────────────────────────────────────────
# 부속 리소스
# ─────────────────────────────────────────────────────────────
def bottlenecks(run: dict, day: str, agg: dict | None = None) -> dict:
    root = Path(run["root"])
    tm = read_json(root / "timing" / f"day_{day}.json")
    if tm is None:
        # timing 리포트는 "그 날이 끝나야" 쓰인다. 중단 run 에는 없다.
        # 대신 metrics 의 phase 레벨 timing_t_* 만 서버에서 다시 접어 부분 순위를 만든다.
        agg = agg or aggregate_day(root / "metrics" / f"day_{day}.jsonl")
        rank = sorted(
            ({"path": "phase." + k.removeprefix("timing_"),
              "total_sec": s["total"], "avg_sec": s["avg"], "p95_sec": s["p95"]}
             for k, s in agg["distributions"].items() if k.startswith("timing_t_")),
            key=lambda x: -x["total_sec"],
        )
        return {
            "run_id": run["run_id"], "day": day,
            "available": False,
            "reason": "timing/day_%s.json 없음 — 시뮬이 그 날을 끝내지 못했다" % day,
            "degraded": True,
            "degraded_note": ("phase 레벨(4개)만 metrics 에서 재계산. "
                              "stage1.*/stage2.*/dawn.* 하위 경로와 cache·policy_payment 는 미확인."),
            "agents_ok": agg["agents_ok"],
            "agents_error": agg["agents_error"],
            "bottleneck_rank": None,
            "cache": None, "policy_payment": None, "counters": None, "timings": None,
            "fallback_rank": rank,
            "fallback_source": f"metrics/day_{day}.jsonl (timing_t_* 필드)",
            "unknown": ["bottleneck_rank", "cache", "policy_payment",
                        "counters", "timings"],
        }
    return {
        "run_id": run["run_id"], "day": day,
        "available": True,
        "reason": None,
        "degraded": False,
        "degraded_note": None,
        "agents_ok": tm.get("agents_ok"),
        "agents_error": tm.get("agents_error"),
        "bottleneck_rank": tm.get("bottleneck_rank"),
        "cache": tm.get("cache"),
        "policy_payment": tm.get("policy_payment"),
        "counters": tm.get("counters"),
        "timings": tm.get("timings"),
        "fallback_rank": None,
        "fallback_source": None,
        "unknown": [],
    }


def slow_page(run: dict, day: str, limit: int = SLOW_PAGE) -> dict:
    root = Path(run["root"])
    path = root / "timing" / f"slow_{day}.json"
    rows = read_json(path)
    # 임계값(SLOW_DAWN_SEC/SLOW_STAGE1_SEC/SLOW_STAGE2_SEC)은 파일에 기록되지 않는다.
    # 실행 시 값은 available 여부와 무관하게 **항상 미확인**이다 (CONTRACT §2.5).
    if rows is None:
        return {"run_id": run["run_id"], "day": day, "available": False,
                "reason": "timing/slow_%s.json 없음" % day,
                "total": None, "limit": limit,
                "sorted_by": "max(slow.*) desc",
                "phase_counts": None, "items": [],
                "unknown": ["total", "phase_counts", "items",
                            "slow_thresholds_sec"]}
    def worst(r):
        return max((r.get("slow") or {}).values() or [0])
    top = sorted(rows, key=worst, reverse=True)[:limit]
    return {
        "run_id": run["run_id"], "day": day, "available": True,
        "reason": None,
        "total": len(rows), "limit": limit, "sorted_by": "max(slow.*) desc",
        "phase_counts": dict(Counter(k for r in rows for k in (r.get("slow") or {}))),
        "items": top,
        "unknown": ["slow_thresholds_sec"],
    }


def failed_page(run: dict, day: str) -> dict:
    """§3.7 — checkpoints/failed_<day>.json 원본.

    "0건"과 "미확인"은 다른 상태다.
      파일 있음 + []  → total: 0,    unknown: []            (그 날은 끝났고 실패가 없었다)
      파일 없음       → total: null, unknown: ["failed_checkpoint"]  (끝나질 않아 기록 없음)
    404 를 내지 않는다 (§4.1.1).
    """
    path = Path(run["root"]) / "checkpoints" / f"failed_{day}.json"
    rows = read_json(path)
    if rows is None:
        return {
            "run_id": run["run_id"], "day": day,
            "source_file": f"checkpoints/failed_{day}.json",
            "available": False,
            "reason": (f"checkpoints/failed_{day}.json 없음 — "
                       "이 파일은 일자 종료 시에만 기록된다"),
            "total": None, "items": [],
            "unknown": ["failed_checkpoint"],
        }
    return {
        "run_id": run["run_id"], "day": day,
        "source_file": f"checkpoints/failed_{day}.json",
        "available": True,
        "reason": None,
        "total": len(rows), "items": rows,
        "unknown": [],
    }


def failures_page(run: dict, limit: int = FAILURE_PAGE) -> dict:
    root = Path(run["run_id"] and run["root"])
    path = Path(root) / "stage1_failures.jsonl"
    if not path.exists():
        return {"run_id": run["run_id"], "available": False,
                "reason": "stage1_failures.jsonl 없음", "total": None,
                "by_day": None, "by_error_type": None, "limit": limit, "items": [],
                "unknown": ["total", "by_day", "by_error_type", "items"]}
    by_day, by_type = Counter(), Counter()
    items = []
    total = 0
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += 1
            by_day[r.get("day")] += 1
            by_type[r.get("error_type")] += 1
            if len(items) < limit:
                items.append(r)
    return {
        "run_id": run["run_id"], "available": True, "reason": None, "total": total,
        "by_day": dict(sorted(by_day.items())),
        "by_error_type": dict(by_type),
        "limit": limit, "items": items,
        "unknown": [],
    }


def events_summary(run: dict) -> dict:
    root = Path(run["root"])
    path = root / "events.jsonl"
    poi = read_json(root / "poi_summary.json")
    if not path.exists():
        return {"run_id": run["run_id"], "available": False,
                "reason": "events.jsonl 없음 — 런 종료 후 별도 export 단계(export_run.py) 산출물",
                "source": None,
                "poi_summary": poi,
                "totals": None, "day_type_counts": None,
                "policy_paid_by_policy_id": None,
                "by_day": None, "by_l1": None, "by_day_l1": None,
                "null_only_fields": None,
                "unknown": ["totals", "day_type_counts", "policy_paid_by_policy_id",
                            "by_day", "by_l1", "by_day_l1"]
                           + ([] if poi else ["poi_summary"])}

    by_day = defaultdict(lambda: Counter())
    by_l1 = defaultdict(lambda: Counter())
    by_day_l1 = defaultdict(lambda: Counter())
    policy_ids = Counter()
    day_type = Counter()
    total = Counter()
    seen_keys: set[str] = set()
    non_null = Counter()          # 키별 non-null 관측 수 → null_only_fields 를 실측으로 도출
    n = 0
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            seen_keys |= set(r)
            for k, v in r.items():
                if v is not None:
                    non_null[k] += 1
            day, l1 = r.get("day"), r.get("l1")
            amt = r.get("amt") or 0
            ex = r.get("ex") or 0
            elig = 1 if r.get("elig") else 0
            wba = 1 if r.get("wba") is True else 0
            pol = 0
            try:
                sp = json.loads(r.get("sp") or "{}")
                for pid, v in sp.items():
                    policy_ids[pid] += v
                    pol += v
            except (json.JSONDecodeError, AttributeError):
                pass
            day_type[r.get("day_type")] += 1
            for bucket, key in ((by_day, day), (by_l1, l1)):
                bucket[key]["events"] += 1
                bucket[key]["amt"] += amt
                bucket[key]["policy_paid"] += pol
                bucket[key]["extra_spent"] += ex
                bucket[key]["coupon_eligible_events"] += elig
                bucket[key]["would_buy_anyway"] += wba
            b = by_day_l1[(day, l1)]
            b["events"] += 1
            b["amt"] += amt
            b["policy_paid"] += pol
            total["events"] += 1
            total["amt"] += amt
            total["policy_paid"] += pol
            total["extra_spent"] += ex
            total["coupon_eligible_events"] += elig
            total["would_buy_anyway"] += wba

    null_only = sorted(k for k in seen_keys if not non_null[k])
    return {
        "run_id": run["run_id"], "available": True, "reason": None,
        "source": "events.jsonl (export_run.py 산출)",
        "poi_summary": poi,
        "totals": dict(total),
        "day_type_counts": dict(day_type),
        "policy_paid_by_policy_id": dict(policy_ids),
        "by_day": [{"day": d, **dict(c)} for d, c in sorted(by_day.items())],
        "by_l1": [{"l1": k, **dict(c)} for k, c in
                  sorted(by_l1.items(), key=lambda kv: -kv[1]["amt"])],
        "by_day_l1": [{"day": d, "l1": l, **dict(c)}
                      for (d, l), c in sorted(by_day_l1.items())],
        # 하드코딩이 아니라 전 행 스캔 결과다. "전부 null 이라 조인에 못 쓴다"는 경고.
        "null_only_fields": null_only,
        "unknown": list(null_only) + ([] if poi else ["poi_summary"]),
    }


# ─────────────────────────────────────────────────────────────
# 정책
# ─────────────────────────────────────────────────────────────
# policy_preflight.py 는 등급을 이모지 접두로 찍는다.
#   _PASS="✅"(U+2705)  _WARN="⚠️ "(U+26A0 U+FE0F + 공백)  _FAIL="❌"(U+274C)
# ⚠️ 는 2 코드포인트라 첫 글자만으로 판정한다.
GRADE = {"✅": "pass", "⚠": "warn", "❌": "fail"}

# 프롬프트 미리보기 블록의 시작 표식.
#   policy_preflight.py:138/150 이 `--- {pid} 프롬프트 미리보기 (...) ---` 를 찍고
#   바로 다음 줄부터 dawn_context._format_policy() 의 반환값을 통째로 출력한다.
PREVIEW_MARK = "프롬프트 미리보기"
PREVIEW_PERSONA_RE = re.compile(r"프롬프트 미리보기\s*\((?P<persona>.*?)\)\s*-*\s*$")


def run_preflight(policy_path: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, str(PREFLIGHT), str(policy_path)],
        capture_output=True, text=True, encoding="utf-8", cwd=str(REPO_ROOT),
    )
    checks: list[dict] = []
    preview: list[str] = []
    persona = None
    verdict = None
    in_preview = False

    # 미리보기는 "블록"이다. 줄 접두(`-`)로 줍지 않는다.
    #   _format_policy_facts() 는 2칸 들여쓴 `  배경: ...` 줄을 낸다(dawn_context.py:507).
    #   접두 규칙으로는 이 줄이 통째로 버려져 정책 description 이 미리보기에서 사라진다.
    # 블록은 시작 표식에서 열리고, 등급 줄 / `결과:` / `====` 구분선에서 닫힌다.
    for raw in (proc.stdout or "").splitlines():
        line = raw.rstrip()
        s = line.strip()
        head = s[:1]
        if head in GRADE:
            in_preview = False
            checks.append({"grade": GRADE[head],
                           "message": s[1:].lstrip("️").strip()})
            continue
        if s.startswith("결과:"):
            in_preview = False
            verdict = s.split(":", 1)[1].strip()
            continue
        if PREVIEW_MARK in s and s.startswith("---"):
            in_preview = True
            m = PREVIEW_PERSONA_RE.search(s)
            persona = m.group("persona") if m else None
            continue                      # 헤더는 preflight 라벨이지 카드 원문이 아니다
        if in_preview:
            if s and set(s) == {"="}:
                in_preview = False
                continue
            preview.append(line)          # 들여쓰기 원문 유지 ("  배경: ..." 포함)

    # NEO4J_URI 가 없으면 preflight 의 check_db_wiring 이 통째로 건너뛴다.
    # "applied_to 0건 → 정책이 아무에게도 안 보임" 이라는 치명 결함을 못 본 상태다.
    db_wiring_checked = bool(os.environ.get("NEO4J_URI"))
    return {
        "policy_id": policy_path.stem,
        "exit_code": proc.returncode,
        "ok": proc.returncode == 0,
        "verdict": verdict,
        "counts": dict(Counter(c["grade"] for c in checks)),
        "checks": checks,
        "prompt_preview": "\n".join(preview).strip("\n"),
        "prompt_preview_persona": persona,
        "db_wiring_checked": db_wiring_checked,
        "stderr": (proc.stderr or "").strip()[:2000],
        "command": ["python", "scripts/sim/policy_preflight.py", str(policy_path.name)],
        "unknown": [] if db_wiring_checked else ["db_wiring"],
    }


def effective_grant_key(pol: dict) -> str:
    """preflight 와 **같은 규칙**으로 실효 grant_key 를 도출한다.

    policy_preflight.py:74   grant_key = pol.get("grant_key") or "income"
    policy_preflight.py:133  row["grant_key"] = "spend_decile" if decile_grants else grant_key
    → 파일에 키가 없어도 검증기는 "income" 으로 동작한다. 원본 null 을 그대로 내보내면
      화면이 "기준 미정"으로 오독한다 (실측: P008/P009/P011 파일에 키 자체가 없음).
    """
    if pol.get("decile_grants"):
        return "spend_decile"
    return pol.get("grant_key") or "income"


def policy_index() -> dict:
    items = []
    for p in sorted(POLICY_DIR.glob("P*.json")):
        pol = read_json(p)
        if not pol:
            continue
        items.append({
            "id": pol.get("id"),
            "file": p.name,
            "name": pol.get("name"),
            "type": pol.get("type"),
            "announce_date": pol.get("announce_date"),
            "effective_from": pol.get("effective_from"),
            "effective_until": pol.get("effective_until"),
            "target_districts": pol.get("target_districts"),
            "benefit_categories": pol.get("benefit_categories"),
            # 원본 파일값(없으면 null)과 preflight 가 실제로 쓰는 실효값을 분리한다.
            "grant_key": pol.get("grant_key"),
            "grant_key_effective": effective_grant_key(pol),
            "grant_key_source": "file" if pol.get("grant_key") else "default",
            "poi_restricted": bool(pol.get("poi_restricted")),
            "has_decile_grants": bool(pol.get("decile_grants")),
            "has_income_grants": bool(pol.get("income_grants")),
            "unknown": [],
        })
    return {"total": len(items), "items": items,
            "source_dir": "data/neo4j_load/policies",
            "unknown": []}


# ─────────────────────────────────────────────────────────────
# 실행 lock (B8) — 실제 사고 로그에서 사실만 추출
# ─────────────────────────────────────────────────────────────
def lock_evidence() -> dict:
    log = DATA_ROOT / "logs_scripts" / "chain_p2.log"
    lines = []
    if log.exists():
        with log.open(encoding="utf-8", errors="replace") as fp:
            for i, raw in enumerate(fp, 1):
                s = raw.strip()
                if s.startswith("[") and "]" in s[:20]:
                    lines.append({"line_no": i, "text": s})
    return {
        "source": "logs_scripts/chain_p2.log (실측)",
        "note": ("동일 chain 스크립트가 두 번 기동되어 두 번째 실행의 `neo4j stop` 이 "
                 "첫 실행의 시뮬레이션을 죽였다. lock 리소스가 막아야 하는 정확한 사건."),
        "timeline": lines[:8],
        "timeline_total": len(lines),
        "timeline_limit": 8,
        "killed_run": {
            "run_id": "BASE7500",
            "run_root": str(RUNS["BASE7500"][0]),
            "log": "logs_scripts/run_BASE7500.log",
        },
        # 이 파일의 값은 전부 chain_p2.log 원문 실측이다. 미확인 항목이 없다.
        "unknown": [],
    }


# ─────────────────────────────────────────────────────────────
def main() -> None:
    print(f"source: {DATA_ROOT}")
    print(f"output: {OUT_DIR}\n")

    runs = {rid: scan_run(rid) for rid in RUNS}

    # ── GET /api/runs ────────────────────────────────────────
    index_items = []
    for rid, run in runs.items():
        days = run["days_present"]
        index_items.append({
            "run_id": rid,
            "root": run["root"],
            "status": run["status"],
            "first_day": days[0] if days else None,
            "last_day": days[-1] if days else None,
            "days_present": len(days),
            "days_planned": run["plan"]["planned_days"],
            "agents_target": run["plan"]["agents_target"],
            "completed_at": run["completed_at"],
            "artifacts": run["artifacts"],
            "unknown": [k for k, v in (
                ("days_planned", run["plan"]["planned_days"]),
                ("agents_target", run["plan"]["agents_target"]),
                ("completed_at", run["completed_at"]),
            ) if v is None],
        })
    # 컬렉션 리소스는 최상위와 items[] 원소 **양쪽**에 unknown 을 갖는다 (§4.1.3).
    write("runs.index.json", {"total": len(index_items), "items": index_items,
                              "unknown": []})

    for rid, run in runs.items():
        days = run["days_present"]
        detail = {k: run[k] for k in
                  ("run_id", "root", "status", "artifacts", "days_present",
                   "days_with_timing", "days_with_done_checkpoint",
                   "days_with_failed_checkpoint", "plan", "completed_at",
                   "updated_at", "log_hint")}
        detail["day_summaries"] = [
            {k: v for k, v in s.items() if k != "timing_top"}
            for s in ((run.get("summary") or {}).get("summary") or [])
        ]
        detail["unknown"] = [k for k, v in (
            ("plan.planned_days", run["plan"]["planned_days"]),
            ("plan.agents_target", run["plan"]["agents_target"]),
            ("plan.start_day", run["plan"]["start_day"]),
            ("completed_at", run["completed_at"]),
        ) if v is None]
        write(f"run.{rid}.detail.json", detail)

        prog = [day_progress(run, d) for d in days]
        write(f"run.{rid}.days.json",
              {"run_id": rid, "total": len(prog), "items": prog,
               # 컬렉션 최상위 unknown = 전 일자에 공통으로 걸리는 미확인 항목.
               "unknown": (["agents_target"]
                           if run["plan"]["agents_target"] is None else [])})

        # 일자 상세는 대표 1일만 (용량). BASE=첫날, FINAL=마지막날, BASE7500=유일한 날
        # 훑어서 새로 잡힌 run 은 마지막 날을 쓴다 — 정책 시행 후가 그쪽에 있다.
        pick = {"BASE": days[0], "FINAL": days[-1], "BASE7500": days[0]}.get(rid, days[-1])
        agg = aggregate_day(Path(run["root"]) / "metrics" / f"day_{pick}.jsonl")
        src = Path(run["root"]) / "metrics" / f"day_{pick}.jsonl"
        write(f"run.{rid}.day.{pick}.json", {
            "run_id": rid, "day": pick,
            "source_file": f"metrics/day_{pick}.jsonl",
            "source_bytes": src.stat().st_size,
            "aggregated_server_side": True,
            **agg,
        })
        write(f"run.{rid}.day.{pick}.bottlenecks.json", bottlenecks(run, pick, agg))
        write(f"run.{rid}.day.{pick}.slow.json", slow_page(run, pick))
        write(f"run.{rid}.failures.json", failures_page(run))
        write(f"run.{rid}.events.summary.json", events_summary(run))

        # ── §3.7 failed 리소스의 3가지 상태를 전부 실물로 남긴다 ──
        #   (a) 파일 있음 + 실패 있음   → FINAL 2025-08-03
        #   (b) 파일 있음 + 0건         → BASE  첫날 (정상인데 비어 있는 화면)
        #   (c) 파일 자체 없음          → BASE7500 (일자가 끝나지 않아 기록 안 됨)
        # 하나라도 빠지면 S3/S4 가 (b)와 (c)를 같은 UI로 그려 "실패 0건"과
        # "실패 여부 미확인"을 뭉뚱그린다.
        wrote_nonempty = False
        for d in days:
            fl = read_json(Path(run["root"]) / "checkpoints" / f"failed_{d}.json")
            if fl:
                write(f"run.{rid}.day.{d}.json", {
                    "run_id": rid, "day": d,
                    "source_file": f"metrics/day_{d}.jsonl",
                    "source_bytes": (Path(run["root"]) / "metrics" / f"day_{d}.jsonl").stat().st_size,
                    "aggregated_server_side": True,
                    **aggregate_day(Path(run["root"]) / "metrics" / f"day_{d}.jsonl"),
                })
                write(f"run.{rid}.day.{d}.failed.json", failed_page(run, d))
                wrote_nonempty = True
                break
        if not wrote_nonempty and days:
            write(f"run.{rid}.day.{pick}.failed.json", failed_page(run, pick))

    # ── 정책 ─────────────────────────────────────────────────
    write("policies.index.json", policy_index())
    for p in sorted(POLICY_DIR.glob("P*.json")):
        pol = read_json(p)
        write(f"policy.{p.stem}.detail.json",
              {"file": p.name, "source_dir": "data/neo4j_load/policies",
               "policy": pol,
               "grant_key_effective": effective_grant_key(pol or {}),
               "grant_key_source": "file" if (pol or {}).get("grant_key") else "default",
               # 원본 파일을 통째로 싣는다. 파일에 없는 값은 §3.12 기본값으로 확정된다.
               "unknown": []})
        write(f"policy.{p.stem}.validate.json", run_preflight(p))

    write("runner.lock.evidence.json", lock_evidence())
    print("\ndone.")


if __name__ == "__main__":
    main()
