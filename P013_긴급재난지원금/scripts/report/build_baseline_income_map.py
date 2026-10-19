"""Freeze policy-free citizen spending as an exogenous income schedule.

This calibrates only the simulated purse, never a policy outcome or empirical
effect. Use completed, policy-unexposed days and reuse the exact map in both
treatment arms.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path


def build(archive: Path, start: date, end: date, expected_agents: int,
          max_missing_days_per_agent: int = 0) -> dict:
    by_agent: dict[str, list[int]] = defaultdict(list)
    missing: dict[str, list[str]] = defaultdict(list)
    expected_ids: set[str] | None = None
    with tarfile.open(archive, "r:gz") as tar:
        day = start
        while day <= end:
            name = f"metrics/day_{day.isoformat()}.jsonl"
            stream = tar.extractfile(name)
            if stream is None:
                raise ValueError(f"Missing completed metrics: {name}")
            rows = [json.loads(line) for line in stream]
            ids = [str(row["aid"]) for row in rows]
            if len(rows) != expected_agents or len(set(ids)) != expected_agents:
                raise ValueError(f"Incomplete or duplicate citizens on {day}")
            if expected_ids is not None and set(ids) != expected_ids:
                raise ValueError(f"Citizen set changed on {day}")
            expected_ids = set(ids)
            for row in rows:
                if row.get("status") != "ok":
                    missing[row["aid"]].append(day.isoformat())
                    continue
                if (int(row.get("policy_hits") or 0)
                        or int(row.get("policy_spend_today") or 0)
                        or int(row.get("grant_applied_today") or 0)):
                    raise ValueError(f"Policy exposure in calibration days: {day}, {row['aid']}")
                outflow = row.get("cm_today_total_incl_online")
                if not isinstance(outflow, (int, float)) or outflow < 0:
                    raise ValueError(f"Missing total spending: {day}, {row['aid']}")
                offline = row.get("cm_today_total")
                online = row.get("cm_online_total")
                if (not isinstance(offline, (int, float))
                        or not isinstance(online, (int, float))
                        or abs(outflow - offline - online) > 1):
                    raise ValueError(f"Total spending does not reconcile: {day}, {row['aid']}")
                by_agent[row["aid"]].append(round(outflow))
            day += timedelta(days=1)
    n_days = (end - start).days + 1
    if any(len(days) > max_missing_days_per_agent for days in missing.values()):
        raise ValueError("Too many missing calibration days for a citizen")
    incomes = {aid: round(sum(values) / len(values)) if values else 0
               for aid, values in sorted(by_agent.items())}
    if len(incomes) != expected_agents or any(value <= 0 for value in incomes.values()):
        raise ValueError("Baseline income has zero-valued citizen")
    return {
        "schema": "baseline_income_v1",
        "source_archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "source_days": [start.isoformat(), end.isoformat()],
        "source_metric": "cm_today_total_incl_online",
        "policy_free_success_rows_verified": True,
        "days_per_citizen": n_days,
        "minimum_completed_days": min(len(v) for v in by_agent.values()),
        "missing_days_by_aid": dict(sorted(missing.items())),
        "citizen_count": len(incomes),
        "daily_income_by_aid": incomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--expected-agents", type=int, required=True)
    parser.add_argument("--max-missing-days-per-agent", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.end < args.start or args.expected_agents <= 0
            or args.max_missing_days_per_agent < 0):
        parser.error("Invalid period or citizen count")
    data = build(args.archive, args.start, args.end, args.expected_agents,
                 args.max_missing_days_per_agent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"Frozen {data['citizen_count']} citizens × at least "
          f"{data['minimum_completed_days']}/{data['days_per_citizen']} days "
          f"→ {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
