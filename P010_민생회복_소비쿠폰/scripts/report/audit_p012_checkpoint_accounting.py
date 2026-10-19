"""Completed P012 read-only checkpoints: verify day-to-day cash and month ledger.

The Oct01-15 archive stores compact spend rows; later daily archives store the
complete INCLUDES properties. This auditor accepts both without querying or
mutating the live server.
"""
from __future__ import annotations

import argparse
import gzip
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


def graph_rows(archive: Path, kind: str):
    with tarfile.open(archive, "r:gz") as tar:
        member = next(m for m in tar.getmembers()
                      if m.name.startswith("graph/") and m.name.endswith(f"_{kind}.jsonl.gz"))
        with tar.extractfile(member) as stream:
            for line in gzip.decompress(stream.read()).splitlines():
                yield json.loads(line)


def states(archive: Path, day: str) -> dict[str, dict]:
    return {row["aid"]: row["state"] for row in graph_rows(archive, "state")
            if row["day"] == day}


def audit(previous: Path, current: Path, prev_day: str, day: str) -> dict:
    prev = states(previous, prev_day)
    today = states(current, day)
    gross = defaultdict(int)
    policy = defaultdict(int)
    tx_count = Counter()
    for row in graph_rows(current, "spend"):
        if row["day"] != day:
            continue
        aid = row["aid"]
        props = row.get("spend") or row
        gross[aid] += int(props.get("actual_spent", props.get("amt", 0)) or 0)
        raw = props.get("spent_from_policy") or "{}"
        payments = json.loads(raw) if isinstance(raw, str) else raw
        policy[aid] += sum(int(v) for v in payments.values())
        tx_count[aid] += 1
    common = sorted(prev.keys() & today.keys())
    balance_mismatch = []
    month_mismatch = []
    for aid in common:
        before, after = prev[aid], today[aid]
        paid = gross[aid] - policy[aid] + int(after.get("online_spent") or 0)
        expected_balance = max(0, int(before["balance"]) + int(after.get("income_today") or 0)
                               - paid)
        expected_month = int(before.get("month_spent") or 0) + paid
        if expected_balance != int(after["balance"]):
            balance_mismatch.append({"aid": aid, "expected": expected_balance,
                                     "actual": int(after["balance"]), "gross": gross[aid],
                                     "policy": policy[aid], "online": after.get("online_spent")})
        if expected_month != int(after.get("month_spent") or 0):
            month_mismatch.append({"aid": aid, "expected": expected_month,
                                   "actual": int(after.get("month_spent") or 0)})
    return {
        "previous_day": prev_day, "day": day,
        "previous_states": len(prev), "current_states": len(today),
        "matched_agents": len(common),
        "missing_previous": sorted(today.keys() - prev.keys()),
        "missing_current": sorted(prev.keys() - today.keys()),
        "transactions": sum(tx_count.values()),
        "gross": sum(gross.values()), "policy_payment": sum(policy.values()),
        "balance_mismatch_count": len(balance_mismatch),
        "month_mismatch_count": len(month_mismatch),
        "balance_mismatch_examples": balance_mismatch[:5],
        "month_mismatch_examples": month_mismatch[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    parser.add_argument("--prev-day", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = audit(args.previous, args.current, args.prev_day, args.day)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    print(rendered, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    return 0 if not result["balance_mismatch_count"] and not result["month_mismatch_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
