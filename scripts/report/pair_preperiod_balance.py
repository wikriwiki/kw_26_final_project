"""Audit pre-policy ON/OFF imbalance in an already validated paired ledger.

The daily-gap subtraction is an exploratory sensitivity, not an empirical
effect estimate or a substitute for the preregistered post-policy contrast.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.report import paired_grant_effect as grant


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def window(index: dict, roster: list[str], days: list[str], field: str) -> dict:
    def spent(arm: str, aid: str, day: str) -> int:
        row = index[arm][aid, day]
        if field == "total":
            return row["offline_spent"] + row["online_spent"]
        return row["eligible_offline_spent"]

    on = sum(spent("on", aid, day) for aid in roster for day in days)
    off = sum(spent("off", aid, day) for aid in roster for day in days)
    gap = on - off
    return {"days": days, "on_won": on, "off_won": off, "gap_won": gap,
            "relative_gap": gap / off if off else None,
            "gap_per_citizen_day_won": gap / (len(roster) * len(days))}


def audit(on_path: Path, off_path: Path, roster_path: Path, *,
          start: str, end: str, policy_start: str, policy_id: str) -> dict:
    roster = grant.roster_file(roster_path)
    days = grant.dates(start, end)
    policy_day = date.fromisoformat(policy_start)
    pre = [day for day in days if date.fromisoformat(day) < policy_day]
    post = [day for day in days if date.fromisoformat(day) >= policy_day]
    if not pre or not post:
        raise ValueError("need policy-free and policy-active dates")
    grant.verify_manifests(on_path, off_path, roster=roster, days=days,
                           policy_id=policy_id)
    index = {
        "on": grant._index(grant.read_jsonl(on_path), roster, days, "on", policy_id),
        "off": grant._index(grant.read_jsonl(off_path), roster, days, "off", policy_id),
    }
    for arm in ("on", "off"):
        for aid in roster:
            for day in pre:
                row = index[arm][aid, day]
                if any(row[key] for key in ("grant_received_cumulative",
                                              "grant_remaining", "grant_spent_today")):
                    raise ValueError(f"grant activity before policy start: {arm} {day}")
    outcomes = {}
    for field in ("total", "eligible_offline"):
        before = window(index, roster, pre, field)
        after = window(index, roster, post, field)
        outcomes[field] = {
            "pre": before, "post": after,
            "post_minus_pre_daily_gap_won": (after["gap_per_citizen_day_won"]
                                              - before["gap_per_citizen_day_won"]),
        }
    return {
        "schema": "paired_preperiod_balance_v1", "policy_id": policy_id,
        "citizens": len(roster), "start": start, "end": end,
        "policy_start": policy_start, "outcomes": outcomes,
        "on_ledger_sha256": digest(on_path), "off_ledger_sha256": digest(off_path),
        "roster_sha256": digest(roster_path),
        "interpretation": "exploratory internal pre-period imbalance and daily-gap sensitivity; "
                          "not a matched empirical effect or prompt accuracy score",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--on", type=Path, required=True)
    ap.add_argument("--off", type=Path, required=True)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--policy-start", required=True)
    ap.add_argument("--policy-id", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = audit(a.on, a.off, a.roster, start=a.start, end=a.end,
                   policy_start=a.policy_start, policy_id=a.policy_id)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=a.out.parent,
                                     prefix=".preperiod-", delete=False) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, a.out)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"{a.out}: pre={len(result['outcomes']['total']['pre']['days'])} "
          f"post={len(result['outcomes']['total']['post']['days'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
