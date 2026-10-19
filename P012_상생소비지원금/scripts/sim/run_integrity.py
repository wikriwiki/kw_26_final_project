"""Fail-closed agent-day ledger checks for isolated validation runs."""
from collections import Counter
import math


def require_complete_day(agent_ids, rows):
    expected = set(agent_ids)
    if len(expected) != len(agent_ids):
        raise ValueError("Duplicate registered agent ids")
    rows = list(rows)
    counts = Counter(r.get("aid") for r in rows)
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError("Incomplete/duplicate/unexpected agent-day ledger")
    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"Agent-day failed: {row.get('aid')}")
        values = []
        for field in ["cm_today_total", "cm_online_total", "cm_today_total_incl_online"]:
            value = row.get(field)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Missing/invalid {field}: {row.get('aid')}")
            values.append(value)
        if abs(values[0]+values[1]-values[2]) > .01:
            raise ValueError(f"Offline/online ledger does not reconcile: {row.get('aid')}")
    return {"complete_agents":len(rows),"total_including_online":sum(r["cm_today_total_incl_online"] for r in rows)}
