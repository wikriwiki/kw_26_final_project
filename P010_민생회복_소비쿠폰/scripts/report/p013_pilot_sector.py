"""Read-only paired P013 sector proxy from a completed arm's transaction graph.

Run ``export`` for OFF before its graph is restored and for ON after completion.
``pair`` uses only the two frozen exports; it never calls the model or graph.
These are exploratory, three-day POI-sector proxies, not the KDI card-sales DID.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from pathlib import Path

ROOT = Path(os.environ.get("PILOT_REPO_ROOT", Path(__file__).resolve().parents[2]))
SCORING = ROOT / "data/experiments/scoring_table.json"

import sys
sys.path.insert(0, str(ROOT / "scripts"))
from neo4j_load._common import driver_session  # noqa: E402


QUERY = """
MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
WHERE a.id IN $aids AND toString(pl.day) IN $days
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
WITH a, pl, i, head(collect(c)) AS category
RETURN a.id AS aid, toString(pl.day) AS day,
       i.actual_spent AS amount, category.name AS sector
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        partial.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def groups() -> tuple[list[str], list[str]]:
    table = json.loads(SCORING.read_text(encoding="utf-8"))
    ind = next(x for x in table["EMERGENCY_2020"]["indicators"] if x["id"] == "EM-4")
    return tuple(part.split("|") for part in ind["rank"])


def aggregate(rows: list[dict], roster: list[str], days: list[str],
              definitions: tuple[list[str], list[str]]) -> dict:
    if not roster or len(set(roster)) != len(roster) or not days or len(set(days)) != len(days):
        raise ValueError("frozen nonempty roster and days required")
    result = {aid: {"semidurable_won": 0, "face_service_won": 0} for aid in roster}
    semi, face = map(set, definitions)
    if semi & face:
        raise ValueError("sector groups overlap")
    for row in rows:
        aid, day, amount, sector = (row.get("aid"), row.get("day"),
                                    row.get("amount"), row.get("sector"))
        if aid not in result or day not in days:
            raise ValueError("transaction outside frozen citizen-day window")
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("invalid transaction amount")
        if sector in semi:
            result[aid]["semidurable_won"] += amount
        elif sector in face:
            result[aid]["face_service_won"] += amount
    return result


def pair(on: dict, off: dict, draws: int = 2000, seed: int = 20260927) -> dict:
    for name, arm in (("on", on), ("off", off)):
        if arm.get("schema") != "p013_sector_arm_v1" or arm.get("arm") != name:
            raise ValueError(f"wrong {name} sector artifact")
    if (on.get("roster") != off.get("roster") or on.get("days") != off.get("days")
            or on.get("groups") != off.get("groups")
            or on.get("scoring_table_sha256") != off.get("scoring_table_sha256")):
        raise ValueError("sector arm definitions differ")
    roster = on["roster"]
    if (not roster or len(set(roster)) != len(roster)
            or set(on.get("by_aid", {})) != set(roster)
            or set(off.get("by_aid", {})) != set(roster)):
        raise ValueError("sector arm rosters incomplete")

    def estimate(ids: list[str]):
        out = []
        for key in ("semidurable_won", "face_service_won"):
            a = sum(on["by_aid"][aid][key] for aid in ids)
            b = sum(off["by_aid"][aid][key] for aid in ids)
            if b <= 0:
                return None
            out.append(100 * (a - b) / b)
        return out[0], out[1], out[0] - out[1]

    point = estimate(roster)
    if point is None:
        raise ValueError("zero OFF spending in one sector group")
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        value = estimate(rng.choices(roster, k=len(roster)))
        if value is not None and all(math.isfinite(x) for x in value):
            boot.append(value[2])
    boot.sort()
    ci = ([boot[int(.025 * (len(boot)-1))], boot[int(.975 * (len(boot)-1))]]
          if boot else None)
    return {"schema": "p013_sector_pair_v1", "citizens": len(roster),
            "days": on["days"], "semidurable_relative_change_pct": point[0],
            "face_service_relative_change_pct": point[1],
            "rank_gap_percentage_points": point[2], "citizen_bootstrap_95_interval": ci,
            "bootstrap_valid_draws": len(boot),
            "scoring_table_sha256": on["scoring_table_sha256"],
            "rank_same_as_registered_direction": point[2] > 0,
            "comparison": "internal_same_calendar_poi_sector_proxy; not_external_kdi_estimand"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    exp = sub.add_parser("export")
    exp.add_argument("--arm", choices=("on", "off"), required=True)
    exp.add_argument("--roster", type=Path, required=True)
    exp.add_argument("--days", nargs="+", required=True)
    exp.add_argument("--out", type=Path, required=True)
    paired = sub.add_parser("pair")
    paired.add_argument("--on", type=Path, required=True)
    paired.add_argument("--off", type=Path, required=True)
    paired.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.mode == "export":
        roster = json.loads(a.roster.read_text(encoding="utf-8"))
        with driver_session() as session:
            rows = [dict(r) for r in session.run(QUERY, aids=roster, days=a.days)]
        mapping = groups()
        result = {"schema": "p013_sector_arm_v1", "arm": a.arm,
                  "roster": roster, "days": a.days, "groups": mapping,
                  "source_transactions": len(rows), "by_aid": aggregate(rows, roster, a.days, mapping),
                  "scoring_table_sha256": _sha(SCORING)}
    else:
        result = pair(json.loads(a.on.read_text(encoding="utf-8")),
                      json.loads(a.off.read_text(encoding="utf-8")))
        result["on_sha256"], result["off_sha256"] = _sha(a.on), _sha(a.off)
    _write(a.out, result)
    print(f"{a.out}: {result['schema']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
