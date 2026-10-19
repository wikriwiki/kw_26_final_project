"""Freeze a policy-independent daily budget from stable Agent spending anchors.

This is a synthetic budget replenishment for matched simulation arms, not an
estimate of wages or an empirical spending target. Two completed-day snapshots
must agree on every input field before a map is written.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import tarfile
from pathlib import Path

FORMULA = "round((5*s_daily_wd + 2*s_daily_we)/7)"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def agents(archive: Path) -> dict[str, dict[str, int]]:
    digest = sha(archive)
    manifest_path = archive.with_name(archive.name.removesuffix(".tar.gz") + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("archive_sha256") != digest or manifest.get("bytes") != archive.stat().st_size:
        raise ValueError(f"Agent archive does not match manifest: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.endswith("_agent.jsonl.gz")]
        if len(members) != 1:
            raise ValueError("expected exactly one Agent snapshot")
        stream = tar.extractfile(members[0])
        if stream is None:
            raise ValueError("Agent snapshot cannot be read")
        with gzip.GzipFile(fileobj=stream) as compressed:
            rows = [json.loads(line) for line in compressed if line.strip()]
    result = {}
    for row in rows:
        aid, agent = row.get("aid"), row.get("agent")
        if (not isinstance(aid, str) or not aid or aid in result
                or not isinstance(agent, dict) or agent.get("id") != aid):
            raise ValueError("duplicate or inconsistent Agent identity")
        values = {field: agent.get(field) for field in ("s_daily_wd", "s_daily_we")}
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in values.values()):
            raise ValueError(f"missing/invalid spending anchor for {aid}")
        result[aid] = values
    return result


def build(first: Path, confirmation: Path, roster_path: Path) -> dict:
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    if (not isinstance(roster, list) or not roster
            or any(not isinstance(aid, str) or not aid for aid in roster)
            or len(roster) != len(set(roster))):
        raise ValueError("roster must be a nonempty list of unique citizen IDs")
    first_agents, confirmed_agents = agents(first), agents(confirmation)
    if set(first_agents) != set(roster) or first_agents != confirmed_agents:
        raise ValueError("roster or Agent spending anchors differ across snapshots")
    projection = {aid: first_agents[aid] for aid in sorted(roster)}
    canonical = json.dumps(projection, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    daily = {aid: round((5 * fields["s_daily_wd"] + 2 * fields["s_daily_we"]) / 7)
             for aid, fields in projection.items()}
    if any(value <= 0 for value in daily.values()):
        raise ValueError("nonpositive synthetic daily budget")
    return {
        "schema": "fixed_persona_budget_v1",
        "source_kind": "stable_persona_spending_anchors",
        "source_field_stability_verified": True,
        "policy_outcome_used": False,
        "source_archive_sha256": sha(first),
        "confirmation_archive_sha256": sha(confirmation),
        "source_roster_sha256": sha(roster_path),
        "source_agent_projection_sha256": hashlib.sha256(canonical).hexdigest(),
        "source_fields": ["s_daily_wd", "s_daily_we"],
        "formula": FORMULA,
        "citizen_count": len(roster),
        "total_daily_budget_won": sum(daily.values()),
        "daily_income_by_aid": daily,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--confirm-archive", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite frozen income map: {args.out}")
    result = build(args.archive, args.confirm_archive, args.roster)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    partial = args.out.with_name(args.out.name + f".tmp.{os.getpid()}")
    try:
        partial.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
        partial.replace(args.out)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps({key: result[key] for key in
                      ("schema", "citizen_count", "total_daily_budget_won",
                       "source_agent_projection_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
