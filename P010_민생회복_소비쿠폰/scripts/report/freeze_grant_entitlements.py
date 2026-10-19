"""Freeze a grant's expected citizen payments from an off-server Agent snapshot.

The source is citizen attributes and policy rules, never simulated outcomes.
This makes the entitlement check independent of the treatment ledger.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tarfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from plan_writer import grants_to_apply  # noqa: E402


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def build(archive: Path, member: str, policy_file: Path, *,
          expected_agents: int, require_all_recipients: bool = False) -> dict:
    if expected_agents <= 0:
        raise ValueError("expected agent count must be positive")
    policy = json.loads(policy_file.read_text(encoding="utf-8"))
    pid, effective_day = policy.get("id"), policy.get("effective_from")
    if not isinstance(pid, str) or not pid or policy.get("type") != "grant":
        raise ValueError("selected policy must be a named grant")
    if (not isinstance(effective_day, str)
            or date.fromisoformat(effective_day).isoformat() != effective_day):
        raise ValueError("grant effective date is missing")
    with tarfile.open(archive, "r:gz") as tar:
        if member not in tar.getnames() or not member.endswith("_agent.jsonl.gz"):
            raise ValueError("selected archive member is not an Agent snapshot")
        source = tar.extractfile(member)
        if source is None:
            raise ValueError("Agent snapshot is missing")
        with gzip.GzipFile(fileobj=source) as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != expected_agents:
        raise ValueError("Agent snapshot count differs from expected roster")
    amounts: dict[str, int] = {}
    for row in rows:
        aid, agent = row.get("aid"), row.get("agent")
        if (not isinstance(aid, str) or not aid or aid in amounts
                or not isinstance(agent, dict) or agent.get("id") != aid):
            raise ValueError("duplicate or inconsistent Agent identity")
        got = grants_to_apply([policy], effective_day, {},
                              income=agent.get("p_income_level") or "",
                              spend_decile=agent.get("spending_level_wd"))
        if set(got) not in (set(), {pid}):
            raise ValueError("unexpected grant returned by runtime rule")
        amount = got.get(pid, 0)
        if isinstance(amount, bool) or not isinstance(amount, int) or amount < 0:
            raise ValueError("invalid calculated grant entitlement")
        amounts[aid] = amount
    if require_all_recipients and any(amount <= 0 for amount in amounts.values()):
        raise ValueError("universal grant left a citizen without an entitlement")
    roster = sorted(amounts)
    return {
        "schema": "grant_entitlements_v1",
        "policy_id": pid, "effective_day": effective_day,
        "policy_file_sha256": _sha(policy_file),
        "source_archive_sha256": _sha(archive),
        "source_agent_member": member,
        "roster_sha256": hashlib.sha256(json.dumps(
            roster, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "citizen_count": len(roster),
        "recipient_count": sum(amount > 0 for amount in amounts.values()),
        "expected_issued_won": sum(amounts.values()),
        "amount_by_aid": {aid: amounts[aid] for aid in roster},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--policy-file", type=Path, required=True)
    parser.add_argument("--expected-agents", type=int, required=True)
    parser.add_argument("--require-all-recipients", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--roster-out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.roster_out.exists() or args.out == args.roster_out:
        raise ValueError("refusing to overwrite frozen entitlement or roster file")
    result = build(args.archive, args.member, args.policy_file,
                   expected_agents=args.expected_agents,
                   require_all_recipients=args.require_all_recipients)
    roster = list(result["amount_by_aid"])
    for path, value in ((args.out, result), (args.roster_out, roster)):
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + f".tmp.{os.getpid()}")
        try:
            partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                               encoding="utf-8")
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    print(json.dumps({key: result[key] for key in
                      ("policy_id", "citizen_count", "recipient_count",
                       "expected_issued_won", "roster_sha256")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
