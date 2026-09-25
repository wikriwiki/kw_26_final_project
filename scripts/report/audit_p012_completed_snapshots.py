"""Audit P012 State accumulation against eligible POI transactions in saved snapshots.

Accepts both the early flat transaction export (elig/amt) and later snapshots
with full POI properties. A missing eligibility field is an error: without it,
the archive cannot independently verify the cashback accumulation.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def _archive_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member_rows(archive: tarfile.TarFile, suffix: str):
    names = [name for name in archive.getnames()
             if name.startswith("graph/") and name.endswith(suffix)]
    if len(names) != 1:
        raise ValueError(f"expected one {suffix} member, found {names}")
    stream = archive.extractfile(names[0])
    if stream is None:
        raise ValueError(f"cannot read {names[0]}")
    with gzip.GzipFile(fileobj=stream) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as lines:
            for line in lines:
                if line.strip():
                    yield json.loads(line)


def _transaction(row: dict) -> tuple[bool, int]:
    if "poi" in row and "spend" in row:
        eligible = row["poi"].get("sangsaeng_eligible")
        amount = row["spend"].get("actual_spent")
    else:
        eligible = row.get("elig")
        amount = row.get("amt")
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) or amount < 0 or int(amount) != amount:
        raise ValueError("transaction amount is missing or not a nonnegative integer")
    if (eligible is None and "poi" in row and
            row["poi"].get("type") in {"residence", "workplace"} and amount == 0):
        eligible = False
    if not isinstance(eligible, bool):
        raise ValueError("POI cashback eligibility is missing or not boolean")
    return eligible, int(amount)


def audit(paths: list[Path], expected_per_day: int | None = None,
          strict: bool = False) -> dict:
    if not paths:
        raise ValueError("at least one snapshot archive is required")
    states: dict[tuple[date, str], int] = {}
    eligible_spend: dict[tuple[date, str], int] = defaultdict(int)
    spend_keys: set[tuple[date, str]] = set()
    spend_rows: Counter[date] = Counter()
    inputs = []
    for path in paths:
        digest = _archive_digest(path)
        manifest_path = path.with_name(path.name.removesuffix(".tar.gz") + ".manifest.json")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if digest != manifest.get("archive_sha256") or path.stat().st_size != manifest.get("bytes"):
                raise ValueError(f"archive does not match manifest: {path}")
        else:
            raise ValueError(f"snapshot manifest missing: {manifest_path}")
        inputs.append({"archive": str(path), "sha256": digest})
        with tarfile.open(path, "r:gz") as archive:
            for row in _member_rows(archive, "_state.jsonl.gz"):
                key = (date.fromisoformat(str(row["day"])[:10]), str(row["aid"]))
                if key in states:
                    raise ValueError(f"duplicate State for {key}")
                value = row["state"].get("sangsaeng_month_spent")
                if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
                    raise ValueError(f"invalid State.sangsaeng_month_spent for {key}")
                states[key] = int(value)
            for row in _member_rows(archive, "_spend.jsonl.gz"):
                day = date.fromisoformat(str(row["day"])[:10])
                key = (day, str(row["aid"]))
                eligible, amount = _transaction(row)
                spend_keys.add(key)
                spend_rows[day] += 1
                if eligible:
                    eligible_spend[key] += amount

    days = sorted({day for day, _ in states})
    day_set = set(days)
    missing_calendar_days = []
    for earlier, later in zip(days, days[1:]):
        missing_calendar_days.extend(
            (earlier + timedelta(days=offset)).isoformat()
            for offset in range(1, (later - earlier).days)
        )
    daily = []
    mismatch_examples = []
    compared = 0
    missing_state_days = []
    for day in days:
        aids = {aid for d, aid in states if d == day}
        previous = day - timedelta(days=1)
        previous_observed_same_month = (previous in day_set and
                                        previous.month == day.month and previous.year == day.year)
        previous_aids = {aid for d, aid in states if d == previous} if previous_observed_same_month else set()
        comparable = aids & previous_aids
        mismatches = 0
        delta_total = 0
        eligible_total = 0
        for aid in comparable:
            delta = states[(day, aid)] - states[(previous, aid)]
            eligible = eligible_spend[(day, aid)]
            delta_total += delta
            eligible_total += eligible
            if delta != eligible:
                mismatches += 1
                if len(mismatch_examples) < 20:
                    mismatch_examples.append({"day": day.isoformat(), "aid": aid,
                                              "state_delta": delta, "eligible_spend": eligible})
        if comparable:
            compared += len(comparable)
        if expected_per_day is not None and len(aids) != expected_per_day:
            missing_state_days.append(day.isoformat())
        daily.append({"day": day.isoformat(), "state_agents": len(aids),
                      "spend_rows": spend_rows[day], "compared_agents": len(comparable),
                      "previous_observed_same_month": previous_observed_same_month,
                      "missing_current_from_previous": len(previous_aids - aids),
                      "missing_previous_for_current": len(aids - previous_aids) if previous_aids else None,
                      "state_delta_won": delta_total if comparable else None,
                      "eligible_spend_won": eligible_total if comparable else None,
                      "mismatch_agents": mismatches})
    result = {"inputs": inputs, "days": len(days), "paired_agent_days": compared,
              "mismatch_agents": sum(row["mismatch_agents"] for row in daily),
              "spend_without_state": len(spend_keys - states.keys()),
              "missing_state_days": missing_state_days,
              "missing_calendar_days": missing_calendar_days,
              "mismatch_examples": mismatch_examples, "daily": daily}
    if strict and (result["mismatch_agents"] or result["spend_without_state"] or
                   missing_state_days or missing_calendar_days or
                   (expected_per_day is not None and any(
                       row["compared_agents"] != expected_per_day
                       for row in daily if row["previous_observed_same_month"]))):
        raise ValueError("strict P012 snapshot audit failed: " + json.dumps(
            {key: result[key] for key in ("mismatch_agents", "spend_without_state", "missing_state_days",
                                          "missing_calendar_days", "mismatch_examples")},
            ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", action="append", required=True, type=Path)
    parser.add_argument("--expected-per-day", type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = audit(args.archive, args.expected_per_day, args.strict)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"days": result["days"],
                      "paired_agent_days": result["paired_agent_days"],
                      "mismatch_agents": result["mismatch_agents"],
                      "spend_without_state": result["spend_without_state"],
                      "missing_state_days": result["missing_state_days"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
