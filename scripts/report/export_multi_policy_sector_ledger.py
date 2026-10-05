"""Export complete citizen-day category spending before a policy arm graph reset.

The export is descriptive transaction evidence. Paired effects and empirical
comparisons are computed later, from separately preserved arm ledgers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_policy_daily_ledger import STATE_QUERY, driver_session, receipt_days
from export_cashback_month import verify_cohorts, verify_metric_provenance
from paired_grant_effect import dates, roster_file


SPEND_QUERY = """
MATCH (a:Agent)-[:HAS_PLAN]->(pl:Plan)-[i:INCLUDES]->(p:POI)
WHERE a.id IN $aids AND toString(pl.day) = $day
OPTIONAL MATCH (p)-[:IN_CATEGORY]->(c:Category)
WITH a, i, p, head(collect(c)) AS c
RETURN a.id AS aid, i.actual_spent AS amt,
       i.spent_from_policy AS spent_from_policy,
       i.instant_discount AS instant_discount, i.policy_rebate AS policy_rebate,
       p.sangsaeng_eligible AS sangsaeng_eligible,
       c.name AS sub, c.parent AS l1
"""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _money(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _funding(value: object) -> dict[str, int]:
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError("spent_from_policy must be an object")
    return {str(key): _money(amount, "policy payment") for key, amount in value.items()}


def _metrics(path: Path, roster: list[str]) -> tuple[dict[str, dict], str]:
    seen: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            aid = row.get("aid")
            if aid in seen or row.get("status") != "ok":
                raise ValueError(f"duplicate or non-ok metric: {path} {aid}")
            seen[aid] = row
    if set(seen) != set(roster):
        raise ValueError(f"incomplete metric roster: {path}")
    return seen, _sha(path)


@lru_cache(maxsize=1)
def _excluded_kdi() -> dict[str, dict[str, float]]:
    """{동코드: {KDI업종: 제외분 중 몫}} — BDC dong_consumption 구성비.

    적립 제외분(`online_spent`)은 한 덩어리로만 적혀 있어서 'KDI 제외업종 중
    유통'(K10) 을 셀 수 없었다. 그 덩어리에 **BDC 의 업종 구성비를 입힌다.**

    이것은 시뮬의 선택이 아니라 **자료에서 온 안분**이다. 그래서 이 항의 증가율은
    제외분 전체의 증가율과 같고, 독립된 정보를 주지 않는다 — 채점표가 그렇게 적는다.
    """
    p = Path(__file__).resolve().parents[2] / "data/sangsaeng/dong_eligible_share.json"
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): dict(v.get("excluded_kdi") or {}) for k, v in raw.items()
            if v.get("excluded_kdi")}


def _dong_of(aid: str) -> str:
    parts = str(aid or "").split("_")
    return parts[1] if len(parts) > 1 and parts[1].isdigit() else ""


def aggregate_day(states: list[dict], spends: list[dict], *, roster: list[str],
                  day: str, arm: str, policy_id: str | None) -> list[dict]:
    state_by_aid = {row["aid"]: row for row in states}
    if len(state_by_aid) != len(states) or set(state_by_aid) != set(roster):
        raise ValueError(f"incomplete or duplicate State roster: {day}")
    buckets = {aid: {"offline_spent": 0, "sangsaeng_eligible_offline_spent": 0,
                     "unclassified_won": 0, "by_sub": defaultdict(int),
                     "by_l1": defaultdict(int), "funded_by_sub": defaultdict(int),
                     "eligible_by_sub": defaultdict(int),
                     "eligible_by_l1": defaultdict(int),
                     "discount_by_sub": defaultdict(int), "rebate_by_sub": defaultdict(int),
                     "instant_discount_won": 0, "policy_rebate_won": 0,
                     "policy_funded_won": 0} for aid in roster}
    for spend in spends:
        aid = spend.get("aid")
        if aid not in buckets:
            raise ValueError(f"unregistered transaction: {aid} {day}")
        amount = _money(spend.get("amt"), "transaction amount")
        funding = _funding(spend.get("spent_from_policy"))
        funded = funding.get(policy_id, 0) if policy_id else 0
        if sum(funding.values()) > amount:
            raise ValueError(f"policy payments exceed gross: {aid} {day}")
        if policy_id is None and any(funding.values()):
            raise ValueError(f"control transaction has policy funding: {aid} {day}")
        bucket = buckets[aid]
        bucket["offline_spent"] += amount
        # 업종별 적립/제외 분해 — 이것이 없으면 '제외업종 중 유통'(KDI K10) 을 셀 수 없다.
        eligible = spend.get("sangsaeng_eligible") is True
        if eligible:
            bucket["sangsaeng_eligible_offline_spent"] += amount
        sub, l1 = spend.get("sub"), spend.get("l1")
        if sub:
            bucket["by_sub"][str(sub)] += amount
            bucket["funded_by_sub"][str(sub)] += funded
            if eligible:
                bucket["eligible_by_sub"][str(sub)] += amount
        else:
            bucket["unclassified_won"] += amount
        if l1:
            bucket["by_l1"][str(l1)] += amount
            if eligible:
                bucket["eligible_by_l1"][str(l1)] += amount
        bucket["policy_funded_won"] += funded
        # 결제 즉시 할인(자기부담이 준 금액)과 나중 환급 — 정책 갈래에서만 0 이 아닐 수 있다.
        discount = sum(_funding(spend.get("instant_discount")).values())
        rebate = sum(_funding(spend.get("policy_rebate")).values())
        if policy_id is None and (discount or rebate):
            raise ValueError(f"control transaction has a policy discount or rebate: {aid} {day}")
        if discount > amount:
            raise ValueError(f"discount exceeds gross: {aid} {day}")
        bucket["instant_discount_won"] += discount
        bucket["policy_rebate_won"] += rebate
        if sub:
            bucket["discount_by_sub"][str(sub)] += discount
            bucket["rebate_by_sub"][str(sub)] += rebate
    out = []
    for aid in roster:
        state = state_by_aid[aid]
        bucket = buckets[aid]
        online = _money(state.get("online_spent"), "online spending")
        received = _funding(state.get("grant_received"))
        remaining = _funding(state.get("grant_remaining"))
        if policy_id is None and (any(received.values()) or any(remaining.values())):
            raise ValueError(f"control State has grant funding: {aid} {day}")
        if sum(bucket["by_sub"].values()) + bucket["unclassified_won"] != bucket["offline_spent"]:
            raise ValueError(f"sector spending does not reconcile: {aid} {day}")
        if sum(bucket["eligible_by_sub"].values()) > bucket["sangsaeng_eligible_offline_spent"]:
            raise ValueError(f"eligible sector spending exceeds eligible total: {aid} {day}")
        out.append({"aid": aid, "day": day, "arm": arm, "policy_id": policy_id,
                    "offline_spent": bucket["offline_spent"],
                    "online_spent": online,
                    "total_spent": bucket["offline_spent"] + online,
                    "sangsaeng_eligible_offline_spent": bucket["sangsaeng_eligible_offline_spent"],
                    "unclassified_won": bucket["unclassified_won"],
                    "policy_funded_won": bucket["policy_funded_won"],
                    "instant_discount_won": bucket["instant_discount_won"],
                    "policy_rebate_won": bucket["policy_rebate_won"],
                    "discount_by_sub": dict(sorted((k, v) for k, v in bucket["discount_by_sub"].items() if v)),
                    "rebate_by_sub": dict(sorted((k, v) for k, v in bucket["rebate_by_sub"].items() if v)),
                    "grant_received_cumulative": received.get(policy_id, 0) if policy_id else 0,
                    "grant_remaining": remaining.get(policy_id, 0) if policy_id else 0,
                    "by_sub": dict(sorted(bucket["by_sub"].items())),
                    "by_l1": dict(sorted(bucket["by_l1"].items())),
                    "eligible_by_sub": dict(sorted(bucket["eligible_by_sub"].items())),
                    "eligible_by_l1": dict(sorted(bucket["eligible_by_l1"].items())),
                    # 제외분에 BDC 업종 구성비를 입힌 안분값 — 시뮬의 선택이 아니다.
                    "online_by_kdi_attributed": {
                        k: int(round(online * w))
                        for k, w in sorted(_excluded_kdi().get(_dong_of(aid), {}).items())},
                    "funded_by_sub": dict(sorted(bucket["funded_by_sub"].items()))})
    return out


def export(*, roster: list[str], days: list[str], arm: str, policy_id: str | None,
           metrics_dir: Path, out: Path,
           policy_file: Path | None = None,
           effective_from: str | None = None,
           effective_until: str | None = None) -> dict:
    if not roster or len(set(roster)) != len(roster) or not days or dates(days[0], days[-1]) != days:
        raise ValueError("unique roster and contiguous dates required")
    if arm not in ("on", "off") or (arm == "off" and policy_id is not None):
        raise ValueError("arm/policy mismatch")
    if (effective_from is None) != (effective_until is None):
        raise ValueError("effective_from and effective_until must be supplied together")
    if effective_from and (not policy_id or effective_from > effective_until):
        raise ValueError("invalid policy effective window")
    policy_input = None
    if arm == "on" and policy_id:
        if policy_file is None:
            raise ValueError("policy arm requires its frozen policy input file")
        raw_policy = json.loads(policy_file.read_text(encoding="utf-8"))
        if (raw_policy.get("id") != policy_id
                or raw_policy.get("effective_from") != effective_from
                or raw_policy.get("effective_until") != effective_until):
            raise ValueError("frozen policy input disagrees with requested policy window")
        policy_input = {"path": str(policy_file), "sha256": _sha(policy_file)}
    elif policy_file is not None:
        raise ValueError("control or environment-only arm must not supply a policy input file")
    # 지급 일정이 있으면 사람마다 받는 날부터만 정책이 보인다(dawn_context.visible_from_receipt)
    receipt_day = receipt_days(raw_policy, roster) if policy_input else None
    rows: list[dict] = []
    metrics_hashes = {}
    cohort = verify_cohorts(metrics_dir, days, roster)
    if cohort["prompt_variant"] != "v53":
        raise ValueError("sector export requires the frozen v53 prompt")
    served_path = metrics_dir.parent / "served_model_evidence.json"
    served_evidence = json.loads(served_path.read_text(encoding="utf-8"))
    served_model = "LGAI-EXAONE/EXAONE-4.5-33B-AWQ"
    if (served_evidence.get("served_model_ids") != [served_model]
            or f"--model-path {served_model}" not in
            str(served_evidence.get("server_command") or "")):
        raise ValueError("actual served model evidence is missing or inconsistent")
    served_model_provenance = {"model_id": served_model,
                               "evidence_sha256": _sha(served_path)}
    cohort_hashes = {day: _sha(metrics_dir.parent / f"cohort_{day}.json")
                     for day in days}
    daily_metrics: dict[str, list[dict]] = {}
    requested_models: set[str] = set()
    provenance: dict[str, set] = {key: set() for key in (
        "execution_fingerprint", "paired_environment_fingerprint",
        "source_fingerprint", "experience_environment_id", "experience_run_id")}
    with driver_session() as session:
        graph_policies = [dict(row["policy"]) for row in session.run(
            "MATCH (p:Policy) RETURN properties(p) AS policy")]
        if policy_input:
            if len(graph_policies) != 1 or any(
                str(graph_policies[0].get(key)) != str(raw_policy.get(key))
                for key in ("id", "type", "description", "effective_from", "effective_until")
            ):
                raise ValueError("graph policy does not match frozen input")
        elif graph_policies:
            raise ValueError("graph contains a Policy node in control/environment arm")
        for day in days:
            metrics, metrics_hashes[day] = _metrics(metrics_dir / f"day_{day}.jsonl", roster)
            daily_metrics[day] = list(metrics.values())
            for aid, row in metrics.items():
                requested_model = (row.get("decision_provenance") or {}).get("model_id")
                if not isinstance(requested_model, str) or not requested_model:
                    raise ValueError(f"missing LLM request model in metric: {aid} {day}")
                requested_models.add(requested_model)
                exposed = row.get("experience_policy_ids")
                active = (not effective_from or effective_from <= day <= effective_until)
                if active and receipt_day is not None and arm == "on":
                    due = receipt_day.get(aid)
                    active = due is not None and due <= day
                expected = [policy_id] if arm == "on" and policy_id and active else []
                if not isinstance(exposed, list) or sorted(exposed) != expected:
                    raise ValueError(f"policy exposure mismatch: {aid} {day}")
                for key, values in provenance.items():
                    value = row.get(key)
                    if value is not None or key in ("experience_environment_id",
                                                     "paired_environment_fingerprint"):
                        values.add(value)
            states = [dict(r) for r in session.run(STATE_QUERY, day=day, aids=roster)]
            spends = [dict(r) for r in session.run(SPEND_QUERY, day=day, aids=roster)]
            rows.extend(aggregate_day(states, spends, roster=roster, day=day,
                                      arm=arm, policy_id=policy_id))
    verify_metric_provenance(daily_metrics, cohort)
    if len(requested_models) != 1:
        raise ValueError("LLM request model changed within run")
    if len(rows) != len(roster) * len(days):
        raise ValueError("citizen-day matrix incomplete")
    for key, values in provenance.items():
        if key != "source_fingerprint" and len(values) != 1:
            raise ValueError(f"inconsistent or missing {key}: {len(values)} values")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        temporary.replace(out)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {"schema": "multi_policy_sector_ledger_v1", "arm": arm,
                "policy_id": policy_id, "start": days[0], "end": days[-1],
                "effective_from": effective_from, "effective_until": effective_until,
                "policy_input": policy_input,
                "citizens": len(roster), "days": len(days), "rows": len(rows),
                "roster_sha256": hashlib.sha256(json.dumps(sorted(roster), ensure_ascii=False).encode()).hexdigest(),
                "metrics_sha256": metrics_hashes,
                "cohort_sha256": cohort_hashes,
                "served_model_provenance": served_model_provenance,
                "prompt_provenance": {**cohort,
                                      "requested_model_id": next(iter(requested_models))},
                "provenance": {key: sorted(values, key=lambda x: str(x))
                               for key, values in provenance.items()},
                "output_sha256": _sha(out)}
    manifest_path = out.with_name(out.name + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roster", type=Path, required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--arm", choices=("on", "off"), required=True)
    ap.add_argument("--policy-id")
    ap.add_argument("--policy-file", type=Path)
    ap.add_argument("--effective-from")
    ap.add_argument("--effective-until")
    ap.add_argument("--metrics-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m = export(roster=roster_file(a.roster), days=dates(a.start, a.end), arm=a.arm,
               policy_id=a.policy_id, metrics_dir=a.metrics_dir, out=a.out,
               policy_file=a.policy_file,
               effective_from=a.effective_from,
               effective_until=a.effective_until)
    print(f"{a.out}: {m['rows']} verified citizen-days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
