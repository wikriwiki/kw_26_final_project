"""Dependency-free, strict contracts for captured decisions and scored choices."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date, datetime
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_MODEL = "LGAI-EXAONE/EXAONE-4.0-1.2B"


def plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return plain(value.model_dump())
    if dataclasses.is_dataclass(value):
        return plain(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(plain(v) for v in value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    # Neo4j temporal values are present in Dawn snapshots.
    if hasattr(value, "iso_format"):
        return value.iso_format()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")


def canonical(value: Any) -> str:
    return json.dumps(plain(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@dataclasses.dataclass(frozen=True)
class ChoiceQuestion:
    key: str
    state: str
    instructions: str
    options: dict[str, str]

    def __post_init__(self):
        if not self.key or not self.state or not self.instructions:
            raise ValueError("Question key/state/instructions must be nonempty")
        if not 2 <= len(self.options) <= 52:
            raise ValueError("Each question needs 2..52 options; never silently prune")
        if any(not isinstance(k, str) or not k or not isinstance(v, str)
               for k, v in self.options.items()):
            raise ValueError("Options must map nonempty keys to text")


@dataclasses.dataclass(frozen=True)
class TrainingExample:
    question: ChoiceQuestion
    target: str
    group: str
    kind: str


@dataclasses.dataclass(frozen=True)
class ChoiceScores:
    key: str
    labels: list[str]
    logits: list[float]
    probabilities: list[float]
    latency_seconds: float = 0.0
    input_tokens: int = 0

    def __post_init__(self):
        n = len(self.labels)
        if n < 2 or len(set(self.labels)) != n or len(self.logits) != n or len(self.probabilities) != n:
            raise ValueError("Invalid scored choice shape")
        if not all(math.isfinite(x) for x in self.logits + self.probabilities):
            raise ValueError("Nonfinite choice scores")
        if any(p < 0 or p > 1 for p in self.probabilities) or abs(sum(self.probabilities) - 1) > 1e-5:
            raise ValueError("Invalid choice distribution")


def finite_number(value: Any, low: float, high: float | None = None) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= low and (high is None or value <= high))


def validate_output(snapshot: dict, output: dict) -> list[str]:
    """Reject invalid complete day bundles; never repair or fill defaults."""
    errors = []
    candidates = snapshot.get("candidates", {})
    if not isinstance(candidates, dict) or any(not isinstance(cs, list) or
            any(not isinstance(c, dict) for c in cs) for cs in candidates.values()):
        return ["invalid_candidates"]
    expected = {str(k) for k, cs in candidates.items() if cs}
    picks = output.get("picks")
    if not isinstance(picks, list):
        return ["picks_missing"]
    seen = set()
    chosen_pois = set()
    policy_totals = {}
    policies = {p.get("id"): p for p in (snapshot.get("active_policies") or []) if isinstance(p, dict)}
    remaining = snapshot.get("grant_remaining") or {}
    for p in picks:
        if not isinstance(p, dict) or not isinstance(p.get("order"), int) or isinstance(p.get("order"), bool):
            errors.append("invalid_order")
            continue
        order = str(p["order"])
        if order in seen:
            errors.append("duplicate_order")
        seen.add(order)
        if not isinstance(p.get("poi_id"), str) or not p["poi_id"]:
            errors.append("invalid_poi_id")
            continue
        if p.get("poi_id") in chosen_pois:
            errors.append("duplicate_poi_in_day")
        chosen_pois.add(p.get("poi_id"))
        if p.get("poi_id") not in {c.get("poi_id") for c in candidates.get(order, [])}:
            errors.append("poi_not_in_event_candidates")
        if not finite_number(p.get("actual_spent"), .000001):
            errors.append("invalid_spend")
        if not finite_number(p.get("actual_satisfaction"), 0, 1):
            errors.append("invalid_satisfaction")
        wallet = p.get("policy_spend")
        if wallet is None:
            wallet = {}
        if not isinstance(wallet, dict) or any(not finite_number(v, 0) for v in wallet.values()):
            errors.append("invalid_policy_spend")
        elif finite_number(p.get("actual_spent"), 0) and sum(wallet.values()) > p["actual_spent"]:
            errors.append("policy_spend_exceeds_total")
        if isinstance(wallet, dict):
            for pid, value in wallet.items():
                if not finite_number(value, 0):
                    continue
                policy_totals[pid] = policy_totals.get(pid, 0) + value
                if value and pid not in policies:
                    errors.append("unknown_policy")
                pol = policies.get(pid, {})
                candidate = next((c for c in candidates.get(order, []) if c.get("poi_id") == p.get("poi_id")), {})
                if value and pol.get("poi_restricted") and not candidate.get("coupon_eligible"):
                    errors.append("ineligible_policy_poi")
                if value and pol.get("type") == "grant" and not finite_number(remaining.get(pid), 0):
                    errors.append("missing_policy_balance")
    for pid, amount in policy_totals.items():
        if pid in remaining and (not finite_number(remaining[pid], 0) or amount > remaining[pid]):
            errors.append("policy_balance_exceeded")
    if seen != expected:
        errors.append("incomplete_day_bundle")
    return sorted(set(errors))
