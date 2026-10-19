"""Opt-in factual context and executed-payment accounting for the smoking-ban study.

This module never reads the study's outcomes. It does not alter POI rankings,
visits, spending, or satisfaction. Both arms use the same registered roster.
The runtime manifest contains the cohort and POI classifications, not evaluation
targets. Each arm still requires its own restored Neo4j baseline.
"""
from __future__ import annotations

from collections import defaultdict
from contextvars import ContextVar
from datetime import date
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path


EFFECTIVE_DATE = date(2017, 12, 3)
STUDY_DISTRICTS = frozenset({"11650", "11350", "11710"})
FACILITY_LABELS = {
    "billiard": "당구장", "indoor_golf": "실내 골프연습장", "screen_golf": "스크린골프장",
}
SMOKING_LABELS = {"smoker": "흡연자", "non_smoker": "비흡연자",
                  "unknown": "흡연 여부 정보 없음(성인 통계 적용 대상 아님 또는 나이 미상)"}
_LLM_CONTEXT = ContextVar("no_smoking_llm_context", default=None)


class NoSmokingContext:
    def __init__(self, *, arm: str, cohort: list[dict], pois: list[dict], assignment_seed: int,
                 simulation_seed: int | None = None):
        if arm not in {"off", "on"}:
            raise ValueError("SIM_NO_SMOKING_ARM must be off or on")
        if not isinstance(assignment_seed, int) or isinstance(assignment_seed, bool):
            raise ValueError("assignment_seed must be an integer")
        if simulation_seed is not None and (not isinstance(simulation_seed, int) or isinstance(simulation_seed, bool)):
            raise ValueError("simulation_seed must be an integer")
        if not isinstance(cohort, list) or not cohort or not isinstance(pois, list) or not pois:
            raise ValueError("A nonempty frozen cohort and POI registry are required")
        self.arm = arm
        self.assignment_seed = assignment_seed
        self.simulation_seed = assignment_seed if simulation_seed is None else simulation_seed
        self.people = {}
        for person in cohort:
            aid = person.get("id")
            status = person.get("smoking_status")
            if not isinstance(aid, str) or not aid or aid in self.people:
                raise ValueError("Cohort IDs must be unique nonempty strings")
            if status not in SMOKING_LABELS:
                raise ValueError(f"Invalid or missing smoking_status for {aid}")
            self.people[aid] = status
        self.pois = {}
        for poi in pois:
            pid = poi.get("poi_id")
            if not isinstance(pid, str) or not pid or pid in self.pois:
                raise ValueError("POI IDs must be unique nonempty strings")
            district, facility = poi.get("district_code"), poi.get("facility_type")
            if not isinstance(district, str) or len(district) != 5 or not district.isdigit():
                raise ValueError(f"Explicit five-digit district_code required for {pid}")
            if facility not in {*FACILITY_LABELS, "other"}:
                raise ValueError(f"Explicit facility_type required for {pid}; use other for exclusions")
            # Only these classified facts can ever enter a decision prompt.
            self.pois[pid] = {"poi_id": pid, "district_code": district, "facility_type": facility}
        if not any(self.is_evaluation_poi(pid) for pid in self.pois):
            raise ValueError("Registry has no study-eligible indoor sports POIs")

    @property
    def agent_ids(self) -> list[str]:
        return sorted(self.people)

    def require_graph_roster(self, ids) -> list[str]:
        missing = set(self.people) - set(ids)
        if missing:
            raise ValueError(f"Frozen cohort missing from initialized graph: {len(missing)} agents; "
                             f"examples={sorted(missing)[:5]}")
        return self.agent_ids

    def is_active(self, day: date | str) -> bool:
        day = date.fromisoformat(day) if isinstance(day, str) else day
        return self.arm == "on" and day >= EFFECTIVE_DATE

    def stable_seed(self, *parts) -> int:
        """Arm-independent seeds; server/GPU determinism is a separate limitation."""
        raw = json.dumps([self.simulation_seed, *map(str, parts)], ensure_ascii=False).encode("utf-8")
        return int.from_bytes(hashlib.sha256(raw).digest()[:4], "big") % 2147483647

    def is_evaluation_poi(self, poi_id: str) -> bool:
        poi = self.pois.get(poi_id, {})
        return (poi.get("district_code") in STUDY_DISTRICTS
                and poi.get("facility_type") in FACILITY_LABELS)

    def context_for(self, agent_id: str, day: date | str) -> dict:
        status = self.people[agent_id]  # Missing cohort membership must fail closed.
        active = self.is_active(day)
        rule = (
            "오늘 실험 대상으로 확인된 노원구·서초구·송파구의 당구장과 실내 골프연습장"
            "(스크린골프장 포함)은 금연구역이다. 시설의 이용 공간에서 담배를 피울 수 없다. "
            "별도 적법한 흡연실이 있는 경우 그 안에서만 흡연할 수 있지만, 각 시설의 흡연실 유무는 "
            "정보가 없으므로 가정하지 않는다. 금연 지정은 시설의 영업 중단이 아니다."
            if active else
            "오늘은 실험 대상으로 확인된 노원구·서초구·송파구의 당구장과 실내 골프연습장"
            "(스크린골프장 포함)에 대한 금연구역 확대가 "
            "적용되지 않는 조건이다. 각 시설의 실제 흡연 여부는 별도 정보가 없으면 알 수 없다."
        )
        return {"smoking_status": status, "policy_active": active,
                "prompt": f"흡연 상태: {SMOKING_LABELS[status]} ({status})\n실내 체육시설 이용 규칙: {rule}"}

    def apply(self, ctx, agent_id: str, day: date | str) -> None:
        facts = self.context_for(agent_id, day)
        # The Dawn persona cache is shared; never mutate its dictionary in place.
        ctx.persona = dict(ctx.persona, smoking_status=facts["smoking_status"],
                           _no_smoking_prompt=facts["prompt"])

    def candidate_facts(self, candidate_ids) -> str:
        rows = []
        for pid in sorted(set(candidate_ids)):
            poi = self.pois.get(pid, {})
            label = FACILITY_LABELS.get(poi.get("facility_type"))
            if label and self.is_evaluation_poi(pid):
                rows.append(f"- {pid}: {label}")
        return "\n".join(rows)

    def summarize_receipts(self, receipts: list[dict], agent_id: str, day: date | str) -> dict:
        """Count final positive purchases, not intentions or unique human visitors."""
        by_poi = defaultdict(lambda: {"payment_count": 0, "revenue_krw": 0})
        seen = set()
        for receipt in receipts:
            if not self.is_evaluation_poi(receipt.get("poi_id")):
                continue
            if receipt.get("agent_id") != agent_id or receipt.get("occurred_at") != str(day):
                raise ValueError("Receipt belongs to a different agent or day")
            event_id = receipt.get("event_id")
            if not isinstance(event_id, str) or not event_id or event_id in seen:
                raise ValueError("Missing or duplicate executed receipt ID")
            seen.add(event_id)
            amount = receipt.get("amount")
            if (not isinstance(amount, (int, float)) or isinstance(amount, bool)
                    or not math.isfinite(amount) or amount < 0 or int(amount) != amount):
                raise ValueError("Receipt amount must be a nonnegative whole KRW amount")
            if receipt.get("purchase_status") not in {"purchased", "reduced"} or amount == 0:
                continue
            row = by_poi[receipt["poi_id"]]
            row["payment_count"] += 1
            row["revenue_krw"] += int(amount)
        return {"arm": self.arm, "policy_active": self.is_active(day),
                "smoking_status": self.people[agent_id],
                "payment_count": sum(r["payment_count"] for r in by_poi.values()),
                "revenue_krw": sum(r["revenue_krw"] for r in by_poi.values()),
                "by_poi": [dict(self.pois[pid], **by_poi[pid]) for pid in sorted(by_poi)]}

    def annotate_executed_events(self, events: list[dict]) -> list[dict]:
        """Copy registered POI facts into interview evidence without changing actions.

        The manifest classifies a facility; it does not observe smoke exposure,
        an individual's motives, room availability, or a policy effect.
        """
        manifest_sha = getattr(self, "manifest_sha256", None)
        if (not isinstance(manifest_sha, str) or len(manifest_sha) != 64
                or any(c not in "0123456789abcdef" for c in manifest_sha)):
            raise ValueError("Frozen POI registry hash is required for evidence annotation")
        annotated = []
        for event in events:
            row = dict(event)
            # Only verified registry facts can populate these evidence fields.
            for key in ("facility_type", "district_code", "policy_target", "poi_registry_sha256"):
                row.pop(key, None)
            poi = self.pois.get(row.get("poi_id"))
            if poi:
                row.update(facility_type=poi["facility_type"], district_code=poi["district_code"],
                           policy_target=self.is_evaluation_poi(poi["poi_id"]),
                           poi_registry_sha256=manifest_sha)
            annotated.append(row)
        return annotated


def prompt_for_persona(persona: dict) -> str:
    return persona.get("_no_smoking_prompt") or ""


def begin_llm_scope(identity, day, stage: str) -> None:
    runtime = configured_context()
    _LLM_CONTEXT.set((runtime.stable_seed(identity, day, stage), 0) if runtime else None)


def next_llm_seed() -> int | None:
    context = _LLM_CONTEXT.get()
    if context is None or configured_context() is None:
        return None
    seed, index = context
    _LLM_CONTEXT.set((seed, index + 1))
    return (seed + index) % 2147483647


def clear_llm_scope() -> None:
    _LLM_CONTEXT.set(None)


@lru_cache(maxsize=4)
def _load(path: str, arm: str) -> NoSmokingContext:
    raw = Path(path).read_bytes()
    doc = json.loads(raw)
    allowed = {"experiment_id", "schema_version", "cohort", "pois", "assignment_seed", "simulation_seed"}
    if not isinstance(doc, dict) or set(doc) - allowed:
        raise ValueError("Runtime manifest permits cohort/classification inputs only, not evaluation data")
    if doc.get("experiment_id") != "no_smoking_zone":
        raise ValueError("Unexpected no-smoking experiment_id")
    runtime = NoSmokingContext(arm=arm, cohort=doc["cohort"], pois=doc["pois"],
                               assignment_seed=doc["assignment_seed"], simulation_seed=doc.get("simulation_seed"))
    runtime.manifest_sha256 = hashlib.sha256(raw).hexdigest()
    return runtime


def configured_context() -> NoSmokingContext | None:
    path = os.environ.get("SIM_NO_SMOKING_MANIFEST", "").strip()
    arm = os.environ.get("SIM_NO_SMOKING_ARM", "").strip()
    if not path and not arm:
        return None
    if not path or not arm:
        raise ValueError("SIM_NO_SMOKING_MANIFEST and SIM_NO_SMOKING_ARM are both required")
    return _load(str(Path(path).resolve()), arm)
