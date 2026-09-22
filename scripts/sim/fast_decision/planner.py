"""Conditional finite decisions; shared by inference and teacher example export.

Student outputs are proposals only. All model probabilities are explicitly raw
scores, not empirical visit probabilities or calibrated error estimates.
"""
from __future__ import annotations

import dataclasses
import math
import random

from .contracts import ChoiceQuestion, TrainingExample, canonical, fingerprint, validate_output, finite_number

SPEND_LEVELS = tuple(range(1000, 31000, 1000)) + (35000, 40000, 50000, 60000, 80000,
                                              100000, 150000, 200000, 300000, 500000, 1000000)
SAT_LEVELS = tuple(round(i / 20, 2) for i in range(21))
FACTORS = {
    "known": "과거 방문과 친숙함", "distance": "거리와 일정 제약",
    "satisfaction": "기존 만족도와 선호", "random": "특정 근거 없는 탐색·다양성",
    "DEFER": "위 요인으로 설명할 수 없거나 근거가 부족함",
}


def preflight(snapshot: dict) -> list[str]:
    """Conservative domain bounds, NOT a learned correctness certificate."""
    reasons = []
    state = snapshot.get("state") or {}
    context = snapshot.get("context") or {}
    if not isinstance(context, dict) or not all(isinstance(context.get(k), list) for k in ("memory", "appointment", "social")):
        reasons.append("missing_emotional_social_context")
        context = {}
    if not all(k in state for k in ("mood", "fatigue", "balance")):
        reasons.append("missing_state")
    if not finite_number(state.get("balance"), 0):
        reasons.append("invalid_balance")
    for k in ("mood", "fatigue"):
        v = state.get(k)
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1:
            reasons.append("invalid_emotional_state")
    if snapshot.get("active_policies") or snapshot.get("grant_remaining"):
        reasons.append("policy_context")
    if context.get("appointment") or context.get("social"):
        reasons.append("social_context")
    # Bounds are scope limitations, not empirically calibrated thresholds.
    if isinstance(state.get("mood"), (int, float)) and state["mood"] < .3:
        reasons.append("low_mood")
    if isinstance(state.get("fatigue"), (int, float)) and state["fatigue"] > .7:
        reasons.append("high_fatigue")
    if any(isinstance(m.get("satisfaction"), (int, float)) and m["satisfaction"] < .3
           for m in context.get("memory", []) if isinstance(m, dict)):
        reasons.append("negative_memory")
    events = snapshot.get("stage1", {}).get("events", [])
    for key, cs in snapshot.get("candidates", {}).items():
        if not cs:
            continue
        try:
            event = events[int(key)]
        except (ValueError, IndexError, TypeError):
            reasons.append("invalid_event")
            continue
        if event.get("category") not in ("식사", "카페", "디저트"):
            reasons.append("unvalidated_category")
        if event.get("trigger") not in (None, "none", "lifestyle", "top_category"):
            reasons.append("behavior_change_signal")
        if event.get("pinned_poi") or event.get("with_agents"):
            reasons.append("constrained_social_event")
        ids = [c.get("poi_id") for c in cs]
        if not all(isinstance(i, str) and i and i != "DEFER" for i in ids) or len(set(ids)) != len(ids):
            reasons.append("invalid_candidates")
        if len(cs) > 51:
            reasons.append("too_many_candidates")
    if not any(snapshot.get("candidates", {}).values()):
        reasons.append("no_candidates")
    return sorted(set(reasons))


def _state_fields(snapshot: dict) -> dict:
    # Explicit whitelist: labels, teacher answers and future records cannot leak.
    fields = ("today", "stage1", "persona", "state", "candidates", "recent_poi_ids",
              "active_policies", "grant_remaining", "context")
    data = {k: snapshot.get(k) for k in fields}
    if isinstance(data.get("context"), dict):
        data["context"] = {k: data["context"][k] for k in
                           ("memory", "appointment", "social", "knows_poi_summary", "zone_candidates")
                           if k in data["context"]}
    return data


class PreparedState:
    """Serialize immutable evidence once per proposal, preserving exact JSON.

    Only provisional picks change between dependent questions. Keeping the same
    key order and separators preserves the previously tested prompt bytes.
    """
    def __init__(self, snapshot: dict):
        fields = _state_fields(snapshot)
        before, after = [], []
        for key in sorted(fields):
            entry = canonical(key) + ":" + canonical(fields[key])
            (before if key < "provisional_picks" else after).append(entry)
        self.prefix = "{" + ",".join(before) + ("," if before else "") + '"provisional_picks":'
        self.suffix = ("," if after else "") + ",".join(after) + "}"

    def render(self, picks: list[dict]) -> str:
        return self.prefix + canonical(picks) + self.suffix


def _state(snapshot: dict, picks: list[dict]) -> str:
    return PreparedState(snapshot).render(picks)


def route_question(snapshot: dict, prepared: PreparedState | None = None) -> ChoiceQuestion:
    return ChoiceQuestion("route", (prepared or PreparedState(snapshot)).render([]),
                          "일상적 후보 선택을 진행할 수 있습니까? 새 정책·누적 불만·감정 변화·"
                          "제약 충돌·정보 부족·리뷰 확인 필요가 있으면 보류하세요.",
                          {"PROCEED": "검증 대상 범위의 일상적 선택", "DEFER": "큰 EXAONE의 판단 필요"})


def event_question(snapshot: dict, picks: list[dict], order: int, kind: str,
                   prepared: PreparedState | None = None) -> ChoiceQuestion:
    if kind == "poi":
        options = {c["poi_id"]: canonical(c) for c in snapshot["candidates"][str(order)]}
        options["DEFER"] = "후보를 고를 근거가 부족하거나 추가 리뷰·심층 판단이 필요함"
        instruction = "앞선 거래와 오늘 전체 계획을 고려해 방문할 후보를 고르세요."
    elif kind == "spend":
        options = {str(v): f"총 지출 {v}원" for v in SPEND_LEVELS}
        options["DEFER"] = "금액 범위가 맞지 않거나 판단 근거가 부족함"
        instruction = "선택한 장소·예산·앞선 소비에 맞는 총 지출을 고르세요. 정책 돈을 추가 소비로 간주하지 마세요."
    elif kind == "satisfaction":
        options = {f"{v:.2f}": f"경험 만족도 {v:.2f} (0 매우 불만족, 1 매우 만족)" for v in SAT_LEVELS}
        options["DEFER"] = "만족도를 판단하기에 정보가 부족하거나 추가 검토가 필요함"
        instruction = "선택한 장소·금액·이전 경험·현재 감정을 반영한 경험 만족도를 고르세요."
    elif kind == "factor":
        options, instruction = dict(FACTORS), "제공된 사실 중 선택에 해당하는 요인을 고르세요."
    else:
        raise ValueError(f"Unknown decision kind: {kind}")
    return ChoiceQuestion(f"{order}.{kind}", (prepared or PreparedState(snapshot)).render(picks),
                          f"이벤트 {order}: {instruction}", options)


def _choose(scores, question, mode, seed):
    if scores.key != question.key or scores.labels != list(question.options):
        raise ValueError("Backend returned mismatched question or options")
    if mode == "argmax":
        return scores.labels[max(range(len(scores.logits)), key=scores.logits.__getitem__)]
    if mode != "sample":
        raise ValueError("selection must be argmax or sample")
    rng = random.Random(int(fingerprint([seed, question.key])[:16], 16))
    return rng.choices(scores.labels, weights=scores.probabilities, k=1)[0]


def propose(snapshot: dict, backend, *, selection="argmax", seed=0) -> dict:
    if selection not in ("argmax", "sample"):
        raise ValueError("selection must be argmax or sample")
    reasons = preflight(snapshot)
    result = {"status": "deferred", "reasons": reasons, "output": None, "scores": [],
              "eligible_for_live": False, "probability_semantics": "uncalibrated_model_option_scores",
              "selection": selection, "seed": seed}
    if reasons:
        return result
    prepared = PreparedState(snapshot)
    questions = [route_question(snapshot, prepared)]
    score = backend.score(questions)[0]
    result["scores"].append(dataclasses.asdict(score))
    # Route is a semantic abstention; entropy is not treated as quality risk.
    if _choose(score, questions[0], "argmax", seed) == "DEFER":
        result["reasons"] = ["model_deferred"]
        return result
    picks = []
    stable_seed = [seed, snapshot.get("aid"), snapshot.get("today"), snapshot.get("snapshot_id")]
    for order in sorted(int(k) for k, cs in snapshot["candidates"].items() if cs):
        pick = {"order": order}
        for kind in ("poi", "spend", "satisfaction", "factor"):
            prefix = picks if kind == "poi" else picks + [pick.copy()]
            q = event_question(snapshot, prefix, order, kind, prepared)
            scores = backend.score([q])[0]
            result["scores"].append(dataclasses.asdict(scores))
            label = _choose(scores, q, selection if kind != "factor" else "argmax", stable_seed)
            if label == "DEFER":
                result["reasons"] = [f"model_deferred:{q.key}"]
                return result  # whole bundle discarded
            if kind == "poi":
                if label in {p["poi_id"] for p in picks}:
                    # Same final rejection, without spending three more model
                    # calls on an already invalid second visit.
                    result["reasons"] = ["duplicate_poi_in_day"]
                    return result
                pick["poi_id"] = label
            elif kind == "spend":
                pick["actual_spent"] = int(label)
            elif kind == "satisfaction":
                pick["actual_satisfaction"] = float(label)
            else:
                pick["pick_factor"] = label
        pick["policy_spend"] = {}
        # Trace of model inputs, not a fabricated narrative or human explanation.
        pick["pick_reason"] = f"경량 실험 선택: 후보 {pick['poi_id']}; 요인 코드 {pick['pick_factor']}"
        picks.append(pick)
    output = {"picks": picks, "review_lookup_requests": []}
    errors = validate_output(snapshot, output)
    if errors:
        result["reasons"] = errors
    else:
        result.update(status="proposed", output=output)
    return result


def teacher_examples(snapshot: dict, teacher_output: dict) -> list[TrainingExample]:
    errors = validate_output(snapshot, teacher_output)
    if errors:
        raise ValueError("Invalid teacher bundle: " + ", ".join(errors))
    group = snapshot["aid"]
    blocked = bool(preflight(snapshot) or teacher_output.get("review_lookup_requests"))
    prepared = PreparedState(snapshot)
    examples = [TrainingExample(route_question(snapshot, prepared), "DEFER" if blocked else "PROCEED", group, "route")]
    if blocked:
        return examples
    prefix = []
    for original in sorted(teacher_output["picks"], key=lambda p: p["order"]):
        pick = {"order": original["order"]}
        for kind in ("poi", "spend", "satisfaction", "factor"):
            q = event_question(snapshot, prefix if kind == "poi" else prefix + [pick.copy()], pick["order"], kind, prepared)
            if kind == "poi":
                label = original["poi_id"]
                pick["poi_id"] = label
            elif kind == "spend":
                amount = original["actual_spent"]
                nearest = min(SPEND_LEVELS, key=lambda x: abs(x - amount))
                # Reject out-of-support amounts rather than hide quantization damage.
                label = str(nearest) if abs(nearest - amount) <= max(500, amount * .05) else "DEFER"
                pick["actual_spent"] = nearest
            elif kind == "satisfaction":
                nearest = min(SAT_LEVELS, key=lambda x: abs(x - original["actual_satisfaction"]))
                actual = original["actual_satisfaction"]
                # Do not move labels across the existing low-satisfaction rule.
                label = f"{nearest:.2f}" if (nearest < .3) == (actual < .3) else "DEFER"
                pick["actual_satisfaction"] = nearest
            else:
                label = original.get("pick_factor") or "DEFER"
                if label not in FACTORS:
                    label = "DEFER"
                pick["pick_factor"] = label
            examples.append(TrainingExample(q, label, group, kind))
            if label == "DEFER":
                return examples
        pick["policy_spend"] = {}
        pick["pick_reason"] = f"경량 실험 선택: 후보 {pick['poi_id']}; 요인 코드 {pick['pick_factor']}"
        prefix.append(pick)
    return examples
