"""Bounded public policy arguments: validate structure, never hidden reasoning.

Exact citations link public claims to supplied text; they do not prove that a
claim is entailed, relevant, sincere, or logically sound. The caller separately
verifies packet seals, identity and time limits. No model or network calls occur.
"""
from __future__ import annotations

from collections import Counter
import json
import math

ARGUMENT_CONTRACT_VERSION = 2
STANCES = ("support", "oppose", "mixed", "neutral", "uncertain", "unknown")
BASES = ("recorded_fact", "inference", "value_judgment")
DIRECTIONS = ("for", "against", "uncertain")
LIMITS = {"personal_situation": 3, "considerations": 4, "conditions": 2,
          "uncertainties": 3, "evidence": 3, "reasons": 6,
          "answer_chars": 4000, "stance_quote_chars": 1000, "claim_chars": 800,
          "personal_relevance_chars": 800, "weighing_chars": 1600,
          "condition_chars": 600, "possible_change_chars": 600,
          "uncertainty_chars": 600, "quote_chars": 1000, "evidence_id_chars": 256}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _object(value, keys, name):
    _require(isinstance(value, dict), f"{name} must be an object")
    _require(set(value) == set(keys), f"{name} must contain exactly: {', '.join(keys)}")


def _text(value, limit, name, *, nonempty=False):
    _require(isinstance(value, str) and len(value) <= limit, f"{name} must be text of at most {limit} characters")
    _require(not nonempty or bool(value.strip()), f"{name} must not be blank")


def _array(value, limit, name):
    _require(isinstance(value, list) and len(value) <= limit, f"{name} must be a list with at most {limit} items")
    try:
        serialized = [json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain finite JSON values") from exc
    _require(len(set(serialized)) == len(serialized), f"{name} contains duplicate items")


def _citations(value, evidence, name, *, limit):
    _array(value, limit, name)
    for item in value:
        _object(item, ("evidence_id", "quote"), f"{name} citation")
        _text(item["evidence_id"], LIMITS["evidence_id_chars"], f"{name}.evidence_id", nonempty=True)
        _text(item["quote"], LIMITS["quote_chars"], f"{name}.quote", nonempty=True)
        _require(item["evidence_id"] in evidence, f"{name} cites an unknown evidence_id")
        _require(item["quote"] in evidence[item["evidence_id"]], f"{name} quote must exactly match the cited evidence text")


def validate_argument(response, packet):
    """Return reproducible public-structure flags; schema/citation errors raise.

    Empty optional content is allowed. No minimum word/sentence count, demand for
    both policy directions, or forced change conditions is used. The report is
    not a semantic/logic grade and never changes the respondent's stance.
    """
    _object(response, ("stance", "answer", "stance_quote", "confidence", "reasons", "argument"), "response")
    _require(response["stance"] in STANCES, "response.stance is outside the contract")
    _text(response["answer"], LIMITS["answer_chars"], "answer", nonempty=True)
    _text(response["stance_quote"], LIMITS["stance_quote_chars"], "stance_quote", nonempty=True)
    _require(response["stance_quote"] in response["answer"], "stance_quote must occur exactly in answer")
    certainty = response["confidence"]
    _require(type(certainty) in (int, float) and math.isfinite(certainty) and 0 <= certainty <= 1, "confidence must be a finite number in [0,1]")
    _require(isinstance(packet, dict) and isinstance(packet.get("evidence_items"), list), "packet evidence_items are required")
    evidence = {}
    for item in packet["evidence_items"]:
        _require(isinstance(item, dict) and isinstance(item.get("evidence_id"), str)
                 and item["evidence_id"] and isinstance(item.get("text"), str), "Invalid packet evidence item")
        _require(item["evidence_id"] not in evidence, "Duplicate packet evidence identity")
        evidence[item["evidence_id"]] = item["text"]
    _citations(response["reasons"], evidence, "reasons", limit=LIMITS["reasons"])
    argument = response["argument"]
    _object(argument, ("personal_situation", "considerations", "weighing", "conditions", "uncertainties"), "argument")
    for name in ("personal_situation", "considerations", "conditions", "uncertainties"):
        _array(argument[name], LIMITS[name], name)
    flags = set()
    citations = list(response["reasons"])
    if not argument["personal_situation"]:
        flags.add("personal_situation_not_stated")
    for item in argument["personal_situation"]:
        _object(item, ("claim", "evidence"), "personal_situation item")
        _text(item["claim"], LIMITS["claim_chars"], "personal_situation.claim")
        _citations(item["evidence"], evidence, "personal_situation.evidence", limit=LIMITS["evidence"])
        citations.extend(item["evidence"])
        if not item["claim"].strip():
            flags.add("personal_situation_claim_empty")
        if not item["evidence"]:
            flags.add("personal_situation_without_citation")
    if not argument["considerations"]:
        flags.add("considerations_not_stated")
    for item in argument["considerations"]:
        _object(item, ("claim", "basis", "direction", "personal_relevance", "evidence"), "consideration")
        _text(item["claim"], LIMITS["claim_chars"], "consideration.claim")
        _text(item["personal_relevance"], LIMITS["personal_relevance_chars"], "consideration.personal_relevance")
        _require(item["basis"] in BASES, "consideration.basis is outside the contract")
        _require(item["direction"] in DIRECTIONS, "consideration.direction is outside the contract")
        _citations(item["evidence"], evidence, "consideration.evidence", limit=LIMITS["evidence"])
        citations.extend(item["evidence"])
        if not item["claim"].strip():
            flags.add("consideration_claim_empty")
        if not item["personal_relevance"].strip():
            flags.add("personal_relevance_not_stated")
        if item["basis"] == "recorded_fact" and not item["evidence"]:
            flags.add("recorded_fact_without_citation")
    _text(argument["weighing"], LIMITS["weighing_chars"], "weighing")
    if not argument["weighing"].strip():
        flags.add("weighing_not_stated")
    for item in argument["conditions"]:
        _object(item, ("condition", "possible_change"), "condition item")
        for key in ("condition", "possible_change"):
            _text(item[key], LIMITS[key + "_chars"], key)
            if not item[key].strip():
                flags.add("condition_change_link_empty")
    for text in argument["uncertainties"]:
        _text(text, LIMITS["uncertainty_chars"], "uncertainty")
        if not text.strip():
            flags.add("uncertainty_entry_empty")
    selected = packet.get("selection") or {}
    limited_input = (not evidence or bool(packet.get("missing_days")) or bool(packet.get("missing_night_days"))
                     or (isinstance(selected, dict) and bool(selected.get("omitted_items"))))
    return {"argument_contract_version": ARGUMENT_CONTRACT_VERSION,
            "structural_validity": "valid",
            "quality_status": "needs_enrichment" if flags else "sufficient_for_review",
            "quality_flags": sorted(flags),
            "input_evidence_count": len(evidence),
            "input_evidence_kind_counts": dict(sorted(Counter(str(item.get("kind", "unspecified")) for item in packet["evidence_items"]).items())),
            "limited_input_possible": limited_input,
            "quality_interpretation": "Missing public links invite review, including legitimate short/uncertain answers when input is limited. needs_enrichment is not a failure verdict, rejection, or regeneration instruction; never invent missing experience.",
            "counts": {"personal_situation": len(argument["personal_situation"]),
                       "considerations": len(argument["considerations"]),
                       "conditions": len(argument["conditions"]),
                       "uncertainties": len(argument["uncertainties"]),
                       "citations": len(citations), "distinct_evidence_ids": len({c["evidence_id"] for c in citations}),
                       "consideration_basis": {key: sum(c["basis"] == key for c in argument["considerations"]) for key in BASES},
                       "consideration_direction": {key: sum(c["direction"] == key for c in argument["considerations"]) for key in DIRECTIONS}},
            "review_required": True,
            "review_scope": ["citation entailment and factual meaning", "personal relevance", "consistency between considerations, weighing and declared stance", "whether conditions and uncertainty are honestly expressed"],
            "limitations": ["Exact source matching is not factual or semantic verification.",
                            "Missing public structure is flagged without changing the stance or inventing experience.",
                            "No length score, forced opposing argument, or calibrated logical-quality score is used."]}


def public_argument_text(response):
    """Text-only clustering view; exclude stance, basis, direction and confidence.

    Textual mentions still remain semantic content. Identical text fragments are
    included once so repeated citation placement does not multiply their weight.
    """
    argument = response["argument"]
    parts = [response["answer"]] + [r["quote"] for r in response["reasons"]]
    for item in argument["personal_situation"]:
        parts.append(item["claim"])
        parts.extend(c["quote"] for c in item["evidence"])
    for item in argument["considerations"]:
        parts.extend((item["claim"], item["personal_relevance"]))
        parts.extend(c["quote"] for c in item["evidence"])
    parts.append(argument["weighing"])
    for item in argument["conditions"]:
        parts.extend((item["condition"], item["possible_change"]))
    parts.extend(argument["uncertainties"])
    return " ".join(dict.fromkeys(text for text in parts if text.strip()))
