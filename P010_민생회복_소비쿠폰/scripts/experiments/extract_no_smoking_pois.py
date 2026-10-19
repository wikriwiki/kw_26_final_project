"""Conservatively classify an offline POI export without changing the graph.

The registry is an inference from recorded names/categories, not verification of
registered business type, indoor layout, current operation, or 2017 existence.
Names alone never override conflicting categories or codes. Ambiguous records
remain in the audit and are absent from the runtime target registry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[2]
MAPPING = ROOT / "data/neo4j_load/mapping/mapping_upjong_to_sub.json"
DISTRICTS = {"11350", "11650", "11710"}
RULE_VERSION = "name_category_conservative_v1"
NOTE = (
    "Conservative inference from source POI names and categories; not official "
    "business-registration, indoor-layout, current-operation, or historical-2017 verification. "
    "Pending records are excluded from the runtime registry; coverage is intentionally incomplete."
)
BILLIARD = re.compile(r"당구|빌리어드|빌리아드|빌리어즈|캐롬|포켓볼|billiard|carom")
NONFACILITY = re.compile(r"용품|판매|피팅|시공|설치|제조|유통|도매|쇼핑|장비|시스템")
OUTDOOR_OR_OTHER_GOLF = re.compile(r"실외|야외|인도어|아웃도어|outdoor|파크(?:스크린)?골프")
OTHER_SPORT = re.compile(r"복싱|클라이밍|수영|테니스|탁구|음악|골프|건설|임대|점핑")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_name(value: str) -> str:
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", value).casefold())


def classify(row: dict) -> dict:
    """Return an auditable decision; a category alone is never sufficient."""
    props = row.get("props") or {}
    name = row.get("name")
    code = row.get("upjong_l3") or props.get("upjong_l3")
    code = str(code).strip().upper() if code else None
    result = {
        "poi_id": row.get("poi_id"), "district_code": str(row.get("district_code", "")),
        "raw_name": name, "raw_category": row.get("category"),
        "raw_parent": row.get("parent"), "raw_upjong_l3": code,
        "status": "pending", "facility_type": None, "rule_id": None,
        "flags": {"code_missing": code is None, "category_match": False,
                  "name_rule_match": False, "inference_from_name": True,
                  "registration_verified": False, "current_operation_verified": False},
    }

    def decide(reason: str, status: str = "pending", facility: str | None = None):
        result.update(rule_id=reason, status=status, facility_type=facility)
        return result

    if not isinstance(result["poi_id"], str) or not result["poi_id"].strip():
        return decide("missing_poi_identity")
    if result["district_code"] not in DISTRICTS or props.get("type") != "commerce":
        return decide("outside_district_or_commerce_scope", "excluded")
    if props.get("id") != result["poi_id"] or (props.get("name") is not None and props["name"] != name):
        return decide("conflicting_source_identity_or_name")
    if row.get("upjong_l3") and props.get("upjong_l3") and row["upjong_l3"] != props["upjong_l3"]:
        return decide("conflicting_source_codes")
    if row.get("parent") != "여가" or row.get("category") not in {"당구", "스포츠"}:
        return decide("outside_supported_categories", "excluded")
    result["flags"]["category_match"] = True
    if not isinstance(name, str) or not name.strip():
        return decide("missing_source_name")
    name = normalized_name(name)
    if NONFACILITY.search(name):
        return decide("nonfacility_name_marker_needs_review")
    if OUTDOOR_OR_OTHER_GOLF.search(name):
        return decide("outdoor_or_other_golf_marker_needs_review")

    if row["category"] == "당구":
        if code and code != "R10310":
            return decide("category_code_conflict")
        if OTHER_SPORT.search(name):
            return decide("billiard_category_name_conflict")
        if not BILLIARD.search(name):
            return decide("billiard_name_not_explicit")
        result["flags"]["name_rule_match"] = True
        return decide("billiard_category_and_explicit_name", "included", "billiard")

    if code and code != "R10311":
        return decide("sports_code_not_golf_practice")
    if BILLIARD.search(name):
        return decide("sports_category_billiard_name_conflict")
    if "골프" not in name and "golf" not in name:
        return decide("golf_name_not_explicit")
    # An explicit modality must coexist with golf; brands alone do not qualify.
    if "스크린" in name or "screengolf" in name or "golfscreen" in name:
        result["flags"]["name_rule_match"] = True
        return decide("sports_category_and_explicit_screen_golf_name", "included", "screen_golf")
    if "실내" in name:
        result["flags"]["name_rule_match"] = True
        return decide("sports_category_and_explicit_indoor_golf_name", "included", "indoor_golf")
    return decide("golf_indoor_or_screen_marker_missing")


def extract(rows: list[dict], source_sha256: str, audit_name: str) -> tuple[dict, dict]:
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Input must be an array of candidate POI objects")
    ids = Counter(str(row.get("poi_id")) for row in rows)
    decisions, targets = [], []
    for row in rows:
        decision = classify(row)
        if ids[str(row.get("poi_id"))] > 1:
            decision.update(status="pending", facility_type=None, rule_id="duplicate_poi_identity")
        decisions.append(decision)
        if decision["status"] == "included":
            targets.append({
                "poi_id": decision["poi_id"], "district_code": decision["district_code"],
                "facility_type": decision["facility_type"],
                "classification_source": (
                    f"Recorded name/category inference ({RULE_VERSION}, {decision['rule_id']}); "
                    f"source_sha256={source_sha256}; audit={audit_name}; "
                    "not official registration/current-operation verification"
                ),
            })
    targets.sort(key=lambda item: item["poi_id"])
    decisions.sort(key=lambda item: (str(item["poi_id"]), str(item["raw_name"])))
    counts = Counter(f"{item['district_code']}:{item['facility_type']}" for item in targets)
    registry = {"schema_version": "1.0", "classification_note": NOTE, "pois": targets}
    audit = {
        "schema_version": "1.0", "rule_version": RULE_VERSION, "classification_note": NOTE,
        "source_sha256": source_sha256,
        "rules": {
            "name_normalization": "Unicode NFKC, casefold, remove punctuation/whitespace",
            "billiard_name_regex": BILLIARD.pattern,
            "nonfacility_review_regex": NONFACILITY.pattern,
            "outdoor_or_other_golf_review_regex": OUTDOOR_OR_OTHER_GOLF.pattern,
            "billiard_conflicting_name_regex": OTHER_SPORT.pattern,
            "golf_rule": "sports category AND explicit golf token AND screen/indoor token",
            "category_only_allowed": False,
            "brand_only_allowed": False,
            "conflicting_or_duplicate_records_allowed": False,
        },
        "limitations": [
            "The source export contains category/name conflicts; a category alone cannot certify a facility.",
            "No missing source industry code is reconstructed or represented as observed.",
            "A name-based indoor inference is not direct inspection or registration verification.",
            "Excluded contains all pending records as well as definite out-of-scope records.",
            "This registry is a conservative subset, not a census or a representative sample of all facilities.",
        ],
        "summary": {"input_rows": len(rows), "included": len(targets),
                    "excluded_from_registry": len(rows) - len(targets),
                    "status_counts": dict(sorted(Counter(d["status"] for d in decisions).items())),
                    "rule_counts": dict(sorted(Counter(d["rule_id"] for d in decisions).items())),
                    "district_facility_counts": dict(sorted(counts.items()))},
        "included": [d for d in decisions if d["status"] == "included"],
        "excluded": [d for d in decisions if d["status"] != "included"],
    }
    return registry, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="Runtime registry JSON path")
    parser.add_argument("--audit", type=Path, help="Default: <out stem>.audit.json")
    parser.add_argument("--snapshot-sha256", help="Optional hash of the actual restored dump, not a fabricated marker")
    args = parser.parse_args()
    audit_path = args.audit or args.out.with_name(args.out.stem + ".audit.json")
    if args.snapshot_sha256 and not re.fullmatch(r"[a-fA-F0-9]{64}", args.snapshot_sha256):
        parser.error("--snapshot-sha256 must contain 64 hexadecimal characters")
    if args.out.resolve() == audit_path.resolve() or args.out.exists() or audit_path.exists():
        parser.error("Registry and audit must be distinct new files; refusing overwrite")
    rows = json.loads(args.input.read_text(encoding="utf-8-sig"))
    registry, audit = extract(rows, digest(args.input), audit_path.name)
    audit.update(source_file=str(args.input.resolve()), source_snapshot_sha256=args.snapshot_sha256,
                 mapping_sha256=digest(MAPPING), classifier_sha256=digest(Path(__file__)),
                 source_upjong_codes_observed=sum(bool(d["raw_upjong_l3"]) for d in audit["included"] + audit["excluded"]))
    for path, content in ((args.out, registry), (audit_path, audit)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
