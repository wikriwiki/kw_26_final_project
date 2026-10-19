"""Freeze an experimental P010 input with explicit generic eligibility rules.

The benefit, dates, and citizen-facing policy description remain from P010.json.
Only the pre-existing merchant approximation is represented as data for v53.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))
from coupon_eligibility import _NAME_EXCLUDE, _NAME_VICE, _SUB_EXCLUDE, is_coupon_eligible
from eligibility import Rules, validated_restricted_rules

source = ROOT / "data/neo4j_load/policies/P010.json"
target = ROOT / "data/experiments/P010_v53_policy_20260927.json"
policy = json.loads(source.read_text(encoding="utf-8"))
assert policy["id"] == "P010" and policy["type"] == "grant"
assert policy["effective_from"] == "2025-07-21"
policy["eligible_marker"] = "[쿠폰]"
policy["eligibility"] = {
    "mode": "exclude",
    "exclude": {
        "name_regex": f"(?:{_NAME_EXCLUDE.pattern})|(?:{_NAME_VICE.pattern})",
        "subs_always": {"other": sorted(_SUB_EXCLUDE)},
    },
}
policy.setdefault("_notes", {})["v53_input_copy"] = (
    "2026-09-27: Explicit policy-fact merchant approximation for generic v53. "
    "Same known brand/subcategory exclusions as coupon_eligibility.py; annual merchant "
    "turnover, franchise ownership, and full card eligibility are unobserved. "
    "Not an empirical target or behavioral instruction."
)
rules = validated_restricted_rules(policy["eligibility"])
cases = [
    ("김밥천국 역삼점", "분식"), ("이마트 성수점", "종합소매"),
    ("이마트24 R성수점", "편의점"), ("홈플러스 익스프레스", "슈퍼마켓"),
    ("스타벅스 강남점", "카페"), ("GS25 관악점", "편의점"),
    ("금은방", "시계·귀금속"), ("서울부동산", "부동산"),
    ("황금성 단란주점", "일반주점"), ("복권명당", "기타상품"),
]
for name, sub in cases:
    old = is_coupon_eligible(name, sub)[0]
    new = rules.eligible(name, sub, upjong_l3="G00000")[0]
    assert old == new, (name, sub, old, new)
blob = json.dumps(policy, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
if target.exists():
    assert target.read_bytes() == blob, "frozen P010 policy changed"
else:
    target.write_bytes(blob)
print(f"{target} sha256={hashlib.sha256(blob).hexdigest()} source_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}")
