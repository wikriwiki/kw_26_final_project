"""Freeze the real 2021-10-01 start for a full-calendar-month P012 screen.

The production P012.json uses a 10-15 counterfactual start for its 28-day
diagnostic. This separate input restores the published October start; it does
not include any empirical outcome or policy-specific behavior instruction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "data/neo4j_load/policies/P012.json"
target = ROOT / "data/experiments/P012_v53_october_policy_20260928.json"
policy = json.loads(source.read_text(encoding="utf-8"))
assert policy["id"] == "P012" and policy["type"] == "cashback"
assert policy["effective_from"] == "2021-10-15"
policy["effective_from"] = "2021-10-01"
policy["notes"] = (
    "2021년 상생소비지원금. 2021-10-01부터 11-30까지 카드 실적이 기준월보다 "
    "3%를 넘으면 초과분의 10%를 다음 달에 돌려주며 월 최대 10만원. "
    "대형마트·백화점·대형 온라인몰 등 제외업종은 실적에 포함되지 않는다."
)
policy["_notes"] = {"v53_october_input_copy": (
    "Published 2021-10-01 start for a full-October screen. "
    "No earlier experiment outcomes, counterfactual schedule, or empirical targets."
)}
blob = json.dumps(policy, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
if target.exists():
    assert target.read_bytes() == blob, "frozen P012 input changed"
else:
    target.write_bytes(blob)
print(f"{target} sha256={hashlib.sha256(blob).hexdigest()} source_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}")
