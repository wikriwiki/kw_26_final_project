"""categories.yaml 파싱 + L1→L2 상속 + override 검증.

`02_categories.py` 의 파일명이 숫자로 시작해 import 불가 → importlib 로 로드.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# 02_categories.py 동적 로드
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOADER_PATH = PROJECT_ROOT / "scripts" / "neo4j_load" / "02_categories.py"
spec = importlib.util.spec_from_file_location("_cat_loader_test", LOADER_PATH)
cat_loader = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(cat_loader)  # type: ignore

parse_categories_yaml = cat_loader.parse_categories_yaml
DESIRE_DEFAULTS = cat_loader.DESIRE_DEFAULTS
REAL_YAML = PROJECT_ROOT / "data" / "neo4j_load" / "categories" / "categories.yaml"


# ---------------------------------------------------------------------------
# L1 상속
# ---------------------------------------------------------------------------
def test_l2_inherits_l1_desire_params():
    raw = yaml.safe_load("""
categories:
  - name: 식사
    open: 11
    close: 22
    recovery_tau_days: 3.0
    desire_drop: 0.85
    saturation_n: 12
    sub:
      - 한식
      - 일식
""")
    rows = parse_categories_yaml(raw)
    assert len(rows) == 2
    assert all(r["parent"] == "식사" for r in rows)
    assert all(r["recovery_tau_days"] == 3.0 for r in rows)
    assert all(r["desire_drop"] == 0.85 for r in rows)
    assert all(r["saturation_n"] == 12 for r in rows)
    assert all(r["open_hour"] == 11 for r in rows)


# ---------------------------------------------------------------------------
# L2 dict override
# ---------------------------------------------------------------------------
def test_l2_dict_overrides_some_fields():
    raw = yaml.safe_load("""
categories:
  - name: 식사
    recovery_tau_days: 3.0
    desire_drop: 0.85
    saturation_n: 12
    open: 11
    close: 22
    sub:
      - 한식
      - { name: 분식, saturation_n: 18 }
      - { name: 치킨, recovery_tau_days: 5.0, desire_drop: 0.7 }
""")
    rows = parse_categories_yaml(raw)
    by_name = {r["name"]: r for r in rows}

    # 한식 — 모두 상속
    assert by_name["한식"]["saturation_n"] == 12
    assert by_name["한식"]["recovery_tau_days"] == 3.0

    # 분식 — saturation_n 만 override
    assert by_name["분식"]["saturation_n"] == 18
    assert by_name["분식"]["recovery_tau_days"] == 3.0   # 상속
    assert by_name["분식"]["desire_drop"] == 0.85         # 상속

    # 치킨 — recovery_tau_days, desire_drop 둘 다 override
    assert by_name["치킨"]["recovery_tau_days"] == 5.0
    assert by_name["치킨"]["desire_drop"] == 0.7
    assert by_name["치킨"]["saturation_n"] == 12         # 상속


# ---------------------------------------------------------------------------
# desire 파라미터 누락 시 default
# ---------------------------------------------------------------------------
def test_missing_desire_params_use_defaults():
    raw = yaml.safe_load("""
categories:
  - name: 기타
    open: 9
    close: 18
    sub:
      - 임의
""")
    rows = parse_categories_yaml(raw)
    assert rows[0]["recovery_tau_days"] == DESIRE_DEFAULTS["recovery_tau_days"]
    assert rows[0]["desire_drop"] == DESIRE_DEFAULTS["desire_drop"]
    assert rows[0]["saturation_n"] == DESIRE_DEFAULTS["saturation_n"]


# ---------------------------------------------------------------------------
# 잘못된 입력 — error
# ---------------------------------------------------------------------------
def test_l2_dict_without_name_raises():
    raw = yaml.safe_load("""
categories:
  - name: 식사
    sub:
      - { saturation_n: 5 }
""")
    with pytest.raises(ValueError, match="missing 'name'"):
        parse_categories_yaml(raw)


def test_l2_non_str_non_dict_raises():
    raw = {"categories": [{"name": "식사", "sub": [42]}]}
    with pytest.raises(ValueError, match="must be str or dict"):
        parse_categories_yaml(raw)


# ---------------------------------------------------------------------------
# 실제 yaml 검증 — 12 L1 모두 desire 파라미터가 있는지
# ---------------------------------------------------------------------------
def test_real_yaml_all_l1_have_desire_params():
    raw = yaml.safe_load(REAL_YAML.read_text(encoding="utf-8"))
    rows = parse_categories_yaml(raw)

    # 12 L1
    l1_set = {r["parent"] for r in rows}
    assert l1_set == {
        "식사", "카페", "디저트", "주점", "편의점", "마트",
        "미용", "쇼핑", "여가", "건강", "교육", "기타",
    }

    # 모든 row 에 desire 3종 필수
    for r in rows:
        assert r["recovery_tau_days"] is not None, f"{r['name']}: tau None"
        assert r["desire_drop"] is not None, f"{r['name']}: drop None"
        assert r["saturation_n"] is not None, f"{r['name']}: sat None"

    # 도메인 sanity — drop ∈ [0,1], tau > 0, sat > 0
    for r in rows:
        assert 0.0 <= r["desire_drop"] <= 1.0, f"{r['name']}: drop out of range"
        assert r["recovery_tau_days"] > 0
        assert r["saturation_n"] > 0


# ---------------------------------------------------------------------------
# L1 별 권장값 sanity — 미용·쇼핑 > 식사·카페 (회복 느림)
# ---------------------------------------------------------------------------
def test_real_yaml_tau_ordering_makes_sense():
    raw = yaml.safe_load(REAL_YAML.read_text(encoding="utf-8"))
    rows = parse_categories_yaml(raw)
    tau_by_l1: dict[str, float] = {}
    for r in rows:
        tau_by_l1[r["parent"]] = r["recovery_tau_days"]   # L1 안에선 같다고 가정

    assert tau_by_l1["식사"] < tau_by_l1["미용"], "식사 회복이 미용보다 빨라야"
    assert tau_by_l1["카페"] < tau_by_l1["쇼핑"]
    assert tau_by_l1["편의점"] < tau_by_l1["미용"]
