"""P013 지원금 사용 기전 지표(U1~U4)의 계산이 정의대로인지 — 손으로 셀 수 있는 작은 원장으로."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "report"))
import score_p013_usage as U  # noqa: E402

CONTRACT = json.loads((ROOT / "data/experiments/P013_indicator_contract.json").read_text(encoding="utf-8"))
IND = {i["id"]: i for i in CONTRACT["indicators"]}
POST = ["2020-05-%02d" % d for d in range(11, 25)]          # 지급 뒤 14일 = 온전한 2주
A, B = "AGT_11110515_F_70대이상_002", "AGT_11680640_M_30대_001"


def _ident(n):
    return np.eye(n)                                         # 재표집 대신 각 사람 하나씩 — 구간 계산만 돌린다


def test_u1_cumulative_share_of_allocated_grant():
    pol = {x: {d: {"grant_spent_today": 0} for d in POST} for x in (A, B)}
    pol[A]["2020-05-12"]["grant_spent_today"] = 56000       # 1주차
    pol[B]["2020-05-19"]["grant_spent_today"] = 112000      # 2주차
    grant = np.array([280000.0, 280000.0])
    rows = U.u1(pol, [A, B], POST, grant, np.ones((3, 2)), IND["U1"])
    assert [r["weeks"] for r in rows] == [1, 2]
    assert abs(rows[0]["sim"] - 10.0) < 1e-9                 # 56,000 / 560,000
    assert abs(rows[1]["sim"] - 30.0) < 1e-9                 # 168,000 / 560,000
    assert rows[0]["truth"] == 13.5 and rows[1]["truth"] == 38.7


def test_u2_similarity_ceiling_and_unmapped_go_to_other():
    sec = {x: {d: {"funded_by_sub": {}} for d in POST} for x in (A, B)}
    sec[A]["2020-05-12"]["funded_by_sub"] = {"슈퍼마켓": 300, "한식": 200}
    sec[B]["2020-05-20"]["funded_by_sub"] = {"한식": 400, "반려동물": 100}   # 반려동물 → 기타
    r = U.u2(sec, [A, B], POST, np.ones((3, 2)), IND["U2"])[0]
    assert r["share_sim"]["대중음식점"] == 60.0 and r["share_sim"]["마트/식료품"] == 30.0 and r["share_sim"]["기타"] == 10.0
    t = IND["U2"]["truth_weekly_amounts_억원"]
    tot = sum(v[0] + v[1] for v in t.values())
    gap = sum(t[c][0] + t[c][1] for c in ("안경", "자동차정비/용품", "서점")) / tot
    assert abs(r["ceiling"] - 100 * (1 - gap)) < 1e-9       # 그래프에 없는 업종 몫만큼 상한이 낮다
    assert r["status"] == "일치"                             # 실측 2주 상위 2개 = 대중음식점·마트/식료품


def test_u4_same_dong_and_gu_from_agent_id():
    on = {A: {"profile": {"age": 72}, "plans": [{"day": "2020-05-12", "items": [
        {"dong_code": "11110515", "actual_spent": 1000, "spent_from_policy": '{"P013": 1000}'},   # 같은 동
        {"dong_code": "11110570", "actual_spent": 3000, "spent_from_policy": '{"P013": 3000}'},   # 같은 구, 다른 동
        {"dong_code": "11680640", "actual_spent": 6000, "spent_from_policy": "{}"}]}]}}           # 지원금 아님
    off = {A: {"plans": [{"day": "2020-05-12", "items": [
        {"dong_code": "11110515", "actual_spent": 2000}, {"dong_code": "11680640", "actual_spent": 8000}]}]}}
    g = U.locality_vectors(on, [A], POST, "P013", True)
    o = U.locality_vectors(off, [A], POST, "P013", False)
    assert g.tolist() == [[1000, 4000, 4000]]
    assert o.tolist() == [[2000, 2000, 10000]]


def test_home_dong_refuses_unreadable_id():
    try:
        U.home_dong("AGT_bad_id")
    except ValueError:
        return
    raise AssertionError("읽을 수 없는 id 는 멈춰야 한다")


def test_spearman_handles_ties():
    assert abs(U.spearman(np.array([1, 2, 2, 3]), np.array([10, 20, 20, 30])) - 1.0) < 1e-12
    assert abs(U.spearman(np.array([3, 2, 1]), np.array([1, 2, 3])) + 1.0) < 1e-12
