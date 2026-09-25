"""실측과 정의가 다른 등록 결과를 크기 적중으로 출력하지 않는다."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
path = ROOT / "scripts/report/build_results_overview.py"
spec = importlib.util.spec_from_file_location("build_results_overview", path)
overview = importlib.util.module_from_spec(spec)
spec.loader.exec_module(overview)


def test_정의가_다른_지표의_크기차와_동일눈금_그래프를_막는다():
    indicators = [{"id": "X", "expect": "+", "desc": "대상 지출 (실측 +7.0%)",
                   "empirical_audit": {"comparison": "different_estimand"}}]
    rows = overview.indicator_rows({"X": {"pct": 6.0, "hit": True}}, indicators)
    assert rows[0]["comparable"] is False
    chart = "\n".join(overview.draw(rows))
    table = "\n".join(overview.table(rows))
    assert "직접 같은 눈금으로 비교할 수" in chart
    assert "차이 +" not in chart
    assert "추정량 불일치·내부판정" in table
    assert rows[0]['hit'] is None and rows[0]['internal_hit'] is True
    assert "-1.0%p" not in table


def test_명시적으로_동일추정량인_지표만_크기차를_표시한다():
    indicators = [{"id": "X", "expect": "+", "desc": "대상 지출 (실측 +7.0%)",
                   "empirical_audit": {
                       "comparison": "matched_estimand", "source": "verified-source",
                       "reported_estimand": "same effect", "simulation_estimand": "same effect",
                       "reported_window": "same dates", "simulation_window": "same dates",
                       "reported_population": "same cohort", "simulation_population": "same cohort",
                       "reported_denominator": "same spend", "simulation_denominator": "same spend",
                       "reported_value": 7.0, "reported_unit": "%", "simulation_unit": "%",
                   }}]
    rows = overview.indicator_rows({"X": {"pct": 6.0, "hit": True}}, indicators)
    assert rows[0]["comparable"] is True
    assert "-1.0%p" in "\n".join(overview.table(rows))


def test_옛_한_필드_직접비교_표시는_크기_오차를_허용하지_않는다():
    indicators = [{"id": "X", "expect": "+", "desc": "대상 지출 (실측 +7.0%)",
                   "empirical_audit": {"comparison": "directly_comparable"}}]
    rows = overview.indicator_rows({"X": {"pct": 6.0, "hit": True}}, indicators)
    assert rows[0]["comparable"] is False
    assert "-1.0%p" not in "\n".join(overview.table(rows))


def test_비율이_아닌_원장_평균도_값을_숨기지_않는다():
    indicators = [{"id": "X", "expect": "+", "desc": "한계소비성향 (실측 0.21)",
                   "empirical_audit": {"comparison": "definition_pending"}}]
    rows = overview.indicator_rows({"X": {"mean": 0.216, "ci": [0.18, 0.24],
                                           "hit": True}}, indicators)
    table = "\n".join(overview.table(rows))
    assert "평균 0.216 (단위 확인)" in table
    assert "[+0.18, +0.24]" in table
    assert "원문 방향 미감사·내부판정" in table


def test_원문에_없는_지역상품권_기전은_실측_적중이_아니다():
    indicators = [{"id": "LV-2", "expect": "+", "desc": "거주 동 소비 비중 증가",
                   "empirical_audit": {"comparison": "not_observed"}}]
    rows = overview.indicator_rows({"LV-2": {"pct": 2.5, "hit": True}}, indicators)
    assert rows[0]['hit'] is None
    assert rows[0]['internal_hit'] is True
    assert "원문 미관측·내부가설" in "\n".join(overview.table(rows))


def test_방향_대응을_별도_감사하면_내부_부호를_외부에_쓸_수_있다():
    audit = {"sign_comparison": "matched_direction", "reported_direction": "+"}
    audit.update({f: 'reviewed' for f in (
        'source', 'reported_estimand', 'simulation_estimand',
        'reported_window', 'simulation_window', 'reported_population',
        'simulation_population', 'reported_denominator', 'simulation_denominator')})
    indicators = [{"id": "X", "expect": "+", "desc": "업종 매출 증가",
                   "empirical_audit": audit}]
    rows = overview.indicator_rows({"X": {"pct": 2.5, "hit": True}}, indicators)
    assert rows[0]['hit'] is True
    assert rows[0]['comparable'] is False  # sign alignment is not magnitude alignment


def test_무효_과거값은_부호_일치로_세지_않는다():
    indicators = [{"id": "X", "expect": "+", "desc": "대상 업종 지출 (실측 +11.1%p)",
                   "empirical_audit": {"comparison": "different_estimand"}}]
    rows = overview.indicator_rows(
        {"X": {"pct": 13.1, "ci": [1, 2], "note": "적격 업종 판정 무효"}}, indicators)
    table = "\n".join(overview.table(rows))
    assert rows[0]["sim"] is None
    assert rows[0]["hit"] is None
    assert "무효 (과거값 +13.1%)" in table
    assert "[+1, +2]" not in table
    assert "| 일치 |" not in table
