"""Distancing control changes only the restriction facts, not disease history."""
from datetime import date
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts" / "sim"))

from environments import build_environment  # noqa: E402
from environments.covid_2021 import disease_facts  # noqa: E402


def test_same_day_control_keeps_infection_observations(monkeypatch):
    monkeypatch.setenv("EXP_CASE_TREND", "1")
    day = date(2020, 11, 24)
    restricted = build_environment("covid_2021", day)
    control = build_environment("covid_no_distancing", day)
    shared = disease_facts(day)
    assert shared
    assert restricted["facts"][:len(shared)] == shared
    assert control["facts"][:len(shared)] == shared
    assert any("매장 이용 불가" in fact for fact in restricted["facts"])
    assert all("매장 이용 불가" not in fact for fact in control["facts"])
    assert "추가 거리두기 제한 없는 비교 조건" in control["headline"]
    for text in [control["headline"], *control["facts"]]:
        assert all(target not in text for target in ("14.1", "4.2%", "더 소비", "덜 소비"))


def test_counterfactual_environment_requires_a_known_regime():
    assert build_environment("covid_no_distancing", date(2019, 1, 1)) == {}
