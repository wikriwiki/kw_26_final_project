"""기간·분모가 다른 역사적 결과를 결측이나 직접 오차로 세지 않는다."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "error_budget", ROOT / "scripts/report/error_budget.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_p012_부분월_상한비율은_미측정이_아니라_정의확인():
    rows = {r['id']: r for r in module.collect()}
    for iid in ('P012-4', 'P012-6'):
        assert rows[iid]['kind'] == '정의확인'
        assert rows[iid]['err'] is None
    assert rows['P012-6']['sim'] == 0


def test_현재_직접_비교_가능한_크기_오차는_없다():
    assert all(r['err'] is None for r in module.collect())


def test_새_지표도_정합_감사와_단위가_없으면_오차를_내지_않는다(tmp_path, monkeypatch):
    path = tmp_path / 'scoring.json'
    indicator = {'id': 'X', 'expect': '+', 'desc': '효과 (실측 +7.3%)'}
    scoring = {'EMERGENCY_2020': {
        'indicators': [indicator],
        'result_stage3_dow_2026_09_16': {'X': {'pct': 12.6}},
    }}
    monkeypatch.setattr(module, 'SCORING', path)

    def row():
        path.write_text(json.dumps(scoring, ensure_ascii=False), encoding='utf-8')
        return next(r for r in module.collect() if r['id'] == 'X')

    assert row()['err'] is None
    indicator['empirical_audit'] = {
        'comparison': 'matched_estimand', 'source': 'verified-source',
        'reported_estimand': 'same effect', 'simulation_estimand': 'same effect',
        'reported_window': 'same dates', 'simulation_window': 'same dates',
        'reported_population': 'same cohort', 'simulation_population': 'same cohort',
        'reported_denominator': 'same spend', 'simulation_denominator': 'same spend',
        'reported_value': 7.3, 'reported_unit': '%',
    }
    assert row()['err'] is None  # simulation unit still unspecified
    indicator['empirical_audit']['simulation_unit'] = '%'
    assert row()['err'] == pytest.approx(5.3)
