"""PDF expansions need real numerical truth and intact ledger provenance."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.report.pdf_benchmark_catalog import apply_catalogs, policy_html


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path):
    source = tmp_path / 'original.pdf'
    source.write_bytes(b'original publication')
    ledger = tmp_path / 'sector.ledger.jsonl'
    ledger.write_bytes(b'ledger bytes')
    numeric = _write(tmp_path / 'numeric.json', {'frozen': True})
    entries = []
    for key, label, value in [('NEW-A', '추가 업종', 8.0),
                              ('NEW-B', '분모가 없는 업종', -4.0),
                              ('NEW-C', '상품자료 필요', 6.0)]:
        entries.append({'id': key, 'benchmark_kind': 'sector_change', 'label': label,
                        'independent_outcome_family': key,
                        'empirical': {'value': value, 'unit': '%', 'denominator': '전년 매출',
                                      'population': '서울 카드 가맹점', 'period': '2019~2020',
                                      'method': '전년 비교', 'pdf_page': 4, 'printed_page': 3,
                                      'locator': '표 7', 'approximate': False},
                        'simulation': {'feasibility': 'needs_mapping',
                                       'missing_requirements': ['상품별 구매 원장 필요']},
                        'direct_gap_allowed': False})
    # Text-only truth must be excluded even from the catalog's HTML list.
    entries.append({'id': 'TEXT-ONLY', 'label': '방향만 있음',
                    'empirical': {'value': None, 'unit': '%'},
                    'direct_gap_allowed': False})
    catalog = _write(tmp_path / 'catalog.json', {
        'schema': 'pdf_benchmark_catalog_v1', 'policy': 'P013',
        'timing': 'post-result exploratory',
        'source': {'path': str(source), 'sha256': _sha(source), 'pages': 8},
        'entries': entries})
    simulation = _write(tmp_path / 'simulation.json', {
        'schema': 'pdf_benchmark_simulation_v1', 'policy': 'EMERGENCY_2020',
        'catalog_path': str(catalog), 'catalog_sha256': _sha(catalog),
        'numeric_path': str(numeric), 'numeric_sha256': _sha(numeric),
        'source_evidence': [{'path': str(ledger), 'sha256': _sha(ledger)}],
        'rows': [{'id': 'NEW-A', 'simulation': 3.25, 'simulation_unit': '%',
                  'formula': '(ON/OFF-1)*100',
                  'raw_components': {'on_won': 413, 'off_won': 400},
                  'sample_citizens': 40, 'scope_note': '같은 3일 ON/OFF 대리값',
                  'quality_notes': '전년 비교와 정의가 다름', 'direct_gap_allowed': False},
                 {'id': 'NEW-B', 'simulation': None, 'simulation_unit': '%',
                  'raw_components': {'on_won': 0, 'off_won': 0},
                  'sample_citizens': 40, 'quality_notes': 'OFF 분모 0',
                  'direct_gap_allowed': False}]})
    frozen = {'indicator_count': 14, 'simulated_count': 14, 'direct_gap_count': 0,
              'rows': [{'id': 'EM-2', 'truth': 6.2, 'simulation': 1.4}],
              'score_files': [{'path': str(numeric), 'sha256': _sha(numeric)}]}
    return frozen, catalog, simulation, source, ledger


def test_catalog_sidecar_preserves_scores_and_only_pairs_supported_numeric_values(tmp_path):
    frozen, catalog, simulation, _, _ = _fixture(tmp_path)
    original = deepcopy(frozen)
    result = apply_catalogs(frozen, [catalog], [simulation], tmp_path)
    assert frozen == original
    assert result['rows'] == original['rows']
    assert result['indicator_count'] == 14 and result['direct_gap_count'] == 0
    assert result['pdf_benchmark_counts'] == {'EMERGENCY_2020': {
        'empirical': 3, 'numeric_pairs': 1, 'raw_only': 1, 'needs_data': 1,
        'additional_outcome_families': 3, 'model_reference_records': 0}}
    rendered = policy_html(result, 'EMERGENCY_2020')
    assert '실측 8%' in rendered and '시뮬 3.25%' in rendered
    assert 'ON 원화: 0원' in rendered and 'OFF 원화: 0원' in rendered
    assert 'OFF 분모 0' in rendered
    assert '상품별 구매 원장 필요' in rendered
    assert 'TEXT-ONLY' not in rendered
    assert '실측 —' not in rendered and '시뮬 —' not in rendered
    assert '원문 숫자' not in rendered or '결과 후 탐색' in rendered
    assert '서울 성별·연령·행정동·소득' in rendered
    assert '실측 분포를 새로 맞춘 표본의 결과가 아닙니다' in rendered
    assert 'href="file:' in rendered and 'PDF 4쪽' in rendered


@pytest.mark.parametrize('damaged', ['source', 'catalog', 'ledger', 'numeric'])
def test_expansion_rejects_source_catalog_ledger_or_score_sha_change(tmp_path, damaged):
    frozen, catalog, simulation, source, ledger = _fixture(tmp_path)
    target = {'source': source, 'catalog': catalog, 'ledger': ledger,
              'numeric': tmp_path / 'numeric.json'}[damaged]
    if damaged == 'catalog':
        payload = json.loads(catalog.read_text(encoding='utf-8'))
        payload['entries'][0]['empirical']['value'] = 9.0
        _write(catalog, payload)
    else:
        target.write_bytes(target.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='SHA256 mismatch|does not bind'):
        apply_catalogs(frozen, [catalog], [simulation], tmp_path)


def test_expansion_refuses_unknown_ids_and_direct_gap_approval(tmp_path):
    frozen, catalog, simulation, _, _ = _fixture(tmp_path)
    payload = json.loads(simulation.read_text(encoding='utf-8'))
    payload['rows'][0]['id'] = 'UNREGISTERED-SIDECAR-ID'
    _write(simulation, payload)
    with pytest.raises(ValueError, match='unknown or duplicate'):
        apply_catalogs(frozen, [catalog], [simulation], tmp_path)
    payload['rows'][0]['id'] = 'NEW-A'
    payload['rows'][0]['direct_gap_allowed'] = True
    _write(simulation, payload)
    with pytest.raises(ValueError, match='authorize direct gaps'):
        apply_catalogs(frozen, [catalog], [simulation], tmp_path)


def test_catalog_only_lists_truth_and_requirements_without_simulation_cards(tmp_path):
    frozen, catalog, _, _, _ = _fixture(tmp_path)
    result = apply_catalogs(frozen, [catalog], [], tmp_path)
    rendered = policy_html(result, 'EMERGENCY_2020')
    assert '추가 원문 실측 3개' in rendered
    assert '<div class="row' not in rendered
    assert '시뮬 —' not in rendered
    assert result['pdf_benchmark_counts']['EMERGENCY_2020']['needs_data'] == 3


def test_nonfinite_simulation_does_not_silently_create_a_numeric_card(tmp_path):
    frozen, catalog, simulation, _, _ = _fixture(tmp_path)
    payload = json.loads(simulation.read_text(encoding='utf-8'))
    payload['rows'][0]['simulation'] = float('nan')
    _write(simulation, payload)
    with pytest.raises(ValueError, match='finite or null'):
        apply_catalogs(frozen, [catalog], [simulation], tmp_path)


def test_source_model_estimate_is_not_counted_as_empirical_validation(tmp_path):
    frozen, catalog, _, _, _ = _fixture(tmp_path)
    payload = json.loads(catalog.read_text(encoding='utf-8'))
    payload['entries'][0]['benchmark_kind'] = 'source_model_estimate'
    _write(catalog, payload)
    result = apply_catalogs(frozen, [catalog], [], tmp_path)
    counts = result['pdf_benchmark_counts']['EMERGENCY_2020']
    assert counts['empirical'] == 2 and counts['model_reference_records'] == 1
    assert counts['additional_outcome_families'] == 2
    rendered = policy_html(result, 'EMERGENCY_2020')
    assert '원문 실측 기록 2개' in rendered
    assert '원문 모형 추정 참고값 1개 · 실측 검증 대상에서 제외' in rendered


def test_alternative_proxy_does_not_replace_strict_null_or_inflate_pair_count(tmp_path):
    frozen, catalog, simulation, _, _ = _fixture(tmp_path)
    payload = json.loads(simulation.read_text(encoding='utf-8'))
    payload['rows'][1]['alternative_proxies'] = [{
        'simulation': 6.5, 'simulation_unit': '%',
        'raw_components': {'on_won': 1065, 'off_won': 1000},
        'scope_note': '더 넓은 업종 범위라 원문과 다름'}]
    _write(simulation, payload)
    result = apply_catalogs(frozen, [catalog], [simulation], tmp_path)
    assert result['pdf_benchmark_counts']['EMERGENCY_2020']['numeric_pairs'] == 1
    original = result['pdf_benchmark_catalogs'][0]['entries'][1]['proxy']
    assert original['simulation'] is None
    markup = policy_html(result, 'EMERGENCY_2020')
    assert '시뮬 6.5%' in markup and '더 넓은 업종 범위라 원문과 다름' in markup
    assert '엄격 지표의 미산출을 이 숫자로 대체하지 않으며' in markup
