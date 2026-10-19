"""Synthetic probes check plumbing and reviewer safeguards, not model quality."""
import copy
import json
from types import SimpleNamespace

from scripts.experiments import collect_policy_stances as collector
from scripts.experiments import probe_stance_reasoning as probe


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [0] * ((len(json.dumps(messages, ensure_ascii=False)) + 3) // 4)


def test_matched_cases_expose_situation_without_prescribing_stance():
    cases = probe.probe_cases()
    assert len(cases) == 4
    assert all(c['synthetic_fixture'] and c['expected_stance'] is None for c in cases)
    personas = [c['packet']['evidence_items'][0]['value']['persona'] for c in cases]
    assert personas[0]['smoking_status'] == personas[1]['smoking_status']
    assert personas[0]['daily_wd'] != personas[1]['daily_wd']
    assert personas[0]['nv_hobbies'] != personas[1]['nv_hobbies']
    smoking = copy.deepcopy(personas[0])
    smoking['smoking_status'] = 'non_smoker'
    assert smoking == personas[2]
    assert personas[3]['smoking_status'] == 'unknown'


def test_default_probe_is_offline_and_cannot_award_semantic_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(collector, 'collect_one', lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden')))
    folder = tmp_path / 'probe'
    assert probe.main(['--out', str(folder)]) == 0
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    review = json.loads((folder / 'review.json').read_text(encoding='utf-8'))
    assert summary['llm_calls_requested'] == 0
    assert not summary['semantic_quality_passed'] and not summary['empirical_evaluation_eligible']
    assert all(r['review_status'] == 'unreviewed' and set(r['scores'].values()) == {None} for r in review['cases'])
    assert 'context_sensitivity' in review['rubric']


def test_unelaborated_answer_is_preserved_and_flagged_without_auto_rewrite(tmp_path):
    response = {'stance': 'support', 'answer': '찬성합니다.', 'stance_quote': '찬성합니다',
                'confidence': 0.7, 'reasons': [],
                'argument': {'personal_situation': [], 'considerations': [], 'weighing': '',
                             'conditions': [], 'uncertainties': []}}
    calls = []
    def call(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model='LG-fixture', choices=[SimpleNamespace(finish_reason='stop',
               message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)))])
    record = collector.collect_one(probe.probe_cases()[0]['packet'], Tokenizer(), call,
                                   tmp_path, 'LG-fixture', synthetic_fixture=True)
    assert len(calls) == 1 and calls[0]['max_tokens'] == 1600
    assert record['response_status'] == 'answered' and record['response'] == response
    assert record['argument_quality']['quality_status'] == 'needs_enrichment'
    assert record['argument_quality']['review_required']
    assert 'weighing_not_stated' in record['argument_quality']['quality_flags']
    assert record['provenance']['argument_contract_version'] == 2


def test_small_evidence_does_not_require_invented_tradeoff_or_private_facts(tmp_path):
    response = {'stance': 'uncertain', 'answer': '이용 경험을 확인할 수 없어 판단을 유보합니다.',
                'stance_quote': '판단을 유보합니다', 'confidence': 0.8, 'reasons': [],
                'argument': {'personal_situation': [], 'considerations': [],
                             'weighing': '현재 자료로 이 규칙이 내 생활에 미칠 영향을 알기 어렵다.',
                             'conditions': [], 'uncertainties': ['나의 시설 이용 여부는 알 수 없다.']}}
    def call(*args, **kwargs):
        return SimpleNamespace(model='LG-fixture', choices=[SimpleNamespace(finish_reason='stop',
               message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)))])
    record = collector.collect_one(probe.probe_cases()[3]['packet'], Tokenizer(), call,
                                   tmp_path, 'LG-fixture', synthetic_fixture=True)
    assert record['response_status'] == 'answered'
    assert record['response']['stance'] == 'uncertain'
    assert record['response']['argument']['conditions'] == []
    assert not any('length' in flag or 'opposing' in flag for flag in record['argument_quality']['quality_flags'])


def test_execute_probe_keeps_all_four_sealed_records_and_unscored_reviews(tmp_path, monkeypatch):
    import llm_client
    import prompt_budget
    from evidence_integrity import verify
    model = 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ'
    response = {'stance': 'uncertain', 'answer': '현재 자료로는 판단을 유보합니다.',
                'stance_quote': '판단을 유보합니다', 'confidence': 0.8, 'reasons': [],
                'argument': {'personal_situation': [], 'considerations': [],
                             'weighing': '주어진 자료만으로 내 생활에 미칠 영향을 단정하기 어렵다.',
                             'conditions': [], 'uncertainties': ['실제 시설의 상황은 모른다.']}}
    # The transport stub verifies persistence, not actual model-quality scores.
    calls = []
    def call(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model=model, choices=[SimpleNamespace(finish_reason='stop',
               message=SimpleNamespace(content=json.dumps(response, ensure_ascii=False)))])
    monkeypatch.setattr(llm_client, 'get_spec', lambda: SimpleNamespace(hf_id=model))
    monkeypatch.setattr(llm_client, 'healthcheck', lambda: {'served_match': True})
    monkeypatch.setattr(llm_client, 'call_chat', call)
    monkeypatch.setattr(prompt_budget, 'load_tokenizer', lambda path: Tokenizer())
    folder = tmp_path / 'executed_stub'
    assert probe.main(['--out', str(folder), '--execute', '--tokenizer-path', str(tmp_path / 'fake')]) == 0
    assert len(calls) == 4
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    review = json.loads((folder / 'review.json').read_text(encoding='utf-8'))
    records = [json.loads(path.read_text(encoding='utf-8')) for path in folder.glob('*/records/*.json')]
    assert len(records) == 4 and all(verify(record) for record in records)
    assert all(record['provenance']['synthetic_fixture'] for record in records)
    assert {r['record_sha256'] for r in review['cases']} == {r['integrity_sha256'] for r in records}
    assert summary['response_counts'] == {'answered': 4}
    assert summary['status'] == 'human_semantic_review_required' and not summary['semantic_quality_passed']
