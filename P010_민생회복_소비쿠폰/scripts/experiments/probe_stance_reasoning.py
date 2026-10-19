"""Small synthetic personal-context probe; no expected support/opposition labels.

Plan/render offline by default. --execute calls the frozen LG server, retaining
public outputs and an unscored review sheet. Structural validity is never a
semantic-quality pass. Fixtures are excluded from empirical study results.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts/sim'))
from evidence_integrity import canonical, digest, seal
from scripts.experiments import collect_policy_stances as collector

RUBRIC = {
    'personal_specificity': 'Does the public position use relevant supplied personal facts rather than only a smoking/demographic label?',
    'evidence_relevance': 'Does each cited fact actually support the associated claim, with purchase versus exposure/motive kept distinct?',
    'logical_connection': 'Does the stated importance of the facts connect to the final position without unsupported causal certainty or contradictions?',
    'fact_inference_value_separation': 'Are recorded facts, predictions, current values and hypothetical conditions distinguished?',
    'tradeoff_and_uncertainty': 'Are material competing concerns and unknowns handled when relevant, without forcing both sides or inventing conditions?',
    'context_sensitivity': 'Across matched cases, are relevant context differences reflected in reasons? Identical stance labels alone are not a failure.',
}


def probe_cases():
    """Hand-authored diagnostics, not citizens or answers from the study data."""
    profiles = [
        ('smoker_frequent_evening_player', {
            'age': 35, 'smoking_status': 'smoker', 'job': '교대근무', 'daily_wd': 18000,
            'daily_we': 22000, 'nv_hobbies': '퇴근 후 당구', 'commute_min': 55,
            'nv_summary': '주중에는 퇴근 뒤 짧은 여가 시간이 있다.'}, True,
         'Use the recorded leisure/time context without treating smoking as an opposition vote.'),
        ('smoker_budget_walking', {
            'age': 35, 'smoking_status': 'smoker', 'job': '사무직', 'daily_wd': 5000,
            'daily_we': 7000, 'nv_hobbies': '집 근처 걷기', 'commute_min': 10,
            'nv_summary': '일상 지출을 아끼며 가까운 곳에서 여가를 보낸다.'}, False,
         'Same smoking label as the first case; personal relevance may differ, with no assumed facility use.'),
        ('non_smoker_frequent_evening_player', {
            'age': 35, 'smoking_status': 'non_smoker', 'job': '교대근무', 'daily_wd': 18000,
            'daily_we': 22000, 'nv_hobbies': '퇴근 후 당구', 'commute_min': 55,
            'nv_summary': '주중에는 퇴근 뒤 짧은 여가 시간이 있다.'}, True,
         'Only smoking differs from the first profile; neither support nor opposition is prescribed.'),
        ('unknown_sparse_history', {'age': None, 'smoking_status': 'unknown'}, None,
         'Missing facts must not become invented health/family/visit history; a short uncertain position is legitimate.'),
    ]
    cases = []
    for case_id, persona, played, focus in profiles:
        days = ['2017-11-21', '2017-12-05', '2017-12-16']
        items = []
        context = {'persona': persona, 'state': ({'balance': 45000, 'energy': 0.6, 'fatigue': 0.4}
                   if played is True else {'balance': 9000} if played is False else {}),
                   'memory': [], 'policy': [], 'appointment': []}
        def item(eid, day, kind, value):
            return {'evidence_id': case_id + '_' + eid, 'day': day, 'kind': kind,
                    'value': value, 'text': canonical(value),
                    'source_ref': {'synthetic_fixture': True, 'author': 'handwritten_context_probe'}}
        items.append(item('context', days[-1], 'context', context))
        if played is True:
            for index, day in enumerate(days[:2]):
                items.append(item(f'receipt_{index}', day, 'executed_receipt', {
                    'event_id': f'FIXTURE_{case_id}_{index}', 'poi_id': 'FIXTURE_BILLIARD',
                    'facility_type': 'billiard', 'district_code': '11650',
                    'amount': 12000, 'purchase_status': 'purchased',
                    'observed_at': day, 'policy_active': day >= '2017-12-03'}))
        elif played is False:
            items.append(item('receipt', days[0], 'executed_receipt', {
                'event_id': f'FIXTURE_{case_id}', 'poi_id': 'FIXTURE_CAFE',
                'facility_type': 'other', 'district_code': '11650',
                'amount': 3000, 'purchase_status': 'purchased', 'observed_at': days[0]}))
        packet = seal({'schema_version': 1, 'kind': 'grounded_interview_packet',
                       'run_id': 'SYNTHETIC_REASONING_PROBE', 'arm': 'on', 'agent_id': case_id,
                       'through_day': days[-1], 'days': days, 'missing_days': [], 'missing_night_days': [],
                       'cohort_sha256': digest([p[0] for p in profiles]),
                       'source_sha256': digest(profiles), 'evidence_items': items,
                       'limitations': ['Hand-authored fixture, never an observed human or completed simulation.']})
        cases.append({'case_id': case_id, 'packet': packet, 'review_focus': focus,
                      'synthetic_fixture': True, 'expected_stance': None})
    return copy.deepcopy(cases)


def review_sheet(case, record=None):
    return {'case_id': case['case_id'], 'record_id': record['record_id'] if record else None,
            'record_sha256': record['integrity_sha256'] if record else None,
            'review_focus': case['review_focus'], 'reviewer': None,
            'scores': {key: None for key in RUBRIC}, 'unsupported_or_contradictory_claims': [],
            'notes': None, 'review_status': 'unreviewed'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', help='Call the frozen LG server for four synthetic probes')
    parser.add_argument('--tokenizer-path', type=Path, help='Pinned local tokenizer; also enables offline rendered-input checks')
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error('Use a new output directory; probe outputs are immutable')
    tokenizer = None
    if args.execute or args.tokenizer_path:
        from prompt_budget import download_tokenizer, load_tokenizer
        path = args.tokenizer_path or ROOT / 'output/experiments/no_smoking_zone/runtime/tokenizer'
        if not args.tokenizer_path:
            download_tokenizer(path)
        tokenizer = load_tokenizer(path)
    call = None
    if args.execute:
        from llm_client import get_spec, healthcheck, call_chat
        if get_spec().hf_id != 'LGAI-EXAONE/EXAONE-4.5-33B-AWQ' or not healthcheck().get('served_match'):
            parser.error('The frozen LG model must be healthy before executing probes')
        call = call_chat
    args.out.mkdir(parents=True)
    rows, reviews = [], []
    for case in probe_cases():
        row = {'case_id': case['case_id'], 'response_status': 'not_called'}
        record = None
        if tokenizer:
            selected = collector.bounded_packet(case['packet'], tokenizer, 8192 - collector.MAX_OUTPUT_TOKENS - 128)
            row.update(input_tokens=collector.prompt_tokens(tokenizer, selected), selection=selected['selection'])
            collector.write_new(args.out / 'rendered' / (case['case_id'] + '.json'),
                                seal({'case_id': case['case_id'], 'packet': selected,
                                      'messages': collector.messages(selected), 'synthetic_fixture': True}))
        if args.execute:
            record = collector.collect_one(case['packet'], tokenizer, call, args.out / case['case_id'],
                'LGAI-EXAONE/EXAONE-4.5-33B-AWQ', synthetic_fixture=True)
            row.update(response_status=record['response_status'], argument_quality=record.get('argument_quality'))
        reviews.append(review_sheet(case, record))
        rows.append(row)
    summary = seal({'synthetic_fixture': True, 'empirical_evaluation_eligible': False,
                    'llm_calls_requested': len(rows) if args.execute else 0,
                    'question_id': collector.QUESTION_ID, 'question_sha256': collector.QUESTION_SHA256,
                    'argument_contract_version': collector.ARGUMENT_CONTRACT_VERSION,
                    'reserved_output_tokens': collector.MAX_OUTPUT_TOKENS,
                    'status': 'human_semantic_review_required' if args.execute else 'offline_plan_or_render_only',
                    'response_counts': dict(Counter(r['response_status'] for r in rows)), 'cases': rows,
                    'semantic_quality_passed': False, 'actual_stance_accuracy_measured': False})
    collector.write_new(args.out / 'summary.json', summary)
    collector.write_new(args.out / 'review.json', seal({'rubric': RUBRIC,
        'score_scale': {'0': 'fails', '1': 'partial', '2': 'meets', 'null': 'unreviewed'},
        'instructions': 'Read the complete input and public response. Exact quotes and detailed JSON alone do not prove logical support. No target stance is supplied.',
        'cases': reviews}))
    print(canonical(summary))
    return 2 if any(r['response_status'] == 'error' for r in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
