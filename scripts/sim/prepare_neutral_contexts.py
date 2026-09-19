"""Rebuild frozen personas with explicit synthetic states; no DB or LLM calls."""
import argparse
import copy
from datetime import date
import hashlib
import json
import os
from pathlib import Path
from validate_prompt_v3 import ROOT, atomic, digest, policy_row
from neutral_context import initial_state, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--personas-source', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--reference-dir', type=Path, default=ROOT / 'output/stats')
    args = ap.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Set PYTHONHASHSEED=0 before process launch')
    for k, v in {'EXP_POLICY_ANONYMOUS': '1', 'EXP_DURABLES': '1', 'EXP_CATLINE': 'fold', 'EXP_SANGSAENG_BASE_RATIO': '0.268'}.items():
        os.environ[k] = v
    from dawn_context import DawnContext, _build_zone_candidates, _sangsaeng_monthly_anchor, _with_params
    from environments import build_environment
    from reference_preflight import inspect_references
    out = Path(args.out)
    if out.exists():
        raise ValueError('Refusing overwrite')
    raw = Path(args.personas_source).read_bytes(); source = json.loads(raw)
    references = inspect_references(args.reference_dir, require_code_geography=True)
    config = json.loads((ROOT / 'data/experiments/validation_v3.json').read_text(encoding='utf-8'))
    cells = []
    for p in source['personas']:
        for case in config['cases']:
            day = date.fromisoformat(case['date'])
            zones = _build_zone_candidates(p, day, stats_dir=args.reference_dir)
            anchor = _sangsaeng_monthly_anchor(p)
            for arm in ['off', 'on']:
                pol = policy_row(case['policy'], day) if arm == 'on' else None
                grant = (pol['id'], 280000) if pol and pol['type'] == 'grant' else None
                state = initial_state(p, day, anchor, grant)
                env = build_environment('covid_2021', day)
                if case['id'] == 'distancing':
                    common = [s for s in env.get('facts', []) if s.startswith('서울 신규 확진')]
                    rules = ['식당 매장 취식은 21시까지, 이후 포장·배달만 가능', '카페는 시간과 무관하게 매장 이용 불가, 포장·배달만 가능'] if arm == 'on' else ['식당·카페의 추가 방역 영업시간 제한 없음. 각 매장 고유 운영시간은 유지']
                    env = {'headline': '감염병 유행 중인 서울', 'facts': common + rules}
                ctx = DawnContext(persona=copy.deepcopy(p), state=state, zone_candidates=zones, policy=[pol] if pol else [], environment=env)
                cb = None
                if pol and pol['type'] == 'cashback':
                    params = _with_params(pol)
                    cb = {'id': pol['id'], 'eligible_month_spent_won': state['sangsaeng_month_spent'],
                          'total_month_spent_won': state['month_spent'], 'reference_monthly_eligible_spend_won': anchor,
                          'threshold_won': int(round(anchor * float(params.get('threshold_ratio') or 1.03))),
                          'refund_rate': float(params.get('rate') if params.get('rate') is not None else .1),
                          'monthly_cap_won': int(params.get('cap') or 100000), 'available_refund_balance_won': 0,
                          'timing': '현재 적립 조건이며 환급 시점은 정책 본문을 따른다.'}
                user = render(ctx.to_prompt_blocks(day), today=day, day_type='weekday' if day.weekday() < 5 else 'weekend',
                              zones=[z['code'] for z in zones], cashback_status=cb)
                cells.append({'aid': p['id'], 'case': case['id'], 'arm': arm, 'date': day.isoformat(),
                              'zones': [z['code'] for z in zones], 'user': user, 'context_sha256': digest(user), 'synthetic_state': state})
    atomic(out, {'personas': source['personas'], 'cells': cells, 'reference_inputs': references,
        'provenance': {'personas_source_sha256': hashlib.sha256(raw).hexdigest(),
                       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       'renderer_sha256': hashlib.sha256(Path(__file__).with_name('neutral_context.py').read_bytes()).hexdigest(),
                       'scope': 'Read-only preparation. No LLM calls. Common synthetic prehistory, current POI geography, no behavioural spending-pace message.',
                       'assumptions': ['Cash balance is 39 weekday spending anchors, not observed wealth.',
                                       'Eligible-share factor 0.268 is inherited model calibration, not an observed individual baseline or causal effect.',
                                       'Grant uses the inherited uniform per-person conversion; not a household-level policy estimate.',
                                       'Still no empirical longitudinal history or historically reconstructed merchant menu prices.']}})
    print(json.dumps({'contexts': len(cells), 'people': len(source['personas']), 'sha256': hashlib.sha256(out.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
