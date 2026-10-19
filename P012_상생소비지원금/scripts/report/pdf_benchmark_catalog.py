"""Display post-result PDF benchmarks without rescoring the frozen experiment.

Catalogs describe empirical values and their definitions. Optional simulation
sidecars bind CPU-only ledger calculations to the catalog and source SHA256s.
Neither input is supplied to a citizen prompt. Numerical pairs remain exploratory
and never acquire a direct gap, a direction hit, or a registered score.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import html
import json
import math
from pathlib import Path


ALIASES = {
    'P013': 'EMERGENCY_2020', 'P014': 'LOCAL_VOUCHER',
    'P015': 'SECTOR_VOUCHER_2020', 'DISTANCING': 'DISTANCING_2020',
    'DIST': 'DISTANCING_2020', 'distancing': 'DISTANCING_2020',
}
KNOWN_POLICIES = {
    'P010', 'P012', 'EMERGENCY_2020', 'DISTANCING_2020',
    'LOCAL_VOUCHER', 'P016', 'SECTOR_VOUCHER_2020', 'GATHERING_2020',
}
MODEL_REFERENCE_KINDS = {'source_model_estimate', 'aggregate_model_derived_effect'}
BENCHMARK_ROLES = {
    'policy_effect': '원문 정책 효과 추정값; 현 시뮬과 같은 추정량인지 별도 감사 필요',
    'descriptive_sales_change': '원문 관측 매출 변화; 코로나·지원금·규제 등의 영향이 섞여 정책 효과와 다름',
    'regression_association': '원문 상관관계 회귀계수; 정책의 인과효과로 해석하지 않음',
    'conditional_purchase_level': '성별·나이 등 조건부 구매수준 진단; 정책 효과 지표와 분리',
    'specification_sensitivity': '원문 추정 규격의 민감도; 같은 결과의 여러 추정치를 독립 실험으로 세지 않음',
    'voucher_use_composition': '쿠폰 사용액의 업종 구성비; 추가 소비율과 다른 분모',
    'voucher_use_timing': '쿠폰 사용 시점의 분포; 장기 사용 원장이 있어야 대응 가능',
    'cashback_distribution': '수령자 금액 분포; 실제 지급과 규칙상 발생추정의 차이 확인 필요',
    'sample_spending_diagnostic': '표본 지출 수준 진단; 정책 효과 점수와 분리',
    'participation_or_delivery': '전국 참여·지급 행정실적; 소규모 시민 표본의 총량과 직접 비교 불가',
}
RAW_LABELS = {
    'on': 'ON', 'off': 'OFF', 'on_won': 'ON 원화', 'off_won': 'OFF 원화',
    'on_amount_won': 'ON 금액', 'off_amount_won': 'OFF 금액',
    'on_count': 'ON 관측 수', 'off_count': 'OFF 관측 수',
    'on_receipts': 'ON 영수증 수', 'off_receipts': 'OFF 영수증 수',
    'on_citizens': 'ON 시민 수', 'off_citizens': 'OFF 시민 수',
    'difference_won': 'ON−OFF 금액 차이', 'delta_won': 'ON−OFF 금액 차이',
    'won': '원화', 'count': '관측 수', 'share': '비중',
    'denominator_won': '원화 분모', 'grant_issued_won': '명목 지급액',
    'policy_funded_won': '정책지갑 결제액', 'total_won': '총액',
    'recipients': '수령자 수(비중의 분모)', 'sample_citizens': '전체 시민 수',
    'citizen_days': '시민×일', 'positive_receipts': '양수 영수증 수',
    'unique_citizens': '고유 시민 수', 'percentage_points': '%p',
    'pre': '사전', 'post': '사후',
    'reason': '미산출 사유', 'bootstrap_valid_draws': '유효 재표집 수',
    'denominator': '분모', 'numerator': '분자',
    'citizen_days_each_arm': '팔별 시민×일 관측 수',
    'mapped_subclasses': '합산한 POI 하위 업종',
    'total_cashback_accrual_won': '월말 캐시백 총 발생추정액',
    'lower_won': '금액 구간 하한', 'upper_won': '금액 구간 상한',
    'cashback_zero_citizens': '캐시백 발생추정이 0인 시민 수',
    'all_policy_funded_won': '정책지갑 결제 총액',
    'policy_funded_positive_citizen_days': '정책지갑 결제가 양수인 시민×일 수',
    'catalog_poi_support_count': '고정 POI 자료의 해당 업종 판매처 수',
}
KOREAN_TEXT = {
    'v53 first round, Seoul 80 citizens; observed 2025-07-21..2025-07-23; wallet payments / issued grants.':
        'v53 1차 쿠폰, 서울 시민 80명의 2025-07-21~23 관측입니다. 정책지갑 결제액을 명목 지급액으로 나눈 값입니다.',
    'Current roster is spending-decile stratified only; this is not income-distribution matching.':
        '현재 시민은 지출 10분위로만 나눈 표본이며 실측 소득 분포를 맞춘 표본이 아닙니다.',
    'Post-result exploratory. Empirical figure 19 is approximate, first/second-round weighted survey use/plans.':
        '결과 후 탐색입니다. 실측 그림 19는 1·2차를 가중한 설문 사용·계획의 그림 판독 근사값입니다.',
    'Only two positive policy-funded citizen-days; most zeros reflect where these few wallet payments occurred, not a tested population response.':
        '정책지갑 결제가 양수인 시민×일은 2개뿐입니다. 다수의 0은 이 소수 결제가 해당 업종에 없었다는 뜻이며 모집단의 정책 무반응을 검증한 값이 아닙니다.',
    'Denominator is issued grant amount, not redeemed amount. No same-estimand gap/directional-hit/size accuracy.':
        '분모는 실제 사용액이 아니라 명목 지급액입니다. 동일 추정량의 오차·방향 적중·크기 정확도로 채점하지 않습니다.',
    'October 2021, 12 identical individual citizens ×31 days, paired ON/OFF;':
        '2021년 10월 같은 시민 12명×31일의 정책 ON/OFF 쌍체 비교입니다.',
    'P012-flag eligible offline spending.': 'P012 적격 표시가 있는 오프라인 지출입니다.',
    'Post-result exploratory; household/recipient/month/year estimand differs; bank sector crosswalk is provisional.':
        '결과 후 탐색이며 가구·수급자·월·연도를 대조하는 원문의 추정량과 다릅니다. 카드사와 POI의 업종 대응도 잠정적입니다.',
    'Post-result exploratory; source is household recipient/nonrecipient triple difference with September and 2019 controls; simulation is different paired percent growth.':
        '결과 후 탐색입니다. 실측은 수급·비수급 가구와 9월·2019년을 대조한 삼중차분이고 시뮬은 같은 시민의 ON/OFF 지출 변화율이므로 서로 다른 계산입니다.',
    'Source log coefficient is not raw percent growth. Card-bank sector crosswalk is provisional, not verified.':
        '원문의 로그회귀계수를 그대로 퍼센트 성장률로 읽으면 안 됩니다. 카드사와 POI 업종의 정확한 대응도 검증되지 않았습니다.',
    'by_sub group sums include offline purchases irrespective of per-transaction P012 eligibility; per-sub eligibility is not stored in this ledger.':
        '하위 업종 지출 합에는 거래별 P012 적격 여부와 관계없이 오프라인 구매가 들어갑니다. 이 원장에는 하위 업종별 적격 금액이 따로 저장되지 않았습니다.',
    '2021-10-31 rule-based cashback accrual distribution, positive-accrual denominator; actual payment not simulated.':
        '2021-10-31 말의 규칙 기반 캐시백 발생추정 분포입니다. 분모는 발생추정액이 양수인 사람이며 실제 지급 거래를 시뮬한 값이 아닙니다.',
    'Post-result exploratory; source finalized paid recipient distribution is reconstructed from rounded administrative counts.':
        '결과 후 탐색입니다. 실측은 최종 지급 수령자 분포를 원문의 반올림된 행정 집계로 재구성한 값입니다.',
    'Only 11 positive recipients out of 12; a zero sample bin is not evidence of a zero population proportion.':
        '12명 중 양수 발생추정 수령자는 11명뿐입니다. 어떤 금액 구간이 0명이라고 해서 모집단의 그 비중도 0이라는 뜻은 아닙니다.',
    'October end-month accrued reward per positive-accrual individual, not actual next-month payment.':
        '10월 말 발생추정액을 양수 발생추정 수령자로 나눈 평균이며 다음 달의 실제 지급 관측이 아닙니다.',
    'Post-result exploratory; source mean is actual nationwide administrative paid reward / positive recipient-months.':
        '결과 후 탐색입니다. 실측 평균은 전국 행정 집계의 실제 지급액을 양수 수령 월별 인원으로 나눈 값입니다.',
    'Aggregate grant-normalized paired October response per rule-accrual; not IV or a marginal receipt effect.':
        '10월 ON/OFF의 총지출 차이를 규칙 발생추정액으로 나눈 값입니다. 원문의 도구변수 추정이나 수령액 1원 증가의 한계 효과와 다릅니다.',
    'Post-result exploratory; self-selection and household/year/source-instrument estimation are not reproduced.':
        '결과 후 탐색이며 자기 선택과 가구·연도·도구변수에 대한 원문의 추정 절차를 재현하지 않았습니다.',
    'The denominator is rule-accrual not an actual payment transaction; no 2SLS accuracy or multiplier validation.':
        '분모는 실제 지급 거래가 아니라 규칙에 따른 발생추정액입니다. 2단계 최소제곱 추정의 정확도나 소비 배율 검증으로 읽지 않습니다.',
    'First/second survey issuance-weighted item use or intended use':
        '1·2차 설문을 지급액 비중으로 가중한 품목별 사용·사용 계획',
    'First survey 2025-08-13..20; second 2025-10-27..11-07; realized plus planned voucher use':
        '1차 설문 2025-08-13~20, 2차 설문 2025-10-27~11-07; 실제 사용과 사용 계획을 함께 관측',
    'Total issued coupon amount, NOT total used amount.':
        '총 쿠폰 지급액(실제 사용액이 아님)',
    'Total issued coupon amount, NOT total used amount':
        '총 쿠폰 지급액(실제 사용액이 아님)',
    'Household triple difference: recipient × month × 2021 (vs 2019)':
        '수급·비수급 가구 × 월 × 2021년(2019년 대비)의 삼중차분',
    '2021-10; September is baseline; 2019 comparison year for DDD; July–December panel':
        '2021년 10월 계수; 기준월은 9월, 비교연도는 2019년인 7~12월 가구 패널',
    'Monthly household log card spending.': '가구의 월별 카드 지출 로그값',
    'individual total spend.': '개인 시민의 총지출입니다.',
    'individual online spend.': '개인 시민의 온라인 지출입니다.',
    'FIRST': '1차', 'SECOND': '2차', 'WEIGHTED': '1·2차 가중',
    'model': '모형',
}


def _human_text(value):
    text = _text(value)
    for original, korean in sorted(KOREAN_TEXT.items(), key=lambda item: len(item[0]), reverse=True):
        text = text.replace(original, korean)
    return text


def is_number(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def numeric_observation(value):
    if is_number(value):
        return True
    if isinstance(value, (list, tuple)):
        return bool(value) and all(is_number(item) for item in value)
    if isinstance(value, dict):
        return bool(value) and all(is_number(item) for item in value.values())
    return False


def _path(value, root):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _display_path(path, root):
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normal_policy(value):
    policy = ALIASES.get(value, value)
    if policy not in KNOWN_POLICIES:
        raise ValueError(f'unknown PDF benchmark policy: {value}')
    return policy


def _check_source(item, root, cache):
    if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
        raise ValueError('PDF benchmark evidence requires path and sha256')
    path = _path(item['path'], root)
    expected = str(item['sha256']).lower()
    if path not in cache:
        cache[path] = _sha(path)
    if cache[path] != expected:
        raise ValueError(f'PDF benchmark evidence SHA256 mismatch: {path}')
    return {**item, 'path': _display_path(path, root), 'sha256': expected,
            'uri': path.as_uri()}


def _proxy(simulation):
    current = simulation.get('current_proxy')
    if isinstance(current, dict):
        return deepcopy(current)
    # Older callers may embed a calculated proxy directly in the catalog.
    if 'value' in simulation or 'simulation' in simulation:
        return deepcopy(simulation)
    return None


def _validate_raw(value, benchmark_id):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f'nonfinite raw observation in PDF benchmark: {benchmark_id}')
    if isinstance(value, dict):
        for item in value.values():
            _validate_raw(item, benchmark_id)
    elif isinstance(value, list):
        for item in value:
            _validate_raw(item, benchmark_id)


def apply_catalogs(report, catalog_paths, simulation_paths, root):
    """Return an extended copy; frozen rows, scores and totals are untouched."""
    if not catalog_paths and simulation_paths:
        raise ValueError('--benchmark-simulation requires --benchmark-catalog')
    out = deepcopy(report)
    cache, catalogs, lookup = {}, [], {}
    for path in catalog_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema') != 'pdf_benchmark_catalog_v1':
            raise ValueError(f'unsupported PDF benchmark catalog schema: {path}')
        policy = _normal_policy(payload.get('policy'))
        if not payload.get('timing') or 'post' not in str(payload['timing']).lower():
            raise ValueError('PDF benchmark expansion must be disclosed as post-result exploratory')
        source = _check_source(payload.get('source'), root, cache)
        entries = []
        for original in payload.get('entries') or []:
            entry = deepcopy(original)
            key = (policy, entry.get('id'))
            if not entry.get('id') or key in lookup:
                raise ValueError(f'duplicate or absent PDF benchmark id: {key}')
            empirical = entry.get('empirical') or {}
            if not numeric_observation(empirical.get('value')):
                # Empirical text without a numeric observation is not an
                # external numerical benchmark and does not enter the HTML.
                continue
            if not empirical.get('unit') or not entry.get('label'):
                raise ValueError(f'PDF benchmark needs label and empirical unit: {key}')
            if entry.get('direct_gap_allowed') is not False:
                raise ValueError(f'PDF expansion cannot inherit direct comparison approval: {key}')
            entry.update(policy=policy, source=source,
                         catalog_path=_display_path(path, root),
                         catalog_sha256=_sha(path),
                         timing=payload['timing'],
                         proxy=_proxy(entry.get('simulation') or {}))
            lookup[key] = entry
            entries.append(entry)
        catalogs.append({'policy': policy, 'path': _display_path(path, root),
                         'sha256': _sha(path), 'source': source,
                         'timing': payload['timing'], 'entries': entries})
    sidecars = []
    for path in simulation_paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if payload.get('schema') != 'pdf_benchmark_simulation_v1':
            raise ValueError(f'unsupported PDF benchmark simulation schema: {path}')
        policy = _normal_policy(payload.get('policy'))
        catalog_path = _path(payload.get('catalog_path', ''), root)
        catalog_sha = str(payload.get('catalog_sha256', '')).lower()
        matching = [item for item in catalogs if item['policy'] == policy
                    and _path(item['path'], root) == catalog_path
                    and item['sha256'] == catalog_sha]
        if len(matching) != 1:
            raise ValueError(f'simulation sidecar does not bind a supplied catalog: {path}')
        evidence = [_check_source(item, root, cache)
                    for item in payload.get('source_evidence') or []]
        if not evidence:
            raise ValueError('PDF benchmark simulation requires verified source_evidence')
        if payload.get('numeric_path') or payload.get('numeric_sha256'):
            score = _check_source({'path': payload.get('numeric_path'),
                                   'sha256': payload.get('numeric_sha256')}, root, cache)
            if not any(item['sha256'] == score['sha256']
                       and _path(item['path'], root) == _path(score['path'], root)
                       for item in report.get('score_files') or []):
                raise ValueError('PDF expansion numeric score is not an input to this report')
        ids = set()
        for original in payload.get('rows') or payload.get('entries') or []:
            row = deepcopy(original)
            key = (policy, row.get('id'))
            if key not in lookup or row.get('id') in ids:
                raise ValueError(f'unknown or duplicate PDF simulation id: {key}')
            if row.get('direct_gap_allowed') is not False:
                raise ValueError(f'PDF simulation cannot authorize direct gaps: {key}')
            ids.add(row['id'])
            row.update(evidence=evidence,
                       sidecar_path=_display_path(path, root), sidecar_sha256=_sha(path))
            lookup[key]['proxy'] = row
        sidecars.append({'policy': policy, 'path': _display_path(path, root),
                         'sha256': _sha(path), 'catalog_sha256': catalog_sha,
                         'source_evidence': evidence,
                         'metadata': {key: value for key, value in payload.items()
                                      if key not in ('rows', 'entries', 'source_evidence')}})
    counts, families, family_unknown = {}, {}, set()
    for entry in lookup.values():
        proxy = entry['proxy']
        if proxy:
            value = proxy.get('simulation', proxy.get('value'))
            if value is not None and not is_number(value):
                raise ValueError(f'PDF simulation value must be finite or null: {entry["id"]}')
            unit = proxy.get('simulation_unit', proxy.get('unit'))
            if is_number(value) and not unit:
                raise ValueError(f'PDF simulation numeric value needs a unit: {entry["id"]}')
            for item in proxy.get('evidence') or []:
                _check_source(item, root, cache)
            if is_number(value) and not proxy.get('evidence'):
                raise ValueError(f'PDF simulation numeric value lacks ledger SHA evidence: {entry["id"]}')
            proxy['simulation'], proxy['simulation_unit'] = value, unit
            proxy['direct_gap_allowed'] = False
            proxy['direction_comparable'] = False
            interval = proxy.get('ci', proxy.get('citizen_bootstrap_95_interval'))
            if interval is not None and not (isinstance(interval, list) and len(interval) == 2
                    and all(is_number(item) for item in interval) and interval[0] <= interval[1]):
                raise ValueError(f'invalid PDF simulation interval: {entry["id"]}')
            proxy['ci'] = interval
            _validate_raw(proxy.get('raw_components') or proxy.get('raw_values') or {}, entry['id'])
            for alternate in proxy.get('alternative_proxies') or []:
                alternate_value = alternate.get('simulation', alternate.get('value'))
                alternate_unit = alternate.get('simulation_unit', alternate.get('unit'))
                if alternate_value is not None and not is_number(alternate_value):
                    raise ValueError(f'nonfinite alternative PDF proxy: {entry["id"]}')
                if is_number(alternate_value) and not alternate_unit:
                    raise ValueError(f'alternative PDF proxy needs a unit: {entry["id"]}')
                if alternate.get('direct_gap_allowed') is True:
                    raise ValueError('alternative PDF proxy cannot authorize direct gaps')
                _validate_raw(alternate.get('raw_components') or {}, entry['id'])
        item = counts.setdefault(entry['policy'], {'empirical': 0, 'numeric_pairs': 0,
                                                 'raw_only': 0, 'needs_data': 0,
                                                 'model_reference_records': 0})
        if entry.get('benchmark_kind') in MODEL_REFERENCE_KINDS:
            item['model_reference_records'] += 1
            continue
        item['empirical'] += 1
        if entry.get('count_as_new_indicator') is not False:
            if entry.get('independent_outcome_family'):
                families.setdefault(entry['policy'], set()).add(entry['independent_outcome_family'])
            else:
                family_unknown.add(entry['policy'])
        if proxy and is_number(proxy.get('simulation')):
            item['numeric_pairs'] += 1
        elif proxy and (proxy.get('raw_components') or proxy.get('raw_values')):
            item['raw_only'] += 1
        else:
            item['needs_data'] += 1
    for policy, item in counts.items():
        item['additional_outcome_families'] = (None if policy in family_unknown
                                             else len(families.get(policy) or set()))
    out['pdf_benchmark_catalogs'] = catalogs
    out['pdf_benchmark_simulations'] = sidecars
    out['pdf_benchmark_counts'] = counts
    return out


def _esc(value):
    return html.escape(str(value if value is not None else ''), quote=True)


def _value(value, unit=''):
    if isinstance(value, list):
        return ' ~ '.join(_value(item, unit) for item in value)
    if isinstance(value, dict):
        return ' · '.join(f'{_esc(RAW_LABELS.get(key, key))} {_value(item, unit)}' for key, item in value.items())
    if is_number(value):
        number = (f'{value:,.0f}' if float(value).is_integer() else
                  f'{value:.6g}' if abs(value) < .0001 else
                  f'{value:,.4f}'.rstrip('0').rstrip('.'))
        suffix = {'log-point': ' 로그포인트', 'ratio': ' (비율)', '%p': '%p', '%': '%', '원': '원'}.get(unit, ' ' + str(unit) if unit else '')
        return number + _esc(suffix)
    return _esc(value)


def _text(value):
    if isinstance(value, list):
        return ' · '.join(str(item) for item in value)
    if isinstance(value, dict):
        return ' · '.join(f'{key}: {item}' for key, item in value.items())
    return str(value or '')


def _source_html(entry):
    empirical = entry['empirical']
    locator = []
    if empirical.get('pdf_page') is not None:
        locator.append(f'PDF {empirical["pdf_page"]}쪽')
    if empirical.get('printed_page') is not None:
        locator.append(f'인쇄 {empirical["printed_page"]}쪽')
    if empirical.get('locator'):
        locator.append(str(empirical['locator']))
    return ('<a href="' + _esc(entry['source']['uri']) + '">'
            + _esc(entry['source']['path']) + '</a>'
            + (' · ' + _esc(' / '.join(locator)) if locator else ''))


def _raw_html(values):
    pieces = []
    def visit(value, parts):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ('ci', 'citizen_bootstrap_95_interval'):
                    continue  # The interval is already labelled above the raw amounts.
                visit(item, parts + [RAW_LABELS.get(key, key)])
        elif isinstance(value, list):
            pieces.append('<li>' + _esc(' · '.join(parts)) + ': ' + _esc(_text(value)) + '</li>')
        elif value is not None:
            label = ' · '.join(parts)
            unit = '원' if any('원화' in part or '금액' in part or '지급액' in part or '결제액' in part
                              or part.endswith('총액')
                              or part.endswith('_won') or part.endswith('추정액') for part in parts) else ''
            pieces.append('<li>' + _esc(label) + ': ' + _value(value, unit) + '</li>')
    visit(values, [])
    return '<ul class="raw-values">' + ''.join(pieces) + '</ul>' if pieces else ''


def policy_html(report, policy):
    entries = [entry for catalog in report.get('pdf_benchmark_catalogs') or []
               for entry in catalog['entries'] if entry['policy'] == policy]
    if not entries:
        return ''
    paired, pending, model_refs = [], [], []
    for entry in entries:
        empirical, proxy = entry['empirical'], entry.get('proxy')
        source = _source_html(entry)
        approximate = ' (그림 판독 근사값)' if empirical.get('approximate') else ''
        truth_html = _value(empirical['value'], empirical['unit'])
        if entry.get('benchmark_kind') in MODEL_REFERENCE_KINDS:
            model_refs.append('<li><strong>' + _esc(entry['label']) + '</strong>: '
                              + truth_html + '<p>' + _esc(_text(empirical.get('method')))
                              + '</p><p class="source">' + source + '</p></li>')
            continue
        existing = entry.get('existing_indicator_id')
        family_note = ('<p class="reason">기존 등록 참조 ' + _esc(existing)
                       + '의 원문 상세이며 새 독립 지표로 더하지 않습니다.</p>') if existing else ''
        raw = (proxy or {}).get('raw_components') or (proxy or {}).get('raw_values') or {}
        alternatives = (proxy or {}).get('alternative_proxies') or []
        if not proxy or (not is_number(proxy.get('simulation')) and not raw and not alternatives):
            sim = entry.get('simulation') or {}
            missing = (_text((proxy or {}).get('quality_notes') or (proxy or {}).get('reason'))
                       or _text(sim.get('missing_requirements'))
                       or '원문과 연결되는 시뮬 측정 자료 및 산식을 먼저 확정해야 합니다.')
            if existing:
                missing = (f'기존 등록 참조 {existing}의 원문 세부값입니다. '
                           '위 등록 카드의 동결 수치·범위를 그대로 유지하고 '
                           '다른 업종 범위의 새 값으로 대체하지 않습니다. '
                           '이 항목은 중복된 원문 참조로 별도 독립 지표 수에 더하지 않습니다.')
            pending.append('<li><strong>' + _esc(_human_text(entry['label'])) + '</strong> '
                           '<span class="tru">실측 ' + truth_html + _esc(approximate) + '</span>'
                           '<p>' + _esc(_human_text(missing)) + '</p><p class="source">' + source + '</p></li>')
            continue
        number = proxy.get('simulation')
        numeric_pair = is_number(number)
        headline = ('<span class="sim">시뮬 ' + _value(number, proxy['simulation_unit']) + '</span>'
                    if numeric_pair else
                    '<span class="sim">시뮬 원관측값 아래 표시 · 비교 지표 미산출</span>')
        notes = _human_text(proxy.get('quality_notes_ko') or proxy.get('quality_notes') or proxy.get('reason'))
        scope = _human_text(proxy.get('scope_note_ko') or proxy.get('scope_note') or proxy.get('scope'))
        count = proxy.get('sample_citizens', proxy.get('n'))
        n_html = f'<span class="n">전체 시뮬 시민 n={_esc(count)} (해당 지표 분모는 원관측값 참조)</span>' if count is not None else ''
        interval = proxy.get('ci')
        ci_html = ('<p class="reason">시뮬 시민 재표집 95% 구간(모델·외부 표본 불확실성 미포함): '
                   + _value(interval[0], proxy['simulation_unit']) + ' ~ '
                   + _value(interval[1], proxy['simulation_unit']) + '</p>'
                   if interval else '')
        alternative_html = ''
        for alternate in alternatives:
            val = alternate.get('simulation', alternate.get('value'))
            if not is_number(val):
                continue
            alternative_html += (
                '<div class="balance alternative-proxy"><strong>추가 범위 대리값 '
                '(같은 실측 참조, 정식 검증 아님):</strong> '
                '<span class="sim">시뮬 ' + _value(val, alternate.get('simulation_unit', alternate.get('unit')))
                + '</span><p>' + _esc(_human_text(alternate.get('scope_note')))
                + '</p>' + _raw_html(alternate.get('raw_components') or {})
                + '<p>위 엄격 지표의 미산출을 이 숫자로 대체하지 않으며 '
                  '독립 실험·추가 검증지표 수로 세지 않습니다.</p></div>')
        detail = {key: empirical.get(key) for key in ('unit', 'denominator', 'population', 'period', 'method')}
        detail.update(formula=proxy.get('formula', proxy.get('method')),
                      benchmark_kind=entry.get('benchmark_kind'),
                      catalog_path=entry['catalog_path'], catalog_sha256=entry['catalog_sha256'],
                      simulation_sidecar=proxy.get('sidecar_path'),
                      simulation_sidecar_sha256=proxy.get('sidecar_sha256'),
                      evidence=proxy.get('evidence'), direct_gap_allowed=False)
        paired.append('<div class="row pdf-benchmark"><div class="meta">'
                      '<span class="id">' + _esc(entry['id']) + '</span>'
                      '<span class="desc">' + _esc(_human_text(entry['label'])) + '</span>'
                      '<span class="tag sus">결과 후 확장 탐색</span></div>'
                      + ('<p class="reason"><strong>지표의 역할:</strong> '
                         + _esc(BENCHMARK_ROLES[entry['benchmark_kind']]) + '.</p>'
                         if entry.get('benchmark_kind') in BENCHMARK_ROLES else '')
                      + '<div class="nums"><span class="tru">실측 ' + truth_html + _esc(approximate)
                      + '</span>' + headline + n_html + '</div>' + ci_html + family_note
                      + '<p class="reason">실측 정의: '
                      + _esc(_human_text(empirical.get('method'))) + '; 기간 '
                      + _esc(_human_text(empirical.get('period'))) + '; 분모 '
                      + _esc(_human_text(empirical.get('denominator'))) + '.</p>'
                      + ('<p class="reason">' + _esc(scope) + '</p>' if scope else '')
                      + _raw_html(raw)
                      + alternative_html
                      + ('<p class="opinion">' + _esc(notes) + '</p>' if notes else '')
                      + '<p class="reason">서로 다른 실측·시뮬 정의를 병렬 표시한 사후 탐색입니다. '
                        '정식 오차·방향 적중·프롬프트 점수를 계산하지 않습니다.</p>'
                      + '<p class="source">원문 출처: ' + source + '</p>'
                      + '<details class="technical"><summary>분모·기간·산식·지문 상세</summary><pre>'
                      + _esc(json.dumps(detail, ensure_ascii=False, indent=2)) + '</pre></details></div>')
    missing_html = ('<details class="benchmark-pending"><summary>추가 원문 실측 '
                    + str(len(pending)) + '개: 아직 필요한 측정 데이터</summary>'
                    '<p>실측 숫자는 있지만 현 원장과 연결된 시뮬 산출물이 없어 숫자 비교 카드에 넣지 않았습니다. '
                    '표본 확대만으로 되는지, 분류·공간·상품·기간 자료가 필요한지 각 항목에 적었습니다.</p><ul>'
                    + ''.join(pending) + '</ul></details>') if pending else ''
    sidecars = [item for item in report.get('pdf_benchmark_simulations') or []
                if item['policy'] == policy]
    audit_html = ('<details class="technical"><summary>추가 계산 자료의 출처·결합 감사</summary><pre>'
                  + _esc(json.dumps(sidecars, ensure_ascii=False, indent=2))
                  + '</pre></details>') if sidecars else ''
    model_html = ('<details class="benchmark-pending"><summary>원문 모형 추정 참고값 '
                  + str(len(model_refs)) + '개 · 실측 검증 대상에서 제외</summary>'
                  '<p>이 숫자는 원문 자체 모형에서 계산한 추정 참고값이며 관측 실적이 아닙니다. '
                  '추가 실측 확보 수나 실측·시뮬 숫자쌍에 합산하지 않습니다.</p><ul>'
                  + ''.join(model_refs) + '</ul></details>') if model_refs else ''
    empirical_count = len(entries) - len(model_refs)
    return ('<div class="pdf-expansion" id="pdf-expansion-' + _esc(policy.lower()) + '">'
            '<h3>원문 PDF에서 추가한 실측 지표 · 결과 후 탐색</h3>'
            '<p class="runline">원문 실측 기록 ' + str(empirical_count) + '개를 추가로 감사했습니다. '
            '각 숫자 기록·시계열 점·집단별 셀 수는 서로 독립인 검증지표 수가 아닙니다. '
            '등록 주지표 합계에 더하지 않고 기존 동결 점수를 재채점하지 않습니다. '
            '이 계산의 시민은 기존 소규모 비가중 표본입니다. 서울 성별·연령·행정동·소득 '
            '실측 분포를 새로 맞춘 표본의 결과가 아닙니다.</p>'
            + ''.join(paired) + missing_html + model_html + audit_html + '</div>')
