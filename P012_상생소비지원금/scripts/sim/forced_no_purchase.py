"""Skip model calls only when the purchase protocol has exactly one outcome.

No quoted candidate and no asset acquisition offer means every event must be
zero. This is an execution fact, not an assumption that an inactive citizen
has no needs. Upstream schedule/demand coverage is outside this optimization.
"""
import json
from asset_transaction_contract import inspect


def resolve(case):
    # Even with no goods, an available prepaid offer is a genuine asset choice.
    if case['offers'] or any(event['candidates'] for event in case['events']):return None
    actions=[{'kind':'consume','id':e['id'],'candidate_id':None,'cash_payment':0,
              'wallet_spend':{},'reason':'입력에 구매 후보와 잔액 취득 제안이 없음'} for e in case['events']]
    raw=json.dumps({'actions':actions},ensure_ascii=False);_,ledger=inspect(raw,case)
    return {'raw':raw,'ledger':ledger,'decision_source':'deterministic_unique_no_purchase',
            'model_calls':0,'scope':'Exact unique financial outcome for this supplied candidate set; no claim of complete human demand.'}
