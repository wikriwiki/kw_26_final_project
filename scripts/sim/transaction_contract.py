"""Experimental transaction output contract; no behavioural fallback or retries."""
import json
from transaction_ledger import settle_choices


def schema(case):
    kinds=[]
    for event in case['events']:
        wallet_ids=sorted(case['wallets'])
        kinds.append({'type':'object','properties':{
            'order':{'type':'integer','enum':[event['order']]},
            'poi_id':{'type':['string','null'],'enum':[None]+[c['poi_id'] for c in event['candidates']]},
            'actual_spent':{'type':'integer','minimum':0},
            'policy_spend':{'type':'object','properties':{pid:{'type':'integer','minimum':0} for pid in wallet_ids},'additionalProperties':False},
            'pick_reason':{'type':'string','minLength':1}},
            'required':['order','poi_id','actual_spent','policy_spend','pick_reason'],'additionalProperties':False})
    return {'type':'object','properties':{'picks':{'type':'array','minItems':len(kinds),'maxItems':len(kinds),'items':{'anyOf':kinds}}},'required':['picks'],'additionalProperties':False}


def inspect(raw, case):
    obj=json.loads(raw)
    if set(obj)!={'picks'} or not isinstance(obj['picks'],list):
        raise ValueError('output shape')
    expected={e['order']:e for e in case['events']}
    seen=set(); transactions=[]; eligibility={}
    for pick in obj['picks']:
        if set(pick)!={'order','poi_id','actual_spent','policy_spend','pick_reason'}:
            raise ValueError('pick fields')
        order=pick['order']
        if isinstance(order,bool) or not isinstance(order,int) or order not in expected or order in seen:
            raise ValueError('order roster')
        seen.add(order)
        if not isinstance(pick['pick_reason'],str) or not pick['pick_reason'].strip():
            raise ValueError('empty reason')
        event=expected[order]
        candidates={c['poi_id']:c for c in event['candidates']}
        pid=pick['poi_id']
        if pid is None:
            if pick['actual_spent']!=0 or pick['policy_spend']!={}:
                raise ValueError('no merchant cannot create payment')
            allowed=set()
        else:
            if pid not in candidates:
                raise ValueError('wrong order candidate')
            allowed=set(candidates[pid]['eligible_wallets'])
        key=str(order)
        eligibility[key]=allowed
        transactions.append({'id':key,'channel':event['channel'],'amount':pick['actual_spent'],'policy_spend':pick['policy_spend'],'poi_id':pid})
    if seen!=set(expected):
        raise ValueError('missing order')
    ledger=settle_choices(transactions,cash=case['cash'],wallets=case['wallets'],eligible_wallets_by_transaction=eligibility)
    failures=[]
    for requirement in case.get('requirements',[]):
        pick=next(p for p in obj['picks'] if p['order']==requirement['order'])
        for field in ['poi_id','actual_spent','policy_spend']:
            if field in requirement and pick[field]!=requirement[field]:
                failures.append(f"fact:{requirement['order']}:{field}")
    return obj,ledger,failures
