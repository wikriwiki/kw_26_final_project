"""Typed ordered purchases plus independently positioned asset acquisitions.

Purchase roster order is guaranteed by representation. No preference, amount,
or funding source is chosen by the compiler. Original model JSON stays intact.
"""
import json
from asset_transaction_contract import schema as legacy_schema, inspect as legacy_inspect


def schema(case):
    legacy=legacy_schema(case)
    branches=legacy['properties']['actions']['items']
    branches=branches.get('anyOf',[]) if isinstance(branches,dict) else []
    purchases=[];acquisitions=[]
    for branch in branches:
        props=branch['properties']
        if props['kind']['const']=='consume':
            props={k:v for k,v in props.items() if k!='kind'}
            purchases.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
        else:
            props={k:v for k,v in props.items() if k not in {'kind','id'}}
            props['before_event_id']={'enum':[None]+[e['id'] for e in case['events']]}
            acquisitions.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
    order={'type':'array','minItems':len(purchases),'maxItems':len(purchases),'items':False}
    if purchases:order['prefixItems']=purchases
    assets={'type':'array','minItems':0,'maxItems':len(case['offers']),
            'items':{'anyOf':acquisitions} if acquisitions else False}
    return {'type':'object','properties':{'acquisitions':assets,'purchases':order},
            'required':['acquisitions','purchases'],'additionalProperties':False}


def inspect(raw,case):
    obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={'acquisitions','purchases'}:raise ValueError('Typed transaction shape')
    if not isinstance(obj['acquisitions'],list) or not isinstance(obj['purchases'],list):raise ValueError('Typed transaction lists')
    expected=[e['id'] for e in case['events']]
    if [p.get('id') for p in obj['purchases']]!=expected:raise ValueError('Typed purchase roster')
    before={eid:[] for eid in expected};before[None]=[];seen=set()
    for a in obj['acquisitions']:
        if not isinstance(a,dict) or set(a)!={'offer_id','units','reason','before_event_id'}:raise ValueError('Typed acquisition fields')
        if a['offer_id'] not in case['offers'] or a['offer_id'] in seen:raise ValueError('Unknown/duplicate acquisition')
        if a['before_event_id'] not in before:raise ValueError('Unknown acquisition position')
        seen.add(a['offer_id'])
        before[a['before_event_id']].append({'kind':'acquire_wallet','id':'acquire:'+a['offer_id'],
            'offer_id':a['offer_id'],'units':a['units'],'reason':a['reason']})
    actions=[]
    for p in obj['purchases']:
        if not isinstance(p,dict) or set(p)!={'id','candidate_id','cash_payment','wallet_spend','reason'}:raise ValueError('Typed purchase fields')
        actions.extend(before[p['id']]);actions.append(dict(p,kind='consume'))
    actions.extend(before[None])
    _,ledger=legacy_inspect(json.dumps({'actions':actions},ensure_ascii=False),case)
    return obj,ledger
