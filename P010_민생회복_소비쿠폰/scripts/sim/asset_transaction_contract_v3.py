"""Citizen choices without redundant payment arithmetic or invented factual prose.

For explicitly static, all-day offers and no intraday income: acquisition units
are chosen by the citizen, and acquired just before first chosen redemption (or
after purchases if never used). Cash is the exact quoted-price residual. Neither
purchase choice nor wallet amount is increased, reduced or silently repaired.
"""
import json
from asset_transaction_contract import inspect as legacy_inspect
from transaction_ledger import won


def preflight(case):
    if case['offers'] and case.get('execution_assumptions')!={'offers_available_before_any_purchase':True,'intraday_income':0}:
        raise ValueError('Static offer availability and zero intraday income must be explicit')


def schema(case):
    preflight(case);purchases=[]
    for event in case['events']:
        choices=[]
        for candidate in [None]+event['candidates']:
            wallets={} if candidate is None else {w:{'type':'integer','minimum':1,'maximum':candidate['price_won']} for w in candidate['eligible_wallets']}
            props={'id':{'const':event['id']},'candidate_id':{'const':candidate['id'] if candidate else None},
                   'wallet_spend':{'type':'object','properties':wallets,'additionalProperties':False}}
            choices.append({'type':'object','properties':props,'required':list(props),'additionalProperties':False})
        purchases.append({'anyOf':choices})
    order={'type':'array','minItems':len(purchases),'maxItems':len(purchases),'items':False}
    if purchases:order['prefixItems']=purchases
    offers={oid:{'type':'integer','minimum':0,'maximum':q['max_units']} for oid,q in case['offers'].items()}
    return {'type':'object','properties':{'acquisition_units':{'type':'object','properties':offers,'required':list(offers),'additionalProperties':False},'purchases':order},
            'required':['acquisition_units','purchases'],'additionalProperties':False}


def inspect(raw,case):
    preflight(case);obj=json.loads(raw)
    if not isinstance(obj,dict) or set(obj)!={'acquisition_units','purchases'}:raise ValueError('Choice shape')
    if not isinstance(obj['acquisition_units'],dict) or set(obj['acquisition_units'])!=set(case['offers']):raise ValueError('Offer choice roster')
    if not isinstance(obj['purchases'],list) or len(obj['purchases'])!=len(case['events']):raise ValueError('Purchase choice roster')
    pending={}
    for oid,value in obj['acquisition_units'].items():
        units=won(value,'acquisition units')
        if units>case['offers'][oid]['max_units']:raise ValueError('Acquisition cap exceeded')
        if units:pending[oid]={'kind':'acquire_wallet','id':'acquire:'+oid,'offer_id':oid,'units':units,'reason':'Citizen selected acquisition units; compiler schedules static offer before first redemption or after purchases.'}
    actions=[]
    for choice,event in zip(obj['purchases'],case['events']):
        if not isinstance(choice,dict) or set(choice)!={'id','candidate_id','wallet_spend'} or choice['id']!=event['id']:raise ValueError('Ordered choice fields')
        cid=choice['candidate_id'];candidates={c['id']:c for c in event['candidates']}
        if cid is not None and cid not in candidates:raise ValueError('Unknown quote')
        if not isinstance(choice['wallet_spend'],dict):raise ValueError('Wallet choices')
        payment={w:won(v,'wallet payment') for w,v in choice['wallet_spend'].items()}
        if any(v==0 for v in payment.values()):raise ValueError('Omit unused wallet')
        amount=0 if cid is None else won(candidates[cid]['price_won'],'quote price')
        cash=amount-sum(payment.values())
        if cash<0:raise ValueError('Wallet allocation exceeds quoted consumption')
        for oid in list(pending):
            if case['offers'][oid]['wallet_id'] in payment:actions.append(pending.pop(oid))
        actions.append(dict(choice,kind='consume',cash_payment=cash,reason='Citizen selected quote and wallet amount; cash is exact residual.'))
    actions.extend(pending.values())
    _,ledger=legacy_inspect(json.dumps({'actions':actions},ensure_ascii=False),case)
    ledger['execution_protocol']='v3 static offers, chosen units just before first redemption; exact price residual cash; no generated factual reason.'
    return obj,ledger
