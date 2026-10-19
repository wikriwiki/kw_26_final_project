"""V3 choices with availability-dependent grammar; no choice is auto-repaired.

A wallet not owned or acquired cannot appear in a redemption. Multi-purchase
balance conservation still needs the independent ledger; this grammar does not
claim to solve every arithmetic constraint. Same output/prompt semantics as V3.
"""
from copy import deepcopy
from itertools import product
from asset_transaction_contract_v3 import inspect, schema as schema_v3


def schema(case):
    base=schema_v3(case);offers=case['offers'];ids=list(offers)
    if len(ids)>8:raise ValueError('Availability grammar supports at most eight simultaneous offers')
    opening={w for w,lots in case['wallet_lots'].items() if sum(lot['face'] for lot in lots)>0}
    caps={oid:min(q['max_units'],case['cash']//q['unit_cash_cost']) if q['unit_cash_cost'] else q['max_units'] for oid,q in offers.items()}
    branches=[]
    for active in product(*[([False,True] if caps[oid]>0 else [False]) for oid in ids]):
        branch=deepcopy(base);available=opening|{offers[oid]['wallet_id'] for oid,on in zip(ids,active) if on}
        values=branch['properties']['acquisition_units']['properties']
        for oid,on in zip(ids,active):values[oid]={'type':'integer','minimum':1,'maximum':caps[oid]} if on else {'const':0}
        for event in branch['properties']['purchases'].get('prefixItems',[]):
            for candidate in event['anyOf']:
                props=candidate['properties']['wallet_spend']['properties']
                candidate['properties']['wallet_spend']['properties']={w:v for w,v in props.items() if w in available}
        branches.append(branch)
    return branches[0] if len(branches)==1 else {'anyOf':branches}
