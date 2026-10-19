"""Expose spendable balances; keep cost-basis bookkeeping in the ledger only."""
from copy import deepcopy
from transaction_ledger import won


def build(case):
    result=deepcopy(case)
    balances={}
    for wallet,lots in result.pop('wallet_lots').items():
        total=0
        for lot in lots:
            face=won(lot['face'],'spendable balance');basis=won(lot['own_basis'],'ledger cost basis')
            if basis>face:raise ValueError('Cost basis exceeds current face balance')
            total+=face
        balances[wallet]=total
    result['wallet_balances_won']=balances
    return result
