"""Experimental prepaid-asset accounting; not connected to production yet.

Acquiring a prepaid balance is an asset transfer. Consumption is recorded when
goods/services are purchased. Explicit redemption choices are never increased.
The caller supplies available offers and merchant eligibility from policy facts.
"""
from copy import deepcopy
from transaction_ledger import won


def settle_asset_actions(actions, *, cash, wallet_lots, offers, eligible_wallets_by_purchase):
    opening_cash = cash = won(cash, 'cash')
    lots = deepcopy(wallet_lots)
    for pid, entries in lots.items():
        if not isinstance(pid, str) or not isinstance(entries, list):
            raise ValueError('Invalid wallet lot structure')
        for lot in entries:
            lot['face'] = won(lot['face'], 'face')
            lot['own_basis'] = won(lot['own_basis'], 'own_basis')
            if lot['own_basis'] > lot['face']:
                raise ValueError('Own basis exceeds face value')
    opening_lots = deepcopy(lots)
    seen = set(); purchases = set(); used_units = {}; records = []
    consumption = {'offline': 0, 'online': 0}
    own_consumption = other_consumption = acquisition_cost = concession_issued = direct_cash = 0
    for original in actions:
        action = deepcopy(original)
        aid = action.get('id')
        if not isinstance(aid, str) or not aid or aid in seen:
            raise ValueError('Missing or duplicate action ID')
        seen.add(aid)
        if action['kind'] == 'acquire_wallet':
            offer_id = action['offer_id']
            if offer_id not in offers: raise ValueError('Unavailable offer')
            quote = offers[offer_id]
            units = won(action['units'], 'units')
            if not units: raise ValueError('Empty acquisition should be omitted')
            cap = won(quote['max_units'], 'max_units')
            used_units[offer_id] = used_units.get(offer_id, 0) + units
            if used_units[offer_id] > cap: raise ValueError('Offer cap exceeded')
            face = won(quote['unit_face'], 'unit_face') * units
            cost = won(quote['unit_cash_cost'], 'unit_cash_cost') * units
            if not face or cost > face: raise ValueError('Unsupported quote')
            if cost > cash: raise ValueError('Acquisition exceeds current cash')
            pid = quote['wallet_id']
            if not isinstance(pid, str) or not pid: raise ValueError('Invalid wallet ID')
            cash -= cost; acquisition_cost += cost; concession_issued += face-cost
            lots.setdefault(pid, []).append({'face': face, 'own_basis': cost})
            action.update(wallet_id=pid, face_added=face, cash_cost=cost)
        elif action['kind'] == 'consume':
            purchases.add(aid)
            if aid not in eligible_wallets_by_purchase: raise ValueError('Missing purchase eligibility')
            if action['channel'] not in consumption: raise ValueError('Unsupported channel')
            amount = won(action['amount'], 'amount')
            personal_cash = won(action['cash_payment'], 'cash_payment')
            requested = action['wallet_spend']
            if not isinstance(requested, dict): raise ValueError('Explicit wallet choices required')
            payments = {pid: won(value, 'wallet_spend') for pid, value in requested.items()}
            if personal_cash + sum(payments.values()) != amount: raise ValueError('Funding differs from consumption')
            if personal_cash > cash: raise ValueError('Purchase exceeds current cash')
            own_from_wallets = 0
            for pid, payment in payments.items():
                if pid not in lots or pid not in eligible_wallets_by_purchase[aid]:
                    raise ValueError('Unavailable or ineligible wallet')
                if payment > sum(lot['face'] for lot in lots[pid]): raise ValueError('Wallet funds exceeded')
                remaining = payment
                for lot in lots[pid]:
                    take = min(remaining, lot['face'])
                    if not take: continue
                    basis = lot['own_basis'] * take // lot['face']
                    lot['face'] -= take; lot['own_basis'] -= basis
                    own_from_wallets += basis; remaining -= take
            own = personal_cash + own_from_wallets
            cash -= personal_cash; direct_cash += personal_cash
            own_consumption += own; other_consumption += amount-own
            consumption[action['channel']] += amount
            action.update(own_funded_amount=own, concession_funded_amount=amount-own)
        else:
            raise ValueError('Unsupported action kind')
        records.append(action)
    if purchases != set(eligible_wallets_by_purchase): raise ValueError('Eligibility roster differs from purchases')
    total = sum(consumption.values())
    opening_face = sum(lot['face'] for entries in opening_lots.values() for lot in entries)
    closing_face = sum(lot['face'] for entries in lots.values() for lot in entries)
    assert total == own_consumption + other_consumption
    assert cash + closing_face == opening_cash + opening_face + concession_issued - total
    assert opening_cash-cash == acquisition_cost+direct_cash
    return {'complete': True, 'actions': records, 'opening_cash': opening_cash, 'closing_cash': cash,
            'opening_wallet_lots': opening_lots, 'closing_wallet_lots': lots,
            'offline_consumption': consumption['offline'], 'online_consumption': consumption['online'],
            'total_consumption': total, 'own_funded_consumption': own_consumption,
            'concession_funded_consumption': other_consumption, 'new_concession_face_value': concession_issued,
            'asset_acquisition_cash_outflow': acquisition_cost, 'cash_outflow_total': acquisition_cost+direct_cash,
            'scope': 'Nominal goods/services purchases, cash flow and prepaid assets separately. Concession payer is not inferred. FIFO proportional own basis in integer won; full redemption releases all residual basis. Not a policy-effect or price model.'}
