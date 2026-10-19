"""Known factual physical-state tests, not a policy-effect or demand benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
from daily_resource_contract import settle
from asset_transaction_contract_v4 import inspect
from validate_prompt_v3 import atomic


def prepare():
    definitions=[('arrival_boundary',0,[('09:00','order'),('10:30','use')]),
                 ('existing_stock',1,[('09:00','use'),('12:00','order')]),
                 ('between_uses',1,[('08:00','use'),('09:00','order'),('11:00','use')]),
                 ('late_arrival',0,[('22:00','order'),('23:30','use')])]
    cells=[];witnesses=[]
    for label,stock,schedule in definitions:
        events=[];choices=[]
        for index,(time,kind) in enumerate(schedule):
            candidates=[{'id':'soap_bottle','description':'세제 한 통, 배송비 포함','price_won':12000,'eligible_wallets':[]}] if kind=='order' else []
            event={'id':'event:'+str(index),'time':time,'activity_id':kind,'channel':'online' if kind=='order' else 'offline','candidates':candidates}
            events.append(event)
            # A feasibility witness, retained outside model-facing case. It is
            # not the correct behavioral purchase when existing stock suffices.
            choices.append({'id':event['id'],'candidate_id':'soap_bottle' if kind=='order' else None,'wallet_spend':{}})
        conditions={'provenance':{'kind':'synthetic_assumption','source':'Registered known physical-resource stress case.'},
                    'resources':{'soap':{'unit':'세탁 1회분','opening_quantity':stock}},
                    'activity_consumption':{'use':{'soap':1}},'quote_receipts':{'soap_bottle':{'soap':10}},
                    'quote_receipt_delay_minutes':{'soap_bottle':90},'needs':[]}
        case={'id':label,'context':'합성 시민의 이미 선택된 일정이다. use는 세탁1회를 실제로 실행하며 세제1회분이 필요하다. order는 구매 검토다. 일정 자체는 변경하지 않는다. 물품은 구입90분 후 도착하며 세제 한 통은10회분이라는 실험 가정이다.',
              'cash':24000,'wallet_lots':{},'offers':{},'events':events,'daily_conditions':conditions}
        raw=json.dumps({'acquisition_units':{},'purchases':choices})
        _,ledger=inspect(raw,case);physical=settle(raw,case)
        cells.append({'aid':label,'case':label,'arm':'probe','date':'2026-09-21','transaction_case':case})
        witnesses.append({'id':label,'raw':raw,'ledger':ledger,'resource_ledger':physical})
    return {'cells':cells,'feasibility_witnesses_not_model_input':witnesses,
            'scope':'Four fixed-schedule physical facts, two registered seeds. No policy contrast. No demand inference. Extra affordable purchase is allowed; depletion/arrival constraints must hold.'}


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    args.out.parent.mkdir(parents=True,exist_ok=True);result=prepare();atomic(args.out,result)
    print(json.dumps({'cells':len(result['cells']),'sha256':hashlib.sha256(args.out.read_bytes()).hexdigest()}))
