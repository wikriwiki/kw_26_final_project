"""Build dated merchant receipt references without replacing active calibration.

Amounts/counts describe receipts, not unit prices, individual daily budgets or
causal policy effects. Always preserve period, aggregation and source hashes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile
from build_unit_price import _industry_to_l1


def build(path):
    path=Path(path); groups=defaultdict(lambda:[0,0]); categories=defaultdict(lambda:[0,0])
    periods=Counter(); unmapped=Counter(); seen=set(); ignored=Counter(); rows_count=0
    with zipfile.ZipFile(path) as archive:
        names=[name for name in archive.namelist() if name.lower().endswith('.csv')]
        if not names: raise ValueError('No CSV in source archive')
        members=[]
        for name in names:
            raw=archive.read(name)
            for encoding in ['utf-8-sig','cp949','euc-kr']:
                try: text=raw.decode(encoding); break
                except UnicodeDecodeError: continue
            else: raise ValueError('Unsupported source encoding')
            members.append({'name':name,'sha256':hashlib.sha256(raw).hexdigest(),'encoding':encoding})
            reader=csv.DictReader(io.StringIO(text))
            required={'기준_년분기_코드','행정동_코드','서비스_업종_코드','서비스_업종_코드_명','당월_매출_금액','당월_매출_건수'}
            if not required.issubset(reader.fieldnames): raise ValueError('Unexpected source columns')
            for row in reader:
                rows_count+=1
                period=row['기준_년분기_코드']; dong=row['행정동_코드']; code=row['서비스_업종_코드']
                key=(period,dong,code)
                if key in seen: raise ValueError(f'Duplicate aggregate row: {key}')
                seen.add(key); periods[period]+=1
                amount=int(row['당월_매출_금액']); count=int(row['당월_매출_건수'])
                if amount<0 or count<0: raise ValueError('Negative receipt aggregate')
                if count==0:
                    ignored['zero_count_nonzero_amount' if amount else 'zero_count_zero_amount']+=1
                    continue
                label=_industry_to_l1(row['서비스_업종_코드_명'])
                if label is None:
                    unmapped[row['서비스_업종_코드_명']]+=1; continue
                for target,index in [(groups,(dong,label)),(categories,label)]:
                    target[index][0]+=amount; target[index][1]+=count
    by_dong={}
    for (dong,label),(amount,count) in sorted(groups.items()):
        by_dong.setdefault(dong,{})[label]={'amount_won':amount,'receipts':count,'mean_receipt_won':round(amount/count,2)}
    summary={}
    for label,(amount,count) in sorted(categories.items()):
        cell_means=[a/n for (d,l),(a,n) in groups.items() if l==label]
        summary[label]={'amount_won':amount,'receipts':count,'mean_receipt_won':round(amount/count,2),
                        'median_of_dong_means_won':round(statistics.median(cell_means),2),'dong_count':len(cell_means)}
    return {'source':{'archive':path.name,'archive_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'members':members,
                      'dataset':'서울시 상권분석서비스(추정매출-행정동)','period_rows':dict(sorted(periods.items())),
                      'total_rows':rows_count,'ignored_rows':dict(ignored),'unmapped_industry_rows':dict(unmapped)},
            'definition':'Total estimated merchant receipts / receipt count over available source periods. Not price per item, not individual daily consumption, not policy-effect targets.',
            'limitations':['Dates must be matched to simulation or explicitly treated as temporal transfer.',
                          'Receipt amounts may include multiple items or people; no transaction-level quantiles can be inferred.',
                          'Industry-to-L1 mapping follows the existing builder; unmapped industries remain explicitly listed.',
                          'This artifact does not overwrite active unit_price.json or change any completed experiment.'],
            'categories':summary,'by_dong':by_dong}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',required=True); ap.add_argument('--out',required=True)
    args=ap.parse_args(); out=Path(args.out)
    if out.exists(): raise ValueError('Refusing overwrite')
    result=build(args.zip); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'periods':result['source']['period_rows'],'rows':result['source']['total_rows'],'dongs':len(result['by_dong']),
                      'categories':result['categories'],'sha256':hashlib.sha256(out.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=='__main__': main()
