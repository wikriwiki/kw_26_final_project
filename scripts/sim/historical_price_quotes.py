"""Lossless item quote access for Seoul OA-1170 annual price observations.

Observed sale specifications, not item-name quantities, define each quote.
No inflation conversion, averaging, future lookup, shop eligibility or POI match.
"""
import argparse
from collections import Counter
import csv
from datetime import date
import hashlib
import json
from pathlib import Path


def read_rows(path):
    path=Path(path);raw=path.read_bytes()
    for encoding in ['utf-8-sig','cp949']:
        try:raw.decode(encoding);break
        except UnicodeDecodeError:continue
    else:raise ValueError('Unsupported price encoding')
    with path.open(encoding=encoding,newline='') as fp:rows=list(csv.DictReader(fp))
    needed={'일련번호','시장/마트 번호','시장/마트 이름','품목 번호','품목 이름','실판매규격','가격(원)',
            '년도-월','자치구 이름','자치구 코드','점검일자','시장유형 구분(시장/마트) 이름','비고'}
    if rows and not needed<=set(rows[0]):raise ValueError('Unexpected annual price schema')
    seen=set();quotes=[];excluded=[]
    for row in rows:
        rid=row['일련번호']
        if not rid or rid in seen:raise ValueError('Missing/duplicate observation ID')
        seen.add(rid);day=date.fromisoformat(row['점검일자'])
        if day.isoformat()!=row['점검일자'] or row['년도-월']!=day.isoformat()[:7]:raise ValueError('Contradictory observation date')
        try:price=int(row['가격(원)'])
        except ValueError:
            excluded.append({'row_id':rid,'reason':'non_integer_price'});continue
        if price<=0:
            excluded.append({'row_id':rid,'reason':'non_positive_price'});continue
        if not row['실판매규격'].strip():
            excluded.append({'row_id':rid,'reason':'missing_sale_specification'});continue
        quotes.append({'observation_id':rid,'observed_on':day.isoformat(),'market_id':row['시장/마트 번호'],
            'market_name':row['시장/마트 이름'],'market_type':row['시장유형 구분(시장/마트) 이름'],
            'district_name':row['자치구 이름'],'source_district_code':row['자치구 코드'],
            'item_id':row['품목 번호'],'item_name':row['품목 이름'],'sale_specification':row['실판매규격'],
            'price_won':price,'note':row['비고']})
    return quotes,{'raw_rows':len(rows),'usable_quotes':len(quotes),'encoding':encoding,
        'raw_sha256':hashlib.sha256(raw).hexdigest(),'excluded_rows':excluded,
        'source_url':'https://data.seoul.go.kr/dataList/OA-1170/S/1/datasetView.do',
        'scope':'Observed food sale packages only. Zero-price observations are not free goods. No inferred eligibility, stock, delivery, routing, or nominal-unit conversion.'}


def as_of(quotes, *, today, district_name, max_age_days):
    day=date.fromisoformat(today)
    if isinstance(max_age_days,bool) or not isinstance(max_age_days,int) or max_age_days<0:raise ValueError('Explicit age bound required')
    selected=[q for q in quotes if q['district_name']==district_name and 0<=(day-date.fromisoformat(q['observed_on'])).days<=max_age_days]
    # Keep latest observation only for the SAME shop, item and literal sale unit.
    # Same-day conflicting observations remain distinct records for caller audit.
    latest={}
    for q in selected:
        key=(q['market_id'],q['item_id'],q['sale_specification'])
        latest[key]=max(latest.get(key,''),q['observed_on'])
    return [q for q in selected if q['observed_on']==latest[(q['market_id'],q['item_id'],q['sale_specification'])]]


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():raise ValueError('Refusing overwrite')
    quotes,audit=read_rows(args.source)
    audit['districts']=sorted({q['district_name'] for q in quotes});audit['markets']=len({q['market_id'] for q in quotes})
    audit['excluded_reason_counts']=dict(Counter(r['reason'] for r in audit['excluded_rows']))
    audit['observed_min']=min(q['observed_on'] for q in quotes);audit['observed_max']=max(q['observed_on'] for q in quotes)
    args.out.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='excluded_rows'},ensure_ascii=False))
