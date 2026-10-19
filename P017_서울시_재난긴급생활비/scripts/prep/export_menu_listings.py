"""Read-only, item-preserving export of cached structured merchant menus.

Posted listings are not verified sale quotes, historical prices, policy merchant
eligibility or delivered totals. No median, unit conversion, imputation or OCR.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3


def listed_won(value):
    if isinstance(value,bool):return None
    if isinstance(value,int):return value if value>0 else None
    if isinstance(value,str):
        match=re.fullmatch(r'\s*([1-9]\d*|[1-9]\d{0,2}(?:,\d{3})+)\s*원?\s*',value)
        if match:return int(match[1].replace(',',''))
    return None


def listings(panel,pid,fetched_at):
    # These exact paths were inspected in the cache. Never recursively treat
    # apartment price, blog counts, AI price levels or reviews as menu prices.
    menu=panel.get('menu') or {};accepted=[];rejected=Counter()
    if not isinstance(menu,dict):return [],Counter({'non_object_menu':1})
    for kind in ['menus','yogiyo_menus']:
        group=menu.get(kind) or {}
        if not isinstance(group,dict):rejected['non_object_menu_group']+=1;continue
        items=group.get('items',[])
        if not isinstance(items,list):
            rejected['non_array_items']+=1;continue
        for index,item in enumerate(items):
            if not isinstance(item,dict):rejected['non_object_item']+=1;continue
            name=item.get('name');price=listed_won(item.get('price'))
            if not isinstance(name,str) or not name.strip():rejected['missing_name']+=1;continue
            if price is None:rejected['missing_or_nonpositive_or_ambiguous_price']+=1;continue
            raw=json.dumps(item,ensure_ascii=False,sort_keys=True,separators=(',',':'))
            accepted.append({'listing_id':f'{pid}:{kind}:{index}','kakao_pid':str(pid),
                'source_path':f'menu.{kind}.items[{index}]','fetched_at':fetched_at,
                'source_url':'https://place.map.kakao.com/'+str(pid),
                'category':panel['summary'].get('category') if isinstance(panel.get('summary'),dict) else None,
                'name':name,'listed_price_won':price,
                'listing_channel':'delivery_platform' if kind=='yogiyo_menus' else 'merchant_menu',
                'sold_out_flag':item.get('sold_out_yn'),'ai_flag':item.get('is_ai_mate'),
                'raw_item_sha256':hashlib.sha256(raw.encode()).hexdigest(),'raw_item':item,
                'purchase_ready':False,
                'unresolved':['Literal item serving/size/terms require review; no default per-person unit',
                              'Current availability and merchant-policy eligibility unverified',
                              'Delivery fee, minimum order and discount conditions not inferred' if kind=='yogiyo_menus' else 'Dine-in/takeaway applicability not inferred']})
    return accepted,rejected


def export(database,out):
    database=Path(database).resolve();out=Path(out)
    if out.exists():raise ValueError('Refusing overwrite')
    out.mkdir(parents=True)
    before=database.stat();con=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True)
    con.execute('PRAGMA query_only=ON');con.execute('BEGIN')
    mappings=defaultdict(list)
    for poi,pid,status in con.execute('SELECT poi_id,kakao_pid,status FROM poi_status WHERE kakao_pid IS NOT NULL'):
        mappings[str(pid)].append({'poi_id':poi,'status':status})
    counts=Counter();excluded=Counter();categories=Counter();channels=Counter();dates=Counter();stream_hash=hashlib.sha256()
    target=out/'listings.jsonl';pending=out/'listings.jsonl.pending'
    with pending.open('x',encoding='utf-8',newline='\n') as fp:
        for pid,raw,fetched_at in con.execute('SELECT kakao_pid,raw_json,fetched_at FROM panel3_raw ORDER BY rowid'):
            counts['source_panels']+=1
            stream_hash.update(json.dumps([pid,raw,fetched_at],ensure_ascii=False,separators=(',',':')).encode());stream_hash.update(b'\n')
            try:panel=json.loads(raw)
            except (TypeError,json.JSONDecodeError):excluded['invalid_panel_json']+=1;continue
            if not isinstance(panel,dict):excluded['non_object_panel']+=1;continue
            found,rejected=listings(panel,pid,fetched_at);excluded.update(rejected)
            if found:counts['panels_with_listings']+=1
            raw_hash=hashlib.sha256(raw.encode()).hexdigest()
            for item in found:
                item['source_panel_sha256']=raw_hash;item['candidate_poi_mappings']=mappings[str(pid)]
                fp.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
                counts['listings']+=1;channels[item['listing_channel']]+=1
                cat=item['category'] if isinstance(item['category'],dict) else {}
                categories[cat.get('name1','unknown')]+=1;dates[str(fetched_at)[:10]]+=1
                if item['sold_out_flag']=='Y':counts['sold_out_listings_retained']+=1
                if item['ai_flag'] is True:counts['ai_flagged_listings_retained']+=1
            if counts['source_panels']%10000==0:
                fp.flush();os.fsync(fp.fileno());print(json.dumps(dict(counts)),flush=True)
        fp.flush();os.fsync(fp.fileno())
    con.rollback();con.close();os.replace(pending,target)
    output_hash=hashlib.sha256()
    with target.open('rb') as fp:
        for chunk in iter(lambda:fp.read(1024*1024),b''):output_hash.update(chunk)
    after=database.stat()
    report={'created_at':datetime.now(timezone.utc).isoformat(),'database':str(database),
        'read_only_snapshot':True,'source_order':'rowid','database_size_before':before.st_size,'database_size_after':after.st_size,
        'database_mtime_before_ns':before.st_mtime_ns,'database_mtime_after_ns':after.st_mtime_ns,
        'source_panel_stream_sha256':stream_hash.hexdigest(),'listings_sha256':output_hash.hexdigest(),
        'counts':dict(counts),'excluded':dict(excluded),'top_level_categories':dict(categories),
        'channels':dict(channels),'fetch_dates':dict(dates),
        'scope':'Cached structured posted menu observations, preserving each raw item. Not historical 2020/2021 data, verified transactions, available inventory, legal merchant eligibility or complete delivered quotes. No automatic experiment integration.'}
    (out/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--db',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();export(a.db,a.out)
