from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/prep'))
from export_menu_listings import listed_won,listings,export


def test_ambiguous_or_nonpositive_prices_are_not_coerced():
    assert listed_won(12000)==12000 and listed_won('12,000원')==12000
    for v in [True,0,-1,1.5,'변동','10,000~20,000','1,00','무료','0원','₩12,000']:
        assert listed_won(v) is None


def test_literal_unit_and_conditions_survive_without_discount_or_fee_inference():
    item={'name':'세트 (시간당/6인)','price':60000,'discounted_price':50000,'sold_out_yn':'Y','is_ai_mate':True,'mod_at':'2024-11-04'}
    panel={'menu':{'yogiyo_menus':{'items':[item]}},'summary':{'category':{'name1':'문화'}}}
    rows,bad=listings(panel,'123','2026-06-15 10:00:00')
    assert not bad and len(rows)==1
    r=rows[0];assert r['raw_item']==item and r['listed_price_won']==60000
    assert r['sold_out_flag']=='Y' and not r['purchase_ready'] and r['listing_channel']=='delivery_platform'
    assert r['fetched_at'].startswith('2026') and r['raw_item']['mod_at'].startswith('2024')


def test_only_observed_structured_menu_paths_are_exported():
    panel={'apartment':{'sale_price':1324},'photos':{'counts':{'menu':3}},'reviews':[{'name':'리뷰','price':10000}],
           'menu':{'menus':{'items':[{'name':'메뉴A','price':3000},{'name':'메뉴B','price':'변동'}]}}}
    rows,bad=listings(panel,'1','2026-06-15')
    assert [r['name'] for r in rows]==['메뉴A'] and bad['missing_or_nonpositive_or_ambiguous_price']==1


def test_snapshot_export_preserves_source_db_and_reports_exclusions(tmp_path):
    import hashlib,json,sqlite3
    db=tmp_path/'cache.db';con=sqlite3.connect(db)
    con.execute('create table poi_status(poi_id text,kakao_pid text,status text)')
    con.execute('create table panel3_raw(kakao_pid text,raw_json text,fetched_at text)')
    con.execute('insert into poi_status values(?,?,?)',('COM_1','123','fetched'))
    panel={'menu':{'menus':{'items':[{'name':'한 메뉴','price':3000},{'name':'변동 메뉴','price':'변동'}]}}}
    con.execute('insert into panel3_raw values(?,?,?)',('123',json.dumps(panel),'2026-06-15 10:00:00'))
    con.commit();con.close();before=hashlib.sha256(db.read_bytes()).hexdigest()
    out=tmp_path/'export';export(db,out)
    assert before==hashlib.sha256(db.read_bytes()).hexdigest()
    audit=json.loads((out/'audit.json').read_bytes())
    assert audit['counts']['listings']==1 and audit['excluded']['missing_or_nonpositive_or_ambiguous_price']==1
    row=json.loads((out/'listings.jsonl').read_bytes())
    assert row['candidate_poi_mappings']==[{'poi_id':'COM_1','status':'fetched'}]
    assert not row['purchase_ready'] and row['raw_item']['name']=='한 메뉴'
