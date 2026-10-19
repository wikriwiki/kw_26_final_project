from pathlib import Path
import sys
import csv
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from historical_price_quotes import as_of,read_rows


def test_no_future_or_stale_price_and_units_remain_distinct():
    def q(day,unit):return {'district_name':'강남구','observed_on':day,'market_id':'M','item_id':'I','sale_specification':unit}
    rows=[q('2020-05-01','1kg'),q('2020-05-13','1kg'),q('2020-05-13','1.5kg'),q('2020-05-15','1kg')]
    assert as_of(rows,today='2020-05-14',district_name='강남구',max_age_days=7)==rows[1:3]
    assert as_of(rows,today='2020-05-14',district_name='서초구',max_age_days=7)==[]


def write(path,rows):
    with path.open('w',encoding='cp949',newline='') as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def row():return {'일련번호':'1','시장/마트 번호':'M','시장/마트 이름':'시장','품목 번호':'I','품목 이름':'양파(1.5kg망)',
    '실판매규격':'1kg','가격(원)':'2000','년도-월':'2020-05','자치구 이름':'강남구','자치구 코드':'680000',
    '점검일자':'2020-05-13','시장유형 구분(시장/마트) 이름':'전통시장','비고':'표시 그대로'}


def test_sale_unit_not_inferred_from_item_name_and_zero_not_free(tmp_path):
    p=tmp_path/'p.csv';write(p,[row(),dict(row(),일련번호='2',**{'가격(원)':'0'})])
    q,a=read_rows(p)
    assert q[0]['sale_specification']=='1kg' and q[0]['price_won']==2000
    assert a['raw_rows']==2 and a['usable_quotes']==1 and a['excluded_rows'][0]['reason']=='non_positive_price'


def test_duplicate_observation_ids_are_not_silently_averaged(tmp_path):
    p=tmp_path/'p.csv';write(p,[row(),row()])
    with pytest.raises(ValueError,match='duplicate'):read_rows(p)
