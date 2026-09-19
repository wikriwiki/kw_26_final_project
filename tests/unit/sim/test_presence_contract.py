import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from presence_contract import violations


CELL={'user':'09:00부터 17:00까지 사무실 근무.','required_presence_intervals':[{'start':'09:00','end':'17:00','anchor':'workplace','evidence':'09:00부터 17:00까지 사무실 근무.'}],
      'minimum_transitions':[{'from_anchor':'workplace','to_anchor':'residence','minimum_minutes':40}]}


def events(*pairs):return [{'time':t,'anchor':a} for t,a in pairs]


def test_on_time_arrival_does_not_excuse_midday_home_departure():
    e=events(('07:00','residence'),('09:00','workplace'),('14:00','residence'),('16:00','workplace'),('18:00','residence'))
    assert 'presence_departure' in violations(e,CELL)


def test_implicit_interval_and_commute_both_need_to_fit():
    assert violations(events(('09:00','workplace'),('17:40','residence')),CELL)==[]
    assert violations(events(('09:00','workplace'),('17:20','residence')),CELL)==['presence_travel_overlap']
    assert violations(events(('09:00','workplace'),('12:00','workplace'),('17:00','workplace'),('17:40','residence')),CELL)==[]


def test_start_coverage_and_evidence_required():
    assert 'presence_start' in violations(events(('08:00','residence'),('10:00','workplace')),CELL)
    with pytest.raises(ValueError,match='evidence'):violations([],dict(CELL,user='시간 정보 없음'))
