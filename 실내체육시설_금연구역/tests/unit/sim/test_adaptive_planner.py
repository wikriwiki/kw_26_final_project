import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
import validate_adaptive_planner as runner


CELL={'aid':'A','case':'test','arm':'off','date':'2026-09-21','has_work':False,'zones':['Z'],'user':'오늘 조건'}


def result(ok):return {'eligible':ok,'valid':ok,'raw_valid':ok,'raw':'{}','errors':[] if ok else ['bad'],'attempt_key':'K'}


def test_success_never_retried(tmp_path,monkeypatch):
    calls=[]
    def invoke(*args):calls.append(args);return result(True)
    monkeypatch.setattr(runner,'invoke',invoke)
    monkeypatch.setattr(runner,'feedback',lambda *a,**k:(_ for _ in ()).throw(AssertionError('Success must not need feedback')))
    r=runner.run_cell((1,CELL),{'initial_thinking_tokens':512,'max_shift_minutes':10},None,None,{},tmp_path)
    assert r['eligible'] and not r['repair_attempted'] and len(calls)==1


def test_one_repair_maximum_and_original_failure_preserved(tmp_path,monkeypatch):
    calls=[];(tmp_path/'attempts').mkdir()
    def invoke(*args):calls.append(args);return result(False)
    monkeypatch.setattr(runner,'invoke',invoke);monkeypatch.setattr(runner,'feedback',lambda *a,**k:{'violations':['bad']})
    monkeypatch.setattr(runner,'prefix_for',lambda *a:'PREFIX')
    r=runner.run_cell((1,CELL),{'initial_thinking_tokens':512,'repair_thinking_tokens':2048,'max_shift_minutes':10},None,None,{},tmp_path)
    assert len(calls)==2 and len(r['stages'])==2 and not r['eligible'] and r['repair_attempted']
    assert not r['stages'][0]['eligible'] and list((tmp_path/'attempts').glob('*_feedback.json'))
