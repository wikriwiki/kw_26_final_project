import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
import bounded_reasoning as b

def test_forced_boundary_is_explicit_and_answer_is_constrained(monkeypatch):
    calls=[];saved=[]
    def post(base,payload,timeout):
        calls.append(payload)
        if len(calls)==1:return {'text':'short deliberation','meta_info':{'finish_reason':{'type':'length'},'completion_tokens':10}}
        return {'text':'{}','meta_info':{'finish_reason':{'type':'stop'}}}
    monkeypatch.setattr(b,'post',post)
    first,second=b.run(prefix='P<think>\n',schema={'type':'object'},base='x',seed=7,thinking_tokens=10,
                       answer_tokens=20,sampling={'temperature':1},timeout=60,on_deliberation=saved.append)
    assert first['forced_reasoning_boundary'] and saved==[first]
    assert calls[1]['text']=='P<think>\nshort deliberation\n</think>\n\n'
    assert calls[1]['sampling_params']['max_new_tokens']==20
    assert calls[1]['sampling_params']['json_schema']=='{"type": "object"}'

def test_deliberation_is_saved_before_answer_failure(monkeypatch):
    count=0;saved=[]
    def post(*args):
        nonlocal count
        count+=1
        if count==1:return {'text':'done','meta_info':{'finish_reason':{'type':'stop','matched':'</think>'}}}
        raise RuntimeError('interrupted')
    monkeypatch.setattr(b,'post',post)
    with pytest.raises(RuntimeError):
        b.run(prefix='<think>',schema={},base='x',seed=1,thinking_tokens=10,answer_tokens=10,sampling={},timeout=60,on_deliberation=saved.append)
    assert len(saved)==1 and not saved[0]['forced_reasoning_boundary']

def test_unexpected_eos_is_not_accepted_as_a_reasoning_boundary(monkeypatch):
    monkeypatch.setattr(b,'post',lambda *a:{'text':'','meta_info':{'finish_reason':{'type':'stop','matched':42}}})
    with pytest.raises(ValueError):
        b.run(prefix='<think>',schema={},base='x',seed=1,thinking_tokens=10,answer_tokens=10,sampling={},timeout=60,on_deliberation=lambda x:None)
