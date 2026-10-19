import json
from pathlib import Path

import pytest

import report_purchase_repeats as module


def prepare(tmp_path,monkeypatch,*,bad=False,changed=False):
    pairs=[];results={};linked={}
    for seed,off,on in [(1,10,12),(2,2,0)]:
        plan=tmp_path/str(seed);plan.mkdir();purchase=plan/'purchase'
        config={'seeds':[seed],'model':'fixed','sampling':{'temperature':0.2}}
        (plan/'manifest.json').write_text(json.dumps({'config':config,'code_sha256':{'x':'same'},'system_sha256':'same'}))
        pm={'config':{'seeds':[99],'model':'fixed'},'code_sha256':{'x':'same'},'system_sha256':'changed' if changed and seed==2 else 'same'}
        linked[str(purchase)]=({'cells':['same'],'personas':['same']},{'source_sha256':{'manifest':'reference'}},pm,[])
        results[str(purchase)]={'all_matrices_complete':not(bad and seed==2),'mechanisms':{'case':{'matched_contrasts':{
            'replicates':1,'citizens':12,'days':1,'by_seed':[{'metrics':{'total_consumption':{
                'off_mean':off,'on_mean':on,'difference':on-off}}}]}}}}
        pairs.append((plan,purchase))
    monkeypatch.setattr(module,'linked_rows',lambda a,p:linked[str(p)])
    monkeypatch.setattr(module,'report',lambda a,p:results[str(p)])
    return pairs


def test_opposite_repeats_remain_visible_and_people_are_not_multiplied(tmp_path,monkeypatch):
    r=module.compare(prepare(tmp_path,monkeypatch))['mechanisms']['case']
    m=r['metrics']['total_consumption']
    assert r['independent_citizen_count']==12 and r['planner_repeats']==2
    assert m['difference_range']==[-2,2] and m['mean_difference']==0
    assert m['relative_change_of_means']==0 and not m['all_repeat_signs_equal']
    assert not m['range_is_confidence_interval']


def test_failed_repeat_cannot_be_omitted(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='incomplete replicate'):
        module.compare(prepare(tmp_path,monkeypatch,bad=True))


def test_different_prompt_cannot_be_pooled_as_repeat(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match='settings differ'):
        module.compare(prepare(tmp_path,monkeypatch,changed=True))


def test_duplicate_seed_does_not_create_more_evidence(tmp_path,monkeypatch):
    pairs=prepare(tmp_path,monkeypatch)
    with pytest.raises(ValueError,match='Duplicate'):
        module.compare([pairs[0],pairs[0]])
