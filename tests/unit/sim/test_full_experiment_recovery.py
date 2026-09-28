"""Integrated contract/regression checks for the paid-run failures and recovery."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from execution_errors import PromptBudgetError, fatal_dispatch_error
from grounded_schema import stage1_format, stage2_format, night_format
from day_resume import read_metric_rows
import checkpoint_control


def test_output_grammar_rejects_cross_order_poi_before_generation():
    from jsonschema import Draft202012Validator
    base = {'json_schema': {'schema': {'properties': {'picks': {
        'type': 'array', 'items': {'type': 'object', 'properties': {
            'order': {'type': 'integer'}, 'poi_id': {'type': 'string'}},
            'required': ['order','poi_id'], 'additionalProperties': False}}}}}}
    response = stage2_format(base, [1,2], {1:[{'poi_id':'A'}],2:[{'poi_id':'B'}]}, {'E0001':'fact'})
    grammar = response['json_schema']['schema']['properties']['picks']['items']
    validator = Draft202012Validator(grammar)
    row = {'order':1,'poi_id':'A','pick_reason':'A is allowed','pick_factor':'distance','evidence_ref':'E0001'}
    assert validator.is_valid(row)
    assert not validator.is_valid(dict(row, poi_id='B'))
    assert not validator.is_valid(dict(row, evidence_ref='E9999'))


def test_stage1_schema_cannot_emit_an_unknown_reference():
    from jsonschema import Draft202012Validator
    schema = stage1_format({'E0001':'assigned fact'})['json_schema']['schema']
    event = {'time':'08:00','anchor':'residence','category':'집','intent':'rest',
             'reasoning':'fact','trigger':'none','evidence_ref':'E0001'}
    validator = Draft202012Validator(schema)
    assert validator.is_valid({'events':[event]})
    assert not validator.is_valid({'events':[dict(event,evidence_ref='E9999')]})
    assert not validator.is_valid({'events':[dict(event,time='25:00')]})


def test_permanent_request_is_not_a_transport_retry():
    assert fatal_dispatch_error(PromptBudgetError('over budget'))
    assert fatal_dispatch_error(SimpleNamespace(status_code=400))
    assert not fatal_dispatch_error(SimpleNamespace(status_code=429))


def test_only_torn_final_append_is_recovered_and_raw_bytes_preserved(tmp_path):
    path=tmp_path/'day.jsonl'
    valid=b'{"aid":"A"}\n'
    tail=b'{"aid":"B","reason":"\xea'
    path.write_bytes(valid+tail)
    assert read_metric_rows(path)==[{'aid':'A'}]
    assert path.read_bytes()==valid
    assert next(tmp_path.glob('*.torn-*')).read_bytes()==tail
    path.write_bytes(b'broken\n'+valid)
    with pytest.raises(json.JSONDecodeError):
        read_metric_rows(path)


def test_graph_checkpoint_is_due_before_twelve_hours_and_at_start(tmp_path,monkeypatch):
    monkeypatch.setenv('SIM_POST_DAY_BACKUP_HOOK',str(tmp_path/'hook.py'))
    now=datetime(2026,9,24,tzinfo=timezone.utc)
    assert checkpoint_control.due(tmp_path,now)
    marker=tmp_path/'recoverable_backup.json'
    marker.write_text(json.dumps({'verified_at_utc':(now-timedelta(hours=9)).isoformat()}))
    assert not checkpoint_control.due(tmp_path,now)
    marker.write_text(json.dumps({'verified_at_utc':(now-timedelta(hours=10)).isoformat()}))
    assert checkpoint_control.due(tmp_path,now)
