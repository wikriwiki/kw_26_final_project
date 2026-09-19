import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from asset_day_checkpoint import commit_day

CASE = {'cash': 8000, 'wallet_lots': {}, 'offers': {}, 'events': [{'id': 'walk', 'channel': 'offline', 'candidates': []}]}
RAW = json.dumps({'actions':[{'kind':'consume','id':'walk','candidate_id':None,'cash_payment':0,'wallet_spend':{},'reason':'Free activity'}]})


def test_complete_zero_is_committed_but_not_a_missing_citizen(tmp_path):
    with pytest.raises(ValueError, match='Incomplete'): commit_day(tmp_path, day='2026-09-21', roster=['A'], cases={'A': CASE}, rows=[])
    assert not list(tmp_path.iterdir())
    p = commit_day(tmp_path, day='2026-09-21', roster=['A'], cases={'A': CASE}, rows=[{'aid':'A','complete':True,'raw':RAW}])
    saved = json.loads(p.read_bytes())
    assert saved['ledgers']['A']['total_consumption'] == 0 and saved['closing_states']['A']['cash'] == 8000
    with pytest.raises(ValueError, match='overwrite'): commit_day(tmp_path, day='2026-09-21', roster=['A'], cases={'A': CASE}, rows=[{'aid':'A','complete':True,'raw':RAW}])
    assert json.loads(p.read_bytes()) == saved


def test_invalid_raw_funding_cannot_be_committed_as_complete(tmp_path):
    bad = RAW.replace('"cash_payment": 0', '"cash_payment": 10')
    with pytest.raises(ValueError): commit_day(tmp_path, day='2026-09-21', roster=['A'], cases={'A': CASE}, rows=[{'aid':'A','complete':True,'raw':bad}])
    assert not list(tmp_path.iterdir())


def test_state_chain_preserves_cash_and_requires_explicit_transfer_protocol(tmp_path):
    rows = [{'aid':'A','complete':True,'raw':RAW}]
    prior = commit_day(tmp_path, day='2026-09-21', roster=['A'], cases={'A':CASE}, rows=rows)
    with pytest.raises(ValueError, match='discontinuity'):
        commit_day(tmp_path, day='2026-09-22', roster=['A'], cases={'A':dict(CASE,cash=9000)}, rows=rows, previous=prior)
    following = commit_day(tmp_path, day='2026-09-22', roster=['A'], cases={'A':CASE}, rows=rows, previous=prior)
    assert json.loads(following.read_bytes())['previous_sha256']


def test_skipped_days_and_another_arms_snapshot_cannot_start_next_day(tmp_path):
    rows = [{'aid':'A','complete':True,'raw':RAW}]
    prior = commit_day(tmp_path/'off',day='2026-09-21',roster=['A'],cases={'A':CASE},rows=rows)
    with pytest.raises(ValueError,match='consecutive'):
        commit_day(tmp_path/'off',day='2026-09-23',roster=['A'],cases={'A':CASE},rows=rows,previous=prior)
    with pytest.raises(ValueError,match='another'):
        commit_day(tmp_path/'on',day='2026-09-22',roster=['A'],cases={'A':CASE},rows=rows,previous=prior)


def test_write_failure_does_not_publish_a_partial_day(tmp_path, monkeypatch):
    import asset_day_checkpoint
    def interrupted(*args): raise OSError('simulated interruption before atomic replacement')
    monkeypatch.setattr(asset_day_checkpoint.os, 'replace', interrupted)
    with pytest.raises(OSError,match='interruption'):
        commit_day(tmp_path,day='2026-09-21',roster=['A'],cases={'A':CASE},rows=[{'aid':'A','complete':True,'raw':RAW}])
    assert not list(tmp_path.iterdir())


def test_current_protocol_revalidates_choices_and_preserves_day_chain(tmp_path):
    raw=json.dumps({'acquisition_units':{},'purchases':[{'id':'walk','candidate_id':None,'wallet_spend':{}}]})
    rows=[{'aid':'A','complete':True,'raw':raw,'transaction_protocol':'v4'}]
    prior=commit_day(tmp_path,day='2026-09-21',roster=['A'],cases={'A':CASE},rows=rows)
    following=commit_day(tmp_path,day='2026-09-22',roster=['A'],cases={'A':CASE},rows=rows,previous=prior)
    result=json.loads(following.read_bytes())
    assert result['transaction_protocol']=='v4' and result['closing_states']['A']['cash']==8000
    with pytest.raises(ValueError,match='protocol changed'):
        commit_day(tmp_path,day='2026-09-23',roster=['A'],cases={'A':CASE},rows=[dict(rows[0],raw=RAW,transaction_protocol='v1')],previous=following)


def test_unknown_or_mixed_protocols_cannot_enter_checkpoint(tmp_path):
    with pytest.raises(ValueError,match='Unknown'):
        commit_day(tmp_path,day='2026-09-21',roster=['A'],cases={'A':CASE},rows=[{'aid':'A','complete':True,'raw':RAW,'transaction_protocol':'v999'}])
    rows=[{'aid':'A','complete':True,'raw':RAW,'transaction_protocol':'v1'},{'aid':'B','complete':True,'raw':RAW,'transaction_protocol':'v4'}]
    with pytest.raises(ValueError,match='Mixed'):
        commit_day(tmp_path,day='2026-09-21',roster=['A','B'],cases={'A':CASE,'B':CASE},rows=rows)
