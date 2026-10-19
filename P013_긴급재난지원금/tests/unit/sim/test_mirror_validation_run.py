import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from mirror_validation_run import complete_prefix

def test_partial_last_record_is_not_claimed_complete():
    prefix,rows,n=complete_prefix(b'{"i":1}\n{"i":')
    assert prefix==b'{"i":1}\n' and rows==[{'i':1}] and n==5

def test_changed_or_shortened_history_cannot_overwrite_backup():
    with pytest.raises(ValueError):complete_prefix(b'{"i":2}\n',b'{"i":1}\n')
    with pytest.raises(ValueError):complete_prefix(b'',b'{"i":1}\n')

def test_corrupt_complete_record_is_an_error():
    with pytest.raises(ValueError):complete_prefix(b'not-json\n')
