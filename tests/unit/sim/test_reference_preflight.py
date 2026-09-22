import json,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from reference_preflight import inspect_references,REQUIRED

def test_missing_references_cannot_silently_collapse_candidate_geography(tmp_path):
    with pytest.raises(FileNotFoundError): inspect_references(tmp_path)

def test_present_relative_prices_do_not_claim_empirical_absolute_prices(tmp_path):
    for name,key in REQUIRED.items():
        value={key:{'x':1}} if key else {'x':1}
        (tmp_path/name).write_text(json.dumps(value))
    result=inspect_references(tmp_path)
    assert not result['absolute_category_prices_available'] and len(result['limitations'])==2
    assert len(result['files'])==5

def test_empty_hub_data_is_an_error(tmp_path):
    (tmp_path/'hub_catalog.json').write_text('{"hubs":[]}')
    with pytest.raises(ValueError): inspect_references(tmp_path)
