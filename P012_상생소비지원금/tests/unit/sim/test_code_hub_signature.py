from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from code_hub_signature import build


def test_same_dong_name_cannot_merge_commerce_characteristics():
    rows = [{'code': '11680510', 'l1': '쇼핑', 'n': 30}, {'code': '11620685', 'l1': '건강', 'n': 30}]
    hubs = [{'code': code, 'name': '신사동', 'is_top_hub': True} for code in ['11680510', '11620685']]
    result = build(rows, hubs)['hubs']
    assert result['11680510']['signature'] == '쇼핑' and result['11620685']['signature'] == '건강'


def test_duplicate_group_is_rejected_instead_of_double_counted():
    row = {'code': '11680510', 'l1': '쇼핑', 'n': 30}
    with pytest.raises(ValueError, match='duplicate'): build([row,row], [])
