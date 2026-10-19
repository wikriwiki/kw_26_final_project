from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from validate_purchase_probe import factual_errors


def test_explicit_current_constraints_evaluated_without_turning_feasibility_into_facts():
    assert factual_errors({'total_consumption':0},{'total_consumption':10000})==['fact:total_consumption']
    assert factual_errors({'total_consumption':10000},{'total_consumption':10000})==[]
    assert factual_errors({'total_consumption':0},{})==[]


def test_unknown_target_or_invalid_amount_rejected():
    with pytest.raises(ValueError,match='Unknown'):factual_errors({}, {'desired_effect':1})
    with pytest.raises(ValueError,match='Invalid'):factual_errors({}, {'total_consumption':True})
