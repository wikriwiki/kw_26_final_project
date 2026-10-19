import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from prepare_persona_cohort import select


def test_disjoint_stratified_sampling_preserves_exact_size_and_seed():
    pop={1:['a','b','c','d'],2:['e','f','g','h','i','j']}
    chosen,quota=select(pop,{'a','e'},4,'fixed')
    assert len(chosen)==4 and not set(chosen)&{'a','e'} and quota=={1:2,2:2}
    assert (chosen,quota)==select(pop,{'a','e'},4,'fixed')


def test_overlapping_strata_cannot_silently_duplicate_people():
    with pytest.raises(ValueError,match='multiple strata'):select({1:['a','b'],2:['b','c']},set(),2,'x')
