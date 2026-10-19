from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/sim'))
from planner_run_selection import select_replicate


def fixture():
    cells=[{'aid':'A','case':'m','arm':a} for a in ['off','on']]
    config={'candidates':[{'id':'frozen'}],'seeds':[1,2]}
    rows=[dict(c,replicate=s,variant='frozen',eligible=True,raw=f'{s}:{c["arm"]}') for s in [1,2] for c in cells]
    return rows,cells,config


def test_selected_raw_rows_preserved_exactly_and_other_seed_remains():
    rows,cells,config=fixture();selected=select_replicate(rows,cells,config,2)
    assert selected==rows[2:] and all(a is b for a,b in zip(selected,rows[2:]))
    assert len(rows)==4


@pytest.mark.parametrize('problem',['missing','duplicate','failure_in_other_repeat','wrong_variant','unknown_seed'])
def test_no_cherry_picking_complete_paths(problem):
    rows,cells,config=fixture();seed=2
    if problem=='missing':rows.pop(0)
    elif problem=='duplicate':rows[0]=rows[1]
    elif problem=='failure_in_other_repeat':rows[0]['eligible']=False
    elif problem=='wrong_variant':rows[0]['variant']='unregistered'
    elif problem=='unknown_seed':seed=3
    with pytest.raises(ValueError):select_replicate(rows,cells,config,seed)
