import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from evidence_contract import evidence_atoms, constrain_evidence, ROUTINE
from planning_contract import endpoint_schedule_schema


def test_evidence_is_verbatim_and_decimal_is_not_split():
    user = '## 개인\n현금은 8만원이다. 독서를 즐긴다.\n비율은 0.25이다.\n외출 anchor 허용값: "zone:A"'
    atoms = evidence_atoms(user)
    assert atoms[0] == ROUTINE
    assert '비율은 0.25이다.' in atoms
    assert all(atom in user for atom in atoms[1:])
    assert not any('anchor 허용값' in atom for atom in atoms)


def test_guard_changes_only_evidence_field_and_preserves_original():
    original = endpoint_schedule_schema(['A'], False, False)
    guarded = constrain_evidence(original, [ROUTINE, '독서를 즐긴다.'])
    assert 'enum' not in original['$defs']['home_event']['properties']['reasoning']
    assert guarded['$defs']['home_event']['properties']['reasoning']['enum'] == [ROUTINE, '독서를 즐긴다.']
    assert guarded['properties'] == original['properties']
    assert guarded['$defs']['any_event']['anyOf'][1]['properties']['anchor'] == {'type': 'string', 'enum': ['zone:A']}
