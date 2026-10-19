import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from persona_lineage import read_complete, restore_baseline, audit_persona
from prepare_baseline_persona_context import replace_persona_section


def fixture():
    base = {'agent_id': 'A', 'personal': {'job': '기사', 'gender': 'M', 'age_group': '50대'},
            'residence': {'dong_code': 'D'}, 'personality': {'lifestyle': '가족 중심'},
            'spending': {'daily_spending_weekday': 12000}}
    frozen = {'id': 'A', 'job': '기사', 'gender': 'M', 'age_group': '50대', 'home_dong_code': 'D',
              'lifestyle': '다른 인물', 'nv_summary': '다른 인물의 취미', 'daily_wd': 23000, 'work_poi_id': None}
    return base, frozen


def test_overlay_preserves_quantitative_and_geographic_state_and_audits_removed_fields():
    base, original = fixture()
    restored, audit = restore_baseline([original], {'A': base})
    assert restored[0] == dict(original, lifestyle='가족 중심', nv_summary=None)
    assert original['lifestyle'] == '다른 인물'
    assert audit[0]['quarantined'] == {'lifestyle': '다른 인물', 'nv_summary': '다른 인물의 취미'}
    assert audit[0]['baseline_record']['spending']['daily_spending_weekday'] == 12000


@pytest.mark.parametrize('field', ['job', 'gender', 'age_group', 'home_dong_code'])
def test_overlay_rejects_identity_conflict(field):
    base, original = fixture()
    original[field] = 'unmatched'
    with pytest.raises(ValueError, match='mismatch'):
        restore_baseline([original], {'A': base})


def test_missing_and_duplicate_people_cannot_be_silently_dropped():
    base, original = fixture()
    with pytest.raises(ValueError, match='Missing'):
        restore_baseline([original], {})
    with pytest.raises(ValueError, match='Duplicate'):
        restore_baseline([original, original], {'A': base})


def test_location_mapping_is_explicit_and_audited_never_silently_corrected():
    base, original = fixture()
    original['home_dong_code'] = 'mapped_POI_dong'
    with pytest.raises(ValueError, match='residence mismatch'):
        restore_baseline([original], {'A': base})
    restored, audit = restore_baseline([original], {'A': base}, allow_mapped_residence=True)
    assert restored[0]['home_dong_code'] == 'mapped_POI_dong'
    assert audit[0]['residence_remapped'] is True
    assert audit[0]['baseline_record']['residence']['dong_code'] == 'D'


def test_malformed_and_duplicate_source_rows_are_not_skipped(tmp_path):
    p = tmp_path / 'input.jsonl'
    p.write_text('{"agent_id":"A"}\nBROKEN\n')
    with pytest.raises(ValueError):
        read_complete(p, jsonl=True)
    p.write_text('{"agent_id":"A"}\n{"agent_id":"A"}\n')
    with pytest.raises(ValueError, match='duplicate'):
        read_complete(p, jsonl=True)


def test_stat_size_mismatch_is_refused(tmp_path, monkeypatch):
    p = tmp_path / 'input.json'
    p.write_text('[{"agent_id":"A"}]')
    monkeypatch.setattr(Path, 'read_bytes', lambda self: b'[]')
    with pytest.raises(ValueError, match='Incomplete'):
        read_complete(p)


def test_audit_does_not_equate_uuid_match_with_numeric_consistency():
    base, original = fixture()
    base['_match'] = {'nvidia_uuid': 'new'}
    base['personality']['lifestyle'] = original['lifestyle']
    graph = {'id': 'A', 'lifestyle': original['lifestyle'], 'persona_uuid': 'old', 'nvidia_uuid_v2': 'new'}
    result = audit_persona(original, graph, {'A': base}, source_id='A')
    assert result['rich_and_lifestyle_uuid_conflict']
    assert result['lifestyle_exact_match'] and result['lifestyle_uuid_exact_match']
    assert not result['daily_wd_exact_match']
    assert audit_persona(original, graph, {}, source_id='B')['source_status'] == 'unresolved'


def test_only_persona_block_changes_and_ambiguous_sections_fail():
    text = '## 사회 배경\nfacts\n\n## 시민 정보\nold\n\n## 개인별 정책 상태\npolicy\nmenu'
    assert replace_persona_section(text, 'new') == text.replace('\nold\n', '\nnew\n')
    with pytest.raises(ValueError, match='Ambiguous'):
        replace_persona_section(text + '\n## 시민 정보\nextra', 'new')
