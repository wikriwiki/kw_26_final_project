"""Two cohorts may be pooled only if they were built under the same rules."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from merge_sources import merge  # noqa: E402


def src(aids, fingerprint='aaa', rule='keep', personas=None, before=None):
    return {
        'action_adapter': {'rule': rule, 'source_sha256': fingerprint},
        'case_filter': {'kept': ['distancing'], 'dropped': ['grant'],
                        'cells_before': before if before is not None else len(aids) * 8,
                        'cells_after': len(aids) * 2},
        'personas': personas if personas is not None else [{'id': a} for a in aids],
        'cells': [{'aid': a, 'case': 'distancing', 'arm': arm, 'user': 'u'}
                  for a in aids for arm in ('off', 'on')],
    }


def test_it_pools_the_citizens():
    out = merge([src('ab'), src('cd')], ['a.json', 'b.json'])
    assert {c['aid'] for c in out['cells']} == set('abcd')
    assert out['cohort_merge']['citizens'] == 4
    assert out['cohort_merge']['cells'] == 8


def test_differing_fingerprints_do_not_block_the_merge():
    """Two runs of one pipeline over different people differ exactly here."""
    out = merge([src('ab', fingerprint='111'), src('cd', fingerprint='222')], ['a', 'b'])
    assert out['cohort_merge']['citizens'] == 4


def test_differing_counts_do_not_block_the_merge():
    """A filter that dropped 960 cells and one that dropped 480 applied the same rule."""
    out = merge([src('ab', before=960), src('cd', before=480)], ['a', 'b'])
    assert out['cohort_merge']['citizens'] == 4


def test_a_differing_filter_rule_still_stops_the_merge():
    a, b = src('ab'), src('cd')
    b['case_filter']['kept'] = ['grant']
    with pytest.raises(ValueError, match='same rules'):
        merge([a, b], ['a', 'b'])


def test_a_differing_rule_stops_the_merge():
    with pytest.raises(ValueError, match='same rules'):
        merge([src('ab', rule='keep'), src('cd', rule='drop')], ['a', 'b'])


def test_a_shared_citizen_is_counted_once():
    out = merge([src('ab'), src('bc')], ['a', 'b'])
    assert out['cohort_merge']['citizens'] == 3
    dropped = out['cohort_merge']['duplicate_cells_dropped']
    assert {d['aid'] for d in dropped} == {'b'}
    assert len(dropped) == 2


def test_the_first_source_wins_a_duplicate():
    a, b = src('ab'), src('bc')
    for c in b['cells']:
        c['user'] = 'second'
    out = merge([a, b], ['a', 'b'])
    kept = [c for c in out['cells'] if c['aid'] == 'b']
    assert all(c['user'] == 'u' for c in kept)


def test_personas_are_deduplicated_too():
    out = merge([src('ab'), src('bc')], ['a', 'b'])
    ids = [p['id'] for p in out['personas']]
    assert sorted(ids) == ['a', 'b', 'c'] and len(ids) == len(set(ids))


def test_both_arms_stay_balanced():
    out = merge([src('ab'), src('cd')], ['a', 'b'])
    arms = [c['arm'] for c in out['cells']]
    assert arms.count('off') == arms.count('on') == 4


def test_the_merge_records_what_it_joined():
    out = merge([src('ab'), src('cd')], ['sixty.json', 'onetwenty.json'])
    names = [s['name'] for s in out['cohort_merge']['sources']]
    assert names == ['sixty.json', 'onetwenty.json']
    assert all(s['citizens'] == 2 for s in out['cohort_merge']['sources'])


def test_one_source_is_not_a_merge():
    with pytest.raises(ValueError, match='at least two'):
        merge([src('ab')], ['a'])
