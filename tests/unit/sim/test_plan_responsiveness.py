"""반응성은 영점 위에서만 뜻이 있다 — 그 규율을 코드로 지킨다.

모델은 확률적이라 같은 요청을 두 번 해도 계획이 달라진다. 그래서 off/on 거리
하나만으로는 "제도를 읽었다"고 말할 수 없다. 같은 팔을 다시 뽑은 거리가 영점이다.
"""
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from plan_responsiveness import arm_contrast, coarse, distance, events, floor_contrast, load


def rec(aid, case, arm, evs, *, replicate=1, valid=True):
    return {'aid': aid, 'date': '2020-05-14', 'case': case, 'arm': arm,
            'replicate': replicate, 'valid': valid,
            'execution_plan': {'events': evs}}


def ev(time, act, anchor='home', cat='집', channel=None):
    return {'time': time, 'activity_id': act, 'anchor': anchor,
            'category': cat, 'purchase_channel': channel}


def write(tmp_path, name, records):
    p = tmp_path / name
    with io.open(p, 'w', encoding='utf-8') as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    return str(p)


def test_identical_plans_are_distance_zero():
    a = rec('A', 'grant', 'off', [ev('08:00', 'x'), ev('12:00', 'y')])
    assert distance(events(a), events(a)) == 0.0


def test_a_moved_time_is_total_replacement_strictly_but_no_change_coarsely():
    """엄격 거리가 30분 이동을 완전 교체로 세는 것이 바로 시각둔감 거리를 둔 이유다."""
    a = rec('A', 'grant', 'off', [ev('08:00', 'x')])
    b = rec('A', 'grant', 'on', [ev('08:30', 'x')])
    assert distance(events(a), events(b)) == 1.0
    assert distance(coarse(a), coarse(b)) == 0.0


def test_load_refuses_to_silently_merge_two_replicates(tmp_path):
    """seed 를 섞으면 이 도구가 재려는 비교 자체가 덮어써진다."""
    path = write(tmp_path, 'mix.jsonl', [
        rec('A', 'grant', 'off', [ev('08:00', 'x')], replicate=1),
        rec('A', 'grant', 'off', [ev('09:00', 'y')], replicate=2)])
    with pytest.raises(ValueError, match='select one with path@seed'):
        load(path)
    assert len(load(path + '@1')) == 1


def test_failed_rows_are_dropped_not_counted_as_unchanged(tmp_path):
    """실패한 행을 빈 계획으로 세면 '안 움직였다'가 공짜로 생긴다."""
    path = write(tmp_path, 'f.jsonl', [
        rec('A', 'grant', 'off', [ev('08:00', 'x')]),
        rec('A', 'grant', 'on', [], valid=False)])
    rows = load(path)
    assert ('A', '2020-05-14', 'grant', 'on') not in rows
    assert arm_contrast(rows) == {}


def test_an_unpaired_arm_contributes_nothing(tmp_path):
    path = write(tmp_path, 'u.jsonl', [rec('A', 'grant', 'off', [ev('08:00', 'x')])])
    assert arm_contrast(load(path)) == {}


def test_floor_compares_the_same_arm_across_seeds(tmp_path):
    a = write(tmp_path, 'a.jsonl', [rec('A', 'grant', 'off', [ev('08:00', 'x')], replicate=1)])
    b = write(tmp_path, 'b.jsonl', [rec('A', 'grant', 'off', [ev('08:00', 'z')], replicate=2)])
    got = floor_contrast(load(a), load(b))
    assert got['grant'][0][0] == 1.0


def test_purchase_slots_ignore_non_buying_events():
    a = rec('A', 'grant', 'off', [ev('08:00', 'x'), ev('12:00', 'buy', channel='offline')])
    b = rec('A', 'grant', 'on', [ev('09:00', 'w'), ev('12:00', 'buy', channel='offline')])
    from plan_responsiveness import buying
    assert distance(buying(a), buying(b)) == 0.0
    assert distance(coarse(a), coarse(b)) > 0.0


def test_reachability_needs_an_offline_purchase_at_a_zone(tmp_path):
    """집·직장에서만 사면 제한 지갑은 결제 시점에 제시되지 않는다."""
    from plan_responsiveness import wallet_reachable
    home = rec('A', 'grant', 'on', [ev('20:00', 'home_online_goods', 'residence', channel='online')])
    assert wallet_reachable(home, '11290725') is False
    out = rec('A', 'grant', 'on', [ev('18:00', 'groceries', 'zone:11290580', channel='offline')])
    assert wallet_reachable(out, '11290725') is True


def test_reachability_applies_the_case_specific_zone_width():
    """지원금은 시 단위(2자리), 지역화폐는 자치구 단위(5자리)다."""
    from plan_responsiveness import wallet_reachable
    far = [ev('18:00', 'groceries', 'zone:11680521', channel='offline')]
    assert wallet_reachable(rec('A', 'grant', 'on', far), '11290725') is True
    assert wallet_reachable(rec('A', 'local_voucher', 'on', far), '11290725') is False


def test_a_zone_visit_without_a_purchase_does_not_count():
    from plan_responsiveness import wallet_reachable
    walk = rec('A', 'grant', 'on', [ev('18:00', 'walk', 'zone:11290580', cat='여가')])
    assert wallet_reachable(walk, '11290725') is False


def test_reachability_says_which_rule_it_used(tmp_path):
    from plan_responsiveness import reachability
    path = write(tmp_path, 'r.jsonl', [
        rec('A', 'grant', 'on', [ev('18:00', 'groceries', 'zone:11290580', channel='offline')]),
        rec('A', 'grant', 'off', [ev('20:00', 'home_meal', 'residence')])])
    rows = load(path)
    loose = reachability(rows)
    assert loose['_all']['share'] == 0.5 and 'superset' in loose['_all']['rule']
    exact = reachability(rows, {'A': '11290725'})
    assert exact['grant']['on']['share'] == 1.0 and exact['grant']['off']['share'] == 0.0
