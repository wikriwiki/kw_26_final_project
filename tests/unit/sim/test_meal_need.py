"""끼니 필요는 세탁과 같은 구조여야 하고, 없으면 기존과 글자 하나 달라선 안 된다.

home_meal 은 지금 채널도 자원 소비도 없다 — 집밥이 공짜라 하루가 밖에 나갈 이유가
없다. 세탁이 쓰는 구조(자원 · 소비 · 보충 견적 · 선택적 필요)를 그대로 끼니에
적용한다. 정답지가 재는 업종을 보고 고른 것이 하나도 없어야 한다.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from prepare_daily_conditions import meal_options, prepare

USER_TEXT = '시민 정보\n\n## 오늘\n날짜는 입력에 있다.'


def source_with(job='전기설비 기술사', work=True):
    persona = {'id': 'A', 'job': job, 'home_dong_code': '11290725'}
    if work:
        persona['work_poi_id'] = 'W1'
    return {'personas': [persona],
            'cells': [{'aid': 'A', 'date': '2020-05-14', 'case': 'grant', 'arm': 'off',
                       'user': USER_TEXT, 'zones': ['11290725']}]}


def profiles(job='전기설비 기술사', **extra):
    base = {'job': job, 'detergent_doses': 2,
            'duty': {'kind': 'none', 'start': '09:00', 'end': '17:00'}}
    base.update(extra)
    return {'A': base}


def conditions_of(result):
    return result['cells'][0]['daily_conditions']


def test_without_the_profile_key_nothing_changes():
    """대조군이 기존과 같아야 A/B 가 성립한다."""
    plain = conditions_of(prepare(source_with(), profiles()))
    assert 'meal_stock' not in plain['resources']
    assert [n['id'] for n in plain['needs']] == ['laundry']
    assert 'home_meal' not in plain['activity_consumption']
    assert 'quote:groceries' not in plain['quote_receipts']


def test_the_meal_stock_mirrors_the_laundry_structure():
    c = conditions_of(prepare(source_with(), profiles(meal_stock=0)))
    assert set(c['resources']['meal_stock']) == set(c['resources']['detergent_dose'])
    assert c['activity_consumption']['home_meal'] == {'meal_stock': 1}
    assert c['quote_receipts']['quote:groceries'] == {'meal_stock': 3}
    assert c['needs'][-1]['id'] == 'meals'


def test_eating_at_home_stops_being_free():
    c = conditions_of(prepare(source_with(), profiles(meal_stock=0)))
    assert 'home_meal' in c['activity_consumption'], '집밥이 공짜면 나갈 이유가 없다'


def test_the_need_is_optional_like_the_laundry_one():
    c = conditions_of(prepare(source_with(), profiles(meal_stock=2)))
    meals = c['needs'][-1]
    assert meals['mandatory'] is False
    assert meals['desired_count'] == 2


def test_the_options_include_home_and_away_but_not_other_categories():
    c = conditions_of(prepare(source_with(), profiles(meal_stock=0)))
    opts = set(c['needs'][-1]['fulfilled_by'])
    assert 'home_meal' in opts
    assert {'meal_dine_in', 'meal_takeaway'} <= opts
    for outside in ('groceries', 'hair', 'shopping', 'cafe_dine_in'):
        assert outside not in opts, outside


def test_a_student_does_not_get_the_workplace_options():
    worker = set(conditions_of(prepare(
        source_with(), profiles(meal_stock=0)))['needs'][-1]['fulfilled_by'])
    student = set(conditions_of(prepare(
        source_with(job='고등학생', work=False),
        profiles(job='고등학생', meal_stock=0)))['needs'][-1]['fulfilled_by'])
    assert student <= worker


def test_no_answer_key_sector_is_singled_out():
    """'음식점을 사라'가 아니라 '끼니를 든다'여야 한다."""
    c = conditions_of(prepare(source_with(), profiles(meal_stock=0)))
    text = c['needs'][-1]['description'] + ' '.join(c['assumptions'])
    for word in ('음식점', '카페', '외식', '늘려', '더 쓰', '많이'):
        assert word not in text, word
    assert '집에서 들 수도 있고 밖에서 들 수도 있다' in c['needs'][-1]['description']


def test_meal_options_reads_the_cells_own_catalog(monkeypatch):
    import action_plan_contract
    monkeypatch.setattr(action_plan_contract, 'catalog',
                        lambda cell: {'home_meal': {}, 'meal_dine_in': {}, 'office_meal': {},
                                      'home_delivery': {}, 'groceries': {}, 'hair': {}})
    got = meal_options({}, has_work=True)
    assert set(got) == {'home_meal', 'meal_dine_in', 'office_meal', 'home_delivery'}
