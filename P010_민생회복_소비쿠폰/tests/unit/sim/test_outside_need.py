"""집 밖 볼일은 시민 자신의 구성에서 기계적으로 고른다 — 정답지를 보지 않는다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/sim'))
from add_outside_need import EVIDENCE, SERVICE, add_evidence_line, chosen_errand


def cell_with(profile):
    return {'aid': 'A', 'user': f'소비: 평일 1원\n합성 시민에게 부여한 업종별 지출 구성: {profile}\n'}


def test_it_picks_the_largest_service_share():
    got, share = chosen_errand(cell_with('마트 50%, 건강 27%, 교육 5%'))
    assert got == 'groceries' and share == 50


def test_a_citizen_with_no_outside_share_gets_no_errand():
    """부여된 비중이 0인데 볼일을 만들면 그건 지어낸 것이다."""
    assert chosen_errand(cell_with('주거 60%, 통신 40%')) == (None, 0)


def test_only_home_impossible_activities_are_candidates():
    """집에서 끝낼 수 있는 것을 고르면 하루가 밖에 나갈 이유가 여전히 없다.

    2026-09-22: 후보를 넷에서 열로 넓혔다. 처음의 넷은 내가 쓴 제한이었고 데이터가
    요구한 것이 아니었다 — 예순 중 여덟이 그 때문에 볼일을 못 받았다. 넓힌 뒤에도
    조건은 하나뿐이다: 집에서 끝낼 수 있는 활동은 후보가 될 수 없다.
    """
    assert set(SERVICE) == {'hair', 'health_goods', 'leisure_service', 'education_service',
                            'meal_dine_in', 'groceries', 'convenience', 'cafe_dine_in',
                            'other_service', 'bar'}
    for a in SERVICE:
        assert not a.startswith('home_')
        assert 'delivery' not in a and 'online' not in a


def test_forcing_a_category_is_recorded_as_assigned_not_observed():
    """전원 지정판은 규칙이 아니라 가정이다. 출처가 그렇게 말해야 한다."""
    import add_outside_need as m
    assert m.chosen_errand(cell_with('마트 80%'), 'meal_dine_in') == ('meal_dine_in', 0)


def test_the_choice_does_not_depend_on_any_answer_key_category():
    """정답지는 음식점·카페·소매·이미용·가전·상권유형을 잰다. 고르는 규칙은 그것을 안 본다."""
    import inspect as _inspect
    import add_outside_need as m
    src = _inspect.getsource(m.chosen_errand) + _inspect.getsource(m.assigned_service_shares)
    for word in ('음식점', '카페', '소매', '가전', '상권', '적립'):
        assert word not in src, word


def test_the_evidence_line_says_only_that_it_cannot_be_done_at_home():
    for word in ('사라', '나가라', '늘려', '더 쓰', '마트', '음식점'):
        assert word not in EVIDENCE, word
    assert '집에서 끝낼 수 없고' in EVIDENCE


def test_the_evidence_line_is_not_added_twice():
    user = f'{"## 정책 적용 전 고정한 오늘의 조건"}\n실험 가정: 기존.\n{{"a":1}}\n\n## 오늘\n'
    once = add_evidence_line(user)
    assert once.count(EVIDENCE) == 1
    assert add_evidence_line(once) == once
