"""구성 거리는 양으로 이길 수 없어야 한다 — v30 이 그렇게 이겼고 지표는 나빠졌다."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
from composition_distance import CATEGORY, pooled, total_variation

WANT = {'식사': 0.3, '마트': 0.25, '쇼핑': 0.15, '건강': 0.13,
        '교육': 0.07, '편의점': 0.06, '기타': 0.04}


def test_identical_mixes_are_zero():
    assert total_variation(WANT, WANT) == 0.0


def test_buying_more_of_the_same_thing_does_not_help():
    """한 업종만 늘리면 비중은 그대로이므로 거리도 그대로다."""
    one = {'쇼핑': 1.0}
    assert total_variation(one, WANT) == total_variation({'쇼핑': 1.0}, WANT)
    # 금액을 열 배로 사도 비중이 같으면 같은 거리다
    assert total_variation({'쇼핑': 1.0}, WANT) > 0.5


def test_spreading_across_the_assigned_categories_reduces_it():
    narrow = {'쇼핑': 0.78, '식사': 0.22}
    wide = {'식사': 0.30, '마트': 0.24, '쇼핑': 0.16, '건강': 0.12,
            '교육': 0.07, '편의점': 0.07, '기타': 0.04}
    assert total_variation(wide, WANT) < total_variation(narrow, WANT)


def test_a_missing_category_costs_its_whole_share():
    without = dict(WANT); share = without.pop('건강')
    renorm = {k: v / sum(without.values()) for k, v in without.items()}
    assert total_variation(renorm, WANT) >= share / 2


def test_the_map_covers_every_catalog_item_we_price():
    from score_indicators import SECTOR
    assert set(CATEGORY) == set(SECTOR), set(CATEGORY) ^ set(SECTOR)


def test_no_catalog_item_maps_to_a_category_outside_the_profile():
    """프로필에 없는 이름으로 매핑하면 거리가 공짜로 벌어진다."""
    known = {'식사', '디저트', '마트', '편의점', '쇼핑', '미용', '건강',
             '여가', '교육', '기타', '주점'}
    assert set(CATEGORY.values()) <= known


def test_pooling_weights_agents_equally_by_default():
    got = pooled({'a': {'식사': 1.0}, 'b': {'마트': 1.0}})
    assert abs(got['식사'] - 0.5) < 1e-9 and abs(got['마트'] - 0.5) < 1e-9
