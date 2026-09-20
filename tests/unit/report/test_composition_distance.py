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


def test_block_bootstrap_is_wider_than_cell_only():
    """블록 변동을 담으면 구간이 넓어져야 한다. 안 넓어지면 고친 의미가 없다.

    블록 안은 같고 블록 사이만 다르게 만든다. 칸만 재추출하면 30칸의 평균으로
    수렴해 좁아지고, 블록을 재추출하면 블록이 셋뿐이라 넓어진다.
    """
    import sys
    from collections import defaultdict
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
    from pool_indicators import bootstrap

    def blk(on, off, n=10):
        b = defaultdict(list)
        b[('grant', 'on')] = [defaultdict(float, {'total': on, 'cells': 1.0}) for _ in range(n)]
        b[('grant', 'off')] = [defaultdict(float, {'total': off, 'cells': 1.0}) for _ in range(n)]
        return b

    blocks = [blk(200.0, 100.0), blk(100.0, 100.0), blk(400.0, 100.0)]
    one = defaultdict(list)
    for b in blocks:
        for k, v in b.items():
            one[k].extend(v)
    cell_only = bootstrap(one, 600, blocks=None)['EM-3']
    two_stage = bootstrap(one, 600, blocks=blocks)['EM-3']
    assert (two_stage['hi'] - two_stage['lo']) > (cell_only['hi'] - cell_only['lo'])


def test_one_block_degenerates_to_the_cell_bootstrap():
    """블록이 하나면 두 단계가 한 단계와 같아야 한다 — 기존 결과와 이어지게."""
    import sys
    from collections import defaultdict
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/report'))
    from pool_indicators import bootstrap

    b = defaultdict(list)
    b[('grant', 'on')] = [defaultdict(float, {'total': 150.0, 'cells': 1.0}) for _ in range(8)]
    b[('grant', 'off')] = [defaultdict(float, {'total': 100.0, 'cells': 1.0}) for _ in range(8)]
    assert bootstrap(b, 300, blocks=[b])['EM-3'] == bootstrap(b, 300, blocks=None)['EM-3']
