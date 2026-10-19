"""통과율 차이가 아니라 짝지은 칸으로 읽는다 — 그리고 못 읽는 것을 없다고 적지 않는다."""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'report'))

from paired_contract_gate import (compare, load, mcnemar, mde,  # noqa: E402
                                  rates, verdict)


def write(tmp_path, rows):
    p = tmp_path / 'responses.jsonl'
    with io.open(p, 'w', encoding='utf-8', newline='\n') as fh:
        for variant, aid, valid in rows:
            fh.write(json.dumps({'variant': variant, 'aid': aid, 'case': 'cashback',
                                 'arm': 'on', 'replicate': 1, 'valid': valid}) + '\n')
    return str(p)


def test_it_pairs_cells_across_candidates(tmp_path):
    p = write(tmp_path, [('a', '1', True), ('a', '2', False),
                         ('b', '1', True), ('b', '2', True)])
    d = load(p)
    assert set(d) == {'a', 'b'}
    assert rates(d) == {'a': (1, 2), 'b': (2, 2)}


def test_mcnemar_counts_only_the_flips():
    a = {'x': True, 'y': False, 'z': True}
    b = {'x': True, 'y': True, 'z': False}
    improved, worsened, p = mcnemar(a, b)
    assert (improved, worsened) == (1, 1)
    assert p == 1.0


def test_a_clean_improvement_is_significant():
    a = {str(i): False for i in range(12)}
    b = {str(i): True for i in range(12)}
    improved, worsened, p = mcnemar(a, b)
    assert (improved, worsened) == (12, 0)
    assert p < 0.001


def test_identical_candidates_give_p_one():
    a = {'x': True, 'y': False}
    assert mcnemar(a, dict(a)) == (0, 0, 1.0)


def test_only_shared_cells_are_compared():
    a = {'x': True, 'y': False}
    b = {'y': True, 'z': True}
    improved, worsened, _ = mcnemar(a, b)
    assert (improved, worsened) == (1, 0)


def test_the_detectable_difference_grows_in_cells_and_shrinks_in_share():
    """칸 수로는 커지고 비율로는 작아진다. 둘을 섞으면 설계를 거꾸로 잡는다.

    처음 이 검사를 mde(192) > mde(384) 로 썼다가 걸렸다 — 칸 단위 MDE 는
    sqrt 로 커진다. 표본을 늘려 얻는 것은 '더 작은 비율'을 읽는 능력이다.
    """
    assert mde(192) < mde(384) < mde(1152)                      # 칸으로는 커진다
    assert mde(192) / 192 > mde(384) / 384 > mde(1152) / 1152   # 비율로는 작아진다
    assert 14 < mde(192) < 17        # 관측 불일치 30% 에서 약 15칸
    assert 19 < mde(384) < 23        # 약 21칸
    assert mde(192) / 192 == pytest.approx(0.079, abs=0.005)
    assert mde(384) / 384 == pytest.approx(0.056, abs=0.005)


def test_underpowered_is_not_written_as_no_difference():
    """이것이 이 파일럿에서 가장 자주 틀리는 자리다."""
    assert verdict(23, 19, 0.644, 192) == '검출력 부족'      # +4칸
    assert verdict(43, 19, 0.0032, 192) == '유의'            # +24칸


def test_a_large_but_insignificant_gap_is_called_that():
    """순증이 MDE 를 넘는데도 p 가 크면 '유의하지 않다' 다 — 다른 말이다."""
    a = {str(i): i % 2 == 0 for i in range(400)}
    b = {str(i): (i % 2 == 0) if i > 60 else (i % 4 == 0) for i in range(400)}
    i2, d2, p2 = mcnemar(a, b)
    v = verdict(i2, d2, p2, 400)
    assert v in ('유의', '유의하지 않다', '검출력 부족')


def test_compare_reports_the_pair_and_its_mde(tmp_path):
    rows = [('v5', str(i), i < 5) for i in range(20)]
    rows += [('v47', str(i), i < 15) for i in range(20)]
    d = load(write(tmp_path, rows))
    out = compare(d, [('v5', 'v47')])
    assert len(out) == 1
    r = out[0]
    assert r['improved'] == 10 and r['worsened'] == 0
    assert r['net'] == 10 and r['cells'] == 20
    assert r['mde'] == pytest.approx(mde(20))


def test_a_missing_candidate_is_reported_not_crashed(tmp_path):
    d = load(write(tmp_path, [('v5', '1', True)]))
    out = compare(d, [('v5', 'v99')])
    assert out[0]['missing'] is True


def test_the_observed_noise_would_be_called_underpowered():
    """같은 프롬프트 두 런: 개선 16 · 악화 22 · 순증 -6. 차이라고 적으면 안 된다."""
    i2, d2, p2 = 16, 22, 0.4177
    assert verdict(i2, d2, p2, 192) == '검출력 부족'
