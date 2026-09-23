"""채점 JSON → 채점표 블록 생성기가 **손으로 옮긴 것과 같은가.**

라운드가 끝날 때마다 채점 JSON 의 수치를 채점표에 옮겨 적었다. 그 자리에서
자릿수를 한 번 틀리면 표·그림·판정이 모두 틀린 수를 말하고, 원본과 대조하기
전에는 아무도 모른다.

이 시험은 **이미 손으로 옮겨 커밋한 블록**(라운드3 v5)을 생성기로 다시 만들어
맞대 본다. 같지 않으면 둘 중 하나가 틀린 것이고, 어느 쪽이든 알아야 한다.
"""
from pathlib import Path
import importlib.util
import io
import json

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _gen():
    spec = importlib.util.spec_from_file_location(
        'stb', ROOT / 'scripts/report/score_to_block.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_it_reproduces_the_hand_transcribed_round3_block():
    m = _gen()
    score_path = ROOT / 'output/answerkey/score_r3_v5.json'
    if not score_path.exists():
        pytest.skip('채점 파일이 저장소에 없다')
    score = json.loads(io.open(score_path, encoding='utf-8').read())
    got = m.build(score, '', '')
    sc = json.loads(io.open(ROOT / 'data/experiments/scoring_table.json',
                            encoding='utf-8').read())
    want = sc['DISTANCING_2020']['result_r3_v5']
    for iid in ('DS-1', 'DS-2', 'DS-4'):
        for field in ('mean', 'pct', 'base', 'ci', 'n', 'hit'):
            assert got[iid][field] == want[iid][field], (
                '%s.%s 가 다르다 — 생성 %r · 채점표 %r'
                % (iid, field, got[iid][field], want[iid].get(field)))


def test_shares_keep_four_decimals_and_money_stays_whole():
    """몫과 금액은 자릿수 규칙이 다르다. 섞이면 몫이 0 으로 뭉개진다."""
    m = _gen()
    assert m._round(0.123456, 0.4) == 0.1235      # 몫
    assert m._round(1234.6, 30000) == 1235        # 금액
    assert m._round(-0.00123, 0.766) == -0.0012


def test_rank_rows_keep_their_verdict_without_inventing_numbers():
    """순위·관측부족 지표에 없는 수를 만들어 넣지 않는다."""
    m = _gen()
    got = m.build({'results': [
        {'id': 'DS-3', 'hit': True, 'got': '쇼핑+마트 +3.1% vs 식사 -7.7%'},
        {'id': 'DS-6', 'hit': None, 'got': '관측부족'},
    ]}, '', '')
    assert got['DS-3'] == {'hit': True, 'note': '쇼핑+마트 +3.1% vs 식사 -7.7%'}
    assert got['DS-6']['hit'] is None
    assert 'mean' not in got['DS-3'] and 'ci' not in got['DS-6']
