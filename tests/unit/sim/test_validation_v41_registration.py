"""v41 은 v3 파일럿에서 후보 하나만 바꾼다 — 나머지 설계는 글자까지 같아야 한다.

한 라운드에 한 변수. 표본·seed·날짜·시나리오·팔·온도·토큰·계약 검사 중 하나라도
함께 움직이면 통과율 차이를 후보 탓으로 돌릴 수 없다.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
V3 = json.loads((ROOT / 'data/experiments/validation_v3.json').read_text(encoding='utf-8'))
V41 = json.loads((ROOT / 'data/experiments/validation_v41.json').read_text(encoding='utf-8'))

# 이 라운드가 바꾸기로 선언한 것. 이 목록에 없는 키는 v3 와 같아야 한다.
DECLARED = {'id', 'registered_on', 'candidates', 'phase', 'decision',
            'predicted', 'not_changed'}


@pytest.mark.parametrize('key', sorted(set(V3) - DECLARED))
def test_every_undeclared_field_is_copied_unchanged(key):
    assert V41[key] == V3[key], '%r 이 v3 와 다르다' % key


def test_only_the_candidate_changed():
    assert V3['candidates'] == ['v5', 'v10']
    assert V41['candidates'] == ['v5', 'v40']


def test_the_incumbent_is_re_measured_in_the_same_run():
    """v5 의 예전 128/192 와 비교하지 않는다 — 서버 모델이 그때와 같다는 보장이 없다."""
    assert 'v5' in V41['candidates']


def test_the_registered_gate_is_not_relaxed():
    assert '95' in V41['decision']
    assert V41['primary_metrics'] == V3['primary_metrics']


def test_the_round_predicts_before_it_runs():
    assert V41['predicted']
    assert '기록' in V41['predicted'] or 'recorded' in V41['predicted']


def test_the_truncation_is_left_alone_on_purpose():
    """절단을 함께 고치면 변수가 둘이 된다. 왜 안 고치는지 등록부가 말해야 한다."""
    assert 'truncation' in V41['not_changed']


def test_the_cell_count_is_unchanged():
    def cells(c):
        return c['sample_n'] * len(c['cases']) * len(c['arms']) * len(c['replicate_seeds'])
    assert cells(V41) == cells(V3) == 192


def test_the_runner_still_defaults_to_the_frozen_registration():
    """기본 경로가 바뀌면 과거 런을 재현하는 명령이 조용히 다른 것을 읽는다."""
    src = (ROOT / 'scripts/sim/validate_prompt_v3.py').read_text(encoding='utf-8')
    assert 'default="data/experiments/validation_v3.json"' in src


def test_v3_is_not_edited_in_place():
    """동결된 등록부다. 후보 목록이 바뀌어 있으면 과거 manifest 와 어긋난다."""
    assert V3['candidates'] == ['v5', 'v10']
    assert V3['id'] == 'validation_v3'
