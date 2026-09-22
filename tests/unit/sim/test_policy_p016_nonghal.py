"""P016 농축산물 할인쿠폰 — 코드를 고치지 않고 붙는가, 그리고 정답이 새지 않는가.

이 정책은 **일반화 시험**이다. 기전(`sector_voucher`)·적격 평가기(`eligibility`)·
라벨 레지스트리가 이미 있는 것만으로 새 정책이 붙어야 한다. 정책마다 판정 코드를
새로 쓰면 소비 프롬프트를 일반화해도 배관에서 1:1 결합이 되살아난다.

설계는 팀이 노션 「농축산물 할인지원」에 사전 확정했다. 여기서 지키는 것은 둘이다.

  ① 새 코드 없이 붙는가 — 기전·적격·라벨이 기존 것으로 처리되는가
  ② 정답이 새지 않는가 — 보고서 수치(+6.957 · +11.6 · +4.6 · +1.9)와
     방향 어휘가 모델이 보는 글에 들어가지 않는가
"""
from pathlib import Path
import io
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts' / 'sim'))

import dawn_context as dc                      # noqa: E402
from mechanisms import label, sector_voucher   # noqa: E402

P016 = json.load(io.open(ROOT / 'data/neo4j_load/policies/P016.json', encoding='utf-8'))
SCORING = json.load(io.open(ROOT / 'data/experiments/scoring_table.json', encoding='utf-8'))


def rendered():
    row = dict(P016)
    row['from_'] = P016['effective_from']
    row['until_'] = P016['effective_until']
    row['regions'] = P016['target_districts']
    row['target_l1s'] = P016['benefit_categories']
    return dc._format_policy_facts([row]) + '\n' + sector_voucher.status('P016', P016, {}, {})


# 보고서의 값과, 행동의 방향을 지시하는 말
ANSWER_LEAKS = ('6.957', '11.6', '4.6%', '1.9%p', '늘어난다', '증가한다', '더 쓴다', '소비가 늘')


@pytest.mark.parametrize('leak', ANSWER_LEAKS)
def test_the_report_numbers_never_reach_the_model(leak):
    assert leak not in rendered(), '정답지가 프롬프트로 샌다: %r' % leak


def test_design_notes_stay_out_of_the_prompt():
    """notes 에는 설계 의도가 적혀 있다. 그것이 새면 정답을 주는 것이다."""
    assert '홀드아웃' in P016['notes']          # notes 에는 있고
    assert '홀드아웃' not in rendered()          # 프롬프트에는 없다
    assert '결과보고서' not in rendered()


def test_it_attaches_with_no_new_code():
    """기전·라벨이 기존 레지스트리로 처리된다."""
    assert P016['type'] == 'sector_voucher'
    got = label(P016['type'], P016['name'])
    assert got == '[업종 한정 할인권] 농축산물 할인쿠폰'


def test_the_generic_eligibility_engine_accepts_the_spec():
    """정책 전용 판정 코드를 쓰지 않는다 — include/subs 로 끝나야 한다."""
    from eligibility import Rules
    r = Rules(P016['eligibility'])
    assert r is not None
    spec = P016['eligibility']
    assert spec['mode'] == 'include'
    assert set(spec['include']['subs']) == {'청과', '정육', '슈퍼마켓', '식료품'}
    # 제외 업종은 목록에 없어서 제외된다 — 따로 적지 않는다
    for out in ('수산', '종합소매', '음료소매'):
        assert out not in spec['include']['subs']


def test_the_personal_cap_reaches_the_model():
    """1인 1만원 한도가 이 정책의 핵심 제약이다. 모델이 그것을 봐야 한다."""
    line = sector_voucher.status('P016', P016, {}, {})
    assert '10,000원' in line and '한도' in line


def test_the_rule_is_stated_as_immediate_discount():
    """지갑·캐시백과 갈리는 지점이다 — 결제 그 자리에서 깎인다."""
    txt = rendered()
    assert '20%' in txt
    assert '1만원' in txt or '10,000원' in txt


def test_indicators_are_registered_before_any_run():
    """돌리기 전에 지표가 채점표에 있어야 한다. 사후 추가는 자유도를 늘린다."""
    blk = SCORING['P016']
    ids = [i['id'] for i in blk['indicators']]
    assert ids == ['C1', 'C2', 'C3', 'E1']
    assert blk['prereg'].startswith('2026-09-22')
    assert 'E1' in blk['not_scored']


def test_the_proxy_gap_is_written_down():
    """보고서는 상품 단위, 우리는 POI 단위다. 그 차이를 적어 두지 않으면 잘못 읽는다."""
    blk = SCORING['P016']
    assert '상품 단위' in blk['proxy_note']
    assert 'POI' in blk['proxy_note']
    assert 'DID' in blk['unit_note']


def test_the_window_avoids_a_regime_change():
    """두 창이 거리두기 같은 단계 안에 있어야 규제 변화가 효과로 새지 않는다."""
    assert '2020-11-10' in SCORING['P016']['window']
    assert '2020-11-17' in SCORING['P016']['window']
    assert '1.5단계' in SCORING['P016']['window']


def test_frozen_policies_are_untouched():
    """P010 은 동결이다. 이 작업이 그 렌더를 건드리면 안 된다."""
    assert label('grant', '민생회복 소비쿠폰 1차') == '[지원금] 민생회복 소비쿠폰 1차'
