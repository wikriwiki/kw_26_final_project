"""2026-10-06 본런 전 프롬프트 점검의 고침을 고정한다.

1) 실적 기간이 그 달 말일보다 먼저 끝나는 캐시백(7일 압축월)은 '이번 실적 기간'으로 적고 남은 일수를 그 끝날까지 센다.
   31일짜리 정책의 표시는 그대로다.
2) 지원금 상태 줄에 배분 기준(소비 규모)을 적지 않는다.
3) v53n = v53 에서 '외출을 너무 보수적으로 줄이면 부자연스럽다' 한 구절만 빠진다.
"""
import importlib
import sys
from datetime import date
from pathlib import Path

SIM_DIR = Path(__file__).resolve().parents[3] / "scripts" / "sim"
if str(SIM_DIR) not in sys.path:
    sys.path.insert(0, str(SIM_DIR))

import dawn_context as dc  # noqa: E402


def _row(until, ratio, cap):
    return {"id": "P012", "type": "cashback", "effective_from": "2021-10-01", "effective_until": until,
            "threshold_ratio": ratio, "rate": 0.10, "cap": cap}


PERSONA = {"daily_wd": 30_000, "daily_we": 30_000}


def test_compressed_cashback_uses_the_period_not_the_month():
    text = dc._format_cashback_status("P012", _row("2021-10-07", 0.232581, 22581), PERSONA, {}, date(2021, 10, 1))
    assert "이번 실적 기간 7일 남음" in text
    assert "이번 실적 기간 최대 22,581원" in text
    assert "이번 달" not in text and "월 최대" not in text and "% 문턱" not in text
    assert "-76.7" not in text
    # 실제 제도의 사실(평소보다 3% 넘게)이 기간 환산으로 보인다: 2분기 월평균 x 7/31 위 3%
    assert "이 기간(7일) 환산" in text and "그보다 3% 많은 이번 실적 기간 문턱" in text


def test_full_month_cashback_wording_is_unchanged():
    text = dc._format_cashback_status("P012", _row("2021-10-31", 1.03, 100000), PERSONA, {}, date(2021, 10, 20))
    assert "3% 문턱" in text and "월 최대 100,000원" in text and "이번 달 12일 남음" in text
    assert "실적 기간" not in text


def test_period_end_helper():
    assert dc._cashback_period_end({"effective_until": "2021-10-07"}) == date(2021, 10, 7)
    assert dc._cashback_period_end({"effective_until": "2021-10-31"}) is None
    assert dc._cashback_period_end({"effective_until": "2020-12-31"}) is None
    assert dc._cashback_period_end({}) is None


def test_v53n_drops_only_the_outing_clause():
    v53 = importlib.import_module("prompts.v53").SYSTEM_PROMPT
    v53n = importlib.import_module("prompts.v53n")
    assert "외출을 너무 보수적으로 줄이면 부자연스럽다" in v53
    assert "외출을 너무 보수적으로 줄이면 부자연스럽다" not in v53n.SYSTEM_PROMPT
    assert "사람들은 평일에도 일상적 외출(점심·간식·간단 쇼핑·운동·약 처방)을 한다." in v53n.SYSTEM_PROMPT
    assert len(v53) - len(v53n.SYSTEM_PROMPT) == len(" 외출을 너무 보수적으로 줄이면 부자연스럽다.")
    assert v53n.STAGE2_NEUTRAL is True
