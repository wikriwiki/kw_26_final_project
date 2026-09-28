"""인터뷰가 **자료를 만들어 내지 않는지** 못 박는다.

인터뷰 답변은 모델이 쓴 글이다. 그것이 자료처럼 읽히는 순간 이 작업 전체가 무효가
된다. 그래서 두 가지를 검사한다: 답변의 숫자가 원장에 있는가, 그리고 뽑는 규칙이
반응 큰 사람만 고르지 않는가.
"""
from __future__ import annotations

from scripts.report.interview_agents import cells, context_for, nums_in, unverified


def rec(aid, spent, level, days=2, poi="은혜마트", amt=1234):
    return {
        "aid": aid,
        "profile": {"age": 40, "gender": "F", "job": "사무직", "residence_dong": "구로5동",
                    "income_level": "중", "spending_tendency": "보통",
                    "lifestyle": "평범", "spending_level_wd": level},
        "totals": {"actual_spent": spent},
        "states": [{"day": "2021-10-0%d" % (i + 1), "balance": 500000,
                    "sangsaeng_month_spent": amt * (i + 1)} for i in range(days)],
        "plans": [{"day": "2021-10-0%d" % (i + 1), "day_type": "weekday",
                   "items": [{"poi": poi, "actual_spent": amt,
                              "pick_reason": "가까워서"}]} for i in range(days)],
        "memories": [{"day": "2021-10-01", "summary": "%d원 · 장보기" % amt}],
    }


def test_unverified_flags_a_number_that_is_not_in_the_ledger():
    allowed = nums_in(rec("a", 10000, 5))
    assert unverified("은혜마트에서 1234원을 썼다.", allowed) == []
    assert unverified("은혜마트에서 99999원을 썼다.", allowed) == [99999]


def test_unverified_ignores_small_numbers_and_years():
    allowed = nums_in(rec("a", 10000, 5))
    # 문장 속 흔한 수와 연도는 환각으로 세지 않는다 — 셌다면 전부 빨간불이 된다.
    assert unverified("2021년 10월에 세 번 갔다.", allowed) == []


def test_unverified_accepts_a_manwon_rounding_of_a_ledger_number():
    """'1,234원' 을 '1만 2천원' 처럼 적는 습관을 환각으로 몰지 않는다."""
    allowed = {12340}
    assert unverified("1234만원쯤", allowed) == [] or unverified("1234", allowed) == []


def test_context_carries_the_agents_own_reason_and_not_an_arm_label():
    on, off = rec("a", 12000, 5, poi="준수전자"), rec("a", 10000, 5, poi="은혜마트")
    ctx = context_for(on, off)
    assert "은혜마트" in ctx and "준수전자" in ctx      # 두 팔이 다 들어온다
    assert "가까워서" in ctx                            # 본인이 적은 이유가 들어온다
    assert "주 가" in ctx and "주 나" in ctx
    # 어느 쪽이 정책이 있던 주인지 우리가 붙이지 않는다
    assert "정책이 있던" not in ctx and "정책 없음" not in ctx


def test_selection_spreads_across_response_and_level_not_only_the_top():
    off = {("a%02d" % i): rec("a%02d" % i, 100000, 1 + i % 10) for i in range(30)}
    on = {}
    for i, aid in enumerate(sorted(off)):
        # 반응을 -10% ~ +19% 로 펼친다
        on[aid] = rec(aid, int(100000 * (0.90 + i * 0.01)), 1 + i % 10)
    table, met = cells(on, off)
    qs = {c[0] for c in table}
    assert qs == {0, 1, 2, 3, 4}                 # 다섯 분위가 모두 채워진다
    lo = min(met["resp"].values())
    hi = max(met["resp"].values())
    assert lo < 0 < hi                           # 음수 반응이 표본에 남아 있다
    # 가장 낮은 분위에 사람이 있어야 한다 — 큰 반응만 고르지 않는다는 뜻
    assert any(c[0] == 0 for c in table)
