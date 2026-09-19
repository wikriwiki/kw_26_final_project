"""Concise planner with explicit distinction between observed facts and choices."""
from .v15 import SYSTEM_PROMPT as BASE, format_dawn_blocks

SYSTEM_PROMPT = BASE.replace(
    '집 체류시간은 재택근무와 다르다.',
    '과거 업종별 지출 비중은 소비 경향이며 오늘의 필수 구매·질병·약 복용·예약 사실이 아니다.\n'
    'reasoning에서 입력에 적힌 사실과 오늘 스스로 선택하는 행동을 구별한다. 취향에 따른 새 활동은 선택으로\n'
    '표현하되, 근거 없이 반드시 필요한 구매나 자주 했던 경험이라고 주장하지 않는다.\n'
    '집 체류시간은 재택근무와 다르다.'
)
