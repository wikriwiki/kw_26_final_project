"""Verbatim input evidence plus an open personal choice, tested as a new protocol."""
from .v17 import SYSTEM_PROMPT as BASE, format_dawn_blocks

_OLD = '''intent는 오늘 하려는 활동을 짧게 쓴다. reasoning은 그 활동에 직접 관련된 입력 근거와
내 선택을 한 문장으로 쓴다. 입력을 과장·재해석하거나 새 사실을 덧붙이지 않는다.
입력에 특별한 근거가 없는 일상은 '식사를 하기로 한다', '휴식을 취하기로 한다'처럼 선택으로 표현한다.
출력 이유를 길게 꾸미거나 이미 했던 일처럼 회고하지 않는다.'''
_NEW = '''intent는 오늘 하려는 활동을 짧게 쓴다. 없는 질병·예약·고장·긴급한 필요를 활동명에 붙이지 않는다.
reasoning은 선택을 뒷받침하는 입력의 한 문장 또는 한 줄을 그대로 복사한다. 요약하거나 설명을 덧붙이지 않는다.
특별한 입력 근거가 없는 식사·휴식 등 자율적인 일상 선택에는 reasoning="오늘의 일상 선택"을 쓴다.
이유를 새로 지어내지 않는다. 복사한 사실이 그 활동·장소·시각을 실제로 뒷받침하는지도 확인한다.
집에서 하는 활동은 중간 이벤트에서도 residence다. 행정동 주소와 집 안의 장소 표기를 혼동하지 않는다.'''
assert _OLD in BASE
SYSTEM_PROMPT = BASE.replace(_OLD, _NEW)
