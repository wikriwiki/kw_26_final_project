"""Best formatting baseline with verbatim evidence and explicit event semantics."""
from .v11 import SYSTEM_PROMPT as BASE, format_dawn_blocks

_OLD = 'reasoning은 그 활동을 선택한 입력 근거를 짧게 쓰며 집·직장 이벤트도 빈 문자열이나 null로 생략하지 않는다.'
_NEW = '''reasoning에는 해당 선택을 뒷받침하는 입력의 한 문장 또는 한 줄을 그대로 복사한다.
설명을 보태거나 입력의 의미를 바꾸지 않는다. 별도의 사실 근거가 없는 식사·휴식 등 일상 선택은
reasoning="오늘의 일상 선택"으로 쓴다. intent에도 없는 질병·약 복용·예약·고장·긴급 필요를 붙이지 않는다.
복사한 사실이 실제 선택과 연결되는지 확인한다. 인용했더라도 관련 없는 근거는 쓰지 않는다.'''
assert _OLD in BASE
SYSTEM_PROMPT = BASE.replace(_OLD, _NEW).replace('"reasoning":"집에서 오늘 일정을 준비한다."', '"reasoning":"오늘의 일상 선택"') + '''
장소 구분의 낱개 필드 예시: 집에서 독서하는 활동은 {"anchor":"residence","category":"집"},
사무실 안에서 일하는 활동은 {"anchor":"workplace","category":"직장"}이다.
집이나 직장의 행정동 주소가 제공되어도 내부 활동을 외출 zone으로 바꾸지 않는다.
동네 산책은 실제 외출이다. 집 안에서 쉬는 활동과 구별한다.
time은 intent 활동을 실제 시작하는 시각이다. 입력에 시작 시각이 명시된 약속·출근은 해당 시각에
그 활동의 이벤트를 둔다. 그보다 일찍 이동하거나 기다리는 활동은 별도 이벤트로 표현한다.
trigger=policy는 입력의 공공제도·방역 등 공공 규제가 선택에 영향을 준 경우다.
개인의 취향, 일상 식사, 출근 시간과 회사 내부 근무 조건 자체는 이 의미의 policy가 아니다.
'''
