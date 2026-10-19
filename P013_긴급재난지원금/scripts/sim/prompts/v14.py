"""Universal planning with relative-time serialization, no repaired choices."""
from .v11 import SYSTEM_PROMPT as _BASE

SYSTEM_PROMPT = _BASE.split("[출력 계약]")[0] + """
[출력 계약: 활동과 시간 간격]
JSON 객체에 start_minute, start, activities, finish, daily_propensity를 작성한다.
start는 집에서 하루를 시작하는 활동, finish는 외출을 마치고 집에서 하루를 마무리하는 활동이다.
activities는 그 사이의 활동을 시간순으로 나열한 배열이다. 집안 활동과 직장 활동도 포함할 수 있다.
start와 finish는 anchor=residence, category=집이다. 중간 활동의 장소와 업종은 위 규칙을 따른다.
모든 활동은 anchor, category, intent, reasoning, trigger를 포함한다. reasoning은 빈 값으로 생략하지 않는다.
필요하면 sub_category를 쓴다. trigger는 appointment, rumor, policy, lifestyle, mood, none 중 하나다.

start_minute는 첫 활동의 시작 시각을 자정부터 지난 분으로 나타낸 정수다.
activities의 각 항목과 finish에는 minutes_after_previous를 넣는다. 이것은 바로 앞 활동 시작부터
이 활동 시작까지 지난 분이며, 최소 20이다. 이동·체류를 포함해 그 사람에게 가능한 간격을 고른다.
실제 각 시각은 start_minute에 앞선 간격들을 누적해서 계산된다. 마지막 시각은 1440 미만이어야 한다.
입력에 정해진 약속 시각이나 영업 조건이 있으면 누적된 실제 시각이 그 조건에 맞도록 간격을 정한다.
일정 형식만을 위해 외출·약속·구매를 추가하지 않는다. 집에서 쉬는 하루도 가능하다.
평일 activities는 4~8개, 주말은 2~6개다. start와 finish를 포함하면 평일 6~10개, 주말 4~8개가 된다.
daily_propensity는 이 사람의 구체적 오늘 일정과 형편에 따른 소비 의향을 0~1로 마지막에 적는다.
이는 지출 금액이나 제도의 효과 크기가 아니다. 거시 결과를 예상해 값을 맞추지 않는다.
JSON 객체 하나만 출력한다. HH:MM 형식이나 events 배열을 함께 쓰지 않는다.
"""
