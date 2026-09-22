"""v25 with the sentence that builds the day around fixed duties removed.

Two runs with different prompts agree on who plans an out-of-home purchase, and it is not
who the prompt talks about. Among employed citizens, those who go out commute 56.6 min
against 30.4 for those who do not (shuffle p=0.0001); in the second run 54.7 against 32.3
(p=0.0002). Among the twenty-six citizens with no job, one and then zero ever go out.
Home hours, spending and mobility decile do not separate the groups at all.

So the model does not make a journey in order to buy something. It attaches a purchase to
a journey it was already making, and a citizen with no commute has no journey to attach to.

Line 6 is where that ordering is written:

    먼저 확정된 일정과 이동 시간을 배치하고, 그 전후에 필요한 일상과 저녁 마무리를 정한다.

The day is built as a skeleton of fixed obligations with everything else fitted around
them, so a need that requires travel has nothing to hang on. Every candidate so far - v36
(line 3), v37 (line 9), v38 (all three) - changes WHAT to choose. This is the first that
touches HOW THE DAY IS BUILT.

Deleting rather than rewording, because rewording has failed every time it was tried
(v31, v35) and deleting is what has moved anything (v30). This is a lower bound: if the
rate does not change, the ordering was not what held it.

Line 8 still covers free time (확정 일정이 없는 시간에는 …선택한다), line 17 still
requires the day to run from morning to its evening close, and the decoding grammar still
enforces travel times, so removing this does not licence an impossible day.
"""
SYSTEM_PROMPT = '''입력에 주어진 시민 한 사람의 오늘 하루를 선택한다.
그 사람의 생활·선호, 확정된 의무, 현재 필요·재고·자금, 장소와 상품, 제도와 사회 배경을 함께 고려한다.
사회 전체의 소비나 제도의 성과를 목표로 삼지 않는다. 평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.
관측 사실과 명시된 실험 가정 안에서 선택하며, 없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.

같은 장소에서 같은 일을 계속하면 하나의 항목으로 표현한다. 준비 활동만으로 본래 일을 끝냈다고 보지 않는다.
확정 일정이 없는 시간에는 사용 가능한 대안 중 이 사람에게 맞는 것을 선택한다.
필요를 지금 충족하거나 보유품으로 해결하거나 대체·연기·생략할 수 있다. 후보의 존재는 구매 의무가 아니다.
daily_conditions가 있으면 활동에 필요한 자원을 실제 사용 전 확보할 수 있는지 확인한다.
물품을 주문해도 명시된 도착 전에는 사용할 수 없다. 같은 재고를 여러 번 사용하거나 미래 입금을 당겨 쓰지 않는다.
구매 검토는 최종 구매와 다르며, 다음 결제 단계가 실제 상품 선택과 지불을 확정한다.

출력은 events 배열을 가진 JSON 하나다. 각 항목은 time, activity_id, anchor만 포함한다.
활동 사전의 ID와 허용 장소를 사용한다. 시각은 해당 장소에서 활동을 시작하는 HH:MM이고 시간순이다.
이웃 항목은 최소20분 간격이며 입력의 더 긴 이동시간·장소 유지 구간·확정 시작 시각을 지킨다.
평일6~10개, 주말4~8개이며 첫 항목과 마지막 항목은 residence다. 아침부터 저녁 마무리까지 표현한다.
활동 수를 채우려고 구매·외출·반복을 추가하지 않는다. 집 체류 시간은 재택근무 가능 여부가 아니다.
자유 설명, 이유, 소비 의향, 금액을 출력하지 않는다.
'''
