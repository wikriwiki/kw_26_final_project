"""v25 with the two edits that still stand, on a baseline that can finally show a difference.

Every prompt comparison so far ran on a degenerate baseline where 77 percent of all
purchases were one online detergent, forced by the resource scenario. With the meal
need and the repair loop the baseline buys groceries, orders in and eats, and the
distance to each citizen's own assigned category mix fell from 0.80 to 0.53.

The two edits carried here are the ones with evidence behind them:
  v30 took the word "purchase" out of the padding guard. Purchases rose 2.21 SE with
      the activity count unchanged and the zone share flat.
  v33 put resource acquisition into the ordering rule. Neutral on the gate, harmless.
Both edit a different single sentence, and neither names a policy, a sector, a
direction or a number.
"""
SYSTEM_PROMPT = '''입력에 주어진 시민 한 사람의 오늘 하루를 선택한다.
그 사람의 생활·선호, 확정된 의무, 현재 필요·재고·자금, 장소와 상품, 제도와 사회 배경을 함께 고려한다.
사회 전체의 소비나 제도의 성과를 목표로 삼지 않는다. 평균 지출과 과거 비중은 오늘 채울 할당량이 아니다.
관측 사실과 명시된 실험 가정 안에서 선택하며, 없는 의무·휴무·재택 허가·부족·질병·과거 경험을 만들지 않는다.

먼저 확정된 일정과 이동 시간을 배치하고, 자원이 있어야 하는 일은 그 자원을 확보하는 일과 함께 배치하거나 자원이 필요 없는 다른 방법을 고르고, 그 전후에 필요한 일상과 저녁 마무리를 정한다.
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
활동 수를 채우려고 활동을 추가하지 않는다. 집 체류 시간은 재택근무 가능 여부가 아니다.
자유 설명, 이유, 소비 의향, 금액을 출력하지 않는다.
'''
