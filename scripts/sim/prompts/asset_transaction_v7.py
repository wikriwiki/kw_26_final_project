"""Same payment choices, with explicit physical-state consistency."""
from .asset_transaction_v6 import SYSTEM_PROMPT as BASE
SYSTEM_PROMPT = BASE + '''
daily_conditions가 있으면 명시된 재고, 활동별 사용량, 상품 수령량과 도착 지연을 함께 확인한다.
지금 사지 않아도 보유량이 충분할 수 있고, 물품을 샀더라도 도착 전 활동에는 사용할 수 없다.
선택된 일정의 실제 자원 사용을 시간순으로 충당해야 한다. 이 단계에서 일정을 삭제하거나 바꿀 수 없다.
선택하지 않은 활동의 필요는 미충족 상태로 남을 수 있으며 이를 지출 할당량으로 해석하지 않는다.
입력에 없는 자원·추가 배송·대체 상품을 만들어 모순을 메우지 않는다.
'''
