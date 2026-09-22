"""Same generic choice rules; typed roster and explicit acquisition placement."""
from .asset_transaction_v1 import SYSTEM_PROMPT as BASE

SYSTEM_PROMPT = BASE.split('출력은 actions 배열')[0].replace('먼저 acquire_wallet 행동을 넣고', '먼저 acquisitions에 취득을 선택하고') + '''출력은 acquisitions와 purchases 배열을 가진 JSON 하나다.
purchases에는 모든 입력 이벤트를 원래 순서대로 정확히 한 번씩 넣는다.
각 항목은 id, candidate_id, cash_payment, wallet_spend, reason이다. 구매가 없더라도 항목을 생략하지 않는다.
acquisitions에는 선택한 선불 잔액 취득만 넣는다. 각 offer는 최대 한 번, 여러 단위는 units에 넣는다.
각 취득은 offer_id, units, before_event_id, reason이다. before_event_id는 취득 직후 이어질 이벤트 ID다.
그 이벤트에서 지갑을 쓰려면 그보다 앞에 취득해야 한다. 하루 구매가 끝난 뒤 잔액만 사 둘 때는
before_event_id=null로 표시한다. 취득 목록을 먼저 출력하더라도 실제 취득 시점은 이 필드로 정한다.
필요 없는 취득은 acquisitions=[]로 둔다. 이유에는 확인되지 않은 재고·부족·경험을 사실처럼 쓰지 않는다.
주어진 정보로 선택한 판단과 입력으로 확인한 사실을 구별해 짧게 적는다. 다른 필드는 출력하지 않는다.
'''
