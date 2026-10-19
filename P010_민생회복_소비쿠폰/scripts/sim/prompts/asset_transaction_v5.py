"""Clarify empty input collections and keep financing reasoning concise."""
from .asset_transaction_v4 import SYSTEM_PROMPT as BASE

SYSTEM_PROMPT = BASE + '''
입력 구조: offers는 선불 잔액을 새로 취득하는 제안만 담는다. events 안의 candidates는
상품·서비스 구매 후보이며 offers와 별개다. offers={}이면 선불 취득 제안이 없고
acquisition_units={}이다. 그래도 상품 후보는 있을 수 있다. wallet_lots={}이면 현재
보유한 선불 잔액이 없다. 빈 목록이나 객체를 입력 오류·숨겨진 자금·암묵적 제안으로 해석하지 않는다.
길게 입력을 재서술하지 말고 현재 재원, 취득 비용, 시간순 구매의 자금 잔여를 짧게 점검한다.
구매 후보가 있어도 현재 재원으로 결제할 수 없으면 구매를 선택하지 않는다.
확정된 구매 의무도 없는 자금이나 차입을 만들어 내는 근거가 되지 않는다.
'''
