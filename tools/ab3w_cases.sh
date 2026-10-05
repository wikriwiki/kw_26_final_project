# 3주 A/B 실행기(tools/run_ab3w.sh)의 정책별 설정. 실행기가 source 한다 — 직접 실행하지 않는다.
#
# 정할 것: START(정책 시작일 = 두 갈래가 갈리는 날), POLICY(정책 있음 쪽에만 넣는 파일, 환경형은 비움),
# ENV_PRE·ENV_ON·ENV_OFF(사회 배경 ID), LEDGERS(뽑을 원장), CASE_EXPORTS(이 정책에만 필요한 결제 규칙).
# 결제 규칙은 두 갈래에 똑같이 넣는다 — 정책 없는 쪽에는 지갑이 없어 닿지 않고, 실행 지문이 같아야 짝이 된다.
# 정책 기간(FROM/UNTIL)은 정책 파일의 effective_from/until 을 그대로 쓴다(실행기가 읽는다).
#
# 발표일 효과: 정책은 복제 뒤 정책 있음 쪽에만 들어가므로, 발표~시행 사이의 선반영은 두 갈래 모두 없다
# (사용자 설계 2026-10-05: 정책 주입 전 1주 공통 → 주입 시점부터 적용/미적용).
case "$AB_CASE" in
  p012)
    # 상생소비지원금(카드 캐시백). 정책 전 주(9월)의 지출은 10월 누적에 섞이지 않는다 — 엔진이 매달 1일에
    # 월 누적을 0 으로 되돌린다(plan_writer, date($today).day = 1). 7일 창은 문턱·한도를 7/31 로 줄인
    # 압축월 파일과 짝이다(창과 파일은 함께 움직인다 — 섞으면 문턱이 창에 비해 너무 높아 정책이 사라진다).
    START=2021-10-01; PID=P012
    POLICY=data/experiments/P012_v53_compressed7_main_20260929.json
    # 시험(AB_TEST_SHORT=1)은 짧은 창을 허용한다 — 문턱에 못 닿아 캐시백이 0 이어도 배관 확인은 된다.
    [[ $AB_POST_DAYS == 7 || ${AB_TEST_SHORT:-0} == 1 ]] || { echo "P012 압축월 파일은 7일 창 전용이다 (AB_POST_DAYS=$AB_POST_DAYS)" >&2; exit 2; }
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector cashback"
    CASE_EXPORTS=();;
  p013)
    # 1차 긴급재난지원금(카드 충전). 사람마다 실제 신청·충전 일정대로 받는 날이 다르고(receipt_schedule,
    # 05-12~06-03), 받은 날부터 정책이 보인다. 카드 충전형이라 사용처 결제에서 자동 차감(EXP_PAYMENT_CHOICE=0,
    # 하루 인출 상한 없음 EXP_SPREAD_DAYS=1). 지원금 카드로 낸 몫 0.5617 은 전국 5/24 누적 소진율(행안부 M2)에
    # 맞춘 보정값이다(P013_indicator_contract.json _calibration) — U1 2주차는 독립 검증에서 뺀다.
    # 1주 창(05-11~05-17)에서 받는 사람은 약 절반이다(일정 비율 0.0956 + 0.5094 x 5/6).
    START=2020-05-11; PID=P013
    POLICY=data/experiments/P013_v53_policy_20261003.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector policy"
    CASE_EXPORTS=(EXP_PAYMENT_CHOICE=0 EXP_GRANT_USE=0.5617 EXP_SPREAD_DAYS=1);;
  p010)
    # 민생회복 소비쿠폰 1차(2025-07-21 신청·지급 시작). 사회 배경 없음(평시).
    # 결제 규칙: 신용·체크카드 충전분(수령자의 69.2%, 집행결과 p2)은 사용처 결제에서 자동 차감됐다(정책원문 p3).
    # 건별 선택(엔진 기본값)은 P013 에서 사용 0.1% 로 정책의 결제 규칙과 달랐다 — 자동 차감으로 둔다.
    # 쿠폰은 개인 단위라 P013 의 가구 단위 보정(0.5617)은 쓰지 않는다(엔진 기본 1.0). 하루 인출 상한 없음.
    START=2025-07-21; PID=P010
    POLICY=data/experiments/P010_v53_policy_20260927.json
    ENV_PRE=''; ENV_ON=''; ENV_OFF=''
    LEDGERS="sector policy"
    CASE_EXPORTS=(EXP_PAYMENT_CHOICE=0 EXP_SPREAD_DAYS=1);;
  p016)
    # 농축산물 할인쿠폰 1차(2020-07-30 시작, 결제 즉시 20%, 1인 누적 1만원). 정답지 C1(대형 유통 5사 신선식품
    # 매출)의 대리 지표만 잴 수 있다 — 그래프에 대형마트·온라인몰이 없고(실제 쿠폰 사용의 80%), 시뮬의 할인은
    # 동네 청과·정육·슈퍼·식료품 가게에서만 일어난다. C2·C3 는 정의대로 셀 수 없다(검수 2026-10-05).
    START=2020-07-30; PID=P016
    POLICY=data/neo4j_load/policies/P016.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector"
    CASE_EXPORTS=();;
  distancing)
    # 서울 사회적 거리두기 2단계 격상(2020-11-24). 정책 있음 = 실제 일정(covid_2021, 11-24 부터 2단계),
    # 정책 없음 = 11-23 의 1.5단계를 이어 간다(covid_2020_hold_1123, 확진 소식은 그날 것). 정답지(서울연구원)는
    # 2단계 이상 vs 그 아래 단계를 비교한다. 단계표에 1.5단계 세부 규칙이 없어 대조 쪽은 단계 이름만 보인다.
    # 규칙은 Stage1 프롬프트로만 전달되고 후보 가게를 거르지 않는다(에이전트 하루가 대개 21시 전에 끝난다).
    START=2020-11-24; PID=''; POLICY=''
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2020_hold_1123
    LEDGERS="sector distancing"
    CASE_EXPORTS=();;
  p014)
    # 서울사랑상품권(2020). 상시 제도라 시작일 2020-09-21 은 반사실 발효일이다(9/14 부터 2단계 구간).
    # 할인율은 발행 구마다(기본 7%, 8개 구 10%), 구별 월 70만원, 사는 구·직장 구 상품권 모두 사용 —
    # 엔진이 결제 즉시 할인으로 회계한다. 모든 사람이 상품권을 쓰는 세상 vs 없는 세상이다.
    # 정답지(조세재정연구원 2020)는 서울 밖 시군구 연간 자료라 업종별 방향만 맞댄다.
    START=2020-09-21; PID=P014
    POLICY=data/experiments/P014_ab3w_policy_20261006.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector"
    CASE_EXPORTS=();;
  p015)
    # 8대 소비쿠폰 재개분 중 그래프에서 잴 수 있는 외식·숙박·체육(2020-10-30 외식 재개, 11-02 체육, 11-04 숙박).
    # 업종별 시작일·조건은 정책 사본의 sectors 에 있고 엔진이 날짜별로 켠다. 여행(온라인 상품만)·공연·영화·전시
    # (가게 없음)·농수산물(이 기간 쿠폰 출처 없음)은 넣지 않았다. 모두 응모·당첨된 세상 vs 쿠폰이 없는 세상이다.
    START=2020-10-30; PID=P015
    POLICY=data/experiments/P015_ab3w_policy_20261006.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector"
    CASE_EXPORTS=();;
  *) echo "정책 설정이 없다: $AB_CASE" >&2; exit 2;;
esac
# 시험 전용: 시작일을 옮긴다(예: P013 은 지급이 05-12 부터라 하루 시험이면 05-13, P015 는 세 업종이 모두 켜진 11-06).
# 실행 기록(run_manifest.json 의 start)에 옮긴 날짜가 남는다. 본런에는 쓰지 않는다.
if [[ -n ${AB_START_OVERRIDE:-} ]]; then
  echo "[시험] 시작일 $START → $AB_START_OVERRIDE" >&2
  START=$AB_START_OVERRIDE
fi
