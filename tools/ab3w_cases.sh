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
    # [2026-10-06] 사본 P012_ab3w_policy_20261006 = compressed7_main 과 규칙이 같고, 설명만 압축월 숫자로 맞춘 판(본런 전 프롬프트 점검).
    START=2021-10-01; PID=P012
    # [2026-10-08] 신청제: 신청한 사람에게만 적용된 제도다. 사람마다 모델이 신청 여부를 정한다(실행기 6b).
    ENROLL_PIDS=P012
    POLICY=data/experiments/P012_ab3w_policy_20261006.json
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
    # [2026-10-06] 사본 P013_ab3w_policy_20261006 = v53 사본과 규칙이 같고, 설명에서 '서울 시뮬레이션 … 환산' 문장만 뺀 판.
    START=2020-05-11; PID=P013
    POLICY=data/experiments/P013_ab3w_policy_20261006.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector policy"
    CASE_EXPORTS=(EXP_PAYMENT_CHOICE=0 EXP_GRANT_USE=0.5617 EXP_SPREAD_DAYS=1);;
  p010)
    # 민생회복 소비쿠폰 1차(2025-07-21 신청·지급 시작). 사회 배경 없음(평시).
    # 결제 규칙: 신용·체크카드 충전분(수령자의 69.2%, 집행결과 p2)은 사용처 결제에서 자동 차감됐다(정책원문 p3).
    # 건별 선택(엔진 기본값)은 P013 에서 사용 0.1% 로 정책의 결제 규칙과 달랐다 — 자동 차감으로 둔다.
    # 쿠폰은 개인 단위라 P013 의 가구 단위 보정(0.5617)은 쓰지 않는다(엔진 기본 1.0). 하루 인출 상한 없음.
    # [2026-10-06] 사본 P010_ab3w_policy_20261006 = v53 사본과 규칙이 같고, 설명에서 '서울 시뮬레이션 … 소비 10분위' 문장만 뺀 판.
    START=2025-07-21; PID=P010
    POLICY=data/experiments/P010_ab3w_policy_20261006.json
    ENV_PRE=''; ENV_ON=''; ENV_OFF=''
    LEDGERS="sector policy"
    CASE_EXPORTS=(EXP_PAYMENT_CHOICE=0 EXP_SPREAD_DAYS=1);;
  p016)
    # [2026-10-11] 농축산물 할인 1차(대한민국 농할갑시다 2020-07-30~08-09). 참여 유통업체(이마트·롯데마트·농협하나로마트·
    # GS더프레시) 매장에서 국산 신선 농축산물 값의 20%를 결제할 때 깎는다(유통업체마다 1인 최대 1만원).
    # 품목 금액은 모델이 장보기 결제마다 답한다(EXP_PRODUCE_FIELD, 두 갈래 같은 질문). 대형마트 POI 를 더하고
    # (tools/load_p016_marts.py, 카카오맵 2026-10-10) 장보기 후보에 집·직장 3km 안 가까운 대형 형태 매장(대형마트·기업형 슈퍼, 브랜드 무관) 3곳을 넣는다(EXP_MART_REACH).
    # 배경에 2020 장마철(기상청, 중부 6/24 시작)을 둔다(EXP_SEASON_NOTE). 설계 data/experiments/P016_DESIGN_20261011.md.
    START=2020-07-30; PID=P016
    POLICY=data/experiments/P016_ab3w_policy_20261011.json
    ENV_PRE=covid_2021; ENV_ON=covid_2021; ENV_OFF=covid_2021
    LEDGERS="sector"
    CASE_GRAPH_PREP=tools/load_p016_marts.py
    CASE_POWER_GATE=tools/p016_power_gate.py   # 정책 전 주 뒤 C1 검출 최소 차이 관문(설계 문서 3절)
    CASE_EXPORTS=(EXP_PRODUCE_FIELD=1 EXP_MART_REACH=1 EXP_MART_REACH_KM=3 EXP_MART_REACH_N=3 EXP_MART_REACH_FORMATS=hypermarket,ssm EXP_SEASON_NOTE=1);;
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
