# 디지털 트윈 시뮬레이터 최종 점검 및 수정 설계 보고서

작성일: 2026-09-14  
검토 기준: 현재 작업 디렉터리 HEAD 3051e95와 미커밋 변경을 포함한 파일 상태  
상태: 수정 전 감사 및 구현 설계. 아래 수정은 아직 적용하지 않았다.

## 1. 결론과 범위

현재 시스템은 페르소나, 공간 후보, 정책지갑, 방문 기억, 다음 날 상태, 사회적 연결을 가진 생성형 ABM 기반이다. 단순 대화 시스템으로 되돌릴 이유는 없다. 그러나 행동 선택, 회계상 보정, 실제로 모의 실행된 사건, 주관적 평가가 같은 필드에 섞여 있어 “어떤 경험 때문에 어떤 집단이 정책에 반응했는가”를 검증하기에는 부족하다.

핵심 수정 대상은 감정 프롬프트가 아니라 **의사결정과 실행 사이의 계약, 사건·상태의 시간적 일관성, 실험 재현성**이다. 회계적으로 올바른 결과가 반드시 인간의 자발적 선택을 나타내지는 않는다.

범위는 소비, 장소·시간, 정책 적용, 초기 상태, 기억, 상호작용, 재실행, 난수, 보고서 및 인터뷰 경로다. 운영 서버의 최신 코드·Neo4j 실제 데이터·GPU 지연시간은 확인하지 않았다. Cypher 발견 사항은 정적 분석이며 운영 DB에서 재현한 결과로 표기하지 않는다. 기존 변경은 보존했고 실행 코드는 수정하지 않았다.

근거 수준:
- **재현**: 로컬 순수 함수 또는 별도 Python 프로세스로 확인.
- **정적 확인**: 실행 코드·쿼리에서 동작을 확인. DB 통합 검증은 남아 있음.
- **설계 한계**: 기존 목적에서는 허용 가능한 가정이지만 새로운 정책 반응 목적에는 부적합.
- P0: 신뢰할 수 있는 후속 실험 전에 해결할 항목. P1: 정책 반응 MVP 전에 해결. P2: 범위 확대 전 해결.

## 2. 실행한 검증

다음 기존 테스트를 실행했다.

```text
python -m pytest tests/unit/sim/test_policy_wallet_neutrality.py tests/unit/sim/test_policy_prompt_timing.py tests/unit/report/test_analytics.py tests/unit/report/test_consistency_and_render.py -q --disable-warnings
63 passed in 4.41s
```

통과는 기존 회계·프롬프트 시점·보고서 계약의 일부가 유지된다는 뜻이다. 정책 입장의 현실성이나 실행 중단 복구까지 검증되었다는 뜻은 아니다.

별도 소규모 입력으로 확인한 결과:

| 사례 | 입력 | 결과 | 해석 |
|---|---|---|---|
| R1 | 구매 12,000원, 사용 가능, 지원금 요청 없음 | 지원금 12,000원 배정 | 사용 여부가 자발적 선택은 아님 |
| R2 | 구매 12,000원, 사용 불가, 지원금 요청, 현금 충분 | 자기자금 구매로 전환 | 에이전트의 전환 동의가 구분되지 않음 |
| R3 | R2와 같지만 현금 0원, 만족도 0.9 | 구매액 0원, 만족도 0.9 유지 | 구매 결과와 평가 불일치 |
| R4 | 같은 agent-day의 error 1행, ok 중복 2행 | agents=3, 해당 분위 agents=2 | 사람 수가 로그 행 수로 부풀려짐 |
| R5 | subsidy 10%, 거래 10,000원 | policy_used=1,000, 거래 policy_spend는 빈 값 | cap 집계와 혜택 원장이 분리됨 |
| R6 | 별도 Python 프로세스에서 같은 agent ID hash | 서로 다른 정수 | hash 기반 fallback seed가 프로세스 간 불안정 |

R4는 임시 디렉터리에 만든 인공 로그로 검사했다. 운영 산출물이 실제로 중복되었다고 주장하는 결과는 아니다. R5는 cap 함수의 결과이며, 아래 실행 경로 검토와 함께 해석한다.

## 3. 발견 사항 및 수정 요구

### F01 — 자발적 사용과 자동 정산의 혼동 [P1, 재현·설계 한계]

소비 함수는 지원금 요청을 진단용으로 집계하고, 사용 가능한 거래에 최대흐름으로 정책 결제액을 최대화한다. 자동정산이라는 가정 아래에는 타당하지만 미사용 의사, 사용법 미인지, 불신, 신청·이용 부담을 표현하지 못한다. 자동배정을 곧 정책 지지로 해석해서는 안 된다.

수정: payment_mode를 automatic 또는 agent_choice로 명시한다. 자동 방식은 현실 정책의 실제 자동결제 제도에 해당할 때 사용한다. 선택 방식은 에이전트가 허용한 수단·정책·한도 안에서만 기존 배정 알고리즘을 사용한다. 최대흐름 계산 자체는 보존할 수 있다.

검증: 선택 방식의 명시적 미사용은 0원 유지. 자동 방식은 정책지갑 우선 정산. 두 모드 모두 자기부담+정책부담=구매액.

근거: [consumption.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/consumption.py:371>).

### F02 — 구매 축소·포기와 만족도 불일치 [P0, 재현]

잔액 부족 시 자기자금 필요액을 배분해 구매금액을 축소하지만 기존 만족도를 그대로 둔다. 0원 구매도 실행 상태 구분 없이 이벤트로 남아 후속 방문·기억의 입력이 될 수 있다. 0원이라고 방문 자체가 없었다고 단정해서도 안 된다.

수정: 방문 여부와 구매 여부를 분리한다. 구매 완료·부분 구매·구매 포기·방문 취소를 별도 실행 결과로 남긴다. 사전 만족도는 expected_satisfaction으로만 보관하고 실행 후 평가와 구별한다. 이전 actual_satisfaction을 이름만 바꿔 새로운 정답으로 사용하지 않는다.

검증: 구매 미실행 시 구매 만족도는 미측정. 방문 경험은 방문이 실제 실행된 경우만 유지. 구매 포기와 방문 포기는 서로 다른 결과를 만든다.

근거: [consumption.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/consumption.py:534>), [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:369>).

### F03 — 가격·소비성향의 중복 작용과 우선순위 부재 [P1, 정적 확인·설계 한계]

Stage2는 가격대를 고려한 절대 소비액을 생성하도록 지시받는다. 후처리는 여기에 가격배율과 Stage1 소비성향 배율을 다시 적용한다. 같은 정책 자극이 활동 개수·금액·성향을 함께 움직일 수 있다. 중복 반영의 실제 크기는 별도 실험이 필요하다. 잔액 부족을 비례 축소로 해결하면 식사와 선택적 소비의 우선순위를 표현하지 못한다.

수정: 금액의 의미와 결정권자를 하나로 정한다. 권장 MVP는 Stage2가 최종 가격 기준 구매 희망액을 선택하고 소비 엔진은 예산·수단 제약을 집행하는 방식이다. 가격을 곱해야 하는 수량 모델은 별도 버전으로 분리한다. 일일 예산은 상한이나 선택 제약으로 작동시키며 같은 동기를 재차 배율로 곱하지 않는다. 필수성·최소 구매가능액·축소 가능 여부를 구분한다.

검증: 동일 의도에 가격 또는 propensity만 바꾼 실험, 유동성 경계값, 필수/선택 소비 간 우선순위 검증. 정책 효과의 부호를 사전에 강제하는 테스트는 만들지 않는다.

근거: [stage2_poi.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/stage2_poi.py:359>), [consumption.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/consumption.py:466>).

### F04 — 정책 지역·유형의 실행 계약 불완전 [P0/P1, 정적 확인·일부 재현]

Dawn은 거주·직장 지역으로 정책을 조회한다. 제한 지갑 생성은 사용처와 업종을 넘기지만 지역 제한 필드는 넘기지 않는다. 소비 함수에는 dong_codes 검사 기능이 있으나 지역 제한 정책이 그 기능에 연결되어야 한다. 수혜자 자격 지역과 실제 사용 가능 지역은 서로 다른 속성이다.

subsidy 경로는 cap 사용액을 증가시키지만 현재 process_one의 자기부담 차감은 policy_spend 합계를 기준으로 한다. R5처럼 cap만 증가하면 혜택을 받았다는 집계와 실제 잔액이 어긋날 수 있다. 지급일·환급일을 갖는 별도 미수 혜택 원장도 해당 경로에서 확인되지 않았다.

수정: 대상자 자격, 사용 지역, 업종, 지급/환급 방식, 한도, 유효기간을 공통 정책 계약으로 정의한다. grant와 subsidy는 각기 실행 어댑터를 갖되 최종 원장은 하나로 통합한다. 즉시 할인과 사후 환급을 분리한다. 지원하지 않는 정책 유형은 프롬프트만으로 실행하지 말고 사전 검증에서 차단한다.

검증: 자격 충족·사용지 불충족 사례, 동/자치구 코드 정규화, cap 도달, 중복 정책, 즉시 할인과 지연 환급의 원장 일치.

근거: [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:361>), [consumption.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/consumption.py:162>), [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:320>).

### F05 — 초기 상태·지급 시점·장기 유동성 [P1, 정적 확인·설계 한계]

초기 상태 loader의 DAY_ZERO 기본값과 시뮬레이터 시작 기본값이 다르다. 시작 전날 State가 없으면 Dawn에는 잔액이 없고 소비 함수는 상한 없는 경로로 갈 수 있으며, 밤에는 기본 잔액 150만 원을 적용한다. 동일 일자 안에서 예산 가정이 달라진다.

grant 지급은 effective_from 당일에 한정된다. 정책 시행 후 시작하는 실험에서 기존 수령 상태를 초기화하지 않으면 미수령 상태가 될 수 있다. 현재 일일 상태 갱신은 잔액 차감 중심이며 정기 소득·고정 지출 경로가 보이지 않는다. 월 지출 리셋도 해당 갱신식에는 없다. 짧은 실험 가정을 장기 실험에 그대로 사용하면 유동성 고갈이 정책 불만처럼 나타날 수 있다.

수정: 실행 전 모든 agent의 전일 State와 정책 권리를 검증한다. 누락을 기본값으로 숨기지 않는다. 기존 지급분은 정책 이력 스냅샷으로 초기화한다. 잔액을 자산·가처분예산 중 무엇으로 보는지 확정하고, 실험 기간에 필요한 소득/고정지출을 반영하거나 짧은 기간 모형이라는 범위를 명시한다.

검증: 시작일 변경, 시행 중 정책으로 시작, 전일 State 누락, 월 경계, 소득 유입 전후.

근거: [08_initial_state.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/neo4j_load/08_initial_state.py:27>), [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:245>), [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:424>).

### F06 — 방문 기억의 지연·마지막 날 누락·재실행 중복 [P0, 정적 확인]

오늘 Dawn/계획 뒤에 어제 방문을 finalize한다. 따라서 어제의 구체적 기억·POI 경험은 오늘 선택에 늦는다. main 종료에는 마지막 날 방문을 finalize하는 호출이 없다. Memory의 결정적 ID는 중복 생성을 막지만 KNOWS_POI의 visit_count와 날짜 배열은 같은 finalize 재실행 시 다시 증가한다.

수정: 당일 실행 후 기억·상태를 완료하고 다음 날 시작한다. event ID별 처리 이력을 두거나 원장에서 방문 집계를 재생성한다. 단순히 함수 위치만 옮기면 기존 복구 스크립트와 중복 실행될 수 있으므로 마이그레이션을 함께 설계한다.

검증: D일 방문이 D+1 선택에 보임, 마지막 날 완결, 동일 일자를 두 번 처리해도 집계 불변, 부분 실패 후 복구 결과=정상 완료 결과.

근거: [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:447>), [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:395>).

### F07 — 긍정 경험을 우선 기억하는 구조 [P1, 정적 확인·설계 한계]

방문 기억 importance는 0.5+1.5×만족도이다. 다른 조건이 같으면 만족도가 높을수록 강하게 회상되고, Dawn은 시간 감쇠를 적용한 Top-N을 사용한다. 부정 정책 경험을 보고 싶은 시스템에서 저만족 경험을 구조적으로 덜 회상시키는 가정이다. 반대로 무조건 부정 경험을 우선하는 것도 정답은 아니다.

수정: 중요도를 긍정성에서 분리한다. 목표 방해, 비용, 새로움, 반복, 개인 관련성과 강도로 구성하고 경험 valence는 별도 보관한다. 계수는 공개하고 민감도 분석한다. 동일 경험 반복을 매번 독립 증거처럼 더하지 않는다.

검증: 중요도가 같은 긍정·부정 경험의 검색 기회, 오래된 경험 감쇠, 누적 사건 중복 방지.

근거: [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:376>), [dawn_context.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/dawn_context.py:88>).

### F08 — 시간·공간 실행과 후보 선택의 한계 [P1, 정적 확인·설계 한계]

시각 단조 증가 보정은 이동 가능성을 검증하지 않는다. duration_min은 병합 시 None이다. 후보 거리와 Huff 기반 지역 선택은 있으나 연속 이동·체류·고정 일정 제약의 실행과는 다르다. 동일 지역·업종의 같은 날 이벤트에 후보를 나눠 중복 방문을 막는 것은 인간의 선택이 아니라 후보 생성 규칙이다. Night 만남 노출도 같은 동·시간대 단위여서 실제 같은 장소에서 만났다는 증거로 쓰기 어렵다.

수정: 최소 이동시간 추정과 체류시간, 업무·약속 시간, 도달 가능성 판정을 도입한다. 후보에는 반복 방문과 구매하지 않음도 가능한 선택으로 둔다. 만남 후보 생성과 실제 만남 사건을 구분한다. 대기·재고·혼잡은 검증할 정책에 필요할 때만 구현한다.

검증: 연속 이동 불가능 일정, 체류 중 중복 만남, 단골 재방문, 접근 가능한 대안 없음. 모델이 지원하지 않는 “대기했다/늦었다” 등의 표현은 근거로 채택하지 않는다.

근거: [stage1_intent.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/stage1_intent.py:195>), [stage2_poi.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/stage2_poi.py:155>), [night_interaction.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_interaction.py:55>).

### F09 — 환경 사실·인지·시스템 오류의 혼합 [P1, 정적 확인·설계 한계]

활성 정책 조회가 곧 인지 상태에 반영되고 사용 가능 여부가 후보에 제공된다. 그 상태에서 잘못 출력한 결제 요청을 실제 시민의 오해나 거절 경험으로 기록하면 생성 오류가 사회현상으로 전환된다. 발표 전/후와 시행 전/후도 현재 활성 정책 구간만으로 충분히 구분되지 않는다.

수정: world facts, 개인 observation, belief를 구분한다. 오류 보정은 시스템 로그에만 남긴다. 개인의 오해를 모형화하려면 사전 정의된 정보 노출·인지 과정으로 발생시킨다. 발표 시점의 예상 반응과 시행 후 경험 기반 반응은 분리하여 평가한다.

검증: 미노출 정책 인용 차단, 미래 사건 참조 차단, malformed 출력이 불만 기억을 만들지 않음, 발표 전 혜택 사용 불가.

근거: [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:214>), [dawn_context.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/dawn_context.py:142>).

### F10 — 사회적 기억 내용과 수신 반응 누락 [P1, 정적 확인·설계 한계]

분류기의 reasoning이 수신자의 rumor 기억으로 저장된다. 실제 전달 내용과 수신자의 수용 여부가 분리되지 않는다. 회피 조언도 추천으로 분류하지만 추천 후속 경로는 기존 POI affinity를 일괄 증가시킨다. 회피 조언을 긍정적 선호 강화와 동일하게 처리하는 모순이다. affinity가 현재 선택에 미치는 크기는 별도 확인이 필요하지만 저장 상태의 의미는 이미 어긋난다.

수정: 전달 내용, 출처 사건, 대상, 방향, 수신 여부를 구분한다. 발신자의 평가와 수신자의 입장을 동일시하지 않는다. 수신자의 해석은 다음 기존 의사결정 호출에 반영할 수 있으며 장문 대화 호출은 필수가 아니다. 생성된 initiator/recipient가 입력 pair와 일치하는지도 검증한다.

검증: 부정 조언이 긍정 affinity를 올리지 않음, 미수신 내용은 개인 기억에 없음, 타 agent ID 출력 차단.

근거: [night_intent_llm.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_intent_llm.py:124>), [night_intent_llm.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_intent_llm.py:506>).

### F11 — 사회적 후속 링크가 배치 전체 한 건으로 제한 [P0, 정적 확인]

추천 링크 쿼리와 약속 링크 쿼리는 UNWIND 이후 ORDER BY poi.id LIMIT 1을 적용한다. 개별 입력 행마다 하나를 고르는 구문이 아니라 배치 전체 결과를 한 행으로 제한한다. Conversation/Memory는 여러 건인데 POI 후속 연결은 일부만 생길 수 있다.

수정: 행별 서브쿼리 또는 쓰기 전 유일한 POI ID 해석으로 변경한다. 이름 중복은 임의 선택하지 않고 명시적으로 처리한다.

검증: 서로 다른 추천 3건·약속 3건 입력에 각각 3개 연결, POI 미매칭 1건이 다른 행을 없애지 않음. Neo4j 통합 테스트가 필요하다.

근거: [night_intent_llm.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_intent_llm.py:534>), [night_intent_llm.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_intent_llm.py:562>).

### F12 — 부분 실패·resume·실험 격리 [P0, 정적 확인]

Plan 교체, INCLUDES 쓰기, 기억, State가 여러 DB 호출로 나뉜다. 실행 중단 시 부분 완료가 가능하다. Night2는 해당 날짜 Conversation이 50건 이상이면 전체 skip한다. 50건 미만으로 재실행되면 UUID Conversation과 recipient별 초기화되는 Memory 순번이 충돌·중복을 만들 여지가 있다. 실패 agent가 남아도 main은 다음 날로 진행한다.

출력 디렉터리는 분리할 수 있으나 Plan/State 등 주요 ID는 agent+day이고 Night 조회는 날짜 기반이다. 같은 DB에서 동일 날짜 실험을 여러 번 실행하면 별도 DB/확실한 초기화 없이는 격리되지 않는다.

수정: run_id 또는 실험별 DB로 격리한다. agent-day와 social-pair별 완료 상태 및 입력 해시를 기록한다. 숫자 임계값 대신 처리 대상 집합의 완료를 검사한다. 원장은 append-only, 파생 집계는 재생성 가능하게 하고 완료 manifest를 검증한 뒤 다음 날을 시작한다. 실패자를 제외하고 계속하는 모드는 별도 설정과 결측 보고가 있어야 한다.

검증: 각 저장 단계 강제 중단 후 복구, 대화 일부 실패, 동일 날짜 두 실험 격리, 실패 집단의 비율과 특성 보고.

근거: [plan_writer.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/plan_writer.py:84>), [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:794>), [night_intent_llm.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_intent_llm.py:589>).

### F13 — 난수·fallback·호출비용 재현성 [P1, 일부 재현·정적 확인]

Stage2 fallback의 random.Random(hash(aid))는 Python 프로세스별 hash 무작위화에 영향을 받는다. Night 매칭은 seed 인자를 지원하지만 run_day 호출에서 전달하지 않는다. 실패 재시도는 온도가 바뀌고, 리뷰 조회는 추가 pass를 만들 수 있다. 따라서 “항상 두 번 호출”, “같은 agent면 같은 난수”는 보장되지 않는다.

수정: run seed에서 agent/day/stage별 seed를 안정 해시로 파생한다. 후보 정렬·동률 처리도 고정한다. 모델·프롬프트·샘플링 설정·재시도·fallback·리뷰 pass·토큰을 기록한다. backend의 seed 지원만으로 완전 결정론을 약속하지 않는다. 외부 리뷰·가격 조회는 실험 시작 시 고정한다.

검증: 프로세스 재시작과 worker 수 변경 비교, RNG 결정론 검증, LLM은 복수 seed 변동 보고. 공통난수는 에이전트별 독립 스트림으로 구성해 분기 이후 전역 난수 소비가 섞이지 않게 한다.

근거: [stage2_poi.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/stage2_poi.py:777>), [night_interaction.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/night_interaction.py:392>), [run_simulation.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/run_simulation.py:809>).

### F14 — 보고서 agent 수의 중복·실패 행 포함 [P0, 재현]

read_metrics는 유효 JSON 행마다 agents를 증가시키며 status 필터와 aid별 중복 제거가 없다. 소비 이벤트 집계의 중복 처리와 별개 문제다. runner의 resume 정리가 일부 경로를 완화하지만 분석 함수 자체의 입력 계약을 보장하지는 못한다.

수정: (run_id, aid, day)로 최종 유효 결과를 선택한다. 실패·결측·중복 수는 별도 품질 지표로 반환한다. 예정 모집단, 완료 모집단, 응답 모집단을 구분한다.

검증: error+ok, ok 중복, malformed, aid 누락, 결측 분위, 중복 시간순 우선순위. R4의 유효 agent 수는 1이어야 한다.

근거: [analytics.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/report/analytics.py:233>).

### F15 — 정책 입장·생활 만족·소비 효과의 혼동 [P1, 정적 확인·설계 한계]

기존 인터뷰 대표 추출은 정책 사용과 방문 만족도로 긍정/부정/중립을 구성한다. 웹 인터뷰는 요약 기록에서 답변을 새로 생성한다. 보고서의 소비 DID는 업종 간 전후 변화 분석이며 정책 찬반 지표가 아니다. 집계 수치가 정확해도 정책 입장 검증으로 사용할 수 없다.

수정: 개인별 정책 인지, 입장, 감정, 행동, 경험을 별도 변수로 저장한다. 미인지·미응답을 중립으로 넣지 않는다. 인터뷰는 저장된 입장과 근거의 표현 계층으로 제한하고 재질문 답변을 시뮬레이션 상태에 역주입하지 않는다. 비대상 업종도 대체소비의 영향을 받을 수 있으므로 기존 DID를 자동으로 인과 효과로 해석하지 않는다.

검증: 혜택 사용+반대, 미사용+찬성, 미인지, 불편 경험+정책 목적 동의 사례를 허용한다. 근거 인용률은 추적 가능성 지표이며 인간 반응 정확도와 별개다.

근거: [interview_agent.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/interview_agent.py:306>), [interview.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/web/api/interview.py:159>), [analytics.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/report/analytics.py:425>).

### F16 — 환경·페르소나 자료의 출처와 검증 범위 [P1/P2, 정적 확인·설계 한계]

가격에는 메뉴 실측·지역 평균·해시 fallback이 혼재하고, 사용처는 브랜드·업종 근사를 포함한다. 합성 페르소나의 상세 묘사가 정책 가치관의 관측 근거를 대신하지는 못한다. 자료 시점이 정책 백테스트 이후이면 미래 정보가 환경 데이터로도 들어갈 수 있다.

수정: 가격·자격·페르소나 각 속성에 source, observed_at, estimated 여부를 보관한다. 실제 자료가 없는 가치관은 확정 사실이 아닌 가정으로 취급하고 같은 인구집단 안에서도 이질성을 유지한다. 나이·성별만으로 찬반을 고정하지 않는다. 정책명 비식별화만으로 사전학습 정보 누수가 해결되었다고 주장하지 않는다.

검증: 실측/추정 비율, 환경 가정 교체, 정책명 비노출, 무정책 placebo, 미관측 정책 holdout. 인간 정답 자료가 없으면 표현 타당성만 보고하고 실제 예측 성능이라고 부르지 않는다.

근거: [poi_price.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/poi_price.py:1>), [coupon_eligibility.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/coupon_eligibility.py:9>), [dawn_context.py](<C:/Users/srdyh/OneDrive/사진/바탕 화면/agent_simulate/kw_26_final_project/.claude/worktrees/silly-gagarin-0b181a/scripts/sim/dawn_context.py:34>).

## 4. 목표 구조: 호출 단계가 아니라 데이터 계약을 분리

### 4.1 최소 데이터 계약

| 개체 | 필수 의미 | 생성 책임 |
|---|---|---|
| Decision | 목표·방문 후보·구매 희망액·수단 선호·제약·허용 대처 | 기존 Stage1/2 |
| ExecutionEvent | 시도/실행 여부·구매 상태·금액·정책 적용/거절 사유·실제 시간 | 환경 엔진 |
| Observation | 에이전트가 알게 된 사건·전달 정보·출처·인지 시각 | 노출 규칙 |
| State | 잔액·정책지갑·일정·신체 상태 등 다음 행동의 제약 | 상태 전이 |
| PolicyAppraisal | 정책별 입장·관련 감정·짧은 주장·근거 ID·측정 시각 | 다음 기존 의사결정 호출 |
| SystemDiagnostic | 재시도·fallback·규칙 위반·자료 부족·강제 보정 | 실행기 |
| RunManifest | 버전·seed·자료·프롬프트·모델·표본·완료 상태 | 실행기 |

모든 사건은 run_id, agent_id, event_id, occurred_at을 갖는다. 관측은 observed_at을 별도로 갖는다. 입장은 as_of 시각을 갖고 해당 시점 이전에 관측한 근거만 참조한다. 코드가 정한 사실과 LLM이 해석한 주장은 필드로 구분한다.

### 4.2 하루 실행 순서

1. 전날 완료 manifest 확인 → 지급·만료·정기 수입 등 당일 경계 사건 처리.
2. 기존 Stage1: 과거 관측 기반으로 오늘 의도와 필요한 정책 입장 변화 출력.
3. 후보 생성: 시간·지역·가용성 검증. 반복 방문과 방문 포기도 허용.
4. 기존 Stage2: 실제 후보에서 선택. 구매 희망액·수단·제한된 대처 조건 확정.
5. 코드 실행: 시간순 방문·결제·포기 판정, 사건과 원장 생성.
6. 방문 기억·상태 완료. 기존 Night 상호작용은 완료된 사건만 읽음.
7. 전달 사건 저장, 다음 날 observation으로 반영. 일자 완료 manifest 확정.

오늘 경험의 주관적 해석이 다음 날 반영되는 일일 모형임을 명시한다. 마지막 날 해석이 필요하면 마지막 기존 호출 일부를 terminal 평가로 대체하거나 별도 종결 비용을 예산에 넣는다. 측정되지 않은 마지막 날 입장을 자동 생성하지 않는다.

### 4.3 비용 원칙

- 계획된 Stage1/Stage2 횟수는 유지하되 기존 리뷰 pass·재시도·Night 비용까지 총량에 포함한다.
- 사건 하나마다 LLM을 호출하지 않는다. 관련 관측을 제한된 길이로 묶어 다음 기존 호출에 전달한다.
- 감정 설명은 장문 CoT가 아니라 외부에 표현할 짧은 주장과 근거 ID다.
- 코드가 결정할 수 있는 가격·자격·회계·시간 판정에는 LLM을 사용하지 않는다.
- 허용 대처 예: 자기부담이 사전 허용 한도 이하면 구매, 아니면 포기. 새로운 장소 재탐색은 시간·선택 계약이 지원할 때만 실행한다.
- 호출 수가 같아도 토큰·DB 쓰기·최대 지연이 늘 수 있으므로 소규모 실측 후 규모를 정한다. A100 한 대라는 사실만으로 완료 시간을 단정하지 않는다.

## 5. 호환성과 마이그레이션

1. v1 기록은 legacy_generated라는 출처를 붙여 보존한다. 기존 만족도를 실행 후 평가로 재해석하거나 실패 사건을 사후 복원하지 않는다.
2. v2 Decision/ExecutionEvent를 분리 저장하고 기존 events.jsonl에는 호환 view를 제공한다. 소비 보고서에는 실제 구매 완료/부분 구매 금액만 포함한다.
3. 방문 수·구매 수·구매 포기 수를 분리한다. 기존 방문 통계와 수치가 달라질 수 있음을 버전으로 명시한다.
4. 기존 policy_spend 집계는 확정 원장에서 파생한다. policy_used, cap, wallet, 보고서 금액이 같은 원장을 읽도록 한다.
5. 방문 집계 재생성 및 Night 복구 스크립트는 새 멱등 계약으로 갱신한다. 기존 실행 도중 새 상태 스키마를 혼합하지 않는다.
6. 모델·프롬프트 최적화 전에 실행 의미를 고정한다. 구조 변경 전후 결과를 같은 실험으로 합치지 않는다.
7. 자동지갑·선택지갑 모드는 실험 조건이다. 전환으로 기존 지갑 테스트의 기대값이 달라질 수 있으므로 모드별 테스트로 분리한다.

## 6. 수정 순서와 완료 기준

### 단계 A — 실험 무결성 복구

대상: F02, F04 회계 불일치, F06, F11, F12, F14, 초기 State 누락 검사.

완료 기준:
- 동일 일자 재실행 시 원장·기억·방문·대화 결과 불변.
- 부분 실패 후 복구가 정상 완료 결과와 일치.
- 모든 결제의 원장 보존식 성립.
- 마지막 날 포함 모든 사건·상태 완료.
- agent 수 집계와 완료 모집단 일치.
- DB 통합 테스트에서 3건 배치 연결 3건 보장.

### 단계 B — 정책 하나의 실행 의미 확정

대상: F01, F03, F05, F08, F09.

범위는 사용처 제한 지원금 한 종류로 시작한다. 다른 정책으로 자동 일반화되었다고 주장하지 않는다. 무정책과 정책 조건에서 동일 초기 상태·자료를 사용한다.

완료 기준:
- 미인지·사용·미사용·자기부담 전환·구매 포기의 차이를 기록으로 구분.
- 시스템 오류가 시민 경험이 되지 않음.
- 금액과 가격 결정권자가 하나임.
- 실제 실행 가능한 일정과 기록된 행동이 일치.

### 단계 C — 경험 기반 반응과 사회적 전파

대상: F07, F10, F15, F16.

완료 기준:
- 정책 입장의 근거가 개인의 당시 관측에 존재.
- 찬반을 혜택 사용·생활 만족으로 대체하지 않음.
- 동일 경험에도 이질적 반응을 허용.
- 전달된 정보와 수신자의 해석을 구분.
- 인터뷰 재질문이 상태와 평가 정답을 바꾸지 않음.

### 단계 D — 일반화·성능·자원 검증

대상: F13 및 전체 평가.

권장 예비 규모는 20~50명×3일, 이후 200명×7일이다. 이는 시작점이며 실측 속도와 오류율로 조정한다. 전체 인구 확대 전에 여러 seed에서 반응·소비·결측·호출 비용을 확인한다.

평가를 네 축으로 분리한다:
- 실행 정확성: 회계·시간·멱등·관측 시점 불변식.
- 행동 타당성: 실제 자료와 지출·방문·활동 분포 비교.
- 반응 타당성: 동일 문항·시점·집단의 실제 설문과 입장 분포 비교. 관측 표본의 불확실성도 함께 보고.
- 비용: agent-day당 실제 호출·토큰, 리뷰/재시도 비율, LLM/DB 시간, 완료율.

근거 연결률은 설문 예측 성능을 대신하지 않는다. 시행 후 모의 경험을 발표 직후 설문 정답과 비교하지 않는다. holdout은 정책 또는 정책군 단위로 분리하고 튜닝 결과를 보고 평가 기준을 다시 고르지 않는다.

## 7. 10월 23일 이전 일정 제안

아래는 보장 일정이 아니라 단계별 통과 조건이 있는 실행안이다.

| 기간 | 산출물 |
|---|---|
| 9/14~9/20 | 데이터 계약 확정, 단계 A 무결성 수정·통합 검증 |
| 9/21~9/27 | 단일 정책 실행 모형, 소비·시간·인지 호환 검증 |
| 9/28~10/4 | 경험 기반 입장·사회적 전파·인터뷰 연결 |
| 10/5~10/11 | 소규모 복수 seed, 실제 설문과 정합 가능한 평가셋 고정 |
| 10/12~10/18 | holdout 평가, 성능·자원 측정, 결함 수정 |
| 10/19~10/22 | 버전 동결, 재현 산출물·시연·최종 결과 정리 |

단계 A가 늦어지면 정책 종류·인구 수·사회적 전파 범위를 줄인다. 실행 무결성과 정답의 시점 일치는 줄이지 않는다. 모든 정책을 재현하는 도시 전체 디지털 트윈을 이번 기한의 완료 기준으로 삼지 않는다.

## 8. 최종 판단

유지할 기반은 공간·페르소나·정책지갑·기억·보고서의 원장 중심 집계다. 바꿀 중심은 실행 엔진과 데이터 의미다. “풍부한 말을 하는 에이전트”는 완료 기준이 아니다. **어떤 조건에서 어떤 선택을 했고, 무엇이 실제로 모의 실행되어 어떤 관측과 다음 행동으로 이어졌는지 재현 가능한 에이전트**가 완료 기준이다.

본 보고서는 로컬 코드와 제한된 함수 실행에 대한 최종 감사 결과다. 운영 데이터 품질, Cypher 통합 동작, 모델의 인간 반응 예측력은 구현 후 별도 검증해야 한다.

